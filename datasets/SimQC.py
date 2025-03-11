#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling SimQC dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import yaml
from multiprocessing import Lock


# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import *
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *
from utils.metrics import fast_confusion

from datasets.common import grid_subsampling
from utils.config import bcolors
import OSToolBox as ost

# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class SimQCDataset(PointCloudDataset):
    """Class to handle SimQC dataset."""

    def __init__(self, config, split="train", data_path=None, balance_classes=False, intensity_max=1025):
        PointCloudDataset.__init__(self, "SimQC")

        ##########################
        # Parameters for the files
        ##########################
        
        # Dataset folder
        self.path = data_path

        # Type of task conducted on this dataset
        self.dataset_task = "slam_segmentation"

        # Training or test set
        self.set = "val" if split=="validation" else split
        # Get a list of sequences
        self.data_path = data_path if split=="test" else join(self.path, self.set) 
        # List all files in each sequence
        self.path_files = ost.getFileBySubstr(self.data_path,'.ply')

        self.sequences = []
        for file in self.path_files:
            a = ost.pathRelative(file,2)
            a = os.path.join(a.split("/")[0],a.split("/")[1])
            self.sequences += [a]

        self.intensity_max=intensity_max
        print("Normalization of intensity data at : ", self.intensity_max)

        self.sequences = list(set(self.sequences))

        self.frames = []
        for seq in self.sequences:
            velo_path = join(self.data_path, "sequences", seq)
            frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.ply')])
            self.frames.append(frames)
        

        ###########################
        # Object classes parameters
        ###########################

        # Read labels
        config_file = "SimQC.yaml"
        with open(config_file, "r") as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc["labels"]
            learning_map = doc["learning_map"]
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map.items()}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = []    # [void]

        ##################
        # Other parameters
        ##################
        
        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        ##################
        # Load calibration
        ##################

        # Init variables
        self.all_inds = None
        self.class_proportions = None
        self.class_frames = []
        self.val_confs = []

        # Load everything
        self.load_calib_poses()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials
        self.potentials = torch.from_numpy(
            np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1
        )
        
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        self.balance_classes = balance_classes

        # Choose batch_num in_R and max_in_p depending on validation or training
        if self.set == "train":
            self.batch_num = config.batch_num
            self.max_in_p = config.max_in_points
            self.in_R = config.in_radius
        else:
            self.batch_num = config.val_batch_num
            self.max_in_p = config.max_val_points
            self.in_R = config.val_radius

        # shared epoch indices and classes (in case we want class balanced sampler)
        if config.epoch_steps is not None :
            if split == "train":
                N = int(np.ceil(config.epoch_steps * self.batch_num * 1.1))
            else:
                N = int(np.ceil(config.validation_size * self.batch_num * 1.1))
        else :
            N=len(self.path_files)
            
        self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.epoch_inds = torch.from_numpy(np.zeros((N,), dtype=np.int64))
        self.epoch_labels = torch.from_numpy(np.zeros((N,), dtype=np.int32))
        self.epoch_i.share_memory_()
        self.epoch_inds.share_memory_()
        self.epoch_labels.share_memory_()

        self.worker_waiting = torch.tensor(
            [0 for _ in range(config.input_threads)], dtype=torch.int32
        )
        self.worker_waiting.share_memory_()
        self.worker_lock = Lock()
        
        # self.writer = SummaryWriter(f'runs/testSumWrtr')
        # self.evaluator = iouEval(n_classes=config.num_classes, ignore=self.ignored_labels)
        
        #check
        print("Loaded ",self.set," dataset, found : ",self.__len__()," clouds")
        
        return

    def __len__(self):
        """
        Return the length of data here
        """
        return np.sum([len(f) for f in self.frames])
    
    def qcsfTransform(self,points,drop_ratio=0.1,min_cube_drop=2,max_cube_drop=6,cube_size=4,sigma_cube_size=1,sigma_jittering=0.05,max_intensity=1025):
        ##point drop
        #random point drop        
        dropout_ratio =  np.random.random()*drop_ratio
        drop_idx = np.where(np.random.random((points.shape[0]))<=dropout_ratio)[0]
        points=np.delete(points,drop_idx,0)        
        
        #cuboid drop
        points=ost.randomCuboidDrop(points,min_cube_drop,max_cube_drop,cube_size,sigma_cube_size)
        
        ##position
        # translation :
        translation_xy=np.random.uniform(-20,0,2)
        translation_xyz=np.append(translation_xy,np.random.uniform(-10,10))
        points[:,:3]+=translation_xyz
        scale=[1,1,1]
        #scene flip
        if np.random.random() > 0.5:
            # print("flipped X")
            points[:,0,...] = -1 * points[:,0,...]
            scale[0]=-1
        if np.random.random() > 0.5:
            # print("flipped Y")
            points[:,1,...] = -1 * points[:,1,...]
            scale[1]=-1
        
        # rotation :
        theta = np.random.uniform(0, 2*np.pi)
        R=[[np.cos(theta),-np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0], 
                            [0, 0, 1]]
        points=ost.rotationZ(points, theta)
        # jittering : 
        points=ost.jittering(points,sigma_jittering)
        
        return points,R,scale

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """
        # print("BAAAAATCH I", batch_i)
        batch_iList = []

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
        r_inds_list = []
        r_mask_list = []
        val_labels_list = []
        batch_n = 0
        
        # print("start at ",int(self.batch_limit))
        while True:

            with self.worker_lock:

                # Get potential minimum
                ind = int(self.epoch_inds[self.epoch_i])
                
                # Update epoch indice
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[0]):
                    self.epoch_i -= int(self.epoch_inds.shape[0])
            
            s_ind, f_ind = self.all_inds[ind]
            
            #########################
            # Merge n_frames together
            #########################

            # Initiate merged points
            merged_points = np.zeros((0, 3), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int32)
            merged_coords = np.zeros((0, 3), dtype=np.float32)

            # Get center of the first frame in world coordinates
            p_origin = np.zeros((1, 4))
            p_origin[0, 3] = 1
            
            #pose0 = self.poses[s_ind][f_ind]
            #p0 = p_origin#.dot(pose0.T)[:, :3]
            #p0 = np.squeeze(p0)
            o_pts = None
            o_labels = None

            num_merged = 0

            # Path of points and labels
            seq_path = join(self.data_path,"sequences",self.sequences[s_ind])
            velo_file = join(seq_path, self.frames[s_ind][f_ind] + ".ply")
            # velo_file = "/home/reza/PHD/Data/SimQC_sample/train/sequences/td_5/ab_41/5_41_12.ply"
            # print("velo",velo_file)

            # Read points et data augment if not test
            if self.set == "test":
                data = read_ply(velo_file)
                x = data["x"]
                y = data["y"]
                z = data["z"]
                i = data["intensity"]
                
                points=np.c_[x, y, z]
                intensity=ost.featureAugmentation(i[:, np.newaxis],0,self.intensity_max)
                try : 
                    sem_labels = data["class"].astype(np.int32)
                except :
                    sem_labels = np.zeros((data.shape[0],), dtype=np.int32)
            else:
                data = read_ply(velo_file)
                x = data["x"]
                y = data["y"]
                z = data["z"]
                i = data["intensity"]
                c = data["class"]
                data = np.c_[x, y, z, i, c]
                
                data,R,scale=self.qcsfTransform(data)
                data=ost.featureAugmentation(data,3,self.intensity_max)

                points=data[:,:3]
                intensity = data[:,3]
                intensity = intensity[:, np.newaxis]
                sem_labels = data[:,4]
                sem_labels = sem_labels.astype(np.int32)
            
            
            # In case of validation, keep the original points in memory
            if self.set in ["validation", "test"]:
                o_pts = points.astype(np.float32)
                o_labels = sem_labels.astype(np.int32)
                
            #Conv center 
            wanted_ind = np.random.choice(points.shape[0])
            p0 = points[wanted_ind]
            
            # mask outside raduis points               
            mask = np.sum(np.square(points[:] - p0), axis=1) < self.in_R ** 2
            mask_inds = np.where(mask)[0].astype(np.int32)    
                            
            # Shuffle points
            rand_order = np.random.permutation(mask_inds)
            points = points[rand_order]
            sem_labels = sem_labels[rand_order]
            intensity = intensity[rand_order]
                
            # Increment merge count
            merged_points = np.asarray(points, dtype=np.float32)
            merged_labels = np.hstack((merged_labels, sem_labels))
            merged_coords = np.hstack((merged_points, intensity))
            
            #########################
            # Merge n_frames together
            #########################
            
            # Subsample merged frames
            in_pts, in_fts, in_lbls = grid_subsampling(merged_points, features=merged_coords, labels=merged_labels, sampleDl=self.config.first_subsampling_dl)
            
            # Number collected
            n = in_pts.shape[0]
            
            # Safe check
            if n < 10:
                continue

                        
            # Before augmenting, compute reprojection inds (only for validation and test)
            if self.set in ["validation", "test"]:

                # get val_points that are in range
                radiuses = np.sum(np.square(o_pts - p0), axis=1)
                reproj_mask = radiuses < (0.99 * self.in_R) ** 2

                # Project predictions on the frame points
                search_tree = KDTree(in_pts, leaf_size=50)
                proj_inds = search_tree.query(o_pts[reproj_mask, :], return_distance=False)
                proj_inds = np.squeeze(proj_inds).astype(np.int32)
                
                #fake transfo
                scale=1
                R=0
            else:
                proj_inds = np.zeros((0,))
                reproj_mask = np.zeros((0,))
                
            # Color augmentation
            #if np.random.rand() > self.config.augment_color:
            #    in_fts[:, 3:] *= 0

            # Stack batch
            p_list += [in_pts]
            f_list += [in_fts]
            l_list += [np.squeeze(in_lbls)]
            fi_list += [[s_ind, f_ind]]
            p0_list += [p0]
            s_list += [scale]
            R_list += [R]
            r_inds_list += [proj_inds]
            r_mask_list += [reproj_mask]
            val_labels_list += [o_labels]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                # print("BROKE AT ",batch_n," over ",int(self.batch_limit))
                break

        ###################
        # Concatenate batch
        ###################
        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        frame_inds = np.array(fi_list, dtype=np.int32)
        frame_centers = np.stack(p0_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)
        
        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 2:
            # Use original height coordinate
            stacked_features = np.hstack((stacked_features, features[:, 3:4]))
            
        elif self.config.in_features_dim == 3:
            # Use height + reflectance
            stacked_features = np.hstack((stacked_features, features[:, 2:]))
        elif self.config.in_features_dim == 4:
            # Use all coordinates
            stacked_features = np.hstack((stacked_features, features[:3]))
        elif self.config.in_features_dim == 5:
            # Use all coordinates + reflectance
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError(
                "Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)"
            )

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #
        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points, stacked_features, labels.astype(np.int64), stack_lengths)
        # name="/home/reza/PHD/Sum24/SimQC/KPConv/BBBBBBBBB.ply"
        # if os.path.isfile(name):
        #     name="/home/reza/PHD/Sum24/SimQC/KPConv/CCCCC.ply"
        # ost.writePly(name,[stacked_points,stacked_features,labels],["x","y","z","c","d","lab"])

        # Add scale and rotation for testing
        input_list += [
            scales,
            rots,
            frame_inds,
            frame_centers,
            r_inds_list,
            r_mask_list,
            val_labels_list,
            velo_file,
        ]

        return [self.config.num_layers] + input_list

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        for seq in self.sequences:
            seq_folder = join(self.path, self.set,"sequences", seq)

        ###################################
        # Prepare the indices of all frames
        ###################################

        seq_inds = np.hstack(
            [np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)]
        )
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

        ################################################
        # For each class list the frames containing them
        ################################################

        if self.set in ["train", "validation"]:

            class_frames_bool = np.zeros((0, self.num_classes), dtype=np.bool)
            self.class_proportions = np.zeros((self.num_classes,), dtype=np.int32)

            for s_ind, (seq, seq_frames) in enumerate(zip(self.sequences, self.frames)):

                seq_stat_file = join(self.path,self.set,"seq_stat",seq,"stats_single.pkl")

                # Check if inputs have already been computed
                if exists(seq_stat_file):
                    # Read pkl
                    with open(seq_stat_file, "rb") as f:
                        seq_class_frames, seq_proportions = pickle.load(f)

                else:

                    # Initiate dict
                    print("Preparing seq {:s} class frames. (Long but one time only)".format(seq))

                    # Class frames as a boolean mask
                    seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)

                    # Proportion of each class
                    seq_proportions = np.zeros((self.num_classes,), dtype=np.int32)

                    # Sequence path
                    seq_path = join(self.path, self.set, "sequences", seq)

                    # Read all frames

                    for f_ind, frame_name in enumerate(seq_frames):
                        # Path of points and labels
                        label_file = join(seq_path, frame_name + ".ply")
                        
                        # Read labels
                        sem_labels = read_ply(label_file)["class"].astype(np.int32)
                        # sem_labels = self.learning_map[sem_labels]

                        # Get present labels and there frequency
                        unique, counts = np.unique(sem_labels, return_counts=True)

                        # Add this frame to the frame lists of all class present
                        frame_labels = np.array(
                            [self.label_to_idx[l] for l in unique], dtype=np.int32
                        )
                        seq_class_frames[f_ind, frame_labels] = True

                        # Add proportions
                        seq_proportions[frame_labels] += counts

                    # Save pickle
                    ost.createDir(ost.pathBranch(seq_stat_file))
                    with open(seq_stat_file, "wb") as f:
                        pickle.dump([seq_class_frames, seq_proportions], f)

                class_frames_bool = np.vstack((class_frames_bool, seq_class_frames))
                self.class_proportions += seq_proportions

            # Transform boolean indexing to int indices.
            self.class_frames = []
            for i, c in enumerate(self.label_values):
                if c in self.ignored_labels:
                    self.class_frames.append(torch.zeros((0,), dtype=torch.int64))
                else:
                    integer_inds = np.where(class_frames_bool[:, i])[0]
                    self.class_frames.append(torch.from_numpy(integer_inds.astype(np.int64)))

        # Add variables for validation
        if self.set == "validation":
            self.val_points = []
            self.val_labels = []
            self.val_confs = []

            for s_ind, seq_frames in enumerate(self.frames):
                self.val_confs.append(
                    np.zeros((len(seq_frames), self.num_classes, self.num_classes))
                )

        return




# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class SimQCSampler(Sampler):
    """Sampler for SimQC"""

    def __init__(self, dataset: SimQCDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == "train":
            self.N = dataset.config.epoch_steps if dataset.config.epoch_steps is not None else len(dataset.path_files)
        else:
            self.N = dataset.config.validation_size if dataset.set != "test" else len(dataset.path_files)

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """
        if self.dataset.set != "test":
            if self.dataset.balance_classes:
    
                # Initiate current epoch ind
                self.dataset.epoch_i *= 0
                self.dataset.epoch_inds *= 0
                self.dataset.epoch_labels *= 0
    
                # Number of sphere centers taken per class in each cloud
                num_centers = self.dataset.epoch_inds.shape[0]
    
                # Generate a list of indices balancing classes and respecting potentials
                gen_indices = []
                gen_classes = []
                for i, c in enumerate(self.dataset.label_values):
                    if c not in self.dataset.ignored_labels:
    
                        # Get the potentials of the frames containing this class
                        class_potentials = self.dataset.potentials[
                            self.dataset.class_frames[i]
                        ]
    
                        if class_potentials.shape[0] > 0:
    
                            # Get the indices to generate thanks to potentials
                            used_classes = self.dataset.num_classes - len(
                                self.dataset.ignored_labels
                            )
                            class_n = num_centers // used_classes + 1
                            if class_n < class_potentials.shape[0]:
                                _, class_indices = torch.topk(
                                    class_potentials, class_n, largest=False
                                )
                            else:
                                class_indices = torch.zeros((0,), dtype=torch.int64)
                                while class_indices.shape[0] < class_n:
                                    new_class_inds = torch.randperm(
                                        class_potentials.shape[0]
                                    ).type(torch.int64)
                                    class_indices = torch.cat(
                                        (class_indices, new_class_inds), dim=0
                                    )
                                class_indices = class_indices[:class_n]
                            class_indices = self.dataset.class_frames[i][class_indices]
    
                            # Add the indices to the generated ones
                            gen_indices.append(class_indices)
                            gen_classes.append(class_indices * 0 + c)
    
                            # Update potentials
                            update_inds = torch.unique(class_indices)
                            self.dataset.potentials[update_inds] = torch.ceil(
                                self.dataset.potentials[update_inds]
                            )
                            self.dataset.potentials[update_inds] += torch.from_numpy(
                                np.random.rand(update_inds.shape[0]) * 0.1 + 0.1
                            )
    
                        else:
                            error_message = "\nIt seems there is a problem with the class statistics of your dataset, saved in the variable dataset.class_frames.\n"
                            error_message += "Here are the current statistics:\n"
                            error_message += "{:>15s} {:>15s}\n".format(
                                "Class", "# of frames"
                            )
                            for iii, ccc in enumerate(self.dataset.label_values):
                                error_message += "{:>15s} {:>15d}\n".format(
                                    self.dataset.label_names[iii],
                                    len(self.dataset.class_frames[iii]),
                                )
                            error_message += "\nThis error is raised if one of the classes is not ignored and does not appear in any of the frames of the dataset.\n"
                            raise ValueError(error_message)
    
                # Stack the chosen indices of all classes
                gen_indices = torch.cat(gen_indices, dim=0)
                gen_classes = torch.cat(gen_classes, dim=0)
    
                # Shuffle generated indices
                rand_order = torch.randperm(gen_indices.shape[0])[:num_centers]
                gen_indices = gen_indices[rand_order]
                gen_classes = gen_classes[rand_order]
    
                # Update potentials (Change the order for the next epoch)
                # self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
                # self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)
    
                # Update epoch inds
                self.dataset.epoch_inds += gen_indices
                self.dataset.epoch_labels += gen_classes.type(torch.int32)
    
            else:
                # Initiate current epoch ind
                self.dataset.epoch_i *= 0
                self.dataset.epoch_inds *= 0
                self.dataset.epoch_labels *= 0
    
                # Number of sphere centers taken per class in each cloud
                num_centers = self.dataset.epoch_inds.shape[0]
    
                # Get the list of indices to generate thanks to potentials
                if num_centers < self.dataset.potentials.shape[0]:
                    _, gen_indices = torch.topk(
                        self.dataset.potentials, num_centers, largest=False, sorted=True
                    )
                else:
                    gen_indices = torch.randperm(self.dataset.potentials.shape[0])
                    while gen_indices.shape[0] < num_centers:
                        new_gen_indices = torch.randperm(
                            self.dataset.potentials.shape[0]
                        ).type(torch.int32)
                        gen_indices = torch.cat((gen_indices.long(), new_gen_indices.long()), dim=0)
                    gen_indices = gen_indices[:num_centers]
    
                # Update potentials (Change the order for the next epoch)
                self.dataset.potentials[gen_indices] = torch.ceil(
                    self.dataset.potentials[gen_indices]
                )
                self.dataset.potentials[gen_indices] += torch.from_numpy(
                    np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1
                )
    
                # Update epoch inds
                self.dataset.epoch_inds += gen_indices
        else:
                self.dataset.epoch_inds= np.arange(self.N)      # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N


class SimQCCustomBatch:
    """Custom batch definition with memory pinning for SimQC"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.points = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.neighbors = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.pools = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.upsamples = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.lengths = [
            torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]
        ]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_centers = torch.from_numpy(input_list[ind])
        ind += 1
        self.reproj_inds = input_list[ind]
        ind += 1
        self.reproj_masks = input_list[ind]
        ind += 1
        self.val_labels = input_list[ind]
        ind += 1
        self.fname = input_list[ind]
        
        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.frame_inds = self.frame_inds.pin_memory()
        self.frame_centers = self.frame_centers.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.frame_inds = self.frame_inds.to(device)
        self.frame_centers = self.frame_centers.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements("points", layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements("neighbors", layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements("pools", layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == "points":
            elements = self.points
        elif element_name == "neighbors":
            elements = self.neighbors
        elif element_name == "pools":
            elements = self.pools[:-1]
        else:
            raise ValueError("Unknown element name: {:s}".format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == "pools":
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0 : i0 + length]
                    if element_name == "neighbors":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == "pools":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def SimQCCollate(batch_data):
    return SimQCCustomBatch(batch_data)


