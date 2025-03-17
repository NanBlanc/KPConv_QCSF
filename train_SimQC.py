#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on SimQC dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.SimQC import * 
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN

import wandb

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class SimQCConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SimQC'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 70.0
    val_radius = 70.0
    n_frames = 1
    max_in_points = 12000
    max_val_points = 12000

    # Number of batch (cannot be somthing else than 1)
    batch_num = 1
    val_batch_num = 1
    #accumulation : (to update network after multiple clouds)
    accumulation_steps = 50

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.1

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 4.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.0

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 0.01              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 10

    # Learning rate management
    learning_rate = 1e-3
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    
    
    grad_clip_norm = 100.0

    # Number of steps per epochs if None, will use all data
    epoch_steps = None

    # Number of validation examples per epoch
    validation_size = 500

    # Number of epoch between each checkpoint
    checkpoint_gap = 10    
    
    #Process setup
    input_threads=0
    
    #Experiment setup
    checkpoint=None # if strat from previous checkpoint give path ortherwise None
    saving = True
    saving_path = ost.createDirIncremental("/home/reza/PHD/Sum24/SimQC/KPConv/logs/run")
    project_name= "KPConv-QCSF"
    run_name="train_e10_noint_notr"
    
    use_transform=False
    use_intensity=False
    
    if use_intensity :
        in_features_dim = 2
    else :
        in_features_dim = 1
    

    class_w=[0.01,0.45,0.45,0.09]


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    ############################
    # Initialize the environment
    ############################
    
    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Initialize configuration class
    config = SimQCConfig()

    # init expermient WandB registering
    run = wandb.init(project=config.project_name, dir=config.saving_path, name=config.run_name)

    
    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize datasets
    data_path="/home/reza/PHD/Data/SimQC"
    
    training_dataset = SimQCDataset(config, split='train', data_path=data_path, balance_classes=False)
    val_dataset = SimQCDataset(config, split='val', data_path=data_path, balance_classes=False)

    # Initialize samplers
    training_sampler = SimQCSampler(training_dataset)
    val_sampler = SimQCSampler(val_dataset)
    
    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                  batch_size=1,
                                  sampler=training_sampler,
                                  collate_fn=SimQCCollate,
                                  num_workers=config.input_threads,
                                  pin_memory=True)
        
    val_loader = DataLoader(val_dataset,
                              batch_size=1,
                              sampler=val_sampler,
                              collate_fn=SimQCCollate,
                              num_workers=config.input_threads,
                              pin_memory=True)
        
    # train_batch = next(iter(training_loader))
    # test_batch = next(iter(test_loader))
    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)


    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=config.checkpoint)
    print('Done in {:.2f}s\n'.format(time.time() - t1))
    
    
    print('\nStart training')
    print('**************')
    
    # Training    
    trainer.train(net, training_loader, val_loader, config)
    
    print('DONE')
