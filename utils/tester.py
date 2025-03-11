#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
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


# Basic libs
import sys
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

import OSToolBox as ost
from tqdm import tqdm

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True, output_path=None, save_clouds=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")
        self.output_path=output_path
        print("Test results saved at :",self.output_path)
        self.save_clouds=save_clouds
        if self.save_clouds:
            print("\tClouds and their prediction will be saved")
            
        self.cm=ost.ConfusionMatrix(4, ["sol","ab","pm","vb"],99)

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def QCSF_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Test saving path
        test_path = self.output_path
        cloud_path=ost.createDir(test_path+"/predictions")

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        for i, batch in tqdm(enumerate(test_loader)):

            # New time
            t = t[-1:]
            t += [time.time()]

            if i == 0:
                print('Done in {:.1f}s'.format(t[1] - t[0]))

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            t += [time.time()]

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            s_points = batch.points[0].cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            torch.cuda.synchronize(self.device)
            
            reproj_inds=batch.reproj_inds[0]

            # Predicted labels
            preds = np.argmax(stacked_probs, axis=1).astype(np.int32)

            # Targets
            targets = batch.labels.cpu().numpy()
            # print(np.unique(targets,return_counts=True))

            # Confs
            self.cm.add_batch(targets,preds)
            
            #save cloud
            original=ost.readPly(batch.fname)
            
            ori_preds = preds[reproj_inds]
            ori_targets = targets[reproj_inds].astype(np.int32)
            ori_proba = stacked_probs[reproj_inds]
            
            ost.writePly(cloud_path+"/"+ost.pathLeafExt(batch.fname),[original,ori_preds,ori_targets,ori_proba],["x","y","z","int","lab","pred","lab_reproj","proba_sol","proba_ab","proba_pm","proba_vb"])
            
            
        self.cm.printPerf(test_path)
        
            

        return





















