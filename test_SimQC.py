#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
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
from torch.utils.data import Sampler 

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


    ###########################
    # Call the test initializer
    ###########################
# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    #SELECT CHKPT
    # chkp = "/home/reza/PHD/Sum24/SimQC/KPConv/logs/run_train_e9_int_tr/checkpoints/chkp_best_mVal_IoU_epoch_6.tar"
    chkp="/home/reza/PHD/Sum24/SimQC/KPConv/logs/run_train_e9_noint_notr/checkpoints/chkp_best_mVal_IoU_epoch_6.tar"
    
    #SELECT OUTPUT FOLDER
    output_path="/home/reza/PHD/Sum24/SimQC/KPConv/results"
    output_path=ost.createDirIncremental(output_path+"/inference")
    
    #INTENSTITY MAX VALUE AND DATA
    data_path="/home/reza/PHD/Data/SimQC/test"
    intensity_max=1025
    # data_path="/home/reza/PHD/Data/ALSlike_full_seq/test"
    # intensity_max=125
  

    ############################
    # Initialize the environment
    ############################

    # Initialize configuration class
    config = Config()
    config.load(ost.pathBranch(chkp,2))  
    
    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ##################################
    # Change model parameters for test
    ##################################
    #I DONT

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')


    print(config.dataset)
    # Initiate dataset
    if config.dataset == 'SimQC':
        test_dataset = SimQCDataset(config, split="test", data_path=data_path, intensity_max=intensity_max)
        test_sampler = SimQCSampler(test_dataset)
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)
    
    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SimQCCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()

    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)


    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chkp, output_path=output_path)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Testing
    tester.QCSF_test(net, test_loader, config)