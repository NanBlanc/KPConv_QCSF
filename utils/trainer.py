#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
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
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv

# For loging
import wandb
import pandas as pd

import OSToolBox as ost
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=True, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters()
                        if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nUsing GPU")
        else:
            self.device = torch.device("cpu")
            print("\nNOT Using GPU")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime(
                    'results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        self.best_mVal_IoU = 0
        self.cm_train=ost.ConfusionMatrix(4, ["sol","ab","pm","vb"],99)
        self.accumulation=config.accumulation_steps

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write(
                    'epochs steps LRec out_loss offset_loss train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        #config.epoch_steps = 70
        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(config.max_epoch):
            # Remove File for kill signal
            if epoch == config.max_epoch -1 and exists(PID_file):
                remove(PID_file)
            
            loss_record=[]
            self.step = 0            
            for batch_idx,batch in enumerate(training_loader):

                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)
                loss = net.loss(outputs, batch.labels)
                acc = net.accuracy(outputs, batch.labels)
                
                softmax = torch.nn.Softmax(1)
                stk_probs = softmax(outputs).cpu().detach().numpy()
                preds=np.argmax(stk_probs,axis=1).astype(np.int32)

                self.cm_train.add_batch(batch.labels.cpu().detach().numpy(),preds)

                
                t += [time.time()]
                
                loss = loss / self.accumulation
                loss_record.append(loss.item())
                # Backward + optimize
                loss.backward()


                
                if (batch_idx + 1) % self.accumulation == 0:
                    if config.grad_clip_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm) # What is this?
                        torch.nn.utils.clip_grad_value_(
                            net.parameters(), config.grad_clip_norm)
                    self.optimizer.step()
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                        
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(self.device)
    
                    t += [time.time()]
    
                    # Average timing
                    if self.step < 2:
                        mean_dt = np.array(t[1:]) - np.array(t[:-1])
                    else:
                        mean_dt = 0.9 * mean_dt + 0.1 * \
                            (np.array(t[1:]) - np.array(t[:-1]))
    
                    # Console display (only one per second)
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'e{:03d}-i{:04d} => L={:.3f} LRec={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                        print(message.format(self.epoch, self.step+self.accumulation,
                                              loss.item(),
                                              np.mean(loss_record),
                                              100*acc,
                                              1000 * mean_dt[0],
                                              1000 * mean_dt[1],
                                              1000 * mean_dt[2]))
                        
                    wandb.log({
                        "loss" : np.mean(loss_record), 
                        "output loss" : net.output_loss, 
                        "reg loss" : net.reg_loss, 
                        "accuracy" : acc})
                    
                    # Log file
                    if config.saving:
                        with open(join(config.saving_path, 'training.txt'), "a") as file:
                            message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                            file.write(message.format(self.epoch,
                                                      self.step,
                                                      np.mean(loss_record),
                                                      net.output_loss,
                                                      net.reg_loss,
                                                      acc,
                                                      t[-1] - t0))
                    
                    self.step += self.accumulation
                    del outputs, loss, acc, batch
                    loss_record=[]

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                print()
                for param_group in self.optimizer.param_groups:
                    print("Lr avant maj: ", param_group['lr'])
                    param_group['lr'] *= config.lr_decays[self.epoch]
                    print("Lr apres maj: ", param_group['lr'])
                print()

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}
#
#                # Save current state of the network (for restoring purposes)
                if self.epoch == (config.max_epoch-1):
                    checkpoint_path = join(config.saving_path, "checkpoints", 'lastchkp.tar'.format(self.epoch))
                else:
                    checkpoint_path = join(config.saving_path, "checkpoints", 'chkp.tar'.format(self.epoch))
                torch.save(save_dict, checkpoint_path)
#
#                # Save checkpoints occasionally
#                if (self.epoch + 1) % config.checkpoint_gap == 0:
#                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
#                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            print("Validation step")
            self.validationQCSF(net, val_loader, config, self.epoch, save_dict)
            net.train()
            
            

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------
    def validationQCSF(self, net, val_loader, config: Config, epoch, save_dict):
        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Create folder for validation predictions
        if not exists(join(config.saving_path, 'val_preds')):
            makedirs(join(config.saving_path, 'val_preds'))

        # initiate the dataset validation containers
        val_loader.dataset.val_points = []
        val_loader.dataset.val_labels = []
        
        cm=ost.ConfusionMatrix(val_loader.dataset.num_classes, ["sol","ab","pm","vb"],99)
        
        # Start validation loop
        for i, batch in tqdm(enumerate(val_loader)):


            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stk_probs = softmax(outputs).cpu().detach().numpy()
            labels_list = batch.labels.cpu().numpy().astype(np.int32)
            torch.cuda.synchronize(self.device)
            
            preds=np.argmax(stk_probs,axis=1).astype(np.int32)
            
            cm.add_batch(labels_list,preds)
            
            points=batch.points[0].cpu().numpy()
            # ost.writePly("/home/reza/PHD/Sum24/SimQC/KPConv/vaal.ply",[points,stk_probs,preds,labels_list],["x","y","z","sol","ab","pm","vb","pred","lab"])

        print("Validation ious : ",end=" ")
        ious=cm.class_IoU(show=True)
        mVal_IoU=ious[0]
        # Saving (optionnal)
        if config.saving:            
            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line ='{:.3f} '.format(ious[0])+" "+" ".join(['{:.3f} '.format(i*100) for i in ious[1]])+ '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)
            
            # Save checkpoints when mVal_IoU is the best
            if mVal_IoU > self.best_mVal_IoU:
                self.best_mVal_IoU = mVal_IoU
                checkpoint_directory = join(config.saving_path, 'checkpoints')
                # 'chkp_{:04d}.tar'.format(self.epoch + 1))
                checkpoint_path = join(
                    checkpoint_directory, 'chkp_best_mVal_IoU_epoch_'+str(self.epoch)+'.tar')
                torch.save(save_dict, checkpoint_path)

                epoch_IoU_file = join(config.saving_path,
                                      "checkpoints/epoch_mVal_IoUs.txt")

            # Logging data in WandB
            print("Training ious : ",end=" ")
            ious_train=self.cm_train.class_IoU()
            print("clearing train ious and wandb registering")
            wandb.log({
                "mIoU_train": ious_train[0], 
                "sol_train": ious_train[1][0]*100, 
                "ab_train": ious_train[1][1]*100, 
                "pm_train": ious_train[1][2]*100, 
                "vb_train": ious_train[1][3]*100, 

                                
                "mVal_IoU": mVal_IoU,
                "sol_val": ious[1][0]*100, 
                "ab_val": ious[1][1]*100, 
                "pm_val": ious[1][2]*100, 
                "vb_val": ious[1][3]*100, 
                "best_mVal_IoU": self.best_mVal_IoU})
            self.cm_train.clear()

        return

