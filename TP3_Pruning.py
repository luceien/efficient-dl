############################################# IMPORT ################################################
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary 
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np

from functions import *

def train(model_origin,trainloader,validloader,testloader,device,lr,EPOCHS,earlystop_flag,binary_flag):
    #Function that simplifies readability and output
    model,train_loss_origin,valid_loss_origin,origin_accuracy,\
    best_accuracy,test, execution_time = training_model(model_origin,
                                                    trainloader,
                                                    validloader,
                                                    testloader,
                                                    device,
                                                    learning_rate=lr,
                                                    EPOCHS=EPOCHS,
                                                    earlystop=earlystop_flag,
                                                    patience=30,
                                                    binary=binary_flag)
                                                    
    return [model,train_loss_origin,valid_loss_origin,origin_accuracy,best_accuracy,test, execution_time]

def main():

    #Importing the model
    model_name = 'DenseNet121'
    optimizer_name = 'Adam'
    #model_origin = DenseNet121(10)

    #DenseNet121bis : Model with less filters in block of DenseNet
    model_origin = DenseNet121bis(10)
    #print(summary(model_origin, input_size=(32, 3, 32, 32)))
    model = deepcopy(model_origin)

    model = device(model)

    from models_cifar_10.densenet import DenseNet121, DenseNet121bis
    from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
    from torch.utils.data.dataloader import DataLoader

    ############################################# PARAMETERS ################################################
    EPOCHS,Bsize,lr = 10, 32, 0.001
    #Apply binarization of weight/early stopping during training
    binary_flag, earlystop_flag = False, False
    #Flag for layers to prune
    conv2d_flag,linear_flag,BN_flag = True,False,False
    #Number of different pruning steps
    Niter = 21 #21
    #% of pruning apply to the model
    amount = 0.6

    ################################################ DATA #####################################################
    #Loading data of minicifar
    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True)

    #Plot the accuracy givent pruning %
    #pruning_accuracy(model,trainloader,validloader,testloader,lr,Niter=Niter,EPOCHS=EPOCHS)
    
    #Prune the model and print details of layers pruning
    pruned_model = global_pruning(model,amount,device,conv2d_flag,linear_flag,BN_flag)
    print_prune_details(model)

    #Train the original model
    origin = train(model_origin,trainloader,validloader,testloader,device,lr,EPOCHS,earlystop_flag,binary_flag)
    #Train the pruned model
    pruned = train(pruned_model,trainloader,validloader,testloader,device,lr,EPOCHS,earlystop_flag,binary_flag)

    plot2(origin,pruned,Niter,lr,amount)

if __name__=='__main__':
    main()
    