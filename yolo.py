#Name of Model, optimizer etc.. for Graph file's name
model_name = 'ResNet'
optimizer_name = 'SGD'
import numpy as np
import matplotlib.pyplot as plt
import timeit
import torch
from torchinfo import summary 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.metrics import accuracy_score
from models_cifar_10.densenet import DenseNet121
from models_cifar_10.resnet import ResNet18,ResNet34
#from models_cifar100.densenet import DenseNet121

from functions import *
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

from resnet20 import resnet20
def main():
    #Model chosen from Vgg, Densenet, Resnet
    #model = DenseNet121(100)
    model = resnet20()

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    
    #Import pretrained and add MLP to go from Cifar100 to Cifar10
    #path_model = 'Models/DenseNet_Adam_epochs_100.pth'
    #model = transfer_learning(model,path_model)
    
    model,device = to_device(model)
    #path_model = 'Models/Accuracy_90/SGD_epochs_101_acc94.43.pth'
    #model = load_weights(model,path_model)

    #HyperParameters
    Niter,Bsize,lr = 400 , 32, 0.01#0.01

    earlystop_flag = False
    #Data import
    

    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True) 


    train_model, loss_list_train,loss_list_valid, accuracy_list, best_accuracy, test_accuracy, _ = training_model(model, trainloader,validloader,testloader ,device,lr,Niter,earlystop_flag,optimizer_name)
    print(f'The best accuracy on validation for the saved model is: {best_accuracy}%')

    plot1(loss_list_train,loss_list_valid, accuracy_list, test_accuracy,Niter,lr)


def pruning():
    #Model configuration
    model = ResNet18()
    path_model = 'Models/Accuracy_90/SGD_epochs_300_acc94.63.pth'
    model = load_weights(model,path_model)
    model,device = to_device(model)

    #HyperParameters
    Niter,Bsize,lr = 15 , 32, 0.01
    amount = 0.1
    conv2d_flag,linear_flag,BN_flag = True,True,False
    earlystop_flag = False
    
    model,parameters_to_prune = global_pruning(model,amount,device,conv2d_flag,linear_flag,BN_flag)

    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True)
    print_prune_details(model)
    

    train_model, loss_list_train,loss_list_valid, accuracy_list, best_accuracy, test_accuracy, _ = training_model(model, trainloader,validloader,testloader ,device,lr,Niter,earlystop_flag,parameters_to_prune,optimizer_name)
    print(f'The best accuracy on validation for the saved model is: {best_accuracy}%')
    print_prune_details(model)
    plot1(loss_list_train,loss_list_valid, accuracy_list, test_accuracy,Niter,lr)
if __name__=='__main__':
    main()
    #pruning()