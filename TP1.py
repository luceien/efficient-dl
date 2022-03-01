#Name of Model, optimizer etc.. for Graph file's name
model_name = 'DenseNet'
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


def main():
    #Model chosen from Vgg, Densenet, Resnet
    #model = DenseNet121(100)
    model = ResNet18()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    
    #Import pretrained and add MLP to go from Cifar100 to Cifar10
    #model = transfer_learning(model)
    
    model,device = to_device(model)
    #HyperParameters
    Niter,Bsize,lr = 50 , 32, 0.01

    earlystop_flag = False
    #Data import
    from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
    from torch.utils.data.dataloader import DataLoader

    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True) 

    #best_accuracy = pretrained_model(test_loader=testloader,num_classes=100)
    #train_model, loss_list_train,loss_list_valid, accuracy, best_accuracy = train_model(model, trainloader,validloader,testloader,lr,Niter)

    train_model, loss_list_train,loss_list_valid, accuracy_list, best_accuracy, test_accuracy, _ = training_model(model, trainloader,validloader,testloader ,device,lr,Niter,earlystop_flag)
    print(f'The best accuracy on validation for the saved model is: {best_accuracy}%')

    plot1(loss_list_train,loss_list_valid, accuracy_list, test_accuracy,Niter,lr)
    
if __name__=='__main__':
    main()