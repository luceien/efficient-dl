#%%
model_name = 'DenseNet121'
optimizer_name = 'Adam'





#%%
from sklearn.metrics import accuracy_score
from models_cifar_10.densenet import DenseNet121

from scipy.__config__ import get_info
from torchvision.datasets import CIFAR10
import numpy as np 
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models

import torchvision.transforms as transforms
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.quantization



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = DenseNet121()
state_dict = torch.load('./Models/DenseNet121_adam_epochs_50.pth',  map_location=device)['net']
model.load_state_dict(state_dict)
#model.eval()

#%%


def train_model(model, train_loader, valid_loader, test_loader, learning_rate, EPOCHS, patience=30):
  loss_list_train = []
  loss_list_valid = []
  accuracy_list = []
  early_stop = [1000,0]

  #Optimizer (Adam better)
  if optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  #Loss
  criterion = nn.CrossEntropyLoss()

  for epoch in range(EPOCHS):
    print(f"Epoch n° : {epoch}/{EPOCHS} commencée")
    loss_train = 0
    
    #Training
    for i, data in tqdm(enumerate(train_loader, 0)):  
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        #Clear the gradients
        optimizer.zero_grad()
        
        #Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        #Calculate gradients
        loss.backward()
        #Update weights
        optimizer.step()
        loss_train += loss.item()
        nb_batch = i     

    loss_list_train.append(loss_train/i)
    print("\n","loss par epoch train =",loss_train/(nb_batch+1))

    #Validation 
    loss_valid = 0
    model.eval()
    for i, data in tqdm(enumerate(valid_loader, 0)):  
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        target = model(inputs)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate Loss
        loss_valid += loss.item()
        nb_batch = i  
        #print("\n","loss par Batch valid=",loss.item(),"\n")       
    
    loss_list_valid.append(loss_valid/i)
    print("\n","loss par epoch valid =",loss_valid/(nb_batch+1))

    #Early-Stopping
    if loss_valid/(nb_batch+1) < early_stop[0]:
        early_stop[0] = loss_valid/(nb_batch+1)
        early_stop[1] = 0

    else:
        early_stop[1] += 1

    print(f'Validation loss did not change for {early_stop[1]} epochs')

    #Test
    correct = 0
    total = 0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in tqdm(test_loader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))
    accuracy_list.append(accuracy)

    #End training if early stop reach the patience
    if early_stop[1] == patience:
        break 

    
  return model, loss_list_train,loss_list_valid, accuracy_list 



#%%

#Part 1 - Quantization to half and integer precision




