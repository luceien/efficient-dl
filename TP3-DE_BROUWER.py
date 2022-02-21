#%%
model_name = 'DenseNet121'
optimizer_name = 'Adam'
learning_rate = 0.001
optimizer_name = "Adam"
batch_size = 32
n_epochs = 20
path_model = 'Models/DenseNet121_Adam_epochs_50.pth'





#%%
from xml.dom import ValidationErr
from sklearn.metrics import accuracy_score
from models_cifar_10.densenet import DenseNet121

from scipy.__config__ import get_info
from torchvision.datasets import CIFAR10
import numpy as np 
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models
#from torchinfo import summary

import torchvision.transforms as transforms
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.nn.utils.prune as prune
import time
from copy import deepcopy



start = time.time()

trainloader = DataLoader(minicifar_train, batch_size=batch_size, sampler=train_sampler)
validloader = DataLoader(minicifar_train, batch_size=batch_size, sampler=valid_sampler)
testloader = DataLoader(minicifar_test, batch_size=batch_size, shuffle=True) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.eval()






#%%

def import_transfer_learning_model(path = 'Models/DenseNet121_Adam_epochs_50.pth'):
    
    #Load of weights
    model = torch.load(path, map_location=device)

    n_inputs, n_classes = 1024, 4
    #Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    #Replaicng the last layer with a NN
    model.linear = nn.Sequential(
                        nn.Linear(n_inputs, 128), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(128, n_classes),                   
                        nn.LogSoftmax(dim=1))

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    print(f'{total_trainable_params:,} training parameters.')
    return model


def train_model(model, train_loader, valid_loader, test_loader, EPOCHS, patience=30):
  loss_list_train = []
  loss_list_valid = []
  accuracy_list = []
  save_value = 0
  early_stop = [1000,0]

  for epoch in range(EPOCHS):
    print(f"Epoch n° : {epoch+1}/{EPOCHS} commencée")
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

    if save_value < accuracy:
        torch.save(model.state_dict(), f'Models/{model_name}_{optimizer_name}_epochs_{n_epochs}.pth')
        print("Weights saved !")
        save_value = accuracy

    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    accuracy_list.append(accuracy)

    #End training if early stop reach the patience
    if early_stop[1] == patience:
        break 

    
  return model, loss_list_train,loss_list_valid, accuracy_list, save_value 



def pruning(model):

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)

    prune.remove(module, 'weight')

    return model




#%%
#TRAINING
model = import_transfer_learning_model(path=path_model)

if optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()



model = pruning(model)



model, loss_list_train, loss_list_valid, accuracy_list,best_accuracy = train_model(model,
                                                                                trainloader,
                                                                                validloader,
                                                                                testloader,
                                                                                n_epochs)


#Time
stop = time.time()
execution_time = stop - start

print(f"Program Executed in {execution_time}s")



epochs = [k+1 for k in range(len(loss_list_train))]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axes[0].plot(epochs,loss_list_train,color='b',label='Training loss')
axes[0].plot(epochs,loss_list_valid, color='r',label='Validation loss')
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title(f'{model_name} with {n_epochs} epochs')
axes[0].legend(loc='upper right')

axes[1].plot(epochs,accuracy_list,color='b',label='Accuracy')
axes[1].scatter(accuracy_list.index(best_accuracy)+1,best_accuracy,c='r',label=f'Best accuracy : {best_accuracy}%')
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('Accuracy in %', fontsize=14)
axes[1].legend(loc='upper left')

# Save figure
fig.savefig(f'Images/Binary/Loss_binary_{optimizer_name}_time{int(execution_time)}s_epochs_{n_epochs}_lr_{learning_rate}.png')