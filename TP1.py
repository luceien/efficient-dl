#%%
#Name of Model, optimizer etc.. for Graph file's name
model_name = 'DenseNet121_PT'
optimizer_name = 'Adam'

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
#from models_cifar_10.densenet import DenseNet121
from models_cifar100.densenet import DenseNet121

#Model chosen from Vgg, Densenet, Resnet
model = DenseNet121()
print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters())}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training run on : {device}")
model.to(device)

def train_model(model, train_loader,valid_loader,test_loader,learning_rate,  EPOCHS,patience=30):
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
#Transfer Learning

def pretrained_model(test_loader):
    model =  DenseNet121()
    dict = torch.load('models_cifar100/DenseNet121_model_cifar100_lr_0.01.pth')
    model.load_state_dict(dict['net'])
    model.eval()
    model = model.to(device)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    for param in model.parameters():
        param.requires_grad = False

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
    print(f'Accuracy of the network on test images: {accuracy}%')
    return accuracy

def transfer_learning(model, train_loader,valid_loader,test_loader,learning_rate,  EPOCHS,patience=30):
    #Load of weights
    dict = torch.load('models_cifar100/DenseNet121_model_cifar100_lr_0.01.pth')
    model.load_state_dict(dict['net'])

    n_inputs, n_classes = 1024, 4
    #Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    #Replaicng the last layer with a NN
    model.linear = nn.Sequential(
                        nn.Linear(n_inputs, 256), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, n_classes),                   
                        nn.LogSoftmax(dim=1))

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    #NN parameters 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    loss_list_train = []
    loss_list_valid = []
    accuracy_list = []
    early_stop = [1000,0]

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
        #if early_stop[1] == patience:
        #   break 

    
    return model, loss_list_train,loss_list_valid, accuracy_list 

#%%
#HyperParameters
Niter,Bsize,lr = 500 , 32, 0.0001

#Data import
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True) 

train_model, loss_list_train,loss_list_valid, accuracy = transfer_learning(model, trainloader,validloader,testloader,lr,Niter)
# %%

#Register plot of Accuracy and loss
import matplotlib.pyplot as plt
#import torchvision
import numpy as np
epochs = [k+1 for k in range(len(loss_list_train))]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axes[0].plot(epochs,loss_list_train,color='b',label='Training loss')
axes[0].plot(epochs,loss_list_valid, color='r',label='Validation loss')
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title(f'{model_name} with {Niter} epochs')
axes[0].legend(loc='upper right')

axes[1].plot(epochs,accuracy,color='b',label='Accuracy')
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('Accuracy in %', fontsize=14)
axes[1].legend(loc='upper right')

# Save figure
fig.savefig(f'Images/Loss_{model_name}_{optimizer_name}_epochs_{Niter}_lr_{lr}.png')