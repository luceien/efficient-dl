model_name = 'DenseNet121'
optimizer_name = 'Adam'
#IMPORT
import numpy as np
import torch
from torchinfo import summary 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import timeit
import matplotlib.pyplot as plt
#import torchvision
import numpy as np

def train_model(model, train_loader,valid_loader,test_loader,learning_rate,  EPOCHS,earlystop=True,patience=30,binary=True)  -> tuple(['models_cifar_10.densenet.DenseNet',list,list,list,float]):
  loss_list_train = []
  loss_list_valid = []
  accuracy_list = []
  save_value = 0
  early_stop = [1000,0]

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  #Optimizer (Adam better)
  if optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  #Loss
  criterion = nn.CrossEntropyLoss()
  if binary:
    model_bc = BC(model)
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

        #Apply binarization
        if binary:
            model_bc.binarization()

            #Forward + backward + optimize
            outputs = model_bc.forward(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs,labels)
        #Calculate gradients
        loss.backward()
        
        #Restore before weights' update
        if binary:
            model_bc.restore()

        #Update weights
        optimizer.step()

        if binary:
            model_bc.clip()
        loss_train += loss.item()
        nb_batch = i     

    loss_list_train.append(loss_train/i)
    print("\n","loss par epoch train =",np.round(loss_train/(nb_batch+1),4))

    #Validation 
    loss_valid = 0
    if not binary:
        model.eval()

    for i, data in tqdm(enumerate(valid_loader, 0)):  
        inputs, labels = data
        
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        if binary:
            target = model_bc.forward(inputs)
        else:
            target = model(inputs)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate Loss
        loss_valid += loss.item()
        nb_batch = i  
        #print("\n","loss par Batch valid=",loss.item(),"\n")       
    
    loss_list_valid.append(loss_valid/i)
    print("\n","loss par epoch valid =",np.round(loss_valid/(nb_batch+1),4))

    #Early-Stopping
    if loss_valid/(nb_batch+1) < early_stop[0]:
        early_stop[0] = loss_valid/(nb_batch+1)
        early_stop[1] = 0

    else:
        early_stop[1] += 1

    print(f'Validation loss did not improve for {early_stop[1]} epochs')

    #Test
    correct = 0
    total = 0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in tqdm(test_loader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            if binary:
                outputs = model_bc.forward(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
    print(f'Accuracy of the network on test images: {100 * np.round(correct/total,4)}%')
    accuracy_list.append(accuracy)

    if save_value < accuracy:
        if not binary:
            torch.save(model.state_dict(), f'Models/{model_name}_{optimizer_name}_epochs_{Niter}.pth')
            print("Weights saved! ")
        save_value = accuracy
    #End training if early stop reach the patience
    if earlystop :
        if early_stop[1] == patience:
            break 

#from scikit-learn import classification_report 
  if binary:   
    return model_bc, loss_list_train,loss_list_valid, accuracy_list, save_value
  else:
    return model, loss_list_train,loss_list_valid, accuracy_list, save_value

def half_model(model,test_loader,half=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if half:
       model =  model.half()
    model.eval()
    
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    '''for param in model.parameters():
        param.requires_grad = False'''

    correct = 0
    total = 0
    #with torch.no_grad():  # torch.no_grad for TESTING
    for data in tqdm(test_loader):
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.to(device), labels.to(device)
        if half:
            images = images.half()
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    print(f'Accuracy of the network on test images: {100 * np.round(correct/total,4)}%')
    return accuracy


from models_cifar_10.densenet import DenseNet121
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader
from binaryconnect import *

model = DenseNet121()
#weights = torch.load("Models/DenseNet121_Adam_epochs_25.pth")
#model.load_state_dict(weights['net'])


#accuracy_before_half = weights['accuracy']
#print(f'Accuracy of the saved model : {accuracy_before_half}%')

############### PARAMETERS ##################
Niter,Bsize,lr = 200, 32, 0.001
#torch.save(model,f'Models/HALF_{model_name}_{optimizer_name}_epochs_{Niter}.pth')
#Apply binarization of weight during training
binary_flag = False
earlystop_flag = False

trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True) 

#Time
start = timeit.default_timer()
#Running test on original model
accuracy_before_half = half_model(model,testloader,half=False)

stop = timeit.default_timer()
execution_time = stop - start
print(f"Program Executed in {int(execution_time//60)}min{np.round(execution_time%60,2)}s")

#Time
start = timeit.default_timer()
#Running half on original model

accuracy = half_model(model,testloader,half=True)

stop = timeit.default_timer()
execution_time = stop - start
print(f"Program Executed in {int(execution_time//60)}min{np.round(execution_time%60,2)}s")

#Final score
print(f'The accuracy of the half model is: {accuracy}%\n\
 The old one was : {accuracy_before_half}%')

'''#Time
start = timeit.default_timer()
#Training function
model,loss_list_train,loss_list_valid,\
accuracy_list,best_accuracy= train_model(model,
                                        trainloader,
                                        validloader,
                                        testloader,
                                        learning_rate=lr,
                                        EPOCHS=Niter,
                                        earlystop=earlystop_flag,
                                        patience=30,
                                        binary=binary_flag)

#Time
stop = timeit.default_timer()
execution_time = stop - start


print(f"Program Executed in {execution_time}")

epochs = [k+1 for k in range(len(loss_list_train))]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axes[0].plot(epochs,loss_list_train,color='b',label='Training loss')
axes[0].plot(epochs,loss_list_valid, color='r',label='Validation loss')
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title(f'{model_name} with {Niter} epochs')
axes[0].legend(loc='upper right')

axes[1].plot(epochs,accuracy_list,color='b',label='Accuracy')
axes[1].scatter(accuracy_list.index(best_accuracy)+1,best_accuracy,c='r',label=f'Best accuracy : {best_accuracy}%')
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('Accuracy in %', fontsize=14)
axes[1].legend(loc='upper left')

# Save figure
fig.savefig(f'Images/Binary/Loss_binary_{binary_flag}_{optimizer_name}_time{int(execution_time)}s_epochs_{Niter}_lr_{lr}.png')


'''