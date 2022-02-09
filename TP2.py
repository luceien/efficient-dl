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

from models_cifar_10.densenet import DenseNet121

#from TP1 import train_model
def train_model(model, train_loader,valid_loader,test_loader,learning_rate,  EPOCHS,patience=30):
  loss_list_train = []
  loss_list_valid = []
  accuracy_list = []
  save_value = 0
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
        inputs = inputs.half()
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
    print("\n","loss par epoch train =",np.round(loss_train/(nb_batch+1),4))

    #Validation 
    loss_valid = 0
    model.eval()
    for i, data in tqdm(enumerate(valid_loader, 0)):  
        inputs, labels = data
        inputs = inputs.half()
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
    print("\n","loss par epoch valid =",np.round(loss_valid/(nb_batch+1),4))

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
            inputs, labels = inputs.half()
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
    print(f'Accuracy of the network on test images: {100 * np.round(correct/total,4)}%')
    accuracy_list.append(accuracy)

    if save_value < accuracy:
        torch.save(model.state_dict(), f'Models/{model_name}_{optimizer_name}_epochs_{Niter}.pth')
        print("Weights saved! ")
        save_value = accuracy
    #End training if early stop reach the patience
    if early_stop[1] == patience:
        break 

#from scikit-learn import classification_report    
  return model, loss_list_train,loss_list_valid, accuracy_list, save_value

model = torch.load("Models/DenseNet121_Adam_epochs_30.pth")
model.half()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

Niter,Bsize,lr = 10, 32, 0.001
print(Niter)
torch.save(model,f'Models/HALF_{model_name}_{optimizer_name}_epochs_{Niter}.pth')

trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True) 


train_model, loss_list_train,loss_list_valid, accuracy, best_accuracy = train_model(model, trainloader,validloader,testloader,lr,Niter)