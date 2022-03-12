#%%
from stringprep import map_table_b2
import torch
import torch.nn as nn
import torch.optim as optim
from models_cifar100.densenet import DenseNet121
from torch.optim import lr_scheduler
import time, copy
from torch.utils.data.dataloader import DataLoader
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader = DataLoader(minicifar_train, batch_size=32, sampler = train_sampler)
validloader = DataLoader(minicifar_train, batch_size=32, sampler = valid_sampler)
testloader = DataLoader(minicifar_test, batch_size=32, shuffle = True) 

#%%
#Import model

model_ft = DenseNet121()
state_dict = torch.load('./DenseNet121_model_cifar100_lr_0.01.pth',  map_location=torch.device('cpu'))['net']
model_ft.load_state_dict(state_dict)

for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.linear.in_features
model_ft.linear = nn.Linear(num_ftrs, 10)

#model_ft.eval()
#model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_name = 'Adam'

Niter, Bsize, lr = 4, 32, 0.005



#%%


def test(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in tqdm(testloader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

test(model_ft, testloader)





#%%

def train_model(model, train_loader,valid_loader,test_loader, learning_rate, EPOCHS, patience=30):
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
        loss_valid = 0

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
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            nb_batch = i     

        loss_list_train.append(loss_train/i)
        print("\n","loss par epoch train =",loss_train/(nb_batch+1))

        #Validation 
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

# load best model weights
#model.load_state_dict(best_model_wts)

trained_model, loss_list_train, loss_list_valid, accuracy = train_model(model_ft, trainloader, validloader, testloader, learning_rate=lr, EPOCHS=Niter)



#%%
model_name = 'DenseNet121'
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
#fig.savefig(f'Images/Loss_{model_name}_{optimizer_name}_epochs_{Niter}_lr_{lr}.png')
plt.show()
# %%
