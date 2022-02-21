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

############################################# FUNCTIONS ###############################################

def train_model(model, train_loader,valid_loader,test_loader,learning_rate,  EPOCHS,earlystop=False,patience=30,binary=False)  -> tuple(['models_cifar_10.densenet.DenseNet',list,list,list,float,float]):

  loss_list_train = []
  loss_list_valid = []
  accuracy_list = []
  saved_value = 0
  early_stop = [1000,0]

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'The device use is {device}')
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
  start = timeit.default_timer()
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
            outputs = model.forward(inputs)
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
            target = model.forward(inputs)
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
        for data in tqdm(valid_loader):
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
    print(f'Accuracy of the network on validation images: {100 * np.round(correct/total,4)}%')
    accuracy_list.append(accuracy)

    if saved_value < accuracy:
        if not binary:
            torch.save(model.state_dict(), f'Models/{model_name}_{optimizer_name}_epochs_{Niter}.pth')
            print("Weights saved! ")
        saved_value = accuracy
    #End training if early stop reach the patience
    if earlystop :
        if early_stop[1] == patience:
            break 

  correct = 0
  total = 0
  with torch.no_grad():  # torch.no_grad for TESTING
    for data in tqdm(valid_loader):
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
  test_acc = 100 * np.round(correct/total,4)
  print(f'Accuracy of the network on test images by the end of the training: {test_acc}%','\n')
  stop = timeit.default_timer()
  execution_time = stop - start
  print('Training is done. \n')
  print(f"Training executed in {execution_time//3600}h{execution_time//60}min{np.round(execution_time%60,3)}s")
    #End training if early stop reach the patience
  if binary:   
    return model_bc, loss_list_train,loss_list_valid, accuracy_list, saved_value, test_acc,execution_time
  else:
    return model, loss_list_train,loss_list_valid, accuracy_list, saved_value, test_acc,execution_time

def local_pruning(model):
    for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers 
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
        # prune 40% of connections in all linear layers 
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
    return model

def global_pruning(model,amount):
    parameters_to_prune = []

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m,nn.BatchNorm2d):
            parameters_to_prune.append((m,'weight'))

    prune.global_unstructured(parameters_to_prune,
                            pruning_method=prune.L1Unstructured,
                            amount=amount,)
    '''for name, w in parameters_to_prune:
        prune.remove(name,'weight')'''

    return model

def pruning_accuracy(model,train_loader,valid_loader,test_loader,learning_rate,Niter,EPOCHS):
    
    pruned_model = deepcopy(model)
    list_accuracy = []
    amounts = np.linspace(0,1,Niter)
    for amount in amounts[:-1]:
        print(f'Training n°{amounts.index(amount)+1}/{len(amounts)}')
        pruned_model = global_pruning(pruned_model,amount)
        _,_,_,_,_,accuracy,_=train_model(pruned_model, train_loader,valid_loader,test_loader,learning_rate, EPOCHS)
        list_accuracy.append(accuracy)

    plt.plot(amounts,list_accuracy,c='darkgreen',label='')    
    plt.xlabel("Percentage of prunning")
    plt.ylabel("Accuracy %")
    plt.title(f'Accuracy on {EPOCHS} epochs training given the percentenge of pruning on DenseNet121')
    os.makedirs('Images/Pruning/Accuracy',exist_ok=True)
    plt.savefig(f'Images/Pruning/Accuracy/Accuracy_on_{EPOCHS}_epochs_for_{Niter}_different_pruning.png')

    return None

def print_prune_details(model):
    pruned_weight = 0
    total_weight = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m,nn.BatchNorm2d):
            weight_p = float(torch.sum(m.weight == 0))
            weight_t = float(m.weight.nelement())
            print(f"Sparsity in {m}.weight: \
            {np.round(100. * float(weight_p/weight_t),1)}")
            pruned_weight += weight_p
            total_weight+= weight_t
    print(f'Global sparsity: {np.round(100*float(pruned_weight/total_weight),2)}')
    return None


from models_cifar_10.densenet import DenseNet121
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader
#from binaryconnect import *

if __name__=='__main__':

    #Importing the model
    model_name = 'DenseNet121'
    optimizer_name = 'Adam'
    model_origin = DenseNet121()
    model = deepcopy(model_origin)
    #print(summary(model, input_size=(32, 3, 32, 32)))

    ############################################# PARAMETERS ################################################
    EPOCHS,Bsize,lr = 40, 32, 0.001
    #Apply binarization of weight/early stopping during training
    binary_flag, earlystop_flag = False, False
    #Number of different pruning steps
    Niter = 11 #21
    #% of pruning apply to the model
    amount = 0.6

    ################################################ DATA #####################################################
    #Loading data of minicifar
    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True) 

    ############################################## TRAINING ###################################################
    #Return a graph of accuracy = f(%pruning) in Images/Pruning/Accuracy folder
    #pruning_accuracy(model,trainloader,validloader,testloader,lr,Niter=Niter,EPOCHS=EPOCHS)

    #Prune the model and print details of layers pruning
    pruned_model = global_pruning(model,amount)
    print_prune_details(model)

    #Train the original model
    model,train_loss_origin,valid_loss_origin,origin_accuracy,\
    best_accuracy1,test1, execution_time1 = train_model(model_origin,
                                                    trainloader,
                                                    validloader,
                                                    testloader,
                                                    learning_rate=lr,
                                                    EPOCHS=Niter,
                                                    earlystop=earlystop_flag,
                                                    patience=30,
                                                    binary=binary_flag)
    #Train the pruned model
    model,train_loss_pruned,valid_loss_pruned,pruned_accuracy,\
    best_accuracy2,test2, execution_time2 = train_model(pruned_model,
                                                    trainloader,
                                                    validloader,
                                                    testloader,
                                                    learning_rate=lr,
                                                    EPOCHS=Niter,
                                                    earlystop=earlystop_flag,
                                                    patience=30,
                                                    binary=binary_flag)

    ################################################ PLOT #####################################################
    #Plotitng data from pruned and original training
    original_weights = sum(p.numel() for p in model_origin.parameters())
    epochs = [k+1 for k in range(len(train_loss_pruned))]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    #First figure for training/validation loss
    axes[0].plot(epochs,train_loss_pruned,color='b',label='Training loss pruned')
    axes[0].plot(epochs,valid_loss_pruned, color='r',label='Validation loss pruned')

    axes[0].plot(epochs,train_loss_origin,color='darkgreen',label='Training loss origin')
    axes[0].plot(epochs,valid_loss_origin, color='orange',label='Validation loss origin')
    axes[0].plot([], [], ' ', label=f"Original training done in {np.round(execution_time2,3)}")
    axes[0].plot([], [], ' ', label=f"Pruned training done in {np.round(execution_time1,3)}")

    axes[0].set_xlabel('Epochs', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].set_title(f'{model_name} with {Niter} epochs and {original_weights} parameters. {int((1-amount)*original_weights)} parameters for pruned model')
    axes[0].legend(loc='upper right')

    #Second figure for accuracy of pruned/original model on test set
    axes[1].plot(epochs,pruned_accuracy,color='b',label='Accuracy pruned on validation')
    axes[1].plot(epochs,origin_accuracy,color='darkgreen',label='Accuracy origin on validation')
    axes[1].scatter(pruned_accuracy.index(best_accuracy1)+1,best_accuracy1,c='r',label=f'Best accuracy : {best_accuracy1}% pruned')
    axes[1].scatter(origin_accuracy.index(best_accuracy2)+1,best_accuracy2,c='chocolate',label=f'Best accuracy: {best_accuracy2}% original')
    axes[1].plot([], [], ' ', label=f"Accuracy on test after pruned training:{test1}%")
    axes[1].plot([], [], ' ', label=f"Accuracy on test after original training:{test2}%")
    axes[1].set_xlabel('Epochs', fontsize=14)
    axes[1].set_ylabel('Accuracy in %', fontsize=14)
    axes[1].legend(loc='upper left')

    # Save figure
    fig.savefig(f'Images/Pruning/Loss_{optimizer_name}_epochs_{Niter}_lr_{lr}.png')
    ###############################################################################################################
