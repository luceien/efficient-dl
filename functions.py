import torch
from torchinfo import summary 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tqdm import tqdm
import numpy as np
import timeit
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from copy import deepcopy
######################################################- GENERAL FUNCTIONS -######################################################

def to_device(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training run on : {device}")
    model.to(device)    

    return model,device

def training(model,train_loader,criterion,optimizer,device):

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
    
    print("\n","loss par epoch train =",np.round(loss_train/(nb_batch+1),4))
    loss = loss_train/float(nb_batch)
    return loss

def validation(model,valid_loader,criterion,optimizer,device):
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
  

    print("\n","loss par epoch valid =",np.round(loss_valid/nb_batch+1,4))
    loss = loss_valid/nb_batch
    return loss,nb_batch

def accuracy_validation(model,valid_loader,device):
    correct = 0
    total = 0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in tqdm(valid_loader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
    print(f'Accuracy of the network on validation images: {100 * np.round(correct / total,4)}%')
    return accuracy

def earlystopping(loss_valid,previous,step,nb_batch):
    if loss_valid/(nb_batch+1)  < previous:
        previous = loss_valid/(nb_batch+1)
        step = 0
    else:
        step += 1
    return previous,step

def accuracy_test(model,test_loader,device):
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
    test_acc = 100 * np.round(correct/total,4)
    return test_acc

def save_weights(model,saved_value,accuracy,Niter,model_name='DenseNet',optimizer_name='Adam'):

    if saved_value < accuracy:
        state = {
            'net': model.state_dict(),
            'accuracy': accuracy
        }
        torch.save(state, f'Models/{model_name}_{optimizer_name}_epochs_{Niter}.pth')
        print("Weights saved! ")
        saved_value = accuracy
    return saved_value

def training_model(model, train_loader,valid_loader,test_loader,device,learning_rate, EPOCHS,earlystop,patience=30):
    
    #NN parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = StepLR(optimizer, step_size=70, gamma=0.1)

    loss_list_train, loss_list_valid, accuracy_list, early_stop, saved_value = [], [], [], [1000,0], 0

    start = timeit.default_timer()
    for epoch in range(EPOCHS):
        print(f"Epoch n° : {epoch+1}/{EPOCHS} commencée")

        #Training
        loss_train = training(model,train_loader,criterion,optimizer,device)
        loss_list_train.append(loss_train)

        #Validation 
        loss_valid, nb_batch = validation(model,valid_loader,criterion,optimizer,device)
        loss_list_valid.append(loss_valid)

        #Early-Stopping
        early_stop = earlystopping(loss_valid,early_stop[0],early_stop[1],nb_batch)
        print(f'Validation loss did not change for {early_stop[1]} epochs')

        #Validation accuracy
        accuracy = accuracy_validation(model,valid_loader,device)
        accuracy_list.append(accuracy)

        #End training if early stop reach the patience
        if earlystop:
            if early_stop[1] == patience:
                break 
        #Saving value to compare accuracy for weights saving.
        saved_value = save_weights(model,saved_value,accuracy,EPOCHS)

    #Accuracy on test set by the end of the training
    test_acc = accuracy_test(model,test_loader,device)
    print(f'Accuracy of the network on test images by the end of the training: {test_acc}%','\n')
    
    
    stop = timeit.default_timer()
    execution_time = stop - start
    print('Training is done. \n')
    print(f"Training executed in {int(execution_time//3600)}h{int(execution_time//60)}min{int(np.round(execution_time%60,3))}s")
    
    return model, loss_list_train,loss_list_valid, accuracy_list,saved_value,test_acc,execution_time

######################################################- TP 2 - TRANSFER LEARNING -######################################################

def sequential(model,n_inputs=1024,n_classes=10):
    model.linear = nn.Sequential(
                        nn.Linear(n_inputs, 256), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 32), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(32, n_classes))                 
    return model

def transfer_learning(model):

    #Load weights
    dict = torch.load('models_cifar100/DenseNet121_model_cifar100_lr_0.01.pth')
    model.load_state_dict(dict['net'])

    #Freeze weights
    '''for param in model.parameters():
        param.requires_grad = False'''

    #Add MLP for Transfer Learning
    model = sequential(model,n_inputs=1024,n_classes=10)
    
    # Summary + Find total parameters and trainable parameters
    print(summary(model))
    return model 

def plot1(loss_list_train,loss_list_valid, accuracy, best_accuracy,Niter,lr,model_name='Densenet',optimizer_name='Adam'):
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
    axes[1].plot([], [], ' ', label=f"Accuracy on test after pruned training:{best_accuracy}%")
    axes[1].legend(loc='upper right')

    # Save figure
    fig.savefig(f'Images/Loss_{model_name}_{optimizer_name}_epochs_{Niter}_lr_{lr}.png')

    return None 

######################################################- TP 3 - PRUNING -######################################################

def local_pruning(model):
    for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers 
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
        # prune 40% of connections in all linear layers 
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
    return model

def global_pruning(model,amount,device,conv2d_flag,linear_flag,BN_flag):
    parameters_to_prune = []

    for m in model.modules():
        if (isinstance(m, nn.Conv2d) and conv2d_flag) or (isinstance(m, nn.Linear) and linear_flag) or (isinstance(m,nn.BatchNorm2d) and BN_flag):
        #if isinstance(m, nn.Conv2d):# or isinstance(m, nn.Linear) or isinstance(m,nn.BatchNorm2d):
            parameters_to_prune.append((m,'weight'))

    prune.global_unstructured(parameters_to_prune,
                            pruning_method=prune.L1Unstructured,
                            amount=amount,)
    '''for name, w in parameters_to_prune:
        prune.remove(name,'weight')'''
    model = model.to(device)
    return model

def pruning_accuracy(model,train_loader,valid_loader,test_loader,learning_rate,Niter,EPOCHS):
    #Plot the accuracy givent pruning %
    pruned_model = deepcopy(model)
    list_accuracy = []
    amounts = np.linspace(0,1,Niter)
    index=0
    for amount in amounts[:-1]:
        index+=1
        print(f'Training n°{index}/{amounts.shape[0]}')
        pruned_model = global_pruning(pruned_model,amount,conv2d_flag,linear_flag,BN_flag)
        _,_,_,_,_,accuracy,_=training(pruned_model, train_loader,valid_loader,test_loader,learning_rate, EPOCHS)
        list_accuracy.append(accuracy)

    plt.plot(amounts[:-1],list_accuracy,c='darkgreen',label='')    
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

def plot2(origin,pruned,Niter,lr,amount,model_name='Densenet',optimizer_name='Adam'):
    #[model,train_loss_origin, valid_loss_origin, origin_accuracy, best_accuracy, test_accuracy, execution_time]

    #Plotitng data from pruned and original training
    original_weights = sum(p.numel() for p in origin[0].parameters())
    epochs = [k+1 for k in range(len(pruned[1]))]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    #First figure for training/validation loss
    axes[0].plot(epochs,pruned[1],color='b',label='Training loss pruned')
    axes[0].plot(epochs,pruned[2], color='r',label='Validation loss pruned')

    axes[0].plot(epochs,origin[1],color='darkgreen',label='Training loss origin')
    axes[0].plot(epochs,origin[2], color='orange',label='Validation loss origin')
    axes[0].plot([], [], ' ', label=f"Original training done in {np.round(origin[6],3)}")
    axes[0].plot([], [], ' ', label=f"Pruned training done in {np.round(pruned[6],3)}")

    axes[0].set_xlabel('Epochs', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].set_title(f'{model_name} with {Niter} epochs and {original_weights} parameters. {int((1-amount)*original_weights)} parameters for pruned model')
    axes[0].legend(loc='upper right')


    #Second figure for accuracy of pruned/original model on test set
    axes[1].plot(epochs,pruned[3],color='b',label='Accuracy pruned on validation')
    axes[1].plot(epochs,origin[3],color='darkgreen',label='Accuracy origin on validation')
    axes[1].scatter(origin[3].index(origin[4])+1,origin[4],c='chocolate',label=f'Best accuracy: {origin[4]}% original')
    axes[1].scatter(pruned[3].index(pruned[4])+1,pruned[4],c='r',label=f'Best accuracy : {pruned[4]}% pruned')
    axes[1].plot([], [], ' ', label=f"Accuracy on test after pruned training:{pruned[5]}%")
    axes[1].plot([], [], ' ', label=f"Accuracy on test after original training:{origin[5]}%")
    axes[1].set_xlabel('Epochs', fontsize=14)
    axes[1].set_ylabel('Accuracy in %', fontsize=14)
    axes[1].legend(loc='upper left')

    # Save figure
    os.makedirs('Images/Pruning/Loss',exist_ok=True)
    fig.savefig(f'Images/Pruning/Loss/Loss_{optimizer_name}_epochs_{Niter}_lr_{lr}_C2D.png')

    return None