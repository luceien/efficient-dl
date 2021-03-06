from decimal import DecimalException
import torch
#from torchinfo import summary 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import StepLR
from typing import TypeVar
from torch.autograd import Variable
model = TypeVar('model')
device = TypeVar('device')

from tqdm import tqdm
import numpy as np
import timeit
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from copy import deepcopy
######################################################- GENERAL FUNCTIONS -######################################################

def to_device(model) -> tuple((model,device)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Training run on : {device}")
    model.to(device)    

    return model,device

def training(model,train_loader,criterion,optimizer,device) -> float:
    loss_epoch = 0

    #Training
    model.train()
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
        loss_epoch += loss.item()
        #nb_batch = i     
    
    loss_epoch = loss_epoch/len(train_loader)
    print("\n","loss par epoch train =",np.round(loss_epoch,4))
    return loss_epoch

def validation(model,valid_loader,criterion,optimizer,device) -> float:
    loss_epoch = 0

    #Validation
    model.eval()
    for i, data in tqdm(enumerate(valid_loader, 0)):  
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
         
        target = model(inputs)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate Loss
        loss_epoch += loss.item()
        #nb_batch = i  
        #print("\n","loss par Batch valid=",loss.item(),"\n")       
  

    loss_epoch = loss_epoch/len(valid_loader)
    print("\n","loss par epoch valid =",np.round(loss_epoch,4))
    return loss_epoch

def getAccuracy(model,dataloader,device) -> float:
    correct = 0
    total = 0
    
    #Testing
    half=False
    if half:
        model.half()
    model.eval()
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in tqdm(dataloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            if half:
                images = images.half()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * (correct / total)
    return accuracy

def earlystopping(loss_valid,previous,count) -> tuple((float,int)):
    if loss_valid  < previous:
        previous = loss_valid
        count = 0
    else:
        count += 1
    return previous,count

def save_weights(model,ref_accuracy,saved_value,accuracy,Niter,test_loader,device, add ='', model_name='ResNet',optimizer_name='SGD', verbose = 1) -> tuple((float,float)) :
    #Saved value: best validation accuracy so far
    #Accuracy : validation accuracy during the epoch
    #Ref_accuracy : Best test accuracy so far

    if saved_value < accuracy:
        state = {
            'net': model.state_dict(),
            'accuracy': accuracy
        }
        if accuracy > 90 :
            test_accuracy = getAccuracy(model,test_loader,device)
            if verbose > 1 :
                print(f'Accuracy of the network saved on test images: {test_accuracy}%.')
            
            if test_accuracy > ref_accuracy:
                try: 
                    os.remove(f'Models/Accuracy_90/{optimizer_name}_MU_epochs_{Niter}_acc{ref_accuracy}{add}.pth')
                except:
                    pass

                ref_accuracy = test_accuracy
                os.makedirs('Models/Accuracy_90',exist_ok=True)
                torch.save(state, f'Models/Accuracy_90/{optimizer_name}_MU_epochs_{Niter}_acc{ref_accuracy}{add}.pth')
                if verbose > 1 :
                    print("Weights saved! ")
                
                
        saved_value = accuracy
    return saved_value,ref_accuracy

def load_weights(model,path) -> model:
    #Load weights
    dict = torch.load(path)
    model.load_state_dict(dict['net'])
    return model

def training_model(model, train_loader,valid_loader,test_loader,device,learning_rate, EPOCHS,earlystop,patience=25):
    
    #NN parameters
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.AdamWSGD(model.parameters(),lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)#,verbose = True) #Verbose : print the learning rate
    #scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=EPOCHS,verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,8,9], gamma=0.1)
    
    loss_list_train, loss_list_valid, accuracy_list, saved_value = [], [], [], 0
    early_stop =  [np.inf,0] #[Loss reference for EarlyStopping, keep the number of step]
    ref_accuracy = 0

    start = timeit.default_timer()
    for epoch in range(EPOCHS):
        print(f"Epoch n?? : {epoch+1}/{EPOCHS} commenc??e")

        #Training
        loss_train = training(model,train_loader,criterion,optimizer,device)
        loss_list_train.append(loss_train)

        #Validation 
        loss_valid = validation(model,valid_loader,criterion,optimizer,device)
        loss_list_valid.append(loss_valid)

        #Early-Stopping
        early_stop = earlystopping(loss_valid,early_stop[0],early_stop[1])
        print(f'Validation loss did not change for {early_stop[1]} epochs')

        #Validation accuracy
        accuracy = getAccuracy(model,valid_loader,device)
        print(f'Accuracy of the network on validation images: {accuracy}%')
        accuracy_list.append(accuracy)
        scheduler.step()

        #End training if early stop reach the patience
        if earlystop:
            if early_stop[1] == patience:
                break 

        #Saving value to compare accuracy for weights saving.
        saved_value,ref_accuracy = save_weights(model,ref_accuracy,saved_value,accuracy,EPOCHS,test_loader,device)

    #Accuracy on test set by the end of the training
    test_acc = getAccuracy(model,test_loader,device)
    print(f'Accuracy of the network on test images by the end of the training: {test_acc}%','\n')
    
    
    stop = timeit.default_timer()
    execution_time = stop - start
    print('Training is done. \n')
    print(f"Training executed in {int(execution_time//3600)}h{int(execution_time//60)}min{int(np.round(execution_time%60,3))}s")
    
    return model, loss_list_train,loss_list_valid, accuracy_list,saved_value,test_acc,execution_time

######################################################- TP 2 - TRANSFER LEARNING -######################################################

def sequential(model,n_inputs=1024,n_classes=10) -> model:
    #model.linear = nn.Sequential(nn.Linear(n_inputs, n_classes))                 
    '''model.linear = nn.Sequential(
                        nn.Linear(n_inputs, 256), 
                        nn.ReLU(), 
                        nn.Linear(256, n_classes))  '''   
    model.linear = nn.Sequential(
                        nn.Linear(n_inputs, 256), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 32), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(32, n_classes))            
    return model

def transfer_learning(model,model_name,path) -> model:

    model = load_weights(model,path,model_name)
    #Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    #Add MLP for Transfer Learning
    model = sequential(model,n_inputs=512,n_classes=10)
    
    # Summary + Find total parameters and trainable parameters
    #print(summary(model))
    return model 

def plot1(loss_list_train,loss_list_valid, accuracy, best_accuracy,Niter,lr,model_name='ResNet20',optimizer_name='SGD', add ='') -> None:
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
    os.makedirs('Images/Accuracy_90',exist_ok=True)
    fig.savefig(f'Images/Accuracy_90/Loss_{model_name}_{optimizer_name}_epochs_{Niter}_lr_{lr}{add}.png')

    return None 



######################################################- TP 3 - PRUNING -######################################################

def local_pruning(model) -> model:
    for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers 
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
        # prune 40% of connections in all linear layers 
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
    return model

def global_pruning(model,amount,device,conv2d_flag,linear_flag,BN_flag) -> model:
    parameters_to_prune = []

    for m in model.modules():
        if (isinstance(m, nn.Conv2d) and conv2d_flag) or (isinstance(m, nn.Linear) and linear_flag) or (isinstance(m,nn.BatchNorm2d) and BN_flag):
        #if isinstance(m, nn.Conv2d):# or isinstance(m, nn.Linear) or isinstance(m,nn.BatchNorm2d):
            parameters_to_prune.append((m,'weight'))

    prune.global_unstructured(parameters_to_prune,
                            pruning_method=prune.L1Unstructured,
                            amount=amount,)
    for name, w in parameters_to_prune:
        prune.remove(name,'weight')
    model = model.to(device)
    return model

def pruning_accuracy(model,train_loader,valid_loader,test_loader,learning_rate,Niter,EPOCHS) -> None:
    #Plot the accuracy givent pruning %
    pruned_model = deepcopy(model)
    list_accuracy = []
    amounts = np.linspace(0,1,Niter)
    index=0
    for amount in amounts[:-1]:
        index+=1
        print(f'Training n??{index}/{amounts.shape[0]}')
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

def print_prune_details(model) -> None:
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

def plot2(origin,pruned,Niter,lr,amount,model_name='ResNet',optimizer_name='SGD') -> None:
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

def remove_parameters(model):

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def iterative_pruning(model, train_loader, valid_loader, test_loader, device,learning_rate, EPOCHS=20):

    #NN parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)#,verbose = True) #Verbose : print the learning rate
    
    loss_list_train, loss_list_valid, accuracy_list, saved_value, prune_list, number_epochs = [], [], [], 0, [], []
    ref_accuracy = 0

    start = timeit.default_timer()

    model_test = deepcopy(model)
    conv2d_prune_amount, delta_prune, delta_prune_min = 0.75, 0.08, 0.005

    while delta_prune > delta_prune_min :
        print(f"\nPruned percentage is {conv2d_prune_amount}")

        # Global pruning
        parameters_to_prune = []
        for _, module in model_test.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            amount = conv2d_prune_amount)

        print_prune_details(model_test)

        for epoch in range(EPOCHS):
            #print(f"Epoch n?? : {epoch+1}/{EPOCHS} commenc??e")

            #Training
            loss_train = training(model_test,train_loader,criterion,optimizer,device)
            loss_list_train.append(loss_train)

            #Validation 
            loss_valid = validation(model_test,valid_loader,criterion,optimizer,device)
            loss_list_valid.append(loss_valid)

            #Validation accuracy
            accuracy = getAccuracy(model_test,valid_loader,device)
            print(f'Accuracy of the network on validation images: {round(accuracy,2)}%')
            scheduler.step()

            if accuracy > 90:
                accuracy_list.append(accuracy)
                prune_list.append(conv2d_prune_amount)
                number_epochs.append(epoch)
                print(f"Accuracy successfully reached 90% with prune = {conv2d_prune_amount}")
                conv2d_prune_amount += delta_prune
                break
        
        if accuracy < 90:
            print(f"Accuracy did not manage to reach 90% with prune = {conv2d_prune_amount}")
            delta_prune = delta_prune/2
            conv2d_prune_amount += delta_prune



    #Saving value to compare accuracy for weights saving.
    saved_value, ref_accuracy = save_weights(model_test, ref_accuracy, saved_value, accuracy, EPOCHS, test_loader,device)

    #Accuracy on test set by the end of the training
    test_acc = getAccuracy(model_test,test_loader,device)
    print(f'Accuracy of the network on test images by the end of the training: {test_acc}% with prune {round(conv2d_prune_amount,2)}%','\n')
    
    
    stop = timeit.default_timer()
    execution_time = stop - start
    print('Training is done. \n')
    print(f"Training executed in {int(execution_time//3600)}h{int(execution_time//60)}min{int(np.round(execution_time%60,3))}s")
    
    return model_test, accuracy_list,prune_list,number_epochs, number_epochs, saved_value

def plot_prune_accuracy(accuracy_list, prune_list, number_epochs):

    fig, ax = plt.scatter(prune_list, accuracy_list)
    for i, txt in enumerate(number_epochs):
        ax.annotate(txt, (prune_list[i], accuracy_list[i]))

    # Save figure
    os.makedirs('Images/Pruning/Iteration',exist_ok=True)
    fig.savefig(f'Images/Pruning/Iteration/Prune_from_{prune_list[0]}_to_{round(prune_list[-1],2)}.png')



######################################################- TP 4 - MIX-UP -######################################################

def mixup_data(x, y):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(0.1, 0.1)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def training_mixup(model,train_loader,criterion,optimizer,device) -> float:
    loss_epoch = 0

    #Training
    model.train()
    for i, data in tqdm(enumerate(train_loader, 0)):  
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)


        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
        inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
        #Forward + backward + optimize
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        #Clear the gradients
        optimizer.zero_grad()
        #Calculate gradients
        loss.backward()
        #Update weights
        optimizer.step()
        loss_epoch += loss.item()
        #nb_batch = i     
    
    loss_epoch = loss_epoch/len(train_loader)
    print("\n","loss par epoch train =",np.round(loss_epoch,4))
    return loss_epoch

def training_model_with_mixup(model, train_loader,valid_loader,test_loader,device,learning_rate, EPOCHS,earlystop,patience=25):
    
    #NN parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)#,verbose = True) #Verbose : print the learning rate

    loss_list_train, loss_list_valid, accuracy_list, saved_value = [], [], [], 0
    early_stop =  [np.inf,0] #[Loss reference for EarlyStopping, keep the number of step]
    ref_accuracy = 0

    start = timeit.default_timer()
    for epoch in range(EPOCHS):
        print(f"Epoch n?? : {epoch+1}/{EPOCHS} commenc??e")

        #Training
        loss_train = training_mixup(model,train_loader,criterion,optimizer,device)
        loss_list_train.append(loss_train)

        #Validation 
        loss_valid = validation(model,valid_loader,criterion,optimizer,device)
        loss_list_valid.append(loss_valid)

        #Early-Stopping
        early_stop = earlystopping(loss_valid,early_stop[0],early_stop[1])
        print(f'Validation loss did not change for {early_stop[1]} epochs')

        #Validation accuracy
        accuracy = getAccuracy(model,valid_loader,device)
        print(f'Accuracy of the network on validation images: {accuracy}%')
        accuracy_list.append(accuracy)
        scheduler.step()

        #End training if early stop reach the patience
        if earlystop:
            if early_stop[1] == patience:
                break 

        #Saving value to compare accuracy for weights saving.
        saved_value,ref_accuracy = save_weights(model,ref_accuracy,saved_value,accuracy,EPOCHS,test_loader,device)

    #Accuracy on test set by the end of the training
    test_acc = getAccuracy(model,test_loader,device)
    print(f'Accuracy of the network on test images by the end of the training: {test_acc}%','\n')
    
    
    stop = timeit.default_timer()
    execution_time = stop - start
    print('Training is done. \n')
    print(f"Training executed in {int(execution_time//3600)}h{int(execution_time//60)}min{int(np.round(execution_time%60,3))}s")
    
    return model, loss_list_train,loss_list_valid, accuracy_list,saved_value,test_acc,execution_time



######################################################- TP 2-2 - QUANTIZATION -######################################################

#Model class extension to include quantization
class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def training_model_quantized(model, train_loader,valid_loader,test_loader,device,learning_rate, EPOCHS, add = '', verbose =1):
    
    #NN parameters 
    fused_model = deepcopy(model)

    model.train()
    fused_model.train()
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)

    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for _, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)


    # Model and fused model should be equivalent.
    # model.eval()
    # fused_model.eval()

    quantized_model = QuantizedResNet18(model_fp32=fused_model)

    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig
    # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    quantized_model.qconfig = quantization_config


    print("Training QAT Model...\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(quantized_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    loss_list_train, loss_list_valid, accuracy_list, saved_value = [], [], [], 0
    ref_accuracy = 0

    start = timeit.default_timer()


    for epoch in range(EPOCHS):
        print(f"Epoch n?? : {epoch+1}/{EPOCHS} commenc??e")

        #Training
        torch.quantization.prepare_qat(quantized_model, inplace=True)
        loss_train = training(quantized_model,train_loader,criterion,optimizer,device)
        loss_list_train.append(loss_train)

        #Validation 
        quantized_model = torch.quantization.convert(quantized_model, inplace=True)
        loss_valid = validation(quantized_model,valid_loader,criterion,optimizer,device)
        loss_list_valid.append(loss_valid)


        #Validation accuracy
        accuracy = getAccuracy(quantized_model,valid_loader,device)
        print(f'Accuracy of the network on validation images: {accuracy}%')
        accuracy_list.append(accuracy)
        scheduler.step()

        #Saving value to compare accuracy for weights saving.
        saved_value,ref_accuracy = save_weights(quantized_model,ref_accuracy,saved_value,accuracy,EPOCHS,test_loader,device, add = add , verbose=verbose)

    #Accuracy on test set by the end of the training
    test_acc = getAccuracy(quantized_model,test_loader,device)
    print(f'Accuracy of the network on test images by the end of the training: {test_acc}%','\n')
    
    
    stop = timeit.default_timer()
    execution_time = stop - start

    if verbose > 1 :
        print('Training is done. \n')
        print(f"Training executed in {int(execution_time//3600)}h{int(execution_time//60)}min{int(np.round(execution_time%60,3))}s")
    
    return quantized_model, loss_list_train,loss_list_valid, accuracy_list,saved_value,test_acc,execution_time





#