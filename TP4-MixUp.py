#%%
model_name = 'DenseNet121'
optimizer_name = 'Adam'
learning_rate = 0.001
batch_size = 32
n_epochs = 50
path_model ='models_cifar100/DenseNet121_model_cifar100_lr_0.01.pth'

cifar = 4

if cifar ==10:
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
else :
    class_names = ['plane', 'car', 'bird', 'cat']


#%%

if True :
    from models_cifar_10.densenet import DenseNet121,DenseNet121bis

    import numpy as np 
    #from torchinfo import summary
    from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
    from torch.utils.data.dataloader import DataLoader

    import torch.optim as optim
    import torch.nn as nn
    #import torch.nn.functional as F
    import torch
    import random
    from tqdm import tqdm
    import torch.nn.utils.prune as prune
    import time
    from copy import deepcopy

    from sklearn.metrics import confusion_matrix
    import itertools
    import matplotlib.pyplot as plt

    start = time.time()

    trainloader = DataLoader(minicifar_train, batch_size=batch_size, sampler=train_sampler)
    validloader = DataLoader(minicifar_train, batch_size=batch_size, sampler=valid_sampler)
    testloader = DataLoader(minicifar_test, batch_size=batch_size, shuffle=True) 


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used is {device}\n')



#%%
#####################################################
#TRAIN FUNCTION with MIXUP

def mix_criterion(target, labels, label_perm, l):
    return l*criterion(target,  labels) + (1-l)*criterion(target, label_perm)

def train_model(model, train_loader, valid_loader, test_loader, EPOCHS, mixup=False, verbose =2, patience=30):
  loss_list_train = []
  loss_list_valid = []
  accuracy_list = []
  save_value = 0
  early_stop = [1000,0]

  if optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(EPOCHS):
    if verbose > 1 :
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


        if mixup and epoch > 30:
            lam = np.random.rand()
            permutation = torch.randperm(32)
            if torch.cuda.is_available():
                permutation = permutation.cuda()

            mixdata = lam * inputs + (1 - lam) * inputs[permutation, :]
            label, label_perm = labels, labels[permutation]

            optimizer.zero_grad()
            target = model(mixdata)
            lambda_const = random.random()

            loss = mix_criterion(target, label, label_perm, lambda_const)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()    

    loss_list_train.append(loss_train/i)
    if verbose > 1 :
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
    if verbose > 1 :
        print("\n","loss par epoch valid =",loss_valid/(nb_batch+1))

    #Early-Stopping
    if loss_valid/(nb_batch+1) < early_stop[0]:
        early_stop[0] = loss_valid/(nb_batch+1)
        early_stop[1] = 0

    else:
        early_stop[1] += 1

    if verbose > 1 :
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
        if verbose > 1:
            print("Weights saved !")
        save_value = accuracy

    if verbose > 1:
        print(f'Accuracy of the network on test images: {round(accuracy, 2)}%')
    accuracy_list.append(accuracy)

    #End training if early stop reach the patience
    if early_stop[1] == patience:
        break 

    
  return model, loss_list_train,loss_list_valid, accuracy_list, save_value 


#####################################################
#VIZUALIZATION FUNCTIONS

#Function to plot validation and train loss + accuracy on test set
def plot(n_epochs, loss_list_train, loss_list_valid, save=False, title = f'Images/Binary/Loss_binary_{optimizer_name}_epochs_{n_epochs}_lr_{learning_rate}.png'):
    
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
    if save :
        fig.savefig(title)

#Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("normalized confusion matrix")
    else:
        print('confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')

#Function to get evaluation of the model (accuracy per label + overall accuracy)
def evaluation(model, test_loader, criterion):

  # initialize lists to monitor test loss and accuracy
  test_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  model.eval() # prep model for evaluation
  for data, label in tqdm(test_loader):
      data = data.to(device=device, dtype=torch.float32)
      label = label.to(device=device, dtype=torch.long)
      with torch.no_grad():
          output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
      loss = criterion(output, label)
    
      test_loss += loss.item()*data.size(0)
      _, pred = torch.max(output, 1) # convert output probabilities to predicted class
      correct = np.squeeze(pred.eq(label.data.view_as(pred))) # compare predictions to true label
      # calculate test accuracy for each object class
      for i in range(len(label)):
          digit = label.data[i]
          class_correct[digit] += correct[i].item()
          class_total[digit] += 1

  # calculate and print avg test loss
  test_loss = test_loss/len(test_loader.sampler)
  print('\ntest Loss: {:.6f}\n'.format(test_loss))
  for i in range(10):
    try :
          print('test accuracy of %1s: %2d%% (%2d/%2d)' % (class_names[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    except:
        pass 
  accuracy_overall = 100. * np.sum(class_correct) / np.sum(class_total)
  print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)\n' % (accuracy_overall, np.sum(class_correct), np.sum(class_total)))

  return accuracy_overall



#####################################################
#PRUNING FUNCTIONS

#Function to prune locally
def local_pruning(model, amount):
    for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers 
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        # prune 40% of connections in all linear layers 
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

#Function to prune globally
def global_pruning(model,amount,conv2d_flag=True,linear_flag=True,BN_flag=False):
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

    return model




#%%
#####################################################
#TRAINING

model = DenseNet121(cifar)
#model = DenseNet121bis()


#model = import_transfer_learning_model(model, cifar, path=path_model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

model, loss_list_train, loss_list_valid, accuracy_list,best_accuracy = train_model(model,
                                                                                trainloader,
                                                                                validloader,
                                                                                testloader,
                                                                                n_epochs,
                                                                                mixup=True)

acc_overall = evaluation(model, testloader, criterion)

#torch.save(model.state_dict(), f'Models/{model_name}_{optimizer_name}_epochs_{n_epochs}.pth')
#n_model = pruning(model)





#%%
#Time
stop = time.time()
execution_time = stop - start
print(f"Program Executed in {round(execution_time,2)}s")

import matplotlib.pyplot as plt
plot(n_epochs, loss_list_train, loss_list_valid, title = f'Images/Binary/Verif/Loss_{optimizer_name}_epochs_{n_epochs}_lr_{learning_rate}.png',save=True)





















###########################################
# CEMETERY



#[NOT WORKING] Function to plot accuracy of a model according to the percentage pruned
"""def plot_accuracy_prune(nepochs=2, prune='global', freeze=True):

    accu_list = []
    prune_r = [0.01*i for i in range(100)]


    for pc in tqdm(range(len(prune_r))):
        model = DenseNet121()
        model_n = import_transfer_learning_model(model, cifar, path = path_model, freeze=freeze)

        print('Model has been correctly transfered')


        if prune =='global':
            model_n = global_pruning(model_n,prune_r[pc])
        else:
            model_n = local_pruning(model_n,prune_r[pc])
        print('Pruning done !')

        if torch.cuda.is_available() :
            model_n = model_n.to(device)    


        model_n, _, _, _, best_accuracy = train_model(model_n,
                                                    trainloader,
                                                    validloader,
                                                    testloader,
                                                    nepochs,
                                                    patience=100,
                                                    verbose =2)
        
        del model_n

        accu_list.append(best_accuracy)
    

    #Plotting accuracy depending on pruning %
    plt.plot(prune_r, accu_list, color='red')
    plt.xlabel(f'Pruning part ({prune})')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy according to prune ration with {nepochs} additional training epochs.')
    plt.savefig(f'Images/Try/accuracy_prune_{prune}')"""

#First function to prune locally
"""def pruning(model):

    new_model = deepcopy(model)

    for _, module in new_model.named_modules():

        if isinstance(module, torch.nn.BatchNorm2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, "weight")

        elif isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.4)
            prune.remove(module, "weight")
    

        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, "weight")

        #elif isinstance(module, Bottleneck):
        #   for layer in module :
        #        print(list(layer.named_parameters()))
        #    break    

    return new_model"""

#[NOT WORKING] Function to get lists of labels + predicted labels + images
"""def get_predictions(model, loader):

    MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
    STD = torch.tensor([0.2023, 0.1994, 0.2010])

    preds = torch.tensor([], dtype=torch.long)
    targets = torch.tensor([], dtype=torch.long)
    images = torch.tensor([], dtype=torch.float32)

    images_np = []

    for data, label in tqdm(loader):
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)

        with torch.no_grad():
            output = model(data)

        images = torch.cat((images, data.cpu()), dim = 0)
        targets = torch.cat((targets, label.cpu()), dim = 0)
        preds = torch.cat((preds, torch.max(output.cpu(), 1)[1]), dim = 0)

    for x in tqdm(images) :
        x = x * STD[:, None, None] + MEAN[:, None, None]
        x = np.clip(x, 0,1)
        x = x.numpy().transpose(1, 2, 0)
        images_np.append(x)

    return targets.numpy(), preds.numpy(), images_np"""

#Function to import pretrained model and initialize transfer learning
"""def import_transfer_learning_model(model, n_cifar, path = 'Models/DenseNet121_Adam_epochs_50.pth', freeze=False):
    
    #Load of weight
    state_dict = torch.load(path,  map_location=device)
    #print(state_dict.keys())
    model.load_state_dict(state_dict['net'])

    n_inputs, n_classes = 1024, n_cifar

    if freeze :
        #Freeze weights
        for param in model.parameters():
            param.requires_grad = False

    #Replacing the last layer with a NN
    model.linear = nn.Sequential(
                        nn.Linear(n_inputs, 8), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(8, n_classes),                   
                        nn.LogSoftmax(dim=1))

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    print(f'{total_trainable_params:,} training parameters.')

    return model"""

#Uncomment to plot images missclassified in Images/Try/
'''
preds, targets, images = get_predictions(model, testloader)

index = np.where(preds - targets != 0)[0]
plt.figure(figsize=(25, 4))

for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.axis('off')
    plt.imshow(images[index[i]], cmap='gray')
    plt.title("{} ({})".format(class_names[int(preds[index[i]])], class_names[int(targets[index[i]])]), color=("red"))
    plt.savefig('Images/Try/Misclassified.png')
    plt.show()'''

#Uncomment to plot confusion matrix in Images/Try/
'''#Compute and plot confusion matrix
cnf_matrix = confusion_matrix(targets, preds)
np.set_printoptions(precision=2)

plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig('Images/Try/Confusion_Matrix.png')'''
