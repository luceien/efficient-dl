#%%
model_name = 'DenseNet121'
optimizer_name = 'Adam'
learning_rate = 0.0001
batch_size = 64
n_epochs = 50
path_model = 'Models/DenseNet121_Adam_epochs_25.pth'





#%%
import numpy as np
import torch
from torchinfo import summary 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models_cifar_10.densenet import DenseNet121
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler

from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train, batch_size=batch_size, sampler=train_sampler)
validloader = DataLoader(minicifar_train, batch_size=batch_size, sampler=valid_sampler)
testloader = DataLoader(minicifar_test, batch_size=batch_size, shuffle=True) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121()
#print(device)
#model.eval()

print('\n\nImport done\n\n')




#%%

def import_transfer_learning_model(model, path = 'Models/DenseNet121_Adam_epochs_25.pth', freeze= False):
    
    #Load of weights
    state_dict = torch.load(path,  map_location=device)['net']
    model.load_state_dict(state_dict)

    if freeze :
        n_inputs, n_classes = 1024, 4
        #Freeze weights
        for param in model.parameters():
            param.requires_grad = False

        #Replacing the last layer with a NN
        model.linear = nn.Sequential(
                            nn.Linear(n_inputs, 64), 
                            nn.ReLU(), 
                            nn.Dropout(0.4),
                            nn.Linear(64, n_classes),                   
                            nn.LogSoftmax(dim=1))

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)

        print(f'{total_trainable_params:,} training parameters.')


    print(f'\n\n {summary(model)}\n\n')

    return model


def train_model(model, train_loader, valid_loader, test_loader, EPOCHS, half = False, patience=30):

    #Optimizer & criterion
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    loss_list_train = []
    loss_list_valid = []
    accuracy_list = []
    accuracy_list_half = []
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

        #print(f'Inputs : \n\n{inputs}\n\n')
        loss_list_train.append(loss_train/i)
        print("\n","loss par epoch train =",np.round(loss_train/(nb_batch+1),4))

        #Validation 
        loss_valid = 0
        
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
    
        loss_list_valid.append(loss_valid/i)
        print("\n","loss par epoch valid =",np.round(loss_valid/(nb_batch+1),4))


        print(f'Validation loss did not change for {early_stop[1]} epochs')

        #Test
        correct = 0
        total = 0
        with torch.no_grad():  # torch.no_grad for TESTING
            for data in tqdm(test_loader):

                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total

        print(f'Accuracy of the network on test images : {100 * np.round(correct/total,4)}%')
        accuracy_list.append(accuracy)

        '''if half :
            correct = 0
            total = 0
            model2 = model.half().to(device)

            with torch.no_grad():  # torch.no_grad for TESTING
                for data2 in tqdm(test_loader):

                    images2, labels2 = data2
                    images2 = images2.half()

                    if torch.cuda.is_available():
                        images2, labels2 = images2.to(device), labels2.to(device)

                    outputs = model2(images2)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels2.size(0)
                    correct += (predicted == labels2).sum().item()
                    accuracy = 100 * correct / total

            print(f'Accuracy of the network on reduced test images: {100 * np.round(correct/total,4)}%')
            accuracy_list_half.append(accuracy)
            model.float()'''


    #from scikit-learn import classification_report    
    return model, loss_list_train,loss_list_valid, accuracy_list



#%%

#Part 1 - Quantization to half and integer precision
model = import_transfer_learning_model(model, path=path_model, freeze = True)
model.to(device)

print(f'\n\nModel has been correctly load\n\n')


train_model, loss_list_train,loss_list_valid, accuracy = train_model(model, trainloader, validloader, testloader, n_epochs)



'''import matplotlib.pyplot as plt
epochs = [k+1 for k in range(len(loss_list_train))]
plt.plot(epochs, accuracy, color='b')
plt.plot(epochs, accuracy_red, color='r')

plt.savefig(f'Images/Comparison_half_and_not.png')'''

# %%
