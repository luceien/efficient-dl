#%%

from torchvision.datasets import CIFAR10
import numpy as np 
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models

import torchvision.transforms as transforms
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torch

#%%

trainloader = DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)
testloader = DataLoader(minicifar_test,batch_size=32) 

###classes = ('plane', 'car', 'bird', 'cat')


#%%
#Functions to show an image
'''def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images and check the size
dataiter = iter(trainloader)
images, labels = dataiter.next()

print('batch size:', images.size(0))
print('color channels :', images.size(1))
print('Image size:'+ str(images.size(2))+ 'x'+ str(images.size(3)))

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s\t' % classes[labels[j]] for j in range(4)))'''


# %%
#TASK 1 : IMPORT AND TRAIN A MODEL
from densenet import DenseNet121, test

dn = DenseNet121()

#test()
print(f"Number of parameters of densenet121: {sum(p.numel() for p in dn.parameters())}")




#%%
#Import optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(dn.parameters(), lr=0.001, momentum=0.9)


#Train model
n_epochs=50

for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = dn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')






'''
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
'''








#%%

#TASK 2 : FIGURE ACCURACY VS NUMBER OF PARAMETERS

'''from vgg import VGG
from resnet import ResNet18
from preact_resnet import PreActResNet18

vgg = VGG('VGG11')
rn = ResNet18()
prn = PreActResNet18()

#Number of parameters in dn 
n_param = [sum(p.numel() for p in dn.parameters()), 
            sum(p.numel() for p in vgg.parameters()),
            sum(p.numel() for p in rn.parameters()), 
            sum(p.numel() for p in prn.parameters())]

acc = [95.04, 92.64, 93.75, 94.73]
n = ['denseNet121', 'vgg16', 'resnet18', 'preact_resnet']


fig, ax = plt.subplots()
ax.scatter(n_param, acc)

for i, txt in enumerate(n):
    ax.annotate(txt, (n_param[i], acc[i]))

plt.show()'''


# %%
