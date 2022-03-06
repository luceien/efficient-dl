from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader

import torchvision.transforms as transforms
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),

    transforms.RandomHorizontalFlip(),

    transforms.RandomAffine(5),

    transforms.RandomRotation(20),

    transforms.ColorJitter(brightness=(0.4,1), 
                            contrast=(0.4,1), 
                            saturation=(0.3,0.9)
                            ),
    #transforms.RandomInvert(p=0.05),
    transforms.ToTensor(),
    normalize_scratch,
])


rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)

trainloader = DataLoader(c10train,batch_size=9,shuffle=False) ### Shuffle to False so that we always see the same images

from matplotlib import pyplot as plt 

### Let's do a figure for each batch
f = plt.figure(figsize=(10,10))

for i,(data,target) in enumerate(trainloader):
    print(target)
    data = (data.numpy())
    print(data.shape)
    plt.subplot(3,3,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))
    
    plt.subplot(3,3,5)
    plt.imshow(data[4].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,6)
    plt.imshow(data[5].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,7)
    plt.imshow(data[6].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,8)
    plt.imshow(data[7].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,9)
    plt.imshow(data[8].swapaxes(0,2).swapaxes(0,1))

    break
plt.show()
f.savefig('Images/DA/train_DA.png')