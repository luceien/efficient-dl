#Name of Model, optimizer etc.. for Graph file's name
model_name = 'ResNet'
optimizer_name = 'SGD'

from models_cifar_10.resnet import ResNet18

from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

#from functions_lucien import *
from functions import *
from resnet20 import resnet20


def main():
    #Model chosen from Vgg, Densenet, Resnet
    factorization_flag = False
    model = resnet20(factorization_flag)
    model,device = to_device(model)
    #path_model = 'Models\Accuracy_90\resnset20_SGD_epochs_400_acc91.9.pth'
    #model = load_weights(model,path_model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    #HyperParameters
    Niter, Bsize, lr = 399, 32, 0.01
    earlystop_flag = False

    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True) 

    train_model, loss_list_train,loss_list_valid, accuracy_list, best_accuracy, test_accuracy, _ = training_model_with_mixup(model, trainloader,validloader,testloader ,device,lr,Niter,earlystop_flag,optimizer_name)
    print(f'The best accuracy on validation for the saved model is: {best_accuracy}%')
    plot1(loss_list_train,loss_list_valid, accuracy_list, test_accuracy,Niter,lr)



def pruning(amount=0.5):

    factorization_flag = False
    model = resnet20(factorization_flag)
    #path_model = 'Models\Accuracy_90\resnset20_SGD_epochs_400_acc91.9.pth'
    #model = load_weights(model,path_model)

    model, device = to_device(model)

    #HyperParameters
    Niter, Bsize, lr = 398 , 32, 0.01
    conv2d_flag, linear_flag, BN_flag = True, True, False
    earlystop_flag = False
    model = global_pruning(model,amount,device,conv2d_flag,linear_flag,BN_flag)

    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True)
    print_prune_details(model)

    acc = getAccuracy(model,testloader,device)
    print('Accuracy : ',acc)

    train_model, loss_list_train,loss_list_valid, accuracy_list, best_accuracy, test_accuracy, _ = training_model_with_mixup(model, trainloader,validloader,testloader ,device,lr,Niter,earlystop_flag,optimizer_name)
    print(f'The best accuracy on validation for the saved model is: {best_accuracy}%')

    plot1(loss_list_train,loss_list_valid, accuracy_list, test_accuracy,Niter,lr)


def training_iterative_pruning():
    model = ResNet18()
    path_model = 'Models/Accuracy_90/SGD_epochs_101_acc94.43.pth'
    model = load_weights(model,path_model)

    model, device = to_device(model)

    #HyperParameters
    Niter, Bsize, lr = 40 , 32, 0.01
    trainloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=Bsize,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=Bsize,shuffle=True)
    #print_prune_details(model)

    acc = getAccuracy(model,testloader,device)
    print('Initial accuracy : ',acc)

    _, accuracy_list, prune_list, number_epochs, saved_value = iterative_pruning(model, trainloader, validloader, testloader, device, lr)

    print(f'The best accuracy on validation for the saved model is: {saved_value}%')

    plot_prune_accuracy(accuracy_list, prune_list, number_epochs)





if __name__=='__main__':
    main()
    #pruning()
    #training_iterative_pruning()
    #loading()