import os
from models_cifar_10.resnet import ResNet18, ResNet9
from models_cifar_10.densenet import DenseNet121
import torch
import torch.nn as nn

from functions import *
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

def count_conv2d(m, x, y):
    x = x[0] # remove tuple

    fin = m.in_channels
    fout = m.out_channels
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul/2 # FP16
    ops = kernel_mul + kernel_add + bias_ops

    # Factorization
    factorization = m.groups 
    #print("FACTO",factorization)

    # total ops
    num_out_elements = y.numel()
    #print("KFJFKDLSM",num_out_elements)
    total_ops = num_out_elements * ops
    #print("OOPPSS",ops)
    #print(f"Conv2d: S_c={sh}, F_in={fin}, F_out={fout}, P={x.size()[2:].numel()}, params={int(m.total_params.item())}, operations={int(total_ops)}")
    # incase same conv is used multiple times

    m.total_ops += torch.Tensor([int(total_ops/factorization)])

def count_bn2d(m, x, y):
    x = x[0] # remove tuple

    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])
    
    #print(f"Batch norm: F_in={x.size(1)} P={x.size()[2:].numel()}, params={int(m.total_params.item())}, operations={int(total_ops)}")

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])
    #print(f"ReLU: F_in={x.size(1)} P={x.size()[2:].numel()}, params={0}, operations={int(total_ops)}")

def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    #print(f"AvgPool: S={m.kernel_size}, F_in={x.size(1)}, P={x.size()[2:].numel()}, params={0}, operations={int(total_ops)}")

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features/2
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    print(f"Linear: F_in={m.in_features}, F_out={m.out_features}, params={int(m.total_params.item())}, operations={int(total_ops)}")
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    pass
    #print ("Sequential: No additional parameters  / op")

# custom ops could be used to pass variable customized ratios for quantization
def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()]) / 2 # Division Free quantification

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
        else:
            pass
            #print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        #total_params += m.total_params
        #print("M.TOTAL_PARAMS",m.total_params)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m,nn.BatchNorm2d):
            # For pruning
            weight_used = float(torch.sum(m.weight !=0 ))/2 #Weight multiplicated by 2 compared to m.total_params
            #print("WEIGHTED LAYERS",weight_used)
            total_params += weight_used
        else: #For sequential
            #print("SEQUANTIAL",m.total_params)
            total_params += m.total_params

    return total_ops, total_params

import sys, os
p = os.path.abspath('.')
print('GERFZ',p)
sys.path.insert(1, p)

from functions import load_weights, global_pruning, to_device
from models_cifar_10.resnet20 import resnet20
def main():
    # Resnet18 - Reference for CIFAR 10
    ref_params = 5586981
    ref_flops  = 834362880
    # WideResnet-28-10 - Reference for CIFAR 100
    # ref_params = 36500000
    # ref_flops  = 10490000000

    factorization_flag = True
    groups = 2
    #model = resnet20(factorization_flag)
    #model = ResNet18(factorization_flag)
    model = resnet20(factorization_flag,groups)
    

    #HyperParameters
    
    amount = 0.85
    conv2d_flag,linear_flag,BN_flag = True,True,False
    model = global_pruning(model,amount,None,conv2d_flag,linear_flag,BN_flag)
   
    flops, params = profile(model, (1,3,32,32))
  
    flops, params = flops.item(), params.item()

    score_flops = flops / ref_flops
    score_params = params / ref_params
    score = score_flops + score_params
    print(f"Flops: {flops}, Params: {params}")
    print(f"Score flops: {score_flops} Score Params: {score_params}")
    print(f"Final score: {score}")

    #model = resnet20(False,1)
    model = ResNet18(False)
    flops1, params1 = profile(model, (1,3,32,32))
  
    flops1, params1 = flops1.item(), params1.item()

    score_flops1 = flops1 / ref_flops
    score_params1 = params1 / ref_params
    score = score_flops1 + score_params1
    print(f"Flops1: {flops1}, Params1: {params1}")
    print(f"Score flops1: {score_flops1} Score Params1: {score_params1}")
    print(f"Final score: {score}")

    print(f"Reduced by {100*round((1-flops/flops1),6)} the number of flops")
    print(f"Reduced by {100*round((1-params/params1),6)} the number of params")
if __name__ == "__main__":
    main()
