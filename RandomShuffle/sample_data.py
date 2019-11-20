import torch
import os
import IPython
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import math as ms
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import random as rd
from PIL import Image
import IPython
import matplotlib.image as mpimg
import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
  '/home/imagenet/val',
  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
  ]))
train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=1, shuffle=False,
  num_workers=16, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
  datasets.ImageFolder('/home/imagenet/val/', transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
  ])),
  batch_size=1, shuffle=False,
  num_workers=16, pin_memory=True)
def save_img(kk, path, name):
        kk = kk[0,:,:,:]
        k1 = kk[0,:,:]
        k2 = kk[1,:,:]
        k3 = kk[2,:,:]
        k1 = np.around((k1*std[0]+mean[0])*255)
        k2 = np.around((k2*std[1]+mean[1])*255)
        k3 = np.around((k3*std[2]+mean[2])*255)
        r = Image.fromarray(k1).convert('L')
        g = Image.fromarray(k2).convert('L')
        b = Image.fromarray(k3).convert('L')
        raw = Image.merge('RGB', (r, g, b))
        raw.save(path+name+'.png')

from resnet_drop_012_09 import ResNet18
from resnet import resnet18
import utils

net1 = torch.nn.DataParallel(ResNet18().cuda()).eval()
utils.load_state_ckpt('/home1/luotg/Imagenet/01209/model.pth-100', net1)
net2 = torch.nn.DataParallel(resnet18(pretrained=True).cuda()).eval()
net_dict={'01207': net1}

criterion = nn.CrossEntropyLoss()
criterion.cuda()
tn = 4000

index = np.load('index.npy')
count =0

for inputs, labels in val_loader:
    if count == tn:
            break
    inputs = Variable(inputs.cuda(), requires_grad=True)
    outputs = net1(inputs.cuda())
    score, predicted = torch.max(nn.Softmax()(outputs), 1)
    if labels != predicted.data.cpu():
        continue
    if score < 0.99:
        continue
    inputs = Variable(inputs.cuda(), requires_grad=True)
    outputs = net2(inputs.cuda())
    score, predicted = torch.max(nn.Softmax()(outputs), 1)
    predicted = index[predicted.data.cpu().numpy()[0]]
    if labels.data.cpu().numpy()[0] != predicted:
        continue
    if score < 0.99:
        continue
    save_img(inputs.data.cpu().numpy(), 'sample_01209/', str(count)+'_'+str(labels.data.cpu().numpy()[0]))
    count +=1
