'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
import scipy.misc
from PIL import Image
import numpy as np
import IPython

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18 as n1
from resnet_drop_012_05 import ResNet18 as n2
from resnet_drop_012_07 import ResNet18 as n3
from resnet_drop_012_09 import ResNet18 as n4
from utils import progress_bar
import csv

net0 = torch.nn.DataParallel(n1().cuda()).eval()
net0.load_state_dict(torch.load('checkpoint/ckpt3.t7')['net'])
net1 = torch.nn.DataParallel(n2().cuda()).eval()
net1.load_state_dict(torch.load('checkpoint/ckpt_drop_12_05_first.t7')['net'])
net2 = torch.nn.DataParallel(n2().cuda()).eval()
net2.load_state_dict(torch.load('checkpoint/ckpt_drop_12_07_first.t7')['net'])
net3 = torch.nn.DataParallel(n3().cuda()).eval()
net3.load_state_dict(torch.load('checkpoint/ckpt_drop_12_09_first.t7')['net'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
class testset(Dataset):
    def __init__(self, dir1):
        self.img_dir = dir1
        self.imgpath_list = sorted(os.listdir(self.img_dir),key=lambda x:int(x[:-6]))
    def __getitem__(self, index):
        img = scipy.misc.imread(self.img_dir + self.imgpath_list[index])
        img = Image.fromarray(img)
        label = np.int64(self.imgpath_list[index].split('_')[-1].split('.')[0])
        img = transform_test(img)
        return img, label
    def __len__(self):
        return len(self.imgpath_list)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader0 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_2/'), batch_size=128, shuffle=False, num_workers=2)
testloader1 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_4/'), batch_size=128, shuffle=False, num_workers=2)
testloader2 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_8/'), batch_size=128, shuffle=False, num_workers=2)
testloader3 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_16/'), batch_size=128, shuffle=False, num_workers=2)
testloader4 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_24/'), batch_size=128, shuffle=False, num_workers=2)
testloader5 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_32/'), batch_size=128, shuffle=False, num_workers=2)
testloader6 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_40/'), batch_size=128, shuffle=False, num_workers=2)
testloader7 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_48/'), batch_size=128, shuffle=False, num_workers=2)
testloader8 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_56/'), batch_size=128, shuffle=False, num_workers=2)
testloader9 = torch.utils.data.DataLoader(testset('../cifar/test_randomnoise_012_64/'), batch_size=128, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
def test(testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    return acc

net_dict = {'norm':net0, '01205':net1, '01207': net2, '01209':net3}
w = csv.writer(open('../randomnoise_012.csv', 'a'))

for k in net_dict.keys():
    print(k)
    net = net_dict[k]
    acc=[]
    acc.append(k)
    acc.append(test(testloader0))
    acc.append(test(testloader1))
    acc.append(test(testloader2))
    acc.append(test(testloader3))
    acc.append(test(testloader4))
    acc.append(test(testloader5))
    acc.append(test(testloader6))
    acc.append(test(testloader7))
    acc.append(test(testloader8))
    acc.append(test(testloader9))
    w.writerow(acc)
