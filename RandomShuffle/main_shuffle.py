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
import matplotlib.image as mpimg

from utils2 import progress_bar
from resnet_drop_012_09 import ResNet18
from resnet import resnet18
import csv
import utils

net1 = torch.nn.DataParallel(ResNet18().cuda()).eval()
utils.load_state_ckpt('/home1/luotg/Imagenet/01209/model.pth-100', net1)
net2 = torch.nn.DataParallel(resnet18(pretrained=True).cuda()).eval()
index = np.load('index.npy')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class testset(Dataset):
    def __init__(self, dir1):
        self.img_dir = dir1
        self.imgpath_list = sorted(os.listdir(self.img_dir),key=lambda x:int(x.split('_')[0]))
    def __getitem__(self, index):
        ##correct the label for the pretrained model.
        #img = scipy.misc.imread(self.img_dir + self.imgpath_list[index])
        #img = Image.fromarray(img)
        img = mpimg.imread(self.img_dir + self.imgpath_list[index])
        label = np.int64(self.imgpath_list[index].split('_')[-1].split('.')[0])
        img = transform_test(img)
        return img, label
    def __len__(self):
        return len(self.imgpath_list)

testloaderf = torch.utils.data.DataLoader(testset('sample_01209/'), batch_size=128, shuffle=False, num_workers=2)
testloader0 = torch.utils.data.DataLoader(testset('imagenet_shuffle_2/'), batch_size=128, shuffle=False, num_workers=2)
testloader1 = torch.utils.data.DataLoader(testset('imagenet_shuffle_4/'), batch_size=128, shuffle=False, num_workers=2)
testloader2 = torch.utils.data.DataLoader(testset('imagenet_shuffle_8/'), batch_size=128, shuffle=False, num_workers=2)

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
            for i in range(predicted.size()[0]):
                predicted[i] = torch.from_numpy(np.array(index[predicted[i]]))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(total)
    return acc

net_dict = {'norm':net2}
#net_dict = {'01209':net1}
w = csv.writer(open('shuffle.csv', 'a'))
w.writerow(['model','1x1','2x2','4x4','8x8'])

for k in net_dict.keys():
    print(k)
    net = net_dict[k]
    acc=[]
    acc.append(k)
    acc.append(test(testloaderf))
    acc.append(test(testloader0))
    acc.append(test(testloader1))
    acc.append(test(testloader2))
    w.writerow(acc)
