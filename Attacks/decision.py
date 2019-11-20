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
import foolbox
import copy

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
mean_box = np.array(mean).reshape((3,1,1))
std_box = np.array(std).reshape((3,1,1))
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=1, shuffle=True,
            num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True)
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

from resnet import ResNet18
from pytorch_cifar.resnet_drop_012_09 import ResNet18 as n14
from resnet import ResNet18 as n28

net1 = torch.nn.DataParallel(ResNet18().cuda()).eval()
net1.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt4.t7')['net'])
net14 = torch.nn.DataParallel(n10().cuda()).eval()
net14.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_drop_012_09.t7')['net'])
net28 = torch.nn.DataParallel(n28().cuda()).eval()
net28.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_res18_labelsmooth_01.t7')['net'])

net_dict={'res18' : net1}
#net_dict={'012_09' : net14}
#net_dict={'labelsmooth_01' : net28}

path_list = sorted(os.listdir('t_sample_testdata'),key=lambda x:int(x[:-6]))
criterion = nn.CrossEntropyLoss()
criterion.cuda()
tn = 500
fo = open('BOUNDARY/info.txt','a')

up = torch.from_numpy(((np.ones([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()
down = torch.from_numpy(((np.zeros([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()
bound_base = torch.from_numpy((np.ones([3,32,32])/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()

oris = np.zeros([tn,3,32,32])
aes = np.zeros([tn,3,32,32])
ori_labels = np.zeros(tn)
ae_labels = np.zeros(tn)

for k in net_dict.keys():
    print(k)
    net = net_dict[k]
    Stats = np.zeros(2)
    count = 0

    score_arr = np.zeros(tn)
    aescore_arr = np.zeros(tn)
    norm_min_arr = np.zeros(tn)
    norm_max_arr = np.zeros(tn)
    model = foolbox.models.PyTorchModel(net, bounds=(0,1), num_classes=10, preprocessing=(mean_box, std_box))
    attack = foolbox.attacks.BoundaryAttack(model)

    flag = 0
    none_count = 0

    #os.mkdir('DECISION/'+k+'/')
    for i in range(5000):
        if count == tn:
                break
        oriimg = mpimg.imread('t_sample_testdata/'+path_list[i])
        labels = int(path_list[i].split('_')[1].split('.')[0])
        img = transforms.ToTensor()(oriimg).type(torch.FloatTensor)
        img_box = copy.deepcopy(img.numpy())
        oris[count,:]=copy.deepcopy(img_box)
        label_box = labels
        ori_labels[count] = label_box

        img = normalize(img)
        img = img.unsqueeze(0)
        inputs = img
        labels = torch.from_numpy(np.array(labels))
        labels = labels.unsqueeze(0)
        inputs = Variable(inputs.cuda(), requires_grad=True)
        outputs = net(inputs.cuda())
        score, predicted = torch.max(nn.Softmax()(outputs), 1)
        if labels != predicted.data.cpu():
            continue
            #raise NameError('predict wrong')
        score_arr[count] = score.data.cpu().numpy()[0]

        ae = attack(img_box, label_box)
        aes[count,:] = copy.deepcopy(ae)
        if ae is None:
            none_count +=1
            continue

        ae = normalize(torch.from_numpy(ae))
        score, predicted = torch.max(nn.Softmax()(net(ae.unsqueeze(0).cuda())), 1)
        aescore_arr[count] = score.data.cpu().numpy()[0]
        ae_labels[count] = predicted.data.cpu().numpy()[0]
        count += 1
norm = np.square(np.linalg.norm((oris - aes).reshape(tn, -1), axis=1))/3072
IPython.embed()
#res18
#mean   1.2e-05
#median 7.3e-06

#01209
#mean   5.6e-05
#median 3.5e-05

#ls_eps01
#mean   1.1e-05
#median 6.8e-06
