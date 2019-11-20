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

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                # transforms.RandomCrop(32, 4),
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
from senet import SENet18
from densenet import DenseNet121
from resnet import ResNet50
from vgg import VGG
from pytorch_cifar.resnet_drop_012_07 import ResNet18 as n4
from pytorch_cifar.resnet_drop_012_05 import ResNet18 as n5
from pytorch_cifar.resnet_drop_012_09 import ResNet18 as n2
from pytorch_cifar.resnet_drop_012_05_samer2 import ResNet18 as n12
from pytorch_cifar.resnet32_k4 import resnet32_cifar as n15
from pytorch_cifar.resnet32_k10 import resnet32_cifar as n16
from pytorch_cifar.resnet_nobn import ResNet18 as n17
from pytorch_cifar.resnet_nobn import ResNet18 as n18

net1 = torch.nn.DataParallel(ResNet18().cuda()).eval()
net1.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt4.t7')['net'])
net2 = torch.nn.DataParallel(SENet18().cuda()).eval()
net2.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_senet.t7')['net'])
net3 = torch.nn.DataParallel(DenseNet121().cuda()).eval()
net3.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_dense.t7')['net'])
net4 = torch.nn.DataParallel(ResNet50().cuda()).eval()
net4.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_resnet50.t7')['net'])
net5 = torch.nn.DataParallel(VGG('VGG19').cuda()).eval()
net5.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_vgg.t7')['net'])
net6 = torch.nn.DataParallel(n2().cuda()).eval()
net6.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_drop_012_09.t7')['net'])
net8 = torch.nn.DataParallel(n4().cuda()).eval()
net8.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_drop_012_07.t7')['net'])
net9 = torch.nn.DataParallel(n5().cuda()).eval()
net9.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_drop_012_05.t7')['net'])
net19 = torch.nn.DataParallel(n15().cuda()).eval()
net19.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_resnet32_k4_norm.t7')['net'])

net_dict={'32k4':net19}
#net_dict={'res18':net1, 'senet':net2, 'dense':net3, 'res50':net4, 'vgg':net5}
path_list = sorted(os.listdir('t_sample_testdata'),key=lambda x:int(x[:-6]))
criterion = nn.CrossEntropyLoss()
criterion.cuda()
tn = 5000

#parameters = [4, 8, 16, 32]
parameters = [ 1, 2, 4]
up = torch.from_numpy(((np.ones([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()
down = torch.from_numpy(((np.zeros([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()

fo = open('FGSM/info.txt','a')
for k in net_dict.keys():
    net = net_dict[k]
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)
    for parameter in parameters:
        stepsize = parameter

        learning_rate = 1./255/min(std)*stepsize
        Stats = np.zeros(2)
        count = 0

        score_arr = np.zeros(tn)
        aescore_arr = np.zeros(tn)
        norm_min_arr = np.zeros(tn)
        norm_max_arr = np.zeros(tn)

        os.mkdir('FGSM/'+k+'_'+str(stepsize))
        for i in range(tn):
            oriimg = mpimg.imread('t_sample_testdata/'+path_list[i])
            labels = int(path_list[i].split('_')[1].split('.')[0])
            img = transforms.ToTensor()(oriimg).type(torch.FloatTensor)
            img = normalize(img)
            img = img.unsqueeze(0)
            inputs = img
            labels = torch.from_numpy(np.array(labels))
            labels = labels.unsqueeze(0)

            if count == tn:
                    break

            inputs = Variable(inputs.cuda(), requires_grad=True)
            outputs = net(inputs.cuda())
            score, predicted = torch.max(nn.Softmax()(outputs), 1)
            if labels != predicted.data.cpu():
                continue
                raise NameError('predict wrong')
            score_arr[count] = score.data.cpu().numpy()[0]
            #predict_arr[count] = predicted.data.cpu().numpy()[0]

            labels = Variable(labels.cuda())
            net.zero_grad()
            for param in net.parameters():
            	param.requires_grad = False
            inputs_r = inputs.clone()
            outputs = net(inputs.cuda())
            loss = criterion(outputs.cuda(), labels.cuda())
            loss.backward()
            inputs.data.add_(learning_rate*torch.sign(inputs.grad.data))
            inputs.data[inputs.data > up.data] = up.data[inputs.data > up.data]
            inputs.data[inputs.data < down.data] = down.data[inputs.data < down.data]
            outputs = net(inputs.cuda())
            _, predicted = torch.max(outputs.data, 1)
            s1 = (predicted == labels.data)
            s1 = s1.cpu().numpy()
            norm_min_arr[count] = (inputs.data-inputs_r.data).min().cpu().numpy()
            norm_max_arr[count] = (inputs.data-inputs_r.data).max().cpu().numpy()
            if s1:
                Stats[0] += 1
            	#print ('Not Changed')
            else:
            	Stats[1] += 1
            	#print ('Changed')
            score, rank = torch.max(nn.Softmax()(outputs), 1)
            aescore_arr[count] = score.data.cpu().numpy()[0]
            save_img(inputs.data.cpu().numpy(), 'FGSM/'+k+'_'+str(stepsize)+'/', str(count)+'_'+str(labels.data.cpu().numpy()[0])+'_'+str(rank.data.cpu().numpy()[0]))
            count += 1

        fo.write(k+'_'+str(stepsize)+'\n')
        fo.write('Not changed: %.6f\n' % (Stats[0]/tn))
        fo.write('Changed: %.6f\n' % (Stats[1]/tn))
        fo.write('Min Max Norm: %.2f %.2f\n' % (np.min(norm_min_arr)*min(std)*255, np.max(norm_max_arr)*min(std)*255))
        fo.write('Mean Norm and AE Predict score: %.3f %.3f\n\n' % (np.mean(score_arr), np.mean(aescore_arr)))
