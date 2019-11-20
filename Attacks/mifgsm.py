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

net1 = torch.nn.DataParallel(ResNet18().cuda()).eval()
net1.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt4.t7')['net'])
net_dict = {'res18':net1}

path_list = sorted(os.listdir('t_sample_testdata'),key=lambda x:int(x[:-6]))
criterion = nn.CrossEntropyLoss()
criterion.cuda()
tn = 5000

parameters =[#(1, 10,  8.0),
#             (1, 20, 16.0),
#             (2, 10, 16.0),
             (1, 40, 32.0),]
#             (2, 20, 32.0)]

fo = open('MIFGSM/info.txt','a')
randstart = False
#randstart = True

up = torch.from_numpy(((np.ones([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()
down = torch.from_numpy(((np.zeros([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()
bound_base = torch.from_numpy((np.ones([3,32,32])/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)).type(torch.FloatTensor).cuda()

for k in net_dict.keys():
    print(k)
    net = net_dict[k]
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)
    if randstart:
        k += '_randstart'
    for parameter in parameters:
        stepsize = parameter[0]
        steps = parameter[1]
        epsilon = parameter[2]

        learning_rate = torch.from_numpy((1./255/np.array(std)*stepsize).reshape(1, len(std), 1, 1)).type(torch.FloatTensor).cuda()
        #bound = epsilon/255/min(std)
        bound = epsilon/255 * bound_base
        Stats = np.zeros(2)
        count = 0

        score_arr = np.zeros(tn)
        aescore_arr = np.zeros(tn)
        norm_min_arr = np.zeros(tn)
        norm_max_arr = np.zeros(tn)

        os.mkdir('MIFGSM/'+k+'_'+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'/')
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
                #raise NameError('predict wrong')
            score_arr[count] = score.data.cpu().numpy()[0]

            labels = Variable(labels.cuda())
            net.zero_grad()
            for param in net.parameters():
            	param.requires_grad = False
            inputs_r = inputs.clone()
            if randstart:
                inputs.data += torch.zeros(inputs.shape).cuda().uniform_(-epsilon/255, epsilon/255)
                inputs.data[inputs.data > up.data] = up.data[inputs.data > up.data]
                inputs.data[inputs.data < down.data] = down.data[inputs.data < down.data]

            g = 0
            for p in range (steps):
                inputs = Variable(torch.from_numpy(inputs.cpu().data.numpy()).cuda(), requires_grad=True)
            	outputs = net(inputs.cuda())
            	loss = criterion(outputs.cuda(), labels.cuda())
            	loss.backward()
                g = g + inputs.grad.data.cpu().numpy()/torch.norm(inputs.grad.data,1).cpu().data.numpy()
            	inputs.data.add_(learning_rate*torch.sign(torch.from_numpy(g).cuda()))
                inputs.data[inputs.data - inputs_r.data > bound.data] = inputs_r.data[inputs.data - inputs_r.data > bound.data] + bound.data[inputs.data - inputs_r.data > bound.data]
                inputs.data[inputs.data - inputs_r.data < -bound.data] = inputs_r.data[inputs.data - inputs_r.data < -bound.data] - bound.data[inputs.data - inputs_r.data < -bound.data]
                inputs.data[inputs.data > up.data] = up.data[inputs.data > up.data]
                inputs.data[inputs.data < down.data] = down.data[inputs.data < down.data]
                if p == steps-1:
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
                    #aepredict_arr[count] = rank.data.cpu().numpy()[0]
                    aescore_arr[count] = score.data.cpu().numpy()[0]
                    save_img(inputs.data.cpu().numpy(), 'MIFGSM/'+k+'_'+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'/', str(i)+'_'+str(labels.data.cpu().numpy()[0])+'_'+str(rank.data.cpu().numpy()[0]))
            count += 1

        fo.write(k+'_'+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'\n')
        fo.write('Not changed: %.6f\n' % (Stats[0]/count))
        fo.write('Changed: %.6f\n' % (Stats[1]/count))
        fo.write('Min Max Norm: %.2f %.2f\n' % (np.min(norm_min_arr)*min(std)*255, np.max(norm_max_arr)*min(std)*255))
        fo.write('Mean Norm and AE Predict score: %.3f %.3f\n\n' % (np.mean(score_arr), np.mean(aescore_arr)))
        fo.flush()
fo.close()
