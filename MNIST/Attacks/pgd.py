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
import scipy.misc
import matplotlib.image as mpimg
import os
mean = [0.1307]
std = [0.3081]
normalize = transforms.Normalize(mean= [0.1307], std=[0.3081])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081)),
])

device = 'cuda'
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True, **kwargs)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, **kwargs)
from resnet import ResNet18 as n0
from resnet import ResNet50 as n1
from densenet import DenseNet121 as n2
from senet_unfold import SENet18 as n3
from vgg_unfold import VGG as n4

net0 = torch.nn.DataParallel(n0().cuda()).eval()
net0.load_state_dict(torch.load('checkpoint/res18.t7')['net'])
net1 = torch.nn.DataParallel(n1().cuda()).eval()
net1.load_state_dict(torch.load('checkpoint/resnet50.t7')['net'])
net2 = torch.nn.DataParallel(n2().cuda()).eval()
net2.load_state_dict(torch.load('checkpoint/densenet.t7')['net'])
net3 = torch.nn.DataParallel(n3().cuda()).eval()
net3.load_state_dict(torch.load('checkpoint/senet.t7')['net'])
net4 = torch.nn.DataParallel(n4('VGG19').cuda()).eval()
net4.load_state_dict(torch.load('checkpoint/vgg.t7')['net'])
def save_img(kk, path, name):
	kk = kk[0,0,:,:]
	kk = np.around((kk*std[0]+mean[0])*255)
	scipy.misc.imsave(path+name+'.png', kk)

criterion = nn.CrossEntropyLoss()
criterion.cuda()


net_dict={'res18':net0, 'res50':net1, 'dense':net2, 'senet':net3,'vgg':net4}

path_list = sorted(os.listdir('sample_testdata'),key=lambda x:int(x[:-6]))
criterion = nn.CrossEntropyLoss()
criterion.cuda()
tn = 5000

parameters = [(255*0.01, 40, 255*0.3)]

fo = open('PGD/info.txt','a')

up = (1 - np.array(mean))/np.array(std)[0]
down = (0 - np.array(mean))/np.array(std)[0]
up = up[0]
down = down[0]

for k in net_dict.keys():
    net = net_dict[k]
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)
    for parameter in parameters:
        stepsize = parameter[0]
        steps = parameter[1]
        epsilon = parameter[2]

        learning_rate = 1./255/min(std)*stepsize
        bound = epsilon/255/min(std)
        Stats = np.zeros(2)
        count = 0

        score_arr = np.zeros(tn)
        aescore_arr = np.zeros(tn)
        norm_min_arr = np.zeros(tn)
        norm_max_arr = np.zeros(tn)

        os.mkdir('PGD/'+k+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'/')
        arr = np.zeros(tn)
        for i in range(tn):
            oriimg = mpimg.imread('sample_testdata/'+path_list[i])
            labels = int(path_list[i].split('_')[1].split('.')[0])
            img = transforms.ToTensor()(oriimg.reshape(28,28,1)).type(torch.FloatTensor)
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
            #predict_arr[count] = predicted.data.cpu().numpy()[0]

            labels = Variable(labels.cuda())
            net.zero_grad()
            for param in net.parameters():
            	param.requires_grad = False
            inputs_r = inputs.clone()
            for p in range (steps):
                inputs = Variable(torch.from_numpy(inputs.cpu().data.numpy()).cuda(), requires_grad=True)
            	outputs = net(inputs.cuda())
            	loss = criterion(outputs.cuda(), labels.cuda())
            	loss.backward()
            	inputs.data.add_(learning_rate*torch.sign(inputs.grad.data))
                inputs.data[inputs.data - inputs_r.data > bound] = inputs_r.data[inputs.data - inputs_r.data > bound] + bound
                inputs.data[inputs.data - inputs_r.data < -bound] = inputs_r.data[inputs.data - inputs_r.data < -bound] - bound
                inputs.data[inputs.data > up] = up
                inputs.data[inputs.data < down] = down
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
                    save_img(inputs.data.cpu().numpy(), 'PGD/'+k+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'/', str(i)+'_'+str(labels.data.cpu().numpy()[0])+'_'+str(rank.data.cpu().numpy()[0]))
            count += 1

        fo.write(k+'_'+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'\n')
        fo.write('Not changed: %.6f\n' % (Stats[0]/tn))
        fo.write('Changed: %.6f\n' % (Stats[1]/tn))
        fo.write('Min Max Norm: %.2f %.2f\n' % (np.min(norm_min_arr)*min(std)*255, np.max(norm_max_arr)*min(std)*255))
        fo.write('Mean Norm and AE Predict score: %.3f %.3f\n\n' % (np.mean(score_arr), np.mean(aescore_arr)))
