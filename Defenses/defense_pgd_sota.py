from __future__ import print_function
import torch
import os
import IPython
import torch.nn as nn
import torch.nn.parallel
import os
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import numpy as np
import csv
import matplotlib.image as mpimg


normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=2, pin_memory=True)


total_num = 10000
net_names=['res32k4']
parameters = [(2, 7, 8.0)]
from pytorch_cifar.resnet32_k4 import resnet32_cifar as n146

class Ensemble(nn.Module):
    def __init__(self, net1, net2, net3, net4, net5):
        super(Ensemble, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
        self.net4 = net4
        self.net5 = net5
    def forward(self, inputs):
        return (self.net1(inputs) + self.net2(inputs) + self.net3(inputs) + self.net4(inputs) + self.net5(inputs))/5
class Ensemble2(nn.Module):
    def __init__(self, net1, net2):
        super(Ensemble2, self).__init__()
        self.net1 = net1
        self.net2 = net2
    def forward(self, inputs):
        return (self.net1(inputs) + self.net2(inputs))/2
net146 = torch.nn.DataParallel(n146().cuda()).eval()
net146.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt_resnet32_k4_norm_2.t7')['net'])
net_dict = {
'res32k4' : net146,
}
#testdata_path_list = sorted(os.listdir('t_sample_testdata'),key=lambda x:int(x[:-6]))
for evaluate_name in net_dict.keys():
    print(evaluate_name, end=' ')
    net = net_dict[evaluate_name]
    acc = []
    acc.append(evaluate_name)
    meanscore = []
    meanscore.append(evaluate_name)
    aeacc = []
    aeacc.append(evaluate_name)
    numcount = []
    numcount.append(evaluate_name)
    total = 0
    correct = 0
    for k in net_names:
        print(k)
        for parameter in parameters:
            stepsize = parameter[0]
            steps = parameter[1]
            epsilon = parameter[2]
            imgdir = 'PGD/'+k+'_'+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'/'
            path_list = sorted(os.listdir(imgdir), key=lambda x:int(x[:-8]))
            labels = np.zeros(total_num)
            aelabels = np.zeros(total_num)
            successfool = np.zeros(total_num)
            scores = np.zeros(total_num)
            predicts = np.zeros(total_num)
            with torch.no_grad():
                for i,name in enumerate(path_list):
                    oriimg = mpimg.imread(imgdir + name)
                    labels[i] = int(name.split('.')[0].split('_')[1])
                    aelabels[i] = int(name.split('.')[0].split('_')[-1])
                    img = transforms.ToTensor()(oriimg).type(torch.FloatTensor)
                    img = normalize(img)
                    img = img.unsqueeze(0)
                    inputs = Variable(img.cuda())
                    outputs = net(inputs)
                    #outputs = net(inputs,False)
                    score, predicted = torch.max(nn.Softmax()(outputs), 1)
                    scores[i] = score.data.cpu().numpy()
                    predicts[i] = predicted.data.cpu().numpy()
            acc.append(np.mean(predicts==labels))
            aeacc.append(np.mean(predicts==aelabels))
            meanscore.append(np.mean(scores))
            numcount.append(total_num)
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = net(inputs)
            #outputs = net(inputs,False)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc.append(100.*correct/total)
    w0 = csv.writer(open('pgd_acc.csv','a'))
    w0.writerow(acc)
    w1 = csv.writer(open('pgd_aeacc.csv','a'))
    w1.writerow(aeacc)
    w2 = csv.writer(open('pgd_score.csv','a'))
    w2.writerow(meanscore)
    w3 = csv.writer(open('pgd_num.csv','a'))
    w3.writerow(numcount)
