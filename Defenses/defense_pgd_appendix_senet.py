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


total_num = 5000
net_names=['senet']
para_fgsm = [8,16,32]
para_pgd = [(1, 20, 16.0),
            (2, 10, 16.0),
            (1, 40, 32.0)]
para_cw = [40]
from resnet import ResNet18 as n00

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
net00 = torch.nn.DataParallel(n00().cuda()).eval()
net00.load_state_dict(torch.load('./pytorch_cifar/checkpoint/ckpt3.t7')['net'])
net_dict = [
('norm'       , net00),
]
testdata_path_list = sorted(os.listdir('t_sample_testdata'),key=lambda x:int(x[:-6]))
def defense_ae(net, imgdir, successclasstest):
    path_list = sorted(os.listdir(imgdir), key=lambda x:int(x[:-8]))
    labels = np.zeros(total_num)
    aelabels = np.zeros(total_num)
    successfool = np.zeros(total_num)
    scores = np.zeros(total_num)
    predicts = np.zeros(total_num)
    with torch.no_grad():
        for i,name in enumerate(path_list):
            if successclasstest[i] == 0:
                continue
            oriimg = mpimg.imread(imgdir + name)
            labels[i] = int(name.split('.')[0].split('_')[1])
            aelabels[i] = int(name.split('.')[0].split('_')[-1])
            successfool[i] = (labels[i] != aelabels[i])
            if labels[i] == aelabels[i]:
                continue
            img = transforms.ToTensor()(oriimg).type(torch.FloatTensor)
            img = normalize(img)
            img = img.unsqueeze(0)
            inputs = Variable(img.cuda())
            outputs = net(inputs)
            score, predicted = torch.max(nn.Softmax()(outputs), 1)
            scores[i] = score.data.cpu().numpy()
            predicts[i] = predicted.data.cpu().numpy()
    successfool = successfool > 0
    successfool = successfool & successclasstest
    return np.mean(predicts[successfool]==labels[successfool]), np.mean(predicts[successfool]==aelabels[successfool]), np.mean(scores[successfool]), np.sum(successfool)

#for evaluate_name in net_dict.keys():
for ii in range(len(net_dict)):
    evaluate_name = net_dict[ii][0]
    print(evaluate_name, end=' ')
    #net = net_dict[evaluate_name]
    net = net_dict[ii][1]
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
    successclasstest = np.zeros(total_num)
    with torch.no_grad():
        for i in range(total_num):
            oriimg = mpimg.imread('t_sample_testdata/'+testdata_path_list[i])
            labels = int(testdata_path_list[i].split('_')[1].split('.')[0])
            img = transforms.ToTensor()(oriimg).type(torch.FloatTensor)
            img = normalize(img)
            img = img.unsqueeze(0)
            inputs = Variable(img.cuda())
            labels = torch.from_numpy(np.array(labels))
            labels = labels.unsqueeze(0)
            outputs = net(inputs.cuda())
            score, predicted = torch.max(nn.Softmax()(outputs), 1)
            successclasstest[i] = (labels == predicted.data.cpu())
    successclasstest = successclasstest > 0

    for k in net_names:
        print(k, end=' ')
        for parameter in para_fgsm:
            stepsize = parameter
            imgdir = 'FGSM/'+k+'_'+str(stepsize)+'/'
            acc_, aeacc_, meanscore_, numcount_ = defense_ae(net, imgdir, successclasstest)
            acc.append(acc_)
            aeacc.append(aeacc_)
            meanscore.append(meanscore_)
            numcount.append(numcount_)
        for parameter in para_pgd:
            stepsize = parameter[0]
            steps = parameter[1]
            epsilon = parameter[2]
            imgdir = 'PGD/'+k+'_'+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'/'
            acc_, aeacc_, meanscore_, numcount_ = defense_ae(net, imgdir, successclasstest)
            acc.append(acc_)
            aeacc.append(aeacc_)
            meanscore.append(meanscore_)
            numcount.append(numcount_)
        for parameter in para_cw:
            stepsize = parameter
            imgdir = 'CW/'+k+'_'+str(stepsize)+'/'
            acc_, aeacc_, meanscore_, numcount_ = defense_ae(net, imgdir, successclasstest)
            acc.append(acc_)
            aeacc.append(aeacc_)
            meanscore.append(meanscore_)
            numcount.append(numcount_)

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc.append(100.*correct/total)
    w0 = csv.writer(open(net_names[0]+'_'+'pgd_acc.csv','a'))
    w0.writerow(acc)
    w1 = csv.writer(open(net_names[0]+'_'+'pgd_aeacc.csv','a'))
    w1.writerow(aeacc)
    w2 = csv.writer(open(net_names[0]+'_'+'pgd_score.csv','a'))
    w2.writerow(meanscore)
    w3 = csv.writer(open(net_names[0]+'_'+'pgd_num.csv','a'))
    w3.writerow(numcount)
    print(' ',end='\n')
