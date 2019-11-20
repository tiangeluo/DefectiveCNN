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


normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

testloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

total_num = 5000
net_names=['res18', 'res50', 'dense', 'senet','vgg']
parameters =[(255*0.01, 40, 255*0.3)]

from resnet import ResNet18 as n00
from resnet_drop_012_05 import ResNet18 as n05
from resnet_drop_012_07 import ResNet18 as n06
from vgg_unfold            import VGG as n118
from vgg_drop_01_05        import VGG as n119
from vgg_drop_01_07        import VGG as n120
from densenet              import DenseNet121 as n121
from densenet_drop_012_05  import DenseNet121 as n123
from densenet_drop_012_07  import DenseNet121 as n124
from resnet                import ResNet50    as n125
from resnet50_drop_012_05  import ResNet50    as n127
from resnet50_drop_012_07  import ResNet50    as n128
from senet_unfold          import SENet18     as n130
from senet_drop_012_05     import SENet18     as n131
from senet_drop_012_07     import SENet18     as n132


net00 = torch.nn.DataParallel(n00().cuda()).eval()
net00.load_state_dict(torch.load('./checkpoint/norm.t7')['net'])
net05 = torch.nn.DataParallel(n05().cuda()).eval()
net05.load_state_dict(torch.load('./checkpoint/01205.t7')['net'])
net06 = torch.nn.DataParallel(n06().cuda()).eval()
net06.load_state_dict(torch.load('./checkpoint/01207.t7')['net'])
net118 = torch.nn.DataParallel(n118('VGG19').cuda()).eval()
net118.load_state_dict(torch.load('./checkpoint/vgg.t7')['net'])
net119 = torch.nn.DataParallel(n119('VGG19').cuda()).eval()
net119.load_state_dict(torch.load('./checkpoint/vgg_drop_01_05.t7')['net'])
net120 = torch.nn.DataParallel(n120('VGG19').cuda()).eval()
net120.load_state_dict(torch.load('./checkpoint/vgg_drop_01_07.t7')['net'])
net121 = torch.nn.DataParallel(n121().cuda()).eval()
net121.load_state_dict(torch.load('./checkpoint/dense.t7')['net'])
net123 = torch.nn.DataParallel(n123().cuda()).eval()
net123.load_state_dict(torch.load('./checkpoint/dense_drop_012_05.t7')['net'])
net124 = torch.nn.DataParallel(n124().cuda()).eval()
net124.load_state_dict(torch.load('./checkpoint/dense_drop_012_07.t7')['net'])
net125 = torch.nn.DataParallel(n125().cuda()).eval()
net125.load_state_dict(torch.load('./checkpoint/resnet50.t7')['net'])
net127 = torch.nn.DataParallel(n127().cuda()).eval()
net127.load_state_dict(torch.load('./checkpoint/res50_drop_012_05.t7')['net'])
net128 = torch.nn.DataParallel(n128().cuda()).eval()
net128.load_state_dict(torch.load('./checkpoint/res50_drop_012_07.t7')['net'])
net130 = torch.nn.DataParallel(n130().cuda()).eval()
net130.load_state_dict(torch.load('./checkpoint/senet.t7')['net'])
net131 = torch.nn.DataParallel(n131().cuda()).eval()
net131.load_state_dict(torch.load('./checkpoint/senet_drop_012_05.t7')['net'])
net132 = torch.nn.DataParallel(n132().cuda()).eval()
net132.load_state_dict(torch.load('./checkpoint/senet_drop_012_07.t7')['net'])

net_dict=[
('res18' , net00),
('res18_01205'      ,  net05),
('res18_01207'      ,  net06),
('res50_n'          , net125),
('res50_drop_012_05', net127),
('res50_drop_012_07', net128),
('dense_n'          , net121),
('dense_drop_012_05', net123),
('dense_drop_012_07', net124),
('se18_u'           , net130),
('se18_drop_012_05' , net131),
('se18_drop_012_07' , net132),
('vgg_u'            , net118),
('vgg_drop_01_05'   , net119),
('vgg_drop_01_07'   , net120),
]
testdata_path_list = sorted(os.listdir('sample_testdata'),key=lambda x:int(x[:-6]))
for ii in range(len(net_dict)):
    evaluate_name = net_dict[ii][0]
    print(evaluate_name, end=' ')
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
            oriimg = mpimg.imread('sample_testdata/'+testdata_path_list[i])
            labels = int(testdata_path_list[i].split('_')[1].split('.')[0])
            img = transforms.ToTensor()(oriimg.reshape(28,28,1)).type(torch.FloatTensor)
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
        for parameter in parameters:
            stepsize = parameter[0]
            steps = parameter[1]
            epsilon = parameter[2]
            imgdir = 'PGD/'+k+str(stepsize)+'_'+str(steps)+'_'+str(int(epsilon))+'/'
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
                    img = transforms.ToTensor()(oriimg.reshape(28,28,1)).type(torch.FloatTensor)
                    img = normalize(img)
                    img = img.unsqueeze(0)
                    inputs = Variable(img.cuda())
                    outputs = net(inputs)
                    score, predicted = torch.max(nn.Softmax()(outputs), 1)
                    scores[i] = score.data.cpu().numpy()
                    predicts[i] = predicted.data.cpu().numpy()
            successfool = successfool > 0
            successfool = successfool & successclasstest
            acc.append(np.mean(predicts[successfool]==labels[successfool]))
            aeacc.append(np.mean(predicts[successfool]==aelabels[successfool]))
            meanscore.append(np.mean(scores[successfool]))
            numcount.append(np.sum(successfool))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = net(inputs)
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
    print(' ',end='\n')
