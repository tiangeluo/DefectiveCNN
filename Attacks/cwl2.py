import torch
import cw
import numpy as np
import os
from torchvision import transforms
import torchvision
import scipy.misc
import IPython
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
from resnet import ResNet18
from senet import SENet18
from densenet import DenseNet121
from resnet import ResNet50
from vgg import VGG
from pytorch_cifar.resnet_drop_12_09 import ResNet18 as n2
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
def save_img(kk, path, name):
	kk = kk[:,:,:]
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

net_dict={'resnet50:'net4}
confidences= [50.0]
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
fo = open('CW/info.txt', 'a')
bs = 32

class testdata(Dataset):
    def __init__(self):
        self.imgpath_list = sorted(os.listdir('t_sample_testdata'),key=lambda x:int(x[:-6]))
    def __getitem__(self, index):
        img = scipy.misc.imread('t_sample_testdata/'+self.imgpath_list[index])
        label = np.int32(self.imgpath_list[index].split('_')[-1].split('.')[0])
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])(img)
        return img, label
    def __len__(self):
        return len(self.imgpath_list)

testloader = torch.utils.data.DataLoader(testdata(), batch_size=bs, shuffle=False, num_workers=1)
inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
              max((1 - m) / s for m, s in zip(mean, std)))

for k in net_dict.keys():
    print(k)
    net = net_dict[k]
    for c in confidences:
        os.mkdir('CW/'+k+'_'+str(int(c)))
        adversary = cw.L2Adversary(targeted=False,
                                   confidence=c,
                                   search_steps=30,
                                   box=inputs_box,
                                   optimizer_lr=5e-4)
        total = 0
        correct = 0
        count = 0
        norm = np.array([])
        for data in testloader:
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.type(torch.LongTensor).cuda()
            adversarial_examples = adversary(net, inputs, targets, to_numpy=False)
            outputs_norm = net(inputs)
            outputs_ae   = net(adversarial_examples)
            _, labels_norm = outputs_norm.max(1)
            _, labels_ae   = outputs_ae.max(1)
            num = labels_ae.size(0)
            total += num
            correct += labels_norm.eq(labels_ae).sum().item()
            diff = (adversarial_examples.cpu().data.numpy()-inputs.cpu().data.numpy())
            norm = np.append(norm, np.linalg.norm(diff.reshape(num, -1), ord=2, axis=1))
            IPython.embed()
            exit()
            for i in range(num):
                save_img(adversarial_examples.data.cpu().numpy()[i,:], 'CW/'+k+'_'+str(int(c))+'/', str(count+i)+'_'+str(targets.cpu().data.numpy()[i])+'_'+str(labels_ae.cpu().data.numpy()[i]))
            count += num
            if count > 1024:
                break
        print(k,int(c),100.*correct/total,np.mean(norm))
        fo.write(k+'_'+str(int(c))+'\n')
        fo.write('Not fooling: %.3f\n'%(100.*correct/total))
        fo.write('Mean L2 norm: %.3f \n\n '%(np.mean(norm)))
