'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out

class drop(nn.Module):
    def __init__(self, mask):
        super(drop, self).__init__()
        self.mask = mask
    def forward(self, x):
        return x * self.mask

class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16*4
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
#-------------------------------------
        self.conv00 = nn.Conv2d(16, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv01 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv02 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv03 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv04 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv05 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv06 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv07 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv08 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv09 = nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn00 = nn.BatchNorm2d(16*4)
        self.bn01 = nn.BatchNorm2d(16*4)
        self.bn02 = nn.BatchNorm2d(16*4)
        self.bn03 = nn.BatchNorm2d(16*4)
        self.bn04 = nn.BatchNorm2d(16*4)
        self.bn05 = nn.BatchNorm2d(16*4)
        self.bn06 = nn.BatchNorm2d(16*4)
        self.bn07 = nn.BatchNorm2d(16*4)
        self.bn08 = nn.BatchNorm2d(16*4)
        self.bn09 = nn.BatchNorm2d(16*4)
        #self.df = drop(torch.from_numpy( np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_f.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d00 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_00.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d01 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_01.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d02 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_02.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d03 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_03.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d04 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_04.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d05 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_05.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d06 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_06.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d07 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_07.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d08 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_08.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d09 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_09.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.df = drop(torch.from_numpy( np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_f.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d00 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_00.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d01 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_01.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d02 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_02.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d03 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_03.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d04 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_04.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d05 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_05.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d06 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_06.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d07 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_07.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d08 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_08.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d09 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_32_k4_0_9_09.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.shortcut1 = nn.Sequential(nn.Conv2d(16, 16*4, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(16*4))
        #self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32*4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion*4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.df(x)
        out = self.relu(x)

        tmp = out
        out =   self.conv00(out)
        out =     self.bn00(out)
        out =      self.d00(out)
        out = F.relu(out)
        out =   self.conv01(out)
        out =     self.bn01(out)
        out =      self.d01(out)
        out += self.shortcut1(tmp)
        out = F.relu(out)

        tmp = out
        out =   self.conv02(out)
        out =     self.bn02(out)
        out =      self.d02(out)
        out = F.relu(out)
        out =   self.conv03(out)
        out =     self.bn03(out)
        out =      self.d03(out)
        out += tmp
        out = F.relu(out)

        tmp = out
        out =   self.conv04(out)
        out =     self.bn04(out)
        out =      self.d04(out)
        out = F.relu(out)
        out =   self.conv05(out)
        out =     self.bn05(out)
        out =      self.d05(out)
        out += tmp
        out = F.relu(out)

        tmp = out
        out =   self.conv06(out)
        out =     self.bn06(out)
        out =      self.d06(out)
        out = F.relu(out)
        out =   self.conv07(out)
        out =     self.bn07(out)
        out =      self.d07(out)
        out += tmp
        out = F.relu(out)

        tmp = out
        out =   self.conv08(out)
        out =     self.bn08(out)
        out =      self.d08(out)
        out = F.relu(out)
        out =   self.conv09(out)
        out =     self.bn09(out)
        out =      self.d09(out)
        out += tmp
        out = F.relu(out)

        #x = self.layer1(x)
        x = self.layer2(out)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet20_cifar()
    y = net(torch.randn(1, 3, 64, 64))
    print(net)
    print(y.size())
