'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class drop(nn.Module):
    def __init__(self, mask):
        super(drop, self).__init__()
        self.mask = mask
    def forward(self, x):
        return x * self.mask

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 128 * 4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv101 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.conv102 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv103 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn101 = nn.BatchNorm2d(64)
        self.bn102 = nn.BatchNorm2d(64)
        self.bn103 = nn.BatchNorm2d(256)
        self.shortcut10 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                          nn.BatchNorm2d(256))
        self.conv111 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.conv112 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv113 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn111 = nn.BatchNorm2d(64)
        self.bn112 = nn.BatchNorm2d(64)
        self.bn113 = nn.BatchNorm2d(256)
        self.shortcut11 = nn.Sequential()
        self.conv121 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.conv122 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv123 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn121 = nn.BatchNorm2d(64)
        self.bn122 = nn.BatchNorm2d(64)
        self.bn123 = nn.BatchNorm2d(256)
        self.shortcut12 = nn.Sequential()

        self.conv201 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.conv202 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv203 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn201 = nn.BatchNorm2d(128)
        self.bn202 = nn.BatchNorm2d(128)
        self.bn203 = nn.BatchNorm2d(512)
        self.shortcut20 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                          nn.BatchNorm2d(512))
        self.conv211 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.conv212 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv213 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn211 = nn.BatchNorm2d(128)
        self.bn212 = nn.BatchNorm2d(128)
        self.bn213 = nn.BatchNorm2d(512)
        self.shortcut21 = nn.Sequential()
        self.conv221 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.conv222 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv223 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn221 = nn.BatchNorm2d(128)
        self.bn222 = nn.BatchNorm2d(128)
        self.bn223 = nn.BatchNorm2d(512)
        self.shortcut22 = nn.Sequential()
        self.conv231 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.conv232 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv233 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn231 = nn.BatchNorm2d(128)
        self.bn232 = nn.BatchNorm2d(128)
        self.bn233 = nn.BatchNorm2d(512)
        self.shortcut23 = nn.Sequential()

        #self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        #self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.d0 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_0.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d1 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_1.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d2 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_2.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d3 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_3.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d4 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_4.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d5 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_5.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d6 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_6.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d7 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_07_7.npy').astype(float)).type(torch.FloatTensor).cuda())

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.d0(self.bn1(self.conv1(x))))

        tmp = out
        out = F.relu(self.bn101(self.conv101(out)))
        out = F.relu(self.d1(self.bn102(self.conv102(out))))
        out =        self.bn103(self.conv103(out))
        out += self.shortcut10(tmp)
        out = F.relu(out)

        tmp = out
        out = F.relu(self.bn111(self.conv111(out)))
        out = F.relu(self.d2(self.bn112(self.conv112(out))))
        out =        self.bn113(self.conv113(out))
        out += self.shortcut11(tmp)
        out = F.relu(out)
        
        tmp = out
        out = F.relu(self.bn121(self.conv121(out)))
        out = F.relu(self.d3(self.bn122(self.conv122(out))))
        out =        self.bn123(self.conv123(out))
        out += self.shortcut12(tmp)
        out = F.relu(out)

        #out = self.layer2(out)
        tmp = out
        out = F.relu(self.bn201(self.conv201(out)))
        out = F.relu(self.d4(self.bn202(self.conv202(out))))
        out =        self.bn203(self.conv203(out))
        out += self.shortcut20(tmp)
        out = F.relu(out)

        tmp = out
        out = F.relu(self.bn211(self.conv211(out)))
        out = F.relu(self.d5(self.bn212(self.conv212(out))))
        out =        self.bn213(self.conv213(out))
        out += self.shortcut21(tmp)
        out = F.relu(out)


        tmp = out
        out = F.relu(self.bn221(self.conv221(out)))
        out = F.relu(self.d6(self.bn222(self.conv222(out))))
        out =        self.bn223(self.conv223(out))
        out += self.shortcut22(tmp)
        out = F.relu(out)

        tmp = out
        out = F.relu(self.bn231(self.conv231(out)))
        out = F.relu(self.d7(self.bn232(self.conv232(out))))
        out =        self.bn233(self.conv233(out))
        out += self.shortcut23(tmp)
        out = F.relu(out)

        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
