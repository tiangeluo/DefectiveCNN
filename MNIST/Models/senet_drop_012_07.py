'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out
class drop(nn.Module):
    def __init__(self, mask):
        super(drop, self).__init__()
        self.mask = mask
    def forward(self, x):
        return x * self.mask


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        #self.in_planes = 64
        self.in_planes = 128

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.bn11 = nn.BatchNorm2d(64)
        self.bn12 = nn.BatchNorm2d(64)
        self.bn21 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc11 = nn.Conv2d(64, 4, kernel_size=1)
        self.fc12 = nn.Conv2d(4, 64, kernel_size=1)
        self.fc21 = nn.Conv2d(64, 4, kernel_size=1)
        self.fc22 = nn.Conv2d(4, 64, kernel_size=1)

        self.bn31 = nn.BatchNorm2d(64)
        self.bn32 = nn.BatchNorm2d(128)
        self.bn41 = nn.BatchNorm2d(128)
        self.bn42 = nn.BatchNorm2d(128)
        self.conv31 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv41 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc31 = nn.Conv2d(128, 8, kernel_size=1)
        self.fc32 = nn.Conv2d(8, 128, kernel_size=1)
        self.fc41 = nn.Conv2d(128, 8, kernel_size=1)
        self.fc42 = nn.Conv2d(8, 128, kernel_size=1)
        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)

        #self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        #self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.d0 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_0.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d1 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_1.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d2 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_2.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d3 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_3.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d4 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_f.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d5 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_5.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d6 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_6.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d7 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_12_07_7.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d8 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_12_03_8.npy').astype(float)).type(torch.FloatTensor).cuda())

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        tmp = out
        out = F.relu(self.d0(self.bn11(out)))
        shortcut = tmp
        out = self.conv11(out)
        out = self.conv12(F.relu(self.d1(self.bn12(out))))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc11(w))
        w = F.sigmoid(self.fc12(w))
        out = out * w
        out += shortcut


        tmp = out
        out = F.relu(self.d2(self.bn21(out)))
        shortcut = tmp
        out = self.conv21(out)
        out = self.conv22(F.relu(self.d3(self.bn22(out))))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc21(w))
        w = F.sigmoid(self.fc22(w))
        out = out * w
        out += shortcut


        tmp = out
        out = F.relu(self.d4(self.bn31(out)))
        shortcut = self.shortcut(out)
        out = self.conv31(out)
        out = self.conv32(F.relu(self.d5(self.bn32(out))))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc31(w))
        w = F.sigmoid(self.fc32(w))
        out = out * w
        out += shortcut

        tmp = out
        out = F.relu(self.d6(self.bn41(out)))
        shortcut = tmp
        out = self.conv41(out)
        out = self.conv42(F.relu(self.d7(self.bn42(out))))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc41(w))
        w = F.sigmoid(self.fc42(w))
        out = out * w
        out += shortcut

        #out = self.layer1(out)
        #out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SENet18():
    return SENet(PreActBlock, [2,2,2,2])


def test():
    net = SENet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
