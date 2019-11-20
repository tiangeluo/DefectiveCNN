'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class drop(nn.Module):
    def __init__(self, mask):
        super(drop, self).__init__()
        self.mask = mask
    def forward(self, x):
        return x * self.mask


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.bn100 = nn.BatchNorm2d(64)
        self.conv100 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.conv101 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn110 = nn.BatchNorm2d(96)
        self.conv110 = nn.Conv2d(96, 128, kernel_size=1, bias=False)
        self.bn111 = nn.BatchNorm2d(128)
        self.conv111 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)


        self.bn120 = nn.BatchNorm2d(128)
        self.conv120 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn121 = nn.BatchNorm2d(128)
        self.conv121 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn130 = nn.BatchNorm2d(160)
        self.conv130 = nn.Conv2d(160, 128, kernel_size=1, bias=False)
        self.bn131 = nn.BatchNorm2d(128)
        self.conv131 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn140 = nn.BatchNorm2d(192)
        self.conv140 = nn.Conv2d(192, 128, kernel_size=1, bias=False)
        self.bn141 = nn.BatchNorm2d(128)
        self.conv141 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn150 = nn.BatchNorm2d(224)
        self.conv150 = nn.Conv2d(224, 128, kernel_size=1, bias=False)
        self.bn151 = nn.BatchNorm2d(128)
        self.conv151 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn1t = nn.BatchNorm2d(256)
        self.conv1t = nn.Conv2d(256, 128, kernel_size=1, bias=False)

        #self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        #self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes


        self.bn2000 = nn.BatchNorm2d(128)
        self.conv2000 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn2001 = nn.BatchNorm2d(128)
        self.conv2001 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2010 = nn.BatchNorm2d(160)
        self.conv2010 = nn.Conv2d(160, 128, kernel_size=1, bias=False)
        self.bn2011 = nn.BatchNorm2d(128)
        self.conv2011 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2020 = nn.BatchNorm2d(192)
        self.conv2020 = nn.Conv2d(192, 128, kernel_size=1, bias=False)
        self.bn2021 = nn.BatchNorm2d(128)
        self.conv2021 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2030 = nn.BatchNorm2d(224)
        self.conv2030 = nn.Conv2d(224, 128, kernel_size=1, bias=False)
        self.bn2031 = nn.BatchNorm2d(128)
        self.conv2031 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2040 = nn.BatchNorm2d(256)
        self.conv2040 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.bn2041 = nn.BatchNorm2d(128)
        self.conv2041 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2050 = nn.BatchNorm2d(288)
        self.conv2050 = nn.Conv2d(288, 128, kernel_size=1, bias=False)
        self.bn2051 = nn.BatchNorm2d(128)
        self.conv2051 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2060 = nn.BatchNorm2d(320)
        self.conv2060 = nn.Conv2d(320, 128, kernel_size=1, bias=False)
        self.bn2061 = nn.BatchNorm2d(128)
        self.conv2061 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2070 = nn.BatchNorm2d(352)
        self.conv2070 = nn.Conv2d(352, 128, kernel_size=1, bias=False)
        self.bn2071 = nn.BatchNorm2d(128)
        self.conv2071 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2080 = nn.BatchNorm2d(384)
        self.conv2080 = nn.Conv2d(384, 128, kernel_size=1, bias=False)
        self.bn2081 = nn.BatchNorm2d(128)
        self.conv2081 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2090 = nn.BatchNorm2d(416)
        self.conv2090 = nn.Conv2d(416, 128, kernel_size=1, bias=False)
        self.bn2091 = nn.BatchNorm2d(128)
        self.conv2091 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2100 = nn.BatchNorm2d(448)
        self.conv2100 = nn.Conv2d(448, 128, kernel_size=1, bias=False)
        self.bn2101 = nn.BatchNorm2d(128)
        self.conv2101 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2110 = nn.BatchNorm2d(480)
        self.conv2110 = nn.Conv2d(480, 128, kernel_size=1, bias=False)
        self.bn2111 = nn.BatchNorm2d(128)
        self.conv2111 = nn.Conv2d(128, 32, kernel_size=3, padding = 1, bias=False)

        self.bn2t = nn.BatchNorm2d(512)
        self.conv2t = nn.Conv2d(512, 256, kernel_size=1, bias=False)

        #self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        #self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        self.d0 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_0.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d1 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_1.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d2 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_2.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d3 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_3.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d4 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_4.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d5 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_5.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d6 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_6.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d7 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_7.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d8 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_8.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d9 = drop(torch.from_numpy(np.load( '/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_9.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d10 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_10.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d11 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_11.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d12 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_12.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d13 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_13.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d14 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_14.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d15 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_15.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d16 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_16.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d17 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_17.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d18 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_18.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d19 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_dense_12_07_19.npy').astype(float)).type(torch.FloatTensor).cuda())

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        tmp = out
        out = self.conv100(F.relu(self.d0(self.bn100(out))))
        out = self.conv101(F.relu(self.bn101(out)))
        out = torch.cat([out, tmp], 1)
        
        tmp = out
        out = self.conv110(F.relu(self.d1(self.bn110(out))))
        out = self.conv111(F.relu(self.bn111(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv120(F.relu(self.d2(self.bn120(out))))
        out = self.conv121(F.relu(self.bn121(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv130(F.relu(self.d3(self.bn130(out))))
        out = self.conv131(F.relu(self.bn131(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv140(F.relu(self.d4(self.bn140(out))))
        out = self.conv141(F.relu(self.bn141(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv150(F.relu(self.d5(self.bn150(out))))
        out = self.conv151(F.relu(self.bn151(out)))
        out = torch.cat([out, tmp], 1)

        out = self.conv1t(F.relu(self.d6(self.bn1t(out))))
        out = F.avg_pool2d(out, 2)
        #out = self.trans1(self.dense1(out))

        tmp = out
        out = self.conv2000(F.relu(self.d7(self.bn2000(out))))
        out = self.conv2001(F.relu(self.bn2001(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2010(F.relu(self.d8(self.bn2010(out))))
        out = self.conv2011(F.relu(self.bn2011(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2020(F.relu(self.d9(self.bn2020(out))))
        out = self.conv2021(F.relu(self.bn2021(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2030(F.relu(self.d10(self.bn2030(out))))
        out = self.conv2031(F.relu(self.bn2031(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2040(F.relu(self.d11(self.bn2040(out))))
        out = self.conv2041(F.relu(self.bn2041(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2050(F.relu(self.d12(self.bn2050(out))))
        out = self.conv2051(F.relu(self.bn2051(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2060(F.relu(self.d13(self.bn2060(out))))
        out = self.conv2061(F.relu(self.bn2061(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2070(F.relu(self.d14(self.bn2070(out))))
        out = self.conv2071(F.relu(self.bn2071(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2080(F.relu(self.d15(self.bn2080(out))))
        out = self.conv2081(F.relu(self.bn2081(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2090(F.relu(self.d16(self.bn2090(out))))
        out = self.conv2091(F.relu(self.bn2091(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2100(F.relu(self.d17(self.bn2100(out))))
        out = self.conv2101(F.relu(self.bn2101(out)))
        out = torch.cat([out, tmp], 1)

        tmp = out
        out = self.conv2110(F.relu(self.d18(self.bn2110(out))))
        out = self.conv2111(F.relu(self.bn2111(out)))
        out = torch.cat([out, tmp], 1)

        out = self.conv2t(F.relu(self.d19(self.bn2t(out))))
        out = F.avg_pool2d(out, 2)
        #out = self.trans2(self.dense2(out))

        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
