'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class drop(nn.Module):
    def __init__(self, mask):
        super(drop, self).__init__()
        self.mask = mask
    def forward(self, x):
        return x * self.mask

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        #self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        self.conv00 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv01 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn00 = nn.BatchNorm2d(64)
        self.bn01 = nn.BatchNorm2d(64)
        self.mp0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(128)
        self.bn11 = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv20 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn20 = nn.BatchNorm2d(256)
        self.bn21 = nn.BatchNorm2d(256)
        self.bn22 = nn.BatchNorm2d(256)
        self.bn23 = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv30 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn30 = nn.BatchNorm2d(512)
        self.bn31 = nn.BatchNorm2d(512)
        self.bn32 = nn.BatchNorm2d(512)
        self.bn33 = nn.BatchNorm2d(512)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv40 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn40 = nn.BatchNorm2d(512)
        self.bn41 = nn.BatchNorm2d(512)
        self.bn42 = nn.BatchNorm2d(512)
        self.bn43 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ap = nn.AvgPool2d(kernel_size=1, stride=1)
        
        self.d0 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_vgg_012_07_0.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d1 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_vgg_012_07_1.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d2 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_vgg_012_07_2.npy').astype(float)).type(torch.FloatTensor).cuda())
        self.d3 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/mnist/arr/mask_vgg_012_07_3.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d4 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_vgg_012_09_4.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d5 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_vgg_012_09_5.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d6 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_vgg_012_09_6.npy').astype(float)).type(torch.FloatTensor).cuda())
        #self.d7 = drop(torch.from_numpy(np.load('/home/luotg/Projects/Adversarial/arr/mask_vgg_012_09_7.npy').astype(float)).type(torch.FloatTensor).cuda())
    def forward(self, x):
        x = F.relu(self.d0(self.bn00(self.conv00(x))))
        x = F.relu(self.d1(self.bn01(self.conv01(x))))
        x = self.mp0(x)

        x = F.relu(self.d2(self.bn10(self.conv10(x))))
        x = F.relu(self.d3(self.bn11(self.conv11(x))))
        x = self.mp1(x)

        #x = F.relu(self.d4(self.bn20(self.conv20(x))))
        #x = F.relu(self.d5(self.bn21(self.conv21(x))))
        #x = F.relu(self.d6(self.bn22(self.conv22(x))))
        #x = F.relu(self.d7(self.bn23(self.conv23(x))))
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x = self.mp2(x)

        x = F.relu(self.bn30(self.conv30(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x = self.mp3(x)

        x = F.relu(self.bn40(self.conv40(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
#        x = self.mp4(x)
        out = self.ap(x)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
