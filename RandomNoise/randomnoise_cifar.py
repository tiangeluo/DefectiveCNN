import torch
import torchvision
from PIL import Image
import numpy as np
import IPython
import torchvision.transforms as transforms
import matplotlib.image as mpimg
import os
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
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
	raw.save(path+name)
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

epsilon_arr = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
for epsilon in epsilon_arr:
    save_dir = 'cifar/test_randomnoise_012_'+str(int(epsilon))+'/'
    os.mkdir(save_dir)
    sigma = epsilon/2/255/min(std)
    bound = epsilon/255/min(std)
    up = ((np.ones([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1))
    down = ((np.zeros([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1))
    testdata_path_list = sorted(os.listdir('randomnoise_012_sample_testdata'),key=lambda x:int(x[:-6]))

    for path in testdata_path_list:
        oriimg = mpimg.imread('randomnoise_012_sample_testdata/'+path)
        img = transforms.ToTensor()(oriimg).type(torch.FloatTensor)
        img = normalize(img)
        img = img.unsqueeze(0)
        img = img.cpu().data.numpy()[0,:]
        noise = np.random.normal(0, sigma, (3,32,32))
        noise[noise > bound] = bound
        noise[noise < -bound] = -bound
        img = img + noise
        img[img>up] = up[img>up]
        img[img<down] = down[img<down]
        save_img(img, 'cifar/test_randomnoise_012_'+str(int(epsilon))+'/',path)
