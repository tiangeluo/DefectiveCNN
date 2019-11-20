import torch
import torchvision
from PIL import Image
import numpy as np
import IPython
import torchvision.transforms as transforms
import os
import matplotlib.image as mpimg
from copy import deepcopy
def save_img(kk, path, name):
	kk = kk[:,:,:]
	k1 = kk[0,:,:]
	k2 = kk[1,:,:]
	k3 = kk[2,:,:]
	k1 = np.around(k1*255)
	k2 = np.around(k2*255)
	k3 = np.around(k3*255)
	r = Image.fromarray(k1).convert('L')
	g = Image.fromarray(k2).convert('L')
	b = Image.fromarray(k3).convert('L')
	raw = Image.merge('RGB', (r, g, b))
	raw.save(path+name+'.png')
s_arr = [2, 4, 8]
for s in s_arr:
    dirname ='imagenet_shuffle_'+str(s)+'/'
    os.mkdir(dirname)
    size = 224
    xnum = size//s
    yindex = np.tile(np.arange(xnum*xnum), (s*s,1))
    shape = (s, s, xnum, xnum)
    strides = 4 * np.array([size*xnum, xnum, size, 1])
    i = 0
    path_list = sorted(os.listdir('sample_01209'), key = lambda x:int(x.split('_')[0]))
    for path in path_list:
        img = mpimg.imread('sample_01209/'+path)
        label = np.int64(path.split('_')[-1].split('.')[0])
        xindex = np.random.permutation(s*s)[:, np.newaxis]
        newimg = np.zeros([3,224,224])
        for j in range(3):
            op = deepcopy(img[:,:,j])
            squares = np.lib.stride_tricks.as_strided(op, shape, strides)
            t1 = squares.reshape(s*s, xnum*xnum)
            t2 = t1[xindex, yindex].reshape(shape)
            t3 = t2.transpose(0,2,1,3).reshape(size,size)
            newimg[j,:,:]=deepcopy(t3)

        save_img(newimg, dirname,str(i)+'_'+str(label))
        i +=1
