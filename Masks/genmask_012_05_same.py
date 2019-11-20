import numpy as np
import random
import IPython


maskf = np.ones([64,32,32])
mask0 = np.ones([64,32,32])
mask1 = np.ones([64,32,32])
mask2 = np.ones([64,32,32])
mask3 = np.ones([64,32,32])
mask4 = np.ones([128,16,16])
mask5 = np.ones([128,16,16])
mask6 = np.ones([128,16,16])
mask7 = np.ones([128,16,16])
mask8 = np.ones([128,16,16])


maskf.reshape(maskf.shape[0],-1)[np.arange(maskf.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(maskf.shape[1:3])),(np.sum(maskf[0,:])*0.5).astype(int))).reshape(1,-1),maskf.shape[0],axis=0)] = 0
mask0.reshape(mask0.shape[0],-1)[np.arange(mask0.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask0.shape[1:3])),(np.sum(mask0[0,:])*0.5).astype(int))).reshape(1,-1),mask0.shape[0],axis=0)] = 0
mask1.reshape(mask1.shape[0],-1)[np.arange(mask1.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask1.shape[1:3])),(np.sum(mask1[0,:])*0.5).astype(int))).reshape(1,-1),mask1.shape[0],axis=0)] = 0
mask2.reshape(mask2.shape[0],-1)[np.arange(mask2.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask2.shape[1:3])),(np.sum(mask2[0,:])*0.5).astype(int))).reshape(1,-1),mask2.shape[0],axis=0)] = 0
mask3.reshape(mask3.shape[0],-1)[np.arange(mask3.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask3.shape[1:3])),(np.sum(mask3[0,:])*0.5).astype(int))).reshape(1,-1),mask3.shape[0],axis=0)] = 0
mask4.reshape(mask4.shape[0],-1)[np.arange(mask4.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask4.shape[1:3])),(np.sum(mask4[0,:])*0.5).astype(int))).reshape(1,-1),mask4.shape[0],axis=0)] = 0
mask5.reshape(mask5.shape[0],-1)[np.arange(mask5.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask5.shape[1:3])),(np.sum(mask5[0,:])*0.5).astype(int))).reshape(1,-1),mask5.shape[0],axis=0)] = 0
mask6.reshape(mask6.shape[0],-1)[np.arange(mask6.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask6.shape[1:3])),(np.sum(mask6[0,:])*0.5).astype(int))).reshape(1,-1),mask6.shape[0],axis=0)] = 0
mask7.reshape(mask7.shape[0],-1)[np.arange(mask7.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask7.shape[1:3])),(np.sum(mask7[0,:])*0.5).astype(int))).reshape(1,-1),mask7.shape[0],axis=0)] = 0
mask8.reshape(mask8.shape[0],-1)[np.arange(mask8.shape[0])[:, np.newaxis], np.repeat(np.array(random.sample(range(np.prod(mask8.shape[1:3])),(np.sum(mask8[0,:])*0.5).astype(int))).reshape(1,-1),mask8.shape[0],axis=0)] = 0

#print('mask num', np.sum(1-mmask0), 'candidate num', np.sum(l0 < 0.5), 'all num', np.prod(mask0.shape))
#print('mask num', np.sum(1-mmask1), 'candidate num', np.sum(l1 < 0.5), 'all num', np.prod(mask1.shape))
#print('mask num', np.sum(1-mmask2), 'candidate num', np.sum(l2 < 0.5), 'all num', np.prod(mask2.shape))
#print('mask num', np.sum(1-mmask3), 'candidate num', np.sum(l3 < 0.5), 'all num', np.prod(mask3.shape))
#print('mask num', np.sum(1-mmask4), 'candidate num', np.sum(l4 < 0.5), 'all num', np.prod(mask4.shape))
#print('mask num', np.sum(1-mmask5), 'candidate num', np.sum(l5 < 0.5), 'all num', np.prod(mask5.shape))
#print('mask num', np.sum(1-mmask6), 'candidate num', np.sum(l6 < 0.5), 'all num', np.prod(mask6.shape))
#print('mask num', np.sum(1-mmask7), 'candidate num', np.sum(l7 < 0.5), 'all num', np.prod(mask7.shape))
#print('mask num', np.sum(1-mmask8), 'candidate num', np.sum(l8 < 0.5), 'all num', np.prod(mask8.shape))
maskf = np.expand_dims(maskf, 0)
mask0 = np.expand_dims(mask0, 0)
mask1 = np.expand_dims(mask1, 0)
mask2 = np.expand_dims(mask2, 0)
mask3 = np.expand_dims(mask3, 0)
mask4 = np.expand_dims(mask4, 0)
mask5 = np.expand_dims(mask5, 0)
mask6 = np.expand_dims(mask6, 0)
mask7 = np.expand_dims(mask7, 0)
mask8 = np.expand_dims(mask8, 0)
print('mask num', np.sum(1-maskf), 'total num', np.prod(maskf.shape))
print('mask num', np.sum(1-mask0), 'total num', np.prod(mask0.shape))
print('mask num', np.sum(1-mask1), 'total num', np.prod(mask1.shape))
print('mask num', np.sum(1-mask2), 'total num', np.prod(mask2.shape))
print('mask num', np.sum(1-mask3), 'total num', np.prod(mask3.shape))
print('mask num', np.sum(1-mask4), 'total num', np.prod(mask4.shape))
print('mask num', np.sum(1-mask5), 'total num', np.prod(mask5.shape))
print('mask num', np.sum(1-mask6), 'total num', np.prod(mask6.shape))
print('mask num', np.sum(1-mask7), 'total num', np.prod(mask7.shape))
print('mask num', np.sum(1-mask8), 'total num', np.prod(mask8.shape))
np.save('./arr/mask_12_05_same_f.npy', maskf)
np.save('./arr/mask_12_05_same_0.npy', mask0)
np.save('./arr/mask_12_05_same_1.npy', mask1)
np.save('./arr/mask_12_05_same_2.npy', mask2)
np.save('./arr/mask_12_05_same_3.npy', mask3)
np.save('./arr/mask_12_05_same_4.npy', mask4)
np.save('./arr/mask_12_05_same_5.npy', mask5)
np.save('./arr/mask_12_05_same_6.npy', mask6)
np.save('./arr/mask_12_05_same_7.npy', mask7)
np.save('./arr/mask_12_05_same_8.npy', mask8)
