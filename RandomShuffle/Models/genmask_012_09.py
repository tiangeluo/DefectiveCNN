import numpy as np
import random
import IPython


maskf = np.ones([64,112,112])
mask0 = np.ones([64,56,56])
mask1 = np.ones([64,56,56])
mask2 = np.ones([64,56,56])
mask3 = np.ones([64,56,56])
mask4 = np.ones([128,28,28])
mask5 = np.ones([128,28,28])
mask6 = np.ones([128,28,28])
mask7 = np.ones([128,28,28])
mask8 = np.ones([128,28,28])

maskf.reshape(-1)[random.sample(range(np.prod(maskf.shape)), (np.sum(maskf)*0.9).astype(int))]=0
mask0.reshape(-1)[random.sample(range(np.prod(mask0.shape)), (np.sum(mask0)*0.9).astype(int))]=0
mask1.reshape(-1)[random.sample(range(np.prod(mask1.shape)), (np.sum(mask1)*0.9).astype(int))]=0
mask2.reshape(-1)[random.sample(range(np.prod(mask2.shape)), (np.sum(mask2)*0.9).astype(int))]=0
mask3.reshape(-1)[random.sample(range(np.prod(mask3.shape)), (np.sum(mask3)*0.9).astype(int))]=0
mask4.reshape(-1)[random.sample(range(np.prod(mask4.shape)), (np.sum(mask4)*0.9).astype(int))]=0
mask5.reshape(-1)[random.sample(range(np.prod(mask5.shape)), (np.sum(mask5)*0.9).astype(int))]=0
mask6.reshape(-1)[random.sample(range(np.prod(mask6.shape)), (np.sum(mask6)*0.9).astype(int))]=0
mask7.reshape(-1)[random.sample(range(np.prod(mask7.shape)), (np.sum(mask7)*0.9).astype(int))]=0
mask8.reshape(-1)[random.sample(range(np.prod(mask8.shape)), (np.sum(mask8)*0.9).astype(int))]=0

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
print('mask num', np.sum(1-mask0), 'total num', np.prod(mask0.shape))
print('mask num', np.sum(1-mask1), 'total num', np.prod(mask1.shape))
print('mask num', np.sum(1-mask2), 'total num', np.prod(mask2.shape))
print('mask num', np.sum(1-mask3), 'total num', np.prod(mask3.shape))
print('mask num', np.sum(1-mask4), 'total num', np.prod(mask4.shape))
print('mask num', np.sum(1-mask5), 'total num', np.prod(mask5.shape))
print('mask num', np.sum(1-mask6), 'total num', np.prod(mask6.shape))
print('mask num', np.sum(1-mask7), 'total num', np.prod(mask7.shape))
print('mask num', np.sum(1-mask8), 'total num', np.prod(mask8.shape))
np.save('./arr/mask_12_09_f.npy', maskf)
np.save('./arr/mask_12_09_0.npy', mask0)
np.save('./arr/mask_12_09_1.npy', mask1)
np.save('./arr/mask_12_09_2.npy', mask2)
np.save('./arr/mask_12_09_3.npy', mask3)
np.save('./arr/mask_12_09_4.npy', mask4)
np.save('./arr/mask_12_09_5.npy', mask5)
np.save('./arr/mask_12_09_6.npy', mask6)
np.save('./arr/mask_12_09_7.npy', mask7)
np.save('./arr/mask_12_09_8.npy', mask8)
