import numpy as np
import random
import IPython


mask0 = np.ones([64, 28,28])
mask1 = np.ones([96, 28,28])
mask2 = np.ones([128,28,28])
mask3 = np.ones([160,28,28])
mask4 = np.ones([192,28,28])
mask5 = np.ones([224,28,28])
mask6 = np.ones([256,28,28])
mask7 = np.ones([128, 14,14])
mask8 = np.ones([160, 14,14])
mask9 = np.ones([192, 14,14])
mask10 = np.ones([224,14,14])
mask11 = np.ones([256,14,14])
mask12 = np.ones([288,14,14])
mask13 = np.ones([320,14,14])
mask14 = np.ones([352,14,14])
mask15 = np.ones([384,14,14])
mask16 = np.ones([416,14,14])
mask17 = np.ones([448,14,14])
mask18 = np.ones([480,14,14])
mask19 = np.ones([512,14,14])

mask0.reshape(-1)[random.sample(range(np.prod(mask0.shape)), (np.sum(mask0)*0.7).astype(int))]=0
mask1.reshape(-1)[random.sample(range(np.prod(mask1.shape)), (np.sum(mask1)*0.7).astype(int))]=0
mask2.reshape(-1)[random.sample(range(np.prod(mask2.shape)), (np.sum(mask2)*0.7).astype(int))]=0
mask3.reshape(-1)[random.sample(range(np.prod(mask3.shape)), (np.sum(mask3)*0.7).astype(int))]=0
mask4.reshape(-1)[random.sample(range(np.prod(mask4.shape)), (np.sum(mask4)*0.7).astype(int))]=0
mask5.reshape(-1)[random.sample(range(np.prod(mask5.shape)), (np.sum(mask5)*0.7).astype(int))]=0
mask6.reshape(-1)[random.sample(range(np.prod(mask6.shape)), (np.sum(mask6)*0.7).astype(int))]=0
mask7.reshape(-1)[random.sample(range(np.prod(mask7.shape)), (np.sum(mask7)*0.7).astype(int))]=0
mask8.reshape(-1)[random.sample(range(np.prod(mask8.shape)), (np.sum(mask8)*0.7).astype(int))]=0
mask9.reshape(-1)[random.sample(range(np.prod(mask9.shape)), (np.sum(mask9)*0.7).astype(int))]=0
mask10.reshape(-1)[random.sample(range(np.prod(mask10.shape)), (np.sum(mask10)*0.7).astype(int))]=0
mask11.reshape(-1)[random.sample(range(np.prod(mask11.shape)), (np.sum(mask11)*0.7).astype(int))]=0
mask12.reshape(-1)[random.sample(range(np.prod(mask12.shape)), (np.sum(mask12)*0.7).astype(int))]=0
mask13.reshape(-1)[random.sample(range(np.prod(mask13.shape)), (np.sum(mask13)*0.7).astype(int))]=0
mask14.reshape(-1)[random.sample(range(np.prod(mask14.shape)), (np.sum(mask14)*0.7).astype(int))]=0
mask15.reshape(-1)[random.sample(range(np.prod(mask15.shape)), (np.sum(mask15)*0.7).astype(int))]=0
mask16.reshape(-1)[random.sample(range(np.prod(mask16.shape)), (np.sum(mask16)*0.7).astype(int))]=0
mask17.reshape(-1)[random.sample(range(np.prod(mask17.shape)), (np.sum(mask17)*0.7).astype(int))]=0
mask18.reshape(-1)[random.sample(range(np.prod(mask18.shape)), (np.sum(mask18)*0.7).astype(int))]=0
mask19.reshape(-1)[random.sample(range(np.prod(mask19.shape)), (np.sum(mask19)*0.7).astype(int))]=0

#print('mask num', np.sum(1-mmask0), 'candidate num', np.sum(l0 < 0.5), 'all num', np.prod(mask0.shape))
#print('mask num', np.sum(1-mmask1), 'candidate num', np.sum(l1 < 0.5), 'all num', np.prod(mask1.shape))
#print('mask num', np.sum(1-mmask2), 'candidate num', np.sum(l2 < 0.5), 'all num', np.prod(mask2.shape))
#print('mask num', np.sum(1-mmask3), 'candidate num', np.sum(l3 < 0.5), 'all num', np.prod(mask3.shape))
#print('mask num', np.sum(1-mmask4), 'candidate num', np.sum(l4 < 0.5), 'all num', np.prod(mask4.shape))
#print('mask num', np.sum(1-mmask5), 'candidate num', np.sum(l5 < 0.5), 'all num', np.prod(mask5.shape))
#print('mask num', np.sum(1-mmask6), 'candidate num', np.sum(l6 < 0.5), 'all num', np.prod(mask6.shape))
#print('mask num', np.sum(1-mmask7), 'candidate num', np.sum(l7 < 0.5), 'all num', np.prod(mask7.shape))
#print('mask num', np.sum(1-mmask8), 'candidate num', np.sum(l8 < 0.5), 'all num', np.prod(mask8.shape))
mask0 = np.expand_dims(mask0, 0)
mask1 = np.expand_dims(mask1, 0)
mask2 = np.expand_dims(mask2, 0)
mask3 = np.expand_dims(mask3, 0)
mask4 = np.expand_dims(mask4, 0)
mask5 = np.expand_dims(mask5, 0)
mask6 = np.expand_dims(mask6, 0)
mask7 = np.expand_dims(mask7, 0)
mask8 = np.expand_dims(mask8, 0)
mask9 = np.expand_dims(mask9, 0)
mask10 = np.expand_dims(mask10, 0)
mask11 = np.expand_dims(mask11, 0)
mask12 = np.expand_dims(mask12, 0)
mask13 = np.expand_dims(mask13, 0)
mask14 = np.expand_dims(mask14, 0)
mask15 = np.expand_dims(mask15, 0)
mask16 = np.expand_dims(mask16, 0)
mask17 = np.expand_dims(mask17, 0)
mask18 = np.expand_dims(mask18, 0)
mask19 = np.expand_dims(mask19, 0)
print('mask num', np.sum(1-mask0), 'total num', np.prod(mask0.shape))
print('mask num', np.sum(1-mask1), 'total num', np.prod(mask1.shape))
print('mask num', np.sum(1-mask2), 'total num', np.prod(mask2.shape))
print('mask num', np.sum(1-mask3), 'total num', np.prod(mask3.shape))
print('mask num', np.sum(1-mask4), 'total num', np.prod(mask4.shape))
print('mask num', np.sum(1-mask5), 'total num', np.prod(mask5.shape))
print('mask num', np.sum(1-mask6), 'total num', np.prod(mask6.shape))
print('mask num', np.sum(1-mask7), 'total num', np.prod(mask7.shape))
np.save('./arr/mask_dense_12_07_0.npy', mask0)
np.save('./arr/mask_dense_12_07_1.npy', mask1)
np.save('./arr/mask_dense_12_07_2.npy', mask2)
np.save('./arr/mask_dense_12_07_3.npy', mask3)
np.save('./arr/mask_dense_12_07_4.npy', mask4)
np.save('./arr/mask_dense_12_07_5.npy', mask5)
np.save('./arr/mask_dense_12_07_6.npy', mask6)
np.save('./arr/mask_dense_12_07_7.npy', mask7)
np.save('./arr/mask_dense_12_07_8.npy', mask8)
np.save('./arr/mask_dense_12_07_9.npy', mask9)
np.save('./arr/mask_dense_12_07_10.npy', mask10)
np.save('./arr/mask_dense_12_07_11.npy', mask11)
np.save('./arr/mask_dense_12_07_12.npy', mask12)
np.save('./arr/mask_dense_12_07_13.npy', mask13)
np.save('./arr/mask_dense_12_07_14.npy', mask14)
np.save('./arr/mask_dense_12_07_15.npy', mask15)
np.save('./arr/mask_dense_12_07_16.npy', mask16)
np.save('./arr/mask_dense_12_07_17.npy', mask17)
np.save('./arr/mask_dense_12_07_18.npy', mask18)
np.save('./arr/mask_dense_12_07_19.npy', mask19)
