import numpy as np
import random
import IPython


maskf = np.ones([16,32,32])
mask00 = np.ones([16*4,32,32])
mask01 = np.ones([16*4,32,32])
mask02 = np.ones([16*4,32,32])
mask03 = np.ones([16*4,32,32])
mask04 = np.ones([16*4,32,32])
mask05 = np.ones([16*4,32,32])
mask06 = np.ones([16*4,32,32])
mask07 = np.ones([16*4,32,32])
mask08 = np.ones([16*4,32,32])
mask09 = np.ones([16*4,32,32])

maskf.reshape(-1)[random.sample(range(np.prod(maskf.shape)), (np.sum(maskf)*0.9).astype(int))]=0
mask00.reshape(-1)[random.sample(range(np.prod(mask00.shape)), (np.sum(mask00)*0.9).astype(int))]=0
mask01.reshape(-1)[random.sample(range(np.prod(mask01.shape)), (np.sum(mask01)*0.9).astype(int))]=0
mask02.reshape(-1)[random.sample(range(np.prod(mask02.shape)), (np.sum(mask02)*0.9).astype(int))]=0
mask03.reshape(-1)[random.sample(range(np.prod(mask03.shape)), (np.sum(mask03)*0.9).astype(int))]=0
mask04.reshape(-1)[random.sample(range(np.prod(mask04.shape)), (np.sum(mask04)*0.9).astype(int))]=0
mask05.reshape(-1)[random.sample(range(np.prod(mask05.shape)), (np.sum(mask05)*0.9).astype(int))]=0
mask06.reshape(-1)[random.sample(range(np.prod(mask06.shape)), (np.sum(mask06)*0.9).astype(int))]=0
mask07.reshape(-1)[random.sample(range(np.prod(mask07.shape)), (np.sum(mask07)*0.9).astype(int))]=0
mask08.reshape(-1)[random.sample(range(np.prod(mask08.shape)), (np.sum(mask08)*0.9).astype(int))]=0
mask09.reshape(-1)[random.sample(range(np.prod(mask09.shape)), (np.sum(mask09)*0.9).astype(int))]=0
#mask10.reshape(-1)[random.sample(range(np.prod(mask10.shape)), (np.sum(mask10)*0.8).astype(int))]=0
#mask11.reshape(-1)[random.sample(range(np.prod(mask11.shape)), (np.sum(mask11)*0.8).astype(int))]=0
#mask12.reshape(-1)[random.sample(range(np.prod(mask12.shape)), (np.sum(mask12)*0.8).astype(int))]=0
#mask13.reshape(-1)[random.sample(range(np.prod(mask13.shape)), (np.sum(mask13)*0.8).astype(int))]=0
#mask14.reshape(-1)[random.sample(range(np.prod(mask14.shape)), (np.sum(mask14)*0.8).astype(int))]=0
#mask15.reshape(-1)[random.sample(range(np.prod(mask15.shape)), (np.sum(mask15)*0.8).astype(int))]=0
#mask16.reshape(-1)[random.sample(range(np.prod(mask16.shape)), (np.sum(mask16)*0.8).astype(int))]=0
#mask17.reshape(-1)[random.sample(range(np.prod(mask17.shape)), (np.sum(mask17)*0.8).astype(int))]=0
#mask18.reshape(-1)[random.sample(range(np.prod(mask18.shape)), (np.sum(mask18)*0.8).astype(int))]=0
#mask19.reshape(-1)[random.sample(range(np.prod(mask19.shape)), (np.sum(mask19)*0.8).astype(int))]=0
#mask20.reshape(-1)[random.sample(range(np.prod(mask20.shape)), (np.sum(mask20)*0.8).astype(int))]=0
#mask21.reshape(-1)[random.sample(range(np.prod(mask21.shape)), (np.sum(mask21)*0.8).astype(int))]=0
#mask22.reshape(-1)[random.sample(range(np.prod(mask22.shape)), (np.sum(mask22)*0.8).astype(int))]=0
#mask23.reshape(-1)[random.sample(range(np.prod(mask23.shape)), (np.sum(mask23)*0.8).astype(int))]=0
#mask24.reshape(-1)[random.sample(range(np.prod(mask24.shape)), (np.sum(mask24)*0.8).astype(int))]=0
#mask25.reshape(-1)[random.sample(range(np.prod(mask25.shape)), (np.sum(mask25)*0.8).astype(int))]=0
#mask26.reshape(-1)[random.sample(range(np.prod(mask26.shape)), (np.sum(mask26)*0.8).astype(int))]=0
#mask27.reshape(-1)[random.sample(range(np.prod(mask27.shape)), (np.sum(mask27)*0.8).astype(int))]=0
#mask28.reshape(-1)[random.sample(range(np.prod(mask28.shape)), (np.sum(mask28)*0.8).astype(int))]=0
#mask29.reshape(-1)[random.sample(range(np.prod(mask29.shape)), (np.sum(mask29)*0.8).astype(int))]=0
#mask30.reshape(-1)[random.sample(range(np.prod(mask30.shape)), (np.sum(mask30)*0.8).astype(int))]=0
#mask31.reshape(-1)[random.sample(range(np.prod(mask31.shape)), (np.sum(mask31)*0.8).astype(int))]=0
#mask32.reshape(-1)[random.sample(range(np.prod(mask32.shape)), (np.sum(mask32)*0.8).astype(int))]=0
#mask33.reshape(-1)[random.sample(range(np.prod(mask33.shape)), (np.sum(mask33)*0.8).astype(int))]=0
#mask34.reshape(-1)[random.sample(range(np.prod(mask34.shape)), (np.sum(mask34)*0.8).astype(int))]=0
#mask35.reshape(-1)[random.sample(range(np.prod(mask35.shape)), (np.sum(mask35)*0.8).astype(int))]=0
#mask36.reshape(-1)[random.sample(range(np.prod(mask36.shape)), (np.sum(mask36)*0.8).astype(int))]=0
#mask37.reshape(-1)[random.sample(range(np.prod(mask37.shape)), (np.sum(mask37)*0.8).astype(int))]=0
#mask38.reshape(-1)[random.sample(range(np.prod(mask38.shape)), (np.sum(mask38)*0.8).astype(int))]=0
#mask39.reshape(-1)[random.sample(range(np.prod(mask39.shape)), (np.sum(mask39)*0.8).astype(int))]=0
#mask40.reshape(-1)[random.sample(range(np.prod(mask40.shape)), (np.sum(mask40)*0.8).astype(int))]=0
#mask41.reshape(-1)[random.sample(range(np.prod(mask41.shape)), (np.sum(mask41)*0.8).astype(int))]=0
#mask42.reshape(-1)[random.sample(range(np.prod(mask42.shape)), (np.sum(mask42)*0.8).astype(int))]=0
#mask43.reshape(-1)[random.sample(range(np.prod(mask43.shape)), (np.sum(mask43)*0.8).astype(int))]=0
#mask44.reshape(-1)[random.sample(range(np.prod(mask44.shape)), (np.sum(mask44)*0.8).astype(int))]=0
#mask45.reshape(-1)[random.sample(range(np.prod(mask45.shape)), (np.sum(mask45)*0.8).astype(int))]=0
#mask46.reshape(-1)[random.sample(range(np.prod(mask46.shape)), (np.sum(mask46)*0.8).astype(int))]=0
#mask47.reshape(-1)[random.sample(range(np.prod(mask47.shape)), (np.sum(mask47)*0.8).astype(int))]=0
#mask48.reshape(-1)[random.sample(range(np.prod(mask48.shape)), (np.sum(mask48)*0.8).astype(int))]=0
#mask49.reshape(-1)[random.sample(range(np.prod(mask49.shape)), (np.sum(mask49)*0.8).astype(int))]=0
#mask50.reshape(-1)[random.sample(range(np.prod(mask50.shape)), (np.sum(mask50)*0.8).astype(int))]=0
#mask51.reshape(-1)[random.sample(range(np.prod(mask51.shape)), (np.sum(mask51)*0.8).astype(int))]=0
#mask52.reshape(-1)[random.sample(range(np.prod(mask52.shape)), (np.sum(mask52)*0.8).astype(int))]=0
#mask53.reshape(-1)[random.sample(range(np.prod(mask53.shape)), (np.sum(mask53)*0.8).astype(int))]=0

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
mask00 = np.expand_dims(mask00, 0)
mask01 = np.expand_dims(mask01, 0)
mask02 = np.expand_dims(mask02, 0)
mask03 = np.expand_dims(mask03, 0)
mask04 = np.expand_dims(mask04, 0)
mask05 = np.expand_dims(mask05, 0)
mask06 = np.expand_dims(mask06, 0)
mask07 = np.expand_dims(mask07, 0)
mask08 = np.expand_dims(mask08, 0)
mask09 = np.expand_dims(mask09, 0)
#mask10 = np.expand_dims(mask10, 0)
#mask11 = np.expand_dims(mask11, 0)
#mask12 = np.expand_dims(mask12, 0)
#mask13 = np.expand_dims(mask13, 0)
#mask14 = np.expand_dims(mask14, 0)
#mask15 = np.expand_dims(mask15, 0)
#mask16 = np.expand_dims(mask16, 0)
#mask17 = np.expand_dims(mask17, 0)
#mask18 = np.expand_dims(mask18, 0)
#mask19 = np.expand_dims(mask19, 0)
#mask20 = np.expand_dims(mask20, 0)
#mask21 = np.expand_dims(mask21, 0)
#mask22 = np.expand_dims(mask22, 0)
#mask23 = np.expand_dims(mask23, 0)
#mask24 = np.expand_dims(mask24, 0)
#mask25 = np.expand_dims(mask25, 0)
#mask26 = np.expand_dims(mask26, 0)
#mask27 = np.expand_dims(mask27, 0)
#mask28 = np.expand_dims(mask28, 0)
#mask29 = np.expand_dims(mask29, 0)
#mask30 = np.expand_dims(mask30, 0)
#mask31 = np.expand_dims(mask31, 0)
#mask32 = np.expand_dims(mask32, 0)
#mask33 = np.expand_dims(mask33, 0)
#mask34 = np.expand_dims(mask34, 0)
#mask35 = np.expand_dims(mask35, 0)
#mask36 = np.expand_dims(mask36, 0)
#mask37 = np.expand_dims(mask37, 0)
#mask38 = np.expand_dims(mask38, 0)
#mask39 = np.expand_dims(mask39, 0)
#mask40 = np.expand_dims(mask40, 0)
#mask41 = np.expand_dims(mask41, 0)
#mask42 = np.expand_dims(mask42, 0)
#mask43 = np.expand_dims(mask43, 0)
#mask44 = np.expand_dims(mask44, 0)
#mask45 = np.expand_dims(mask45, 0)
#mask46 = np.expand_dims(mask46, 0)
#mask47 = np.expand_dims(mask47, 0)
#mask48 = np.expand_dims(mask48, 0)
#mask49 = np.expand_dims(mask49, 0)
#mask50 = np.expand_dims(mask50, 0)
#mask51 = np.expand_dims(mask51, 0)
#mask52 = np.expand_dims(mask52, 0)
#mask53 = np.expand_dims(mask53, 0)
#print('mask num', np.sum(1-mask0), 'total num', np.prod(mask0.shape))
#print('mask num', np.sum(1-mask1), 'total num', np.prod(mask1.shape))
#print('mask num', np.sum(1-mask2), 'total num', np.prod(mask2.shape))
#print('mask num', np.sum(1-mask3), 'total num', np.prod(mask3.shape))
#print('mask num', np.sum(1-mask4), 'total num', np.prod(mask4.shape))
#print('mask num', np.sum(1-mask5), 'total num', np.prod(mask5.shape))
#print('mask num', np.sum(1-mask6), 'total num', np.prod(mask6.shape))
#print('mask num', np.sum(1-mask7), 'total num', np.prod(mask7.shape))
#print('mask num', np.sum(1-mask8), 'total num', np.prod(mask8.shape))
#-------------------------------------------------
np.save('./arr/mask_32_k4_0_9_f.npy', maskf)
np.save('./arr/mask_32_k4_0_9_00.npy', mask00)
np.save('./arr/mask_32_k4_0_9_01.npy', mask01)
np.save('./arr/mask_32_k4_0_9_02.npy', mask02)
np.save('./arr/mask_32_k4_0_9_03.npy', mask03)
np.save('./arr/mask_32_k4_0_9_04.npy', mask04)
np.save('./arr/mask_32_k4_0_9_05.npy', mask05)
np.save('./arr/mask_32_k4_0_9_06.npy', mask06)
np.save('./arr/mask_32_k4_0_9_07.npy', mask07)
np.save('./arr/mask_32_k4_0_9_08.npy', mask08)
np.save('./arr/mask_32_k4_0_9_09.npy', mask09)
#---------------------------------------------
#np.save('./arr/mask_110_0_8_10.npy', mask10)
#np.save('./arr/mask_110_0_8_11.npy', mask11)
#np.save('./arr/mask_110_0_8_12.npy', mask12)
#np.save('./arr/mask_110_0_8_13.npy', mask13)
#np.save('./arr/mask_110_0_8_14.npy', mask14)
#np.save('./arr/mask_110_0_8_15.npy', mask15)
#np.save('./arr/mask_110_0_8_16.npy', mask16)
#np.save('./arr/mask_110_0_8_17.npy', mask17)
#np.save('./arr/mask_110_0_8_18.npy', mask18)
#np.save('./arr/mask_110_0_8_19.npy', mask19)
#np.save('./arr/mask_110_0_8_20.npy', mask20)
#np.save('./arr/mask_110_0_8_21.npy', mask21)
#np.save('./arr/mask_110_0_8_22.npy', mask22)
#np.save('./arr/mask_110_0_8_23.npy', mask23)
#np.save('./arr/mask_110_0_8_24.npy', mask24)
#np.save('./arr/mask_110_0_8_25.npy', mask25)
#np.save('./arr/mask_110_0_8_26.npy', mask26)
#np.save('./arr/mask_110_0_8_27.npy', mask27)
#np.save('./arr/mask_110_0_8_28.npy', mask28)
#np.save('./arr/mask_110_0_8_29.npy', mask29)
#np.save('./arr/mask_110_0_8_30.npy', mask30)
#np.save('./arr/mask_110_0_8_31.npy', mask31)
#np.save('./arr/mask_110_0_8_32.npy', mask32)
#np.save('./arr/mask_110_0_8_33.npy', mask33)
#np.save('./arr/mask_110_0_8_34.npy', mask34)
#np.save('./arr/mask_110_0_8_35.npy', mask35)
#np.save('./arr/mask_110_0_8_36.npy', mask36)
#np.save('./arr/mask_110_0_8_37.npy', mask37)
#np.save('./arr/mask_110_0_8_38.npy', mask38)
#np.save('./arr/mask_110_0_8_39.npy', mask39)
#np.save('./arr/mask_110_0_8_40.npy', mask40)
#np.save('./arr/mask_110_0_8_41.npy', mask41)
#np.save('./arr/mask_110_0_8_42.npy', mask42)
#np.save('./arr/mask_110_0_8_43.npy', mask43)
#np.save('./arr/mask_110_0_8_44.npy', mask44)
#np.save('./arr/mask_110_0_8_45.npy', mask45)
#np.save('./arr/mask_110_0_8_46.npy', mask46)
#np.save('./arr/mask_110_0_8_47.npy', mask47)
#np.save('./arr/mask_110_0_8_48.npy', mask48)
#np.save('./arr/mask_110_0_8_49.npy', mask49)
#np.save('./arr/mask_110_0_8_50.npy', mask50)
#np.save('./arr/mask_110_0_8_51.npy', mask51)
#np.save('./arr/mask_110_0_8_52.npy', mask52)
#np.save('./arr/mask_110_0_8_53.npy', mask53)
