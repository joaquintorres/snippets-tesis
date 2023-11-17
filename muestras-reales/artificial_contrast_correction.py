#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File stich_arbitrary.py

"""
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage import io
from skimage.color import rgb2gray
import pathlib as Path
import skimage.data as sdata 
from stitch import Patch, Picture

arbimage = rgb2gray(io.imread('~/Pictures/bebop.png'))
shape = (5, 5)
imshape =  arbimage.shape
pic = Picture.from_image(image=arbimage, shape=shape,
                         overlap=5, outfile='./final_image.h5py')
def cut_sample(sample, y0, x0):
    roi = pic._patch_shape[0]
    sy, sx = imshape
    print(y0, x0, roi)
    return sample[y0-roi//2:y0+roi//2, x0-roi//2:x0+roi//2]

i=0
for patch in pic.patches():
    y0, x0 = patch.center
    cutim = cut_sample(arbimage, y0, x0)
    if i %3:
        cutim = cutim*50
    patch.image = cutim
    i +=1
# pic_hr.append(Patch(delta_gk[0]))
pic.store_array()
complete_image = pic.build_complete_array()
print(arbimage.shape)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(arbimage, cmap='gray')
axes[0].set_title("Original Image")
axes[1].imshow(np.abs(complete_image), cmap='gray')
axes[1].set_title("Patches Magnitude")
axes[2].imshow(np.angle(complete_image), cmap='gray')
axes[2].set_title("Patches Phase")

plt.show()
