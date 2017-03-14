# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:54:13 2017

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob

# Read a color image
img_car = mpimg.imread("image0478car.png")
img_nocar = mpimg.imread("extra49nocar.png")
fig = plt.figure(figsize=(12,8))
plt.subplot(121)
plt.imshow(img_car)
plt.title('Car',fontsize = 25)
plt.subplot(122)
plt.imshow(img_nocar)
plt.title('No Car',fontsize = 25)
fig.tight_layout()

def explore_color_hog(feature_image):
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_image = []
    for channe in range(3):
        channe_img = feature_image[:,:,channe]
        # Call our function with vis=True to see an image output
        features, hog = get_hog_features(channe_img, orient, \
                                pix_per_cell, cell_per_block, \
                                vis=True, feature_vec=False)
        hog_image.append(hog)
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.title('origin image ch' + str(channe))
        plt.imshow(channe_img)
        plt.axis('off')
        
        plt.subplot(122)
        plt.title('feature' + str(channe))
        plt.imshow(hog)
        plt.axis('off')
hog_image = explore_color_hog(img_car)