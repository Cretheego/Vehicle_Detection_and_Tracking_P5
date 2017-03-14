# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 07:33:07 2017

@author: Administrator
"""
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from util_functions import *
from classifier import *
import imghdr
import util_functions as uf  
from scipy.ndimage.measurements import label
from sklearn.utils import shuffle

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle

class VehicleDetection():
    def __init__(self, class_svm):
        self.class_svm = class_svm
        self.ystart = 350
        self.ystop = 660
        self.scale = 1.4
        self.heatmap = []
        self.n = 0
        self.backnum = 5    
        self.class_svm.load()
        self.classifier = self.class_svm.classifier
        self.X_scaler = self.class_svm.X_scaler
        self.fea_extra = self.class_svm.fea_extra
        
    def find_cars(self, img):
        xstart = np.int(img.shape[1] / 2)
        xstop = img.shape[1]
        ystart = self.ystart
        ystop  = self.ystop
        scale = self.scale
        orient = self.fea_extra.orient
        hog_channel = self.fea_extra.hog_channel
        pix_per_cell = self.fea_extra.pix_per_cell
        cell_per_block = self.fea_extra.cell_per_block
        spatial_size = self.fea_extra.spatial_size
        hist_bins = self.fea_extra.hist_bins
        colorspace = self.fea_extra.color_space
        
        print(np.shape(img))    
        draw_img = np.copy(img)
        boxes = []
        img_tosearch = img[ystart:ystop,xstart:xstop,:]

        ctrans_tosearch = uf.convert_color(img_tosearch, colorspace)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        # Compute individual channel HOG features for the entire image
        hog1 = uf.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = uf.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = uf.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hogx = (hog1, hog2, hog3)
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                if hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_ = hogx[hog_channel]
                    hog_features = hog_[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                spatial_features = uf.bin_spatial(subimg, size=spatial_size)
                hist_features = uf.color_hist(subimg, nbins=hist_bins)
                  
                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_prediction = self.classifier.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    boxes.append(((xbox_left + xstart, ytop_draw + ystart),\
                                  (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
                    cv2.rectangle(draw_img,(xbox_left + xstart , ytop_draw+ystart),\
                                  (xbox_left+win_draw + xstart,ytop_draw+win_draw+ystart),(0,0,255),2) 
        
        draw_img = self.detec_false_positive(img, boxes)
        return draw_img
        
    def smothing(self,heat):
        self.heatmap.append(heat)
        heatmap = np.array(self.heatmap)
        if self.n < self.backnum:
            self.n += 1
            if self.n == 1:
                heatmap = heatmap.reshape((np.shape(heatmap)[1], np.shape(heatmap)[2]))
            else:
                self.heatmap[-2:(-self.n - 1):-1] = heatmap[-2:(-self.n - 1):-1] *0.8
                heatmap = np.sum(self.heatmap[-1:(-self.n - 1):-1],axis=0)
        else:
            self.heatmap[-2:(-self.backnum - 1):-1] = heatmap[-2:(-self.backnum - 1):-1] *0.8
            heatmap = np.sum(self.heatmap[-1:(-self.n - 1):-1],axis=0)
            self.heatmap.pop(0)
        return heatmap 
        
    def detec_false_positive(self,image,bbox_list):
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = self.add_heat(heat,bbox_list)
        # Visualize the heatmap when displaying    
        #heatmap = np.clip(heat, 0, 255)
        s = [[1,1,1],[1,1,1],[1,1,1]]
        heatmap = self.smothing(heat)
        print('Maximum of heatmap',np.amax(heatmap))
        heatmap = self.apply_threshold(heatmap,1.6)
        labels = label(heatmap)
        #labels = label(heatmap,structure=s)
        print("Number of car",(labels[1]))
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)
        return draw_img
                
    def add_heat(self,heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap# Iterate through list of bboxes

    def apply_threshold(self,heatmap, threshold):
        # Zero out pixels below the threshold
        
        heatmap[heatmap <= threshold] = 0
        
        # Return thresholded map
        return heatmap
    
    def draw_labeled_bboxes(self,img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
        # Return the image
        return img

       
class_svm = Classifier_Svm(0)
if 0:
    class_svm.train()
veh_dec = VehicleDetection(class_svm)

white_output = 'project_video.mp4'
clip1 = VideoFileClip(white_output)
lane_clip = clip1.fl_image(veh_dec.find_cars)
lane_clip.write_videofile("./new_test_video.mp4")

