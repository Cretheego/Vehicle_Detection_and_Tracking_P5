# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:16:13 2017

@author: Administrator
"""

import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import _pickle as pkl
import pickle
import feature_extraction as fe  
from sklearn.utils import shuffle
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split

class Classifier_Svm():
    def __init__(self,flag):
        self.classifier = None
        self.score = 0
        if flag:
            self.data_prepare()
        
    # dara preprocess    
    def data_prepare(self):
        car_images = glob.glob(r"../vehicles/vehicles/*/*.png")
        nocar_images = glob.glob(r"../non-vehicles/non-vehicles/*/*.png")
        cars = []
        notcars = []
        #print("I am here")
        for image in car_images:
            cars.append(image)
        for image in nocar_images:
            notcars.append(image)
        #print("======")
        #print(np.shape(cars))
        #print(np.shape(notcars))    
        cars = shuffle(cars)
        notcars = shuffle(notcars)
        sample_size = 8000
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]
        
        self.fea_extra = fe.FeatureExtraction()
        spatial_feat = True
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off

        car_features = self.fea_extra.extract_features(cars, spatial_feat=spatial_feat, \
                                                  hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = self.fea_extra.extract_features(notcars,spatial_feat=spatial_feat,\
                                                     hist_feat=hist_feat, hog_feat=hog_feat)

        self.X = np.vstack((car_features, notcar_features)).astype(np.float64)  
        print("----",np.shape(self.X))                      
        self.y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
     
    # Training SVM classifier    
    def train(self, test_size=0.2, rand_range=(0, 100)):
        """ Train classifier """
        rand_state = np.random.randint(rand_range[0], rand_range[1])
        """ Train classifier from the given file list """
        # Fit a per-column scaler
        print("===",np.shape(self.X))
        self.X_scaler = StandardScaler().fit(self.X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(self.X)
        
        labels = self.y
        
        scaled_X, labels = shuffle(scaled_X, labels)
        # Split dataset into training and validation sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            scaled_X, labels, test_size=test_size, random_state=rand_state)
        
        self.classifier = LinearSVC()
        # Fit model
        self.classifier.fit(X_train, Y_train)
        # Validate results
        self.score = self.classifier.score(X_test, Y_test)
        print('Test Accuracy of SVC = ', round(self.score, 4))
        self.save()
        # Return model accuracy
        #return self.classifier

    def save(self, fname='svc_pickle.p'):
        """ Persist model """
        # Save model
        obj = {'model': self.classifier, 'scaler': self.X_scaler, 'fea_extra': self.fea_extra}
        with open(fname, 'wb') as f:
            aa = {'model': self.classifier, 'scaler': self.X_scaler, 'fea_extra': self.fea_extra}
            pickle.dump(aa,f)  

    def load(self, fname='svc_pickle.p'):
        """ Load model from file """
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            self.classifier = obj['model']
            self.X_scaler = obj['scaler']
            self.fea_extra = obj['fea_extra']

    def predict(self, features):
        """ Predict if given image contains object of interest """
        # Predict using classifier
        scaled_X = self.X_scaler.transform(np.array(features).reshape(1, -1))
        return self.classifier.predict(scaled_X)
