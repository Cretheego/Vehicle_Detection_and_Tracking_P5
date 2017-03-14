## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.




---
###General overview

The code for this project includes mainly  the following files.

The heat.py is used to build the bounding boxes classified as cars, and heat-map is constructed.

The classifier.py is used to train a Linear SVM classifier on the labeled training set .

The feature_extraction.py combines features from histograms of color, spatial binning, and Histogram of Oriented Gradients (HOG), on the labeled training set of images to create a feature vector. 

The util_functions.py includes  some public functions which implemts color space convertion and feature computation. 

The software pipeline  is written in vehicledetection.py to detect vehicles in a video stream.
