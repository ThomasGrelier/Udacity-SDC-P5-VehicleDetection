# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:51:29 2017

@author: thomas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:34:21 2017

** VEHICLE DETECTION AND TRACKING **
Project #5 of the first term of Udacity's Self Driving Car nanodegree program

Objective of this project is to detect and track vehicles.
Note: This script works only on videos. A different script exists for processing images.

The steps of the pipeline are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

@author: greliert
"""

#%% import useful modules
import numpy as np
import cv2
import pickle
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt

class Vehicle():
    def __init__(self,path_calib,path_detect):
        self.heatmap = np.zeros((720,1280,1)).astype(np.float)
        self.n_frame = 0   # number of frame
        
        # load calibration matrix and distortion coefficients
        self.load_calibration(path_calib)
        # load classifier and feature extraction parameters for vehicle detection
        self.load_param_detect(path_detect)
    
    def load_calibration(self,pathname):
        '''
        Load camera calibration matrix and distortion coefficients
        Input:
        - pathname: path of the pickle file containing the calibration parameters
        '''
        dist_pickle = pickle.load(open(pathname, "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
    
    def load_param_detect(self,pathname):
        '''
        load classifier and feature extraction parameters for car detection
        Input:
        - pathname: path of the pickle file containing parameters
        '''
        dist_pickle = pickle.load( open(pathname, "rb" ) )
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        self.color_space = dist_pickle["color_space"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.hog_channel = dist_pickle["hog_channel"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]
        self.spatial_feat = dist_pickle["spatial_feat"]
        self.hist_feat = dist_pickle["hist_feat"]
        self.hog_feat = dist_pickle["hog_feat"]

    def process_image(self,image):
        '''
        Detect vehicles and plot rectangle around detected cars
        Inputs:
        - image: input image (RGB)
        Outputs:
        - out_img: processed image (RGB)
        '''
        # 1) Apply a distortion correction to raw images.
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        
        # Min and max in y to search in slide_window()
        y_start_stop_list = [[400, 550],[400, 650],[400, None],[400, None]]
        # Scale factor for detection window: size = (64*scale,64*scale): 
        scale_list = [1,2,3,4]
        # Rectangle colors 
        color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        # window overlap
        overlap = 0.75
        # number of heatmap accumulated over time
        n_acc = 10
        # heatmap thresholds to remove false positive
        n_occ_thr = 3
        heat_thr = 0
        heat_acc_thr = 8
        
        windows_ = []
        hot_windows_ = []
        draw_image = np.copy(image)
        draw_image_hot = np.copy(image)
        # repeat search for cars for each window size
        for scale,y_start_stop,color in zip(scale_list,y_start_stop_list,color_list):
            windows, hot_windows = find_cars(image, y_start_stop, scale, overlap, self.svc, self.X_scaler, self.color_space, 
                                             self.orient,self.pix_per_cell, self.cell_per_block, self.hog_channel,
                                             self.spatial_size, self.hist_bins, self.spatial_feat, self.hist_feat,
                                             self.hog_feat)
            windows_+=windows
            hot_windows_+=hot_windows
            # draw all windows / hot windows only
            draw_image = draw_boxes(draw_image, windows, color=color, thick=6)   
            draw_image_hot = draw_boxes(draw_image_hot, hot_windows, color=color, thick=6)   
                 
        # create heat map
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat for each box in box list
        heat = add_heat(heat,hot_windows_)
        # Apply threshold to help remove false positives
        heatmap = apply_threshold(heat,heat_thr)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = np.copy(image)
        #draw_img = draw_labeled_bboxes(draw_img, labels)

        # Append new heatmap to history, and remove oldest one
        self.heatmap = np.append(self.heatmap[:,:,-n_acc+1:],heatmap[:,:,None],2)
               
        # Keep only labels which are present at least in n_occ_thr frames out of n_acc
        # pixel location associated to a new label is computed as the logical or pixel locations over the n_acc 
        # frames
        heatmap_hist = self.heatmap
        labels = label(heatmap_hist)
        n_occ = np.zeros(labels[1])   # nb of frames where label is present
        mask = np.zeros_like(heatmap_hist,dtype=np.bool)
        for i in np.arange(labels[1]):
            for j in np.arange(heatmap_hist.shape[2]):
                mask_i_j = labels[0][:,:,j]==i+1
                if np.any(mask_i_j):
                    n_occ[i]+=1
            if n_occ[i]>=n_occ_thr:  
                # if nb of occurences >= threshold, then keep pixels with this label
                mask = np.logical_or(mask, labels[0]==i+1)
        
        # Create new_heatmap: extraction of selected labelled values in heatmap_hist
        new_heatmap = np.zeros_like(heatmap_hist)
        new_heatmap[mask] = heatmap_hist[mask]
        
        # Accumulate heatmap, apply threshold, and determine new labels
        new_heatmap_acc = np.sum(new_heatmap,2)
        new_heatmap_acc_thr = apply_threshold(new_heatmap_acc,heat_acc_thr)
        new_labels = label(new_heatmap_acc_thr)
        
        draw_img = draw_labeled_bboxes(draw_img, new_labels, color=(255,0,0))
        
        # add it in the upper left corner of video
        if 1:
            # Concatenate 4 images for plotting and save in .jpg file
            p1 = np.concatenate((draw_image_hot,draw_img),0)
            p2 = np.concatenate((heatmap,new_heatmap_acc_thr),0)
            p2 = np.clip(np.dstack((p2,p2,p2))*10,0,255)
            p =  np.concatenate((p1,p2.astype(np.uint8)),1)
            fig = plt.figure(figsize=(9,7))
            plt.axis('off')
            plt.imshow(p)
            draw_img = cv2.resize(p,(500,400))
       
        self.n_frame += 1
        plt.savefig('./images/results'+str(self.n_frame)+'.jpg')
        #print('image ',self.n_frame)
        return draw_img
    
# VIDEO PROCESSING
# Parameters
path_calib = './calib.p'           # Camera calibration directory
path_param = './svc_pickle19.p'           # Camera calibration directory
video_path = "./project_video.mp4"      # Video file path

# Video output path
video_output_path = 'video_output.mp4'
# create a line object
vehicle = Vehicle(path_calib,path_param)   
# Load video
#clip1 = VideoFileClip(video_path)
clip1 = VideoFileClip(video_path).subclip(25,30)

# Process video
out_clip = clip1.fl_image(vehicle.process_image)
# Write output video
out_clip.write_videofile(video_output_path, audio=False)