import pandas as pd
import numpy as np
import cv2 as cv
import glob
import os
import pickle
import matplotlib.pyplot as plt
%matplotlib inline

contour_vals = {}

def get_img_mts(png):
    # read image
    img = cv.imread(png)
    
    # get shape and resize
    shape = img.shape
    ratio = 100.0 / shape[1]
    newdimension = (100, int(shape[0]*ratio))
    res = cv.resize(img, newdimension, interpolation=cv.INTER_AREA)
    
    # transform image to grayscale 
    grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # apply thresholding for img segmentation
    grayimg2 = cv.threshold(grayimg,0,255,cv.THRESH_BINARY)[1]
    
    # find contours
    im2, contours, hierarchy = cv.findContours(grayimg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    # get moments
    cnt = contours[0]
    M = cv.moments(cnt)

    # Feature 1: centroid
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except:
        cx, cy = 0,0   
       
    # Feature 2: area of contour    
    area = cv.contourArea(cnt)
      
    # Feature 3: perimeter of contour
    perimeter = cv.arcLength(cnt,True)
    
    # Feature 4: radius of bounding circle
    (x1,y1),radius = cv.minEnclosingCircle(cnt)
    center = (int(x1),int(y1))
    radius = int(radius)
    
    # Feature 5: aspect ratio (ratio of width to height of bounding rectangle)
    x2,y2,w,h = cv.boundingRect(cnt)
    aspect_ratio = float(w)/h
    
    # Feature 6: extent (ratio of contour area to bounding rectangle area)
    rect_area = w*h
    extent = float(area)/rect_area
    
    # Feature 7: orientation (angle at which object is directed) and major/minor axis lengths
    try:
        (x3,y3), (MA, ma), angle = cv.fitEllipse(cnt)
    except:
        (MA, ma), angle = (0,0),0
    
    # Feature 8: extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])


    # store values
    contour_vals[str(png)]= (cx, cy, area, perimeter, center, radius, aspect_ratio, w, h, extent, MA, ma, angle,
                               leftmost, rightmost, topmost, bottommost)

    return contour_vals




img_directory = ('Directory_where_images_are_stored')

for dirpath, dirname, filenames in os.walk(img_directory):
    for f in sorted(filenames):
        get_img_mts(f) 



with open('letter_features.pickle', 'wb') as l:
     pickle.dump(contour_vals, l)
