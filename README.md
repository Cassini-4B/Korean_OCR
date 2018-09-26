# Korean_OCR
Optical Character Recognition of handwritten Korean letters


This repository contains machine learning/classification models written in python to classify samples of handwritten Korean letters.  My dataset consists of 11050 images: 2210 original images of 26 letters total and 85 different fonts per letter, which were then translated by one row in each of the 4 directions using OpenCV to increase dataset size.

OpenCV was also used to generate features by obtaining image moments from contours in each image.  A total of 15 features were used to fit into the classification models:
1) Area of the contour
2) Perimeter of the contour
3) Center point of the contour (x pixel coordinate)
4) Center point of the contour (y pixel coordinate
5) Aspect ratio (width/height of rectangular bounding box)
6) Radius of bounding circle
7) Major axis of bounding ellipse
8) Minor axis of bounding ellipse
9) Angle of orientation (angle between major axis and image horizontal)
10) Bottommost point of contour (x pixel coordinate)
11) Bottommost point of contour (y pixel coordinate)
12) Rightmost point of contour (x pixel coordinate)
13) Rightmost point of contour (y pixel coordinate)
14) Height of bounding box 
15) Width of bounding box

After trying logistic regression, gaussian naive bayes, K nearest neighbors, SVM and Decision Trees/Random Forest, the best model to classify with was Convolutional Neural Nets.  

Also included is the code for a Flask translation app (for handwritten Korean characters) of an image uploaded to the website.
