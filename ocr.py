import pandas as pd
import numpy as np
import cv2 as cv
import glob
import os
import pickle
from get_letter_features import get_img_mts
from get_letter_classes import get_classes
from CNN_letters import train_cnn_ocr, test_cnn_ocr



image_directory = 'folder where the image samples reside'

# 1)  To generate contour/letter features for each image:
for dirpath, dirname, filenames in os.walk(image_directory):
    for f in sorted(filenames):
        get_img_mts(f) 


# 2) To get a dataframe of target classes for each image:
	get_classes(image_directory)


training = 'folder with all training images'
validation = 'folder with validation images'
testing = 'folder with test images'
w.hd5 = 'filename for model weights'

# 3) To apply a CNN model to generate letter predictions:
	train_cnn_ocr(training, validation, w.hd5)
	test_cnn_ocr(w.hd5, testing, training_set)


