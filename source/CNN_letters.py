import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import os
import re
import pickle
from PIL import Image

# Must have a separate training and validation folder, with subfolders containing image samples for each class
# Must also use an hd5 filename to store model weights

def train_cnn_ocr(trainfolder, valfolder, weights_filename):	
	classifier = Sequential()
	classifier.add(Conv2D(32, (3, 3), input_shape = (100,100, 3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Flatten())
	classifier.add(Dense(units = 128, activation = 'relu'))
	classifier.add(Dense(units = 26, activation = 'softmax'))
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


	train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2)
	training_set = train_datagen.flow_from_directory(trainfolder,target_size = (100, 100),batch_size = 50, class_mode = 'categorical')
	test_datagen = ImageDataGenerator(rescale=1./255)
	valid_set = test_datagen.flow_from_directory(valfolder,target_size = (100,100),batch_size = 34, class_mode = 'categorical')

	
	return training_set, classifier.save_weights(weights_filename)


def test_cnn_ocr( weights_filename, testfolder, training_set):
	classifier.load_weights(weights_filename)

	test_datagen = ImageDataGenerator(rescale=1./255)
   	test_generator = test_datagen.flow_from_directory(testfolder,target_size=(100, 100),
	color_mode="rgb", shuffle = False, class_mode='categorical',batch_size=1)

	filenames = test_generator.filenames
	nb_samples = len(filenames)
	predict = classifier.predict_generator(test_generator,steps = nb_samples)

	pred_prob = np.argmax(predict, axis=-1) #multiple categories

	label_map = (training_set.class_indices)
	label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
	predictions = [label_map[k] for k in pred_prob]

	correct, wrong = 0,0
	for ypred, ytrue in zip(predictions,test_generator.filenames):
	    if ypred== re.findall('[a-z]{1,2}', ytrue.split('_')[1])[0]:
        	correct +=1
	    else:
        	 wrong +=
    
	accuracy = float(correct)/float(len(test_generator.filenames))
	return accuracy 




# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# serialize class indices
with open('train_class_indices.pickle', 'wb') as f:
    pickle.dump(training_set.class_indices, f)

