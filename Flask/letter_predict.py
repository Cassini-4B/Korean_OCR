import pickle
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras import backend as K


# create a function to use the CNN model created for classification to predict what letter is in the image that was uploaded onto the webpage.
def letter_identification(infile):
   
   K.clear_session()

   # read in the model
   # load json and create model
   json_file = open('model.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   loaded_model = model_from_json(loaded_model_json)
   # load weights into new model
   loaded_model.load_weights("korean_ocr_weights.h5")
   # evaluate loaded model on test data
   loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   # read training class indices
   label_map = pd.read_pickle('train_class_indices.pickle')
   # Read image and input into model 
   test_image = image.load_img(infile, target_size = (100,100))
   test_image = image.img_to_array(test_image)
   test_image = np.expand_dims(test_image, axis = 0)
   result = loaded_model.predict(test_image)

   # Make prediction
   prediction1 = loaded_model.predict_classes(test_image)

   # Determine letter
   label_map2 = dict((v,k) for k,v in label_map.items()) #flip k,v

   letter_predicted = (label_map2[int(prediction1)])

    # return a message
   message = "The letter is {}.".format(str(letter_predicted))

   return message
