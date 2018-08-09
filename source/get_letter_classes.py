import pandas as pd
import numpy as np
import re
import glob
import pickle

file_names = []
letter = []
targets = []


def get_classes(img_directory):
	for png in glob.glob(img_directory + '/*.png'):
	    name = png.split('/')
	    file_names.append(name[1])

	for f in file_names:
	    name = re.findall('[a-z]', f.split('_')[0])
	    letter.append(''.join(name))

	for l in letter:
    	targets.append(re.sub('png','',l))

	target_classes = pd.DataFrame({'Filenames':file_names, 'Letter': targets})

	return target_classes


with open('target_classes.pickle', 'wb') as t:
	pickle.dump(target_classes, t)
