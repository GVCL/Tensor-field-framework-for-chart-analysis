from numpy import loadtxt
from keras.models import load_model
from keras.models import model_from_json

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def classifyImage(path):
	# load json and create model
	json_file = open('Chart_Classification/model_bartype_50.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights("Chart_Classification/model_bartype_50.h5")
	print("Loaded model from disk")
	print(loaded_model.summary())

	image = Image.open(path)
	rgb_im = image.convert('RGB')
	image = rgb_im.resize((200, 200), Image.ANTIALIAS)
	image = np.asarray(image)
	image = image[...,:3].reshape(1, 200, 200, 3)
	image = image.astype('float32')
	image /= 255.0

	pred = loaded_model.predict_classes(image)
	if pred == 0:
		bartype="Histogram"
	elif pred == 1:
		bartype = "Horizontal_grouped_bar"
	elif pred == 2:
		bartype = "Horizontal_simple_bar"
	elif pred == 3:
		bartype = "Horizontal_stacked_bar"
	elif pred == 4:
		bartype = "Vertical_grouped_bar"
	elif pred == 5:
		bartype = "Vertical_simple_bar"
	elif pred == 6:
		bartype = "Vertical_stacked_bar"
	else:
		bartype = "other"
	
	return bartype
