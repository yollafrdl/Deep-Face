import os
import numpy as np
import cv2
from matplotlib import pyplot

import faceDetector


import keras
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image


def detect_face(img, detector_backend = 'opencv', grayscale = False, enforce_detection = True):

	img_region = [0, 0, img.shape[0], img.shape[1]]
	face_detector = faceDetector.build_model(detector_backend)

	try:
		detected_face, img_region = faceDetector.detect_face(face_detector, detector_backend, img)
	except: 
		detected_face = None

	if (isinstance(detected_face, np.ndarray)):
		return detected_face, img_region
	else:
		if detected_face == None:
			if enforce_detection != True:
			  return img, img_region
			else:
			  raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False):

	img = cv2.imread(img)
	base_img = img.copy()

	img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection)

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()


	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	factor_0 = target_size[0] / img.shape[0]
	factor_1 = target_size[1] / img.shape[1]
	factor = min(factor_0, factor_1)

	dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
	img = cv2.resize(img, dsize)

	diff_0 = target_size[0] - img.shape[0]
	diff_1 = target_size[1] - img.shape[1]
	if grayscale == False:
		img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
	else:
		img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	
	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)


	img_pixels = image.img_to_array(img) 
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 

	if return_region == True:
		return img_pixels, region
	else:
		return img_pixels