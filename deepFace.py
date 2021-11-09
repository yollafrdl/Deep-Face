import warnings
warnings.filterwarnings("ignore")
import time
from os import path
import numpy as np
from tqdm import tqdm
import pickle
import functions, distance as dst, vggFace
import matplotlib as plt
import cv2

import tensorflow as tf

def build_model(model_name):

	global model_obj 

	models = {
		'VGG-Face': vggFace.loadModel,
	}

	if not "model_obj" in globals():
		model_obj = {}

	if not model_name in model_obj.keys():
		model = models.get(model_name)
		if model:
			model = model()
			model_obj[model_name] = model
		else:
			raise ValueError('Invalid model_name passed - {}'.format(model_name))

	return model_obj[model_name]

def verify(img1_path, img2_path, model_name = 'VGG-Face', distance_metric = 'cosine',  detector_backend = 'opencv', enforce_detection = True):
    model = build_model(model_name)
    img1_representation = represent(img_path = img1_path
						, model_name = model_name
						, enforce_detection = enforce_detection
                        , detector_backend = detector_backend
						)

    img2_representation = represent(img_path = img2_path
                        , model_name = model_name
                        , enforce_detection = enforce_detection
                        , detector_backend = detector_backend
                        )
    
    if distance_metric == 'cosine':
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    
    distance = np.float64(distance)

    threshold = dst.findThreshold(model_name, distance_metric)

    if distance <= threshold:
        identified = True
    else:
        identified = False

    resp_obj = {
        "verified": identified
        , "distance": distance
        , "max_threshold_to_verify": threshold
        , "model": model_name
        , "similarity_metric": distance_metric

    }

    return resp_obj


def represent(img_path, model_name = 'VGG-Face', enforce_detection = True, detector_backend = 'opencv'):
    
    model = build_model(model_name)
    input_shape = model.layers[0].input_shape
    input_shape_x, input_shape_y = input_shape[0][1:3]
    
    img = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend)
        
    img *= 255
    
    embedding = model.predict(img)[0].tolist()
    
    return embedding


img1 = 'D:\Pak Reza\detection\Data\shawn.jpg'
img2 = 'D:\Pak Reza\detection\Data\shawn2.jpg'


out = verify(img1_path = img1, img2_path = img2, distance_metric = 'euclidean', detector_backend = 'opencv', enforce_detection = True)
print(out)