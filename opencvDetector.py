import cv2
import os
import pandas as pd
import faceDetector

def build_model():

	detector ={}

	detector["face_detector"] = build_cascade('haarcascade')
	# detector["eye_detector"] = build_cascade('haarcascade_eye')

	return detector

def build_cascade(model_name = 'haarcascade'):

	if model_name == 'haarcascade':

		face_detector_path = 'Weights/haarcascade_frontalface_default.xml'

		if os.path.isfile(face_detector_path) != True:
			raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")


		face_detector = cv2.CascadeClassifier(face_detector_path)
		return face_detector

def detect_face(detector, img):

	resp = []

	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]

	faces = []
	try:
		#faces = detector["face_detector"].detectMultiScale(img, 1.3, 5)
		faces = detector["face_detector"].detectMultiScale(img, 1.1, 10)
	except:
		pass

	if len(faces) > 0:

		for x,y,w,h in faces:
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]

			img_region = [x, y, w, h]

			resp.append((detected_face, img_region))

	return resp