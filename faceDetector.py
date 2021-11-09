import opencvDetector, mtcnnDetector
from PIL import Image
import math
import numpy as np
import distance

def build_model(detector_backend):

    global face_detector_obj #singleton design pattern

    backends = {
        'opencv': opencvDetector.build_model,
        'mtcnn' : mtcnnDetector.build_model
    }

    if not "face_detector_obj" in globals():
        face_detector_obj = {}

    if not detector_backend in face_detector_obj.keys():
        face_detector = backends.get(detector_backend)

        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
            #print(detector_backend," built")
        else:
            raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_detector_obj[detector_backend]

def detect_face(face_detector, detector_backend, img):

    obj = detect_faces(face_detector, detector_backend, img)

    if len(obj) > 0:
        face, region = obj[0] #discard multiple faces
    else: #len(obj) == 0
        face = None
        region = [0, 0, img.shape[0], img.shape[1]]

    return face, region

def detect_faces(face_detector, detector_backend, img):

    backends = {
        'opencv': opencvDetector.detect_face,
        'mtcnn' : mtcnnDetector.build_model
    }

    detect_face = backends.get(detector_backend)

    if detect_face:
        obj = detect_face(face_detector, img)
        #obj stores list of detected_face and region pair

        return obj
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)
