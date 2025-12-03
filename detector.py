# focusgroup/detector.py
from facenet_pytorch import MTCNN
import torch
import cv2
import numpy as np 

class FaceDetectorMTCNN:
    def __init__(self, device='cpu'):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device) 

    def detect_faces(self, frame):
        """
        Devuelve una lista de tuplas: [((x1, y1, x2, y2), landmarks, confidence), ...]
        donde confidence es la probabilidad de que sea un rostro (float).
        frame: imagen BGR de OpenCV
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CAMBIO CLAVE: Capturamos boxes, probas, y landmarks
        boxes, probas, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True) 
        
        results = []
        if boxes is not None and landmarks is not None and probas is not None and len(boxes) == len(landmarks):
            for box, lm, proba in zip(boxes, landmarks, probas): # Iteramos sobre probas
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2, y2)
                
                # Devolvemos la tupla (bbox, landmarks, proba)
                results.append((bbox, lm, proba)) 
        
        return results