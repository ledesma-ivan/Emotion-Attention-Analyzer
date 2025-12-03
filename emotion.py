# emotion.py
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class EmotionRecognizerVIT:
    def __init__(self, model_name="mo-thecreator/vit-Facial-Expression-Recognition", device="cpu"):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        # Forzamos que labels sea un diccionario id2label válido
        self.labels = getattr(self.model.config, "id2label", None)
        if not self.labels:
            # fallback genérico
            self.labels = {i: str(i) for i in range(self.model.config.num_labels)}

    def predict_face(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return "unknown"
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(face_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.softmax(out.logits[0], dim=0).cpu().numpy()
            idx = int(np.argmax(probs))
            # devolver etiqueta como string legible
            return self.labels.get(idx, str(idx))


