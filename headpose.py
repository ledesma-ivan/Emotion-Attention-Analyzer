# focusgroup/headpose.py
import cv2
import numpy as np
# Eliminamos imports innecesarios para headpose: from facenet_pytorch.models.mtcnn import PNet, RNet, ONet

# Puntos 3D de un modelo de rostro genérico (se asume un rostro promedio)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.double) # CORRECCIÓN: Usar np.double

# ... (código de inicialización de la clase) ...

class HeadPoseEstimator:
    def __init__(self, fx=900, fy=900, cx=450, cy=360):
        # Parámetros intrínsecos de la cámara (aproximados para un frame de 900x720)
        self.camera_matrix = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]], dtype=np.double # CORRECCIÓN: Usar np.double
        )
        self.dist_coeffs = np.zeros((4, 1)) # Asumimos distorsión cero

    def estimate_pose(self, landmarks, frame_size):
        """
        Estima la pose de la cabeza (Yaw, Pitch, Roll) a partir de los 5 landmarks.
        landmarks: numpy array de 5 puntos (x, y) de la cara (e.g., de MTCNN).
        frame_size: (width, height) del frame.
        """
        if landmarks is None or len(landmarks) < 5:
            return None, None, None # Yaw, Pitch, Roll

        # Puntos de la imagen (orden MTCNN: Ojo_I, Ojo_D, Nariz, Boca_I, Boca_D)
        # Aseguramos el tipo de dato para solvePnP
        image_points = np.array([
            landmarks[2], # Nariz (punto 0 en el modelo 3D)
            landmarks[2], # Usamos nariz para Chin (aproximación)
            landmarks[0], # Ojo I (punto 2 en el modelo 3D)
            landmarks[1], # Ojo D (punto 3 en el modelo 3D)
            landmarks[3], # Boca I (punto 4 en el modelo 3D)
            landmarks[4]  # Boca D (punto 5 en el modelo 3D)
        ], dtype=np.double) # CORRECCIÓN: Usar np.double
        
        # Obtenemos rotación y traslación
        # solvePnP puede retornar valores incorrectos si no converge, pero devuelve 'success'
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            MODEL_POINTS, image_points, 
            self.camera_matrix, self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Si solvePnP falla, no hay pose válida (success no siempre es un booleano, pero lo usamos como indicador)
        if rotation_vector is None:
             return None, None, None 

        # Convertimos el vector de rotación a una matriz de rotación
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)

        # Calculamos ángulos de Euler (Yaw, Pitch, Roll)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        # Convertimos a grados
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)

        return yaw, pitch, roll