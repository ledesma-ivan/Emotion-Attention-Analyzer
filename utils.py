# focusgroup/utils.py
import cv2

def draw_bbox(frame, bbox, label=None, color=(0,255,0), thickness=2):
    """
    Dibuja un bounding box en el frame con un label legible,
    utilizando un fondo compacto y texto más pequeño para reducir la superposición.
    bbox: (x1, y1, x2, y2)
    label: texto que aparece arriba del rectángulo
    """
    x1, y1, x2, y2 = bbox
    
    # 1. Dibuja el bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    if label is not None:
        # Parámetros del texto (TAMAÑO REDUCIDO)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4        # Reducido de 0.6 a 0.5 para hacerlo más pequeño
        font_thickness = 1      # Reducido de 2 a 1 (texto más delgado)
        
        # 2. Calcula el tamaño del texto para crear el fondo
        (text_width, text_height), baseline = cv2.getTextSize(
            str(label), font, font_scale, font_thickness
        )
        
        # 3. Define las coordenadas del fondo (AJUSTE DE PADDING)
        # El texto se colocará justo encima del bounding box.
        padding_x = 3  # Relleno horizontal (reducido)
        padding_y = 5  # Espacio vertical (ajustado para que el fondo no toque la caja)

        text_x = x1
        text_y = y1 - padding_y 
        
        box_coords_start = (x1, y1 - text_height - padding_y - 2) # Esquina superior izquierda
        box_coords_end = (x1 + text_width + padding_x, y1 - 2)   # Esquina inferior derecha

        # 4. Dibuja el fondo negro (compacto)
        cv2.rectangle(
            frame, 
            box_coords_start, 
            box_coords_end, 
            (0, 0, 0), # Color de fondo: Negro
            cv2.FILLED # Relleno completo
        )
        
        # 5. Dibuja el texto sobre el fondo
        cv2.putText(
            frame, 
            str(label), 
            (text_x, text_y),
            font, 
            font_scale, 
            (255, 255, 255), # Color del texto: Blanco
            font_thickness
        )