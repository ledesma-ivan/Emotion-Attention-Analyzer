---
# Análisis de Emociones y Atención

Sistema de visión por computadora que detecta rostros, analiza emociones y evalúa el nivel de atención tanto en tiempo real como en procesamiento no en tiempo real, utilizando Python y herramientas de machine learning.

## Funcionalidades

* Detección de rostros en imágenes o video.
* Reconocimiento de emociones básicas.
* Análisis de atención mediante postura de cabeza o mirada.
* Procesamiento en tiempo real.
* Código modular (detector, emotion, headpose, utils, etc.).

## Tecnologías utilizadas

* Python 3
* OpenCV
* Mediapipe o Dlib
* NumPy
* Modelos preentrenados de detección y emociones

## Estructura del proyecto

```
Emotion-Attention-Analyzer/
│── detector.py
│── emotion.py
│── headpose.py
│── tracker.py
│── utils.py
│── main.py
```

## Cómo ejecutar

### Crear entorno virtual

```bash
python -m venv venv
```

O bien, si usas Python 3.12:

```bash
py -3.12 -m venv venv
```

### Activar el entorno virtual (Windows)

```bash
.\venv\Scripts\Activate
```

### Instalar dependencias

```bash
pip install -r .\requirements.txt
```

### Ejecutar el programa

```bash
python main.py
```

## Requisitos

* Python 3.x
* Webcam si se utiliza en tiempo real
* Librerías indicadas en `requirements.txt`

## Licencia

Proyecto de uso académico. Puede ser modificado o ampliado libremente.

---
