import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np # Necesario para el filtrado y cálculo de pose
from collections import defaultdict

# Módulos
from detector import FaceDetectorMTCNN
from tracker import Tracker
from emotion import EmotionRecognizerVIT
from utils import draw_bbox
from report import EmotionStatsReport
from headpose import HeadPoseEstimator # Para cálculo de atención

plt.style.use("dark_background")

# Colores personalizados para modo oscuro
plt.rcParams["text.color"] = "#e6e6e6"
plt.rcParams["axes.labelcolor"] = "#e6e6e6"
plt.rcParams["axes.edgecolor"] = "#888888"
plt.rcParams["axes.facecolor"] = "#2b2b2b"
plt.rcParams["figure.facecolor"] = "#2b2b2b"
plt.rcParams["xtick.color"] = "#e6e6e6"
plt.rcParams["ytick.color"] = "#e6e6e6"


class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Emoción y Atención")
        self.root.geometry("800x900")
        self.video_running = False
        self.cap = None

        self.frame_count = 0
        self.last_emotion = {}   # {pid: "emotion"}
        
        # Variables de Estabilidad y Pose:
        self.yaw_history = defaultdict(lambda: []) 
        self.MAX_YAW_HISTORY = 15 # Ventana de suavizado (15 frames = 0.5 segundos)
        self.max_people_seen = 0 # NUEVO: Contador del máximo de personas vistas

        # Mapa de emociones en español
        self.emotion_map_es = {
            "happy": "Feliz",
            "sad": "Triste",
            "anger": "Enojo",
            "neutral": "Neutral",
            "surprise": "Sorpresa",
            "disgust": "Disgusto",
            "fear": "Miedo"
        }

        # ───────── MODO OSCURO Tkinter + TTK ─────────
        root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use("clam")

        # Colores base
        bg_dark   = "#1e1e1e"
        bg_panel  = "#2b2b2b"
        fg_text   = "#e6e6e6"
        accent    = "#4a90e2"

        # ----- Botones -----
        style.configure(
            "TButton",
            font=("Segoe UI", 11),
            padding=6,
            background=bg_panel,
            foreground=fg_text,
        )
        style.map(
            "TButton",
            background=[("active", accent)],
            foreground=[("active", "#ffffff")]
        )

        # ----- Frames -----

        def set_dark(frame):
            frame.configure(bg=bg_dark)

        # ───────── Layout general ─────────
        self.frame_left = tk.Frame(root, bg=bg_dark)
        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.frame_right = tk.Frame(root, bg=bg_dark)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Panel de video
        self.video_label = tk.Label(self.frame_left, bg=bg_dark)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Botones
        ttk.Button(self.frame_right, text="Usar Cámara", command=self.start_camera).pack(fill='x', pady=5)
        ttk.Button(self.frame_right, text="Cargar Video", command=self.load_video).pack(fill='x', pady=5)
        ttk.Button(self.frame_right, text="Detener", command=self.stop_video).pack(fill='x', pady=5)
        ttk.Button(self.frame_right, text="Exportar Reporte", command=self.export_report).pack(fill='x', pady=5)

        # --- Panel contenedor ---
        self.right_panel = tk.Frame(self.frame_right, bg=bg_dark)
        self.right_panel.pack(fill=tk.BOTH, expand=True)

        # ===============================
        #   FILA 1: Stats (izq) | Atención (der)
        # ===============================

        self.frame_top_left = tk.Frame(self.right_panel, bg=bg_dark)
        self.frame_top_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_top_right = tk.Frame(self.right_panel, bg=bg_dark)
        self.frame_top_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # ---- Estadísticas ----
        tk.Label(self.frame_top_left, text="Estadísticas",bg="#1e1e1e", fg="#e6e6e6", font=("Segoe UI", 12, "bold")).pack()
        self.stats_text = tk.Text(
            self.frame_top_left,
            width=30,
            height=12,
            bg="#1e1e1e",
            fg="#e6e6e6",
            highlightthickness=0, borderwidth=0,
            insertbackground="#e6e6e6",   # cursor visible
            font=("Segoe UI", 12)         
        )
        self.stats_text.pack(fill='both', expand=True)

        # ---- Gráfico de emociones ----
        fig1 = Figure(figsize=(4, 3))
        self.ax_attention = fig1.add_subplot(111)
        self.ax_attention.set_title("Atención Total")
        self.ax_attention.set_facecolor("#2b2b2b")

        self.canvas = FigureCanvasTkAgg(fig1, master=self.frame_top_right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # ===============================
        #   FILA 2: Gráfico Emociones
        # ===============================

        self.frame_bottom = tk.Frame(self.right_panel, bg=bg_dark)
        self.frame_bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        fig2 = Figure(figsize=(8, 5), facecolor="#2b2b2b")  
        self.ax = fig2.add_subplot(111)
        self.ax.set_title("Emociones")


        self.canvas_attention = FigureCanvasTkAgg(fig2, master=self.frame_bottom)
        self.canvas_attention.get_tk_widget().pack(fill='both', expand=True, pady=8)

        # ---- Configurar expansión de grilla ----
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_columnconfigure(1, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=2)   # fila inferior = doble de altura

        # ───────── Inicialización de módulos ─────────
        self.detector = FaceDetectorMTCNN(device='cpu')
        self.tracker = Tracker()
        self.emotion_model = EmotionRecognizerVIT()
        self.pose_estimator = HeadPoseEstimator()

        data = defaultdict(list)
        attention = defaultdict(list)
        self.stats = EmotionStatsReport(data=data, attention=attention)

        self.last_csv_path = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ───────── Video ─────────
    def start_camera(self):
        self.start_video_source(0)

    def load_video(self):
        path = filedialog.askopenfilename(title="Seleccionar video")
        if path:
            self.start_video_source(path)

    def start_video_source(self, source):
        # 1. Detener la fuente anterior
        self.stop_video()
        self.cap = cv2.VideoCapture(source)
        
        # --- LIMPIEZA DE DATOS ACUMULADOS ---
        data = defaultdict(list)
        attention = defaultdict(list)
        self.stats = EmotionStatsReport(data=data, attention=attention) 
        
        self.frame_count = 0 
        self.last_emotion = {} 
        self.yaw_history = defaultdict(lambda: []) # Reiniciar el historial de suavizado de pose
        self.max_people_seen = 0 # Resetear el contador de máximo de personas vistas
        
        # ------------------------------------
        
        # 4. Iniciar el nuevo bucle
        self.video_running = True
        self.update_frame()

    def stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        if self.stats:
            self.last_csv_path = self.stats.save_csv()

    # ───────── Update frame ─────────
    def update_frame(self):
        if not self.video_running or not self.cap or not self.cap.isOpened():
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return

            frame_resized = cv2.resize(frame, (900, 720))
            h, w, _ = frame_resized.shape

            # ---------------------------
            # DETECCIÓN, FILTRADO Y PROCESAMIENTO
            # ---------------------------
            # self.detector.detect_faces devuelve [(bbox, landmarks, proba), ...]
            faces_and_landmarks_probas = self.detector.detect_faces(frame_resized) 
            
            # --- FILTRADO DE ROSTROS POR TAMAÑO Y CONFIANZA ---
            filtered_results = []
            MIN_FACE_SIZE = 10 
            MIN_CONFIDENCE = 0.95 
            MAX_DETECTION_AREA_RATIO = 0.9 
            FRAME_AREA = h * w
            
            if faces_and_landmarks_probas: 
                for bbox, landmarks, proba in faces_and_landmarks_probas: 
                    x1, y1, x2, y2 = bbox
                    w_face = x2 - x1
                    h_face = y2 - y1
                    face_area = w_face * h_face
                    
                    # Filtros de estabilidad:
                    is_high_confidence = (proba >= MIN_CONFIDENCE)
                    is_valid_size = (w_face >= MIN_FACE_SIZE and h_face >= MIN_FACE_SIZE)
                    is_not_too_large = (face_area / FRAME_AREA) < MAX_DETECTION_AREA_RATIO

                    if is_valid_size and is_not_too_large and is_high_confidence:
                        # Usamos (bbox, landmarks) para el bucle de procesamiento
                        filtered_results.append((bbox, landmarks)) 
            
            video_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            active_ids = [] # Contiene los IDs TEMPORALES (índice i) activos en este frame

            # --- CÁLCULO DEL CONTEO ÚNICO (Máximo) ---
            num_people_active = len(filtered_results)
            if num_people_active > self.max_people_seen:
                self.max_people_seen = num_people_active
            # ------------------------------------------

            # Itera sobre las detecciones filtradas
            for i, (bbox, landmarks) in enumerate(filtered_results): 
                x1, y1, x2, y2 = bbox
                pid = i # PID TEMPORAL, PERO ÚNICO EN ESTE FRAME
                
                # Recorte seguro de la cara
                y1_crop, y2_crop = max(0, y1), min(h, y2)
                x1_crop, x2_crop = max(0, x1), min(w, x2)
                
                if y2_crop <= y1_crop or x2_crop <= x1_crop:
                    continue
                    
                face_crop = frame_resized[y1_crop:y2_crop, x1_crop:x2_crop]
                active_ids.append(pid)

                # ---------------------------
                # CÁLCULO DE EMOCIÓN OPTIMIZADO (usa el PID TEMPORAL)
                # ---------------------------
                emotion = self.last_emotion.get(pid, "unknown")
                
                if pid not in self.last_emotion or self.frame_count % 10 == 0:
                    try:
                        new_emotion = self.emotion_model.predict_face(face_crop)
                    except:
                        new_emotion = "unknown"
                    # Guardamos la emoción con el PID temporal de este frame
                    self.last_emotion[pid] = new_emotion 
                    emotion = new_emotion

                # ---------------------------
                # ATENCIÓN (USANDO HEAD POSE Y SUAVIZADO)
                # ---------------------------
                yaw, pitch, roll = self.pose_estimator.estimate_pose(landmarks, (w, h))

                attention_flag = False
                smooth_yaw = None
                
                # Suavizado de Yaw
                if yaw is not None:
                    # Usamos el PID TEMPORAL para acceder a yaw_history
                    hist = self.yaw_history[pid] 
                    hist.append(yaw)
                    
                    if len(hist) > self.MAX_YAW_HISTORY:
                        hist.pop(0) 
                        
                    if len(hist) >= 3: 
                        smooth_yaw = sum(hist) / len(hist)
                    else:
                        smooth_yaw = yaw
                    
                    # Rango de atención ajustado mirada al frente
                    if -100.0 <= smooth_yaw <= 100.0: 
                        attention_flag = True
                        
                # ─ Actualizar stats ─
                self.stats.data[pid].append((emotion, video_time))
                self.stats.attention[pid].append((attention_flag, video_time))

                emotion_es = self.emotion_map_es.get(emotion, emotion)
                
                pose_label = f" | Yaw: {smooth_yaw:.1f}°" if smooth_yaw is not None else " | Yaw: N/A"
                label_text = f"ID:{pid}: {emotion_es}" + (" Atento" if attention_flag else " No Atento") 

                draw_bbox(frame_resized, bbox, label=label_text) 
                
            # ---------------------------
            # ACTUALIZACIÓN DE TKINTER
            # ---------------------------
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            
            # FIX DE PERSISTENCIA
            img_tk = ImageTk.PhotoImage(img_pil) 
            self.video_label.image = img_tk
            self.video_label.config(image=img_tk)

            if self.frame_count % 10 == 0:
                # El conteo total único es el máximo visto hasta ahora
                self.update_stats_panel(active_ids, self.max_people_seen) 
                self.update_chart()
                self.update_attention_chart()


        except Exception as e:
            print("ERROR update_frame:", e)
        
        self.frame_count += 1
        self.root.after(33, self.update_frame) 

    # ───────── Panel lateral ─────────
    # Acepta el conteo total de personas únicas (maximum)
    def update_stats_panel(self, active_ids, unique_people_count): 
        durations = self.stats.compute_durations() 
        self.stats_text.delete("1.0", tk.END) 

        # --- Número de personas detectadas ---
        num_people_active = len(active_ids)
        # Mostrar el conteo máximo visto y el activo
        self.stats_text.insert(tk.END, f"Personas activas: {num_people_active}\n") 
        self.stats_text.insert(tk.END, f"Máximo personas simultáneas: {unique_people_count}\n\n") 

        # --- TÍTULO EXPERIENCIA (Posicionado antes de las duraciones) ---
        self.stats_text.insert(tk.END, "EXPERIENCIA EMOCIONAL MÁXIMA:\n")

        if not durations: 
            self.stats_text.insert(tk.END, "Sin datos todavía...") 
            return 

        for em, sec in durations.items(): 
            em_es = self.emotion_map_es.get(em, em) 
            self.stats_text.insert(tk.END, f"{em_es}: {sec:.2f} s\n")

    # ... (resto de las funciones) ...


    # ───────── Gráfico de emociones ─────────
    def update_chart(self):
        summary = self.stats.get_live_summary()
        emotions = [self.emotion_map_es.get(e,e) for e in summary.keys()]
        counts = list(summary.values())
        colors = {'Feliz':'goldenrod', 'Triste':'cornflowerblue', 'Enojo':'orangered', 
                  'Neutral':'slategray', 'Sorpresa': 'mediumpurple', 'Disgusto':'lightseagreen', 'Miedo':'darkcyan'}

        self.ax.clear()
        self.ax.bar(emotions, counts, color=[colors.get(e,'gray') for e in emotions])
        self.ax.set_ylabel("Cantidad de cuadros", fontsize=10)
        self.ax.set_xlabel("Emoción", fontsize=10)
        self.ax.set_title("Distribución de emociones", fontsize=12, fontweight='bold')
        self.ax.tick_params(axis='x', rotation=0)
        self.ax.margins(x=0.05)
        self.ax.figure.tight_layout()
        self.canvas.draw()

    # ───────── Gráfico de atención ─────────
    def update_attention_chart(self):
        # Contadores globales
        total = 0
        attentive = 0
        for entries in self.stats.attention.values():
            for att_flag, _ in entries:
                total += 1
                if att_flag:
                    attentive += 1
        not_attentive = total - attentive

        # Limpiar el eje
        self.ax_attention.clear()

        if total == 0:
            self.ax_attention.text(
                0.5, 0.5, "Sin datos de atención",
                ha='center', va='center',
                fontsize=12, fontweight="bold"
            )
        else:
            sizes = [attentive, not_attentive]
            labels = ["Atento", "No Atento"]
            colors = ["cadetblue", "lightcoral"]

            # --- Donut ---
            wedges, texts, autotexts = self.ax_attention.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={'width': 0.38, 'edgecolor': 'white'}, 
                pctdistance=0.75
            )

            # Mejorar texto
            for t in texts:
                t.set_fontsize(10)
                t.set_fontweight("bold")
            for t in autotexts:
                t.set_fontsize(9)
                t.set_color("white")
                t.set_fontweight("bold")

        # Título
        self.ax_attention.set_title("Atención Total", fontsize=13, fontweight='bold')

        # Ajuste general para que no se corte
        self.ax_attention.figure.subplots_adjust(
            left=0.05, right=0.95, top=0.88, bottom=0.05
        )

        # Dibujar en canvas
        self.canvas_attention.draw()


    # ───────── Cierre ─────────
    def on_closing(self):
        if self.stats:
            self.stats.save_csv()
        self.root.destroy()

    def export_report(self):
        if self.stats:
            self.stats.save_csv()  # sigue estando bien
            self.stats.generate_pdf_report()

            tk.messagebox.showinfo("Reporte generado", "El PDF fue creado correctamente.")



if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionGUI(root)
    root.mainloop()
