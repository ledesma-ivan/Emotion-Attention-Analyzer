# report.py
import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

class EmotionStatsReport:
    """
    Clase para guardar y graficar emociones y atención recogidas en la GUI.
    """

    def __init__(self, data, attention, save_folder="reports"):
        """
        data: dict pid -> list de (emotion, timestamp)
        attention: dict pid -> list de (att_flag, timestamp)
        """
        self.data = data
        self.attention = attention
        self.save_folder = save_folder
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

        os.makedirs(save_folder, exist_ok=True)

        # ─────────────────────────────
    # Método para gráficos en vivo
    # ─────────────────────────────
    def get_live_summary(self):
        """
        Devuelve un diccionario {emoción: cantidad total de cuadros} combinando todas las personas
        """
        counter = defaultdict(int)
        for pid, entries in self.data.items():
            for emotion, _ in entries:
                counter[emotion] += 1
        return dict(counter)

    def compute_durations(self):
        """
        Calcula la duración máxima por emoción (el tiempo más largo que una sola persona
        mantuvo esa emoción) usando la diferencia entre timestamps consecutivos.
        """
        # emotion_max_time almacenará la duración MÁXIMA para cada emoción encontrada
        emotion_max_time = defaultdict(float) 

        for pid, entries in self.data.items():
            
            # 1. Calculamos la duración de las emociones para la persona actual (PID)
            person_emotion_time = defaultdict(float)
            
            # Aseguramos que las entradas estén ordenadas por timestamp
            entries = sorted(entries, key=lambda x: x[1])
            
            for i in range(len(entries)-1):
                emotion, t = entries[i]
                _, t_next = entries[i+1]
                dt = max(0, t_next - t)
                
                # Sumamos el tiempo de esta emoción SOLO para esta persona
                person_emotion_time[emotion] += dt
            
            # 2. Comparamos los tiempos de esta persona con el máximo global
            for emotion, duration in person_emotion_time.items():
                # Si la duración de esta persona es mayor que el máximo actual, actualizamos
                if duration > emotion_max_time[emotion]:
                    emotion_max_time[emotion] = duration
                    
        return dict(emotion_max_time)

    def save_csv(self, filename="emotion_attention_report.csv"):
        """
        Guarda un CSV con columnas:
        pid, emotion, emotion_timestamp, attention, attention_timestamp
        """
        csv_path = os.path.join(self.save_folder, filename)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pid", "emotion", "emotion_timestamp", "attention", "attention_timestamp"])

            pids = set(list(self.data.keys()) + list(self.attention.keys()))
            for pid in pids:
                emotion_entries = self.data.get(pid, [])
                attention_entries = self.attention.get(pid, [])

                # combinar por índice, rellenando None si no hay dato
                max_len = max(len(emotion_entries), len(attention_entries))
                for i in range(max_len):
                    e, et = (emotion_entries[i] if i < len(emotion_entries) else ("", ""))
                    a, at = (attention_entries[i] if i < len(attention_entries) else ("", ""))
                    writer.writerow([pid, e, et, a, at])

        print(f"[OK] Reporte guardado en {csv_path}")
        return csv_path


    # ------------------------
    #  Gráfico de emociones
    # ------------------------
    def plot_emotion_distribution(self, return_fig=False):
        all_emotions = []
        for entries in self.data.values():
            for emotion, _ in entries:
                all_emotions.append(emotion)

        if not all_emotions:
            print("No hay datos para graficar.")
            return

        labels = list(set(all_emotions))
        values = [all_emotions.count(e) for e in labels]

        colors = {
            "Feliz": "goldenrod",
            "Triste": "cornflowerblue",
            "Enojo": "orangered",
            "Neutral": "slategray",
            "Sorpresa": "mediumpurple",
            "Disgusto": "lightseagreen",
            "Miedo": "crimson"
        }

        fig = plt.figure(figsize=(8,5))
        plt.bar(labels, values, color=[colors.get(l, "gray") for l in labels])
        plt.title("Distribución Total de Emociones")
        plt.xlabel("Emoción")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()


    # ----------------------
    # Gráfico de atención
    # ----------------------
    def plot_attention_distribution(self, return_fig=False):
        total = 0
        attentive = 0
        for entries in self.attention.values():
            for att_flag, _ in entries:
                total += 1
                if att_flag:
                    attentive += 1

        not_attentive = total - attentive

        fig, ax = plt.subplots(figsize=(6,6))

        if total == 0:
            ax.text(0.5, 0.5, "Sin datos de atención", ha='center', va='center')
        else:
            sizes = [attentive, not_attentive]
            labels = ["Atento", "No Atento"]
            colors = ["cadetblue", "lightcoral"]

            ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={'width':0.4}  # donut
            )

        ax.set_title("Atención Total", fontsize=12, fontweight='bold')
        fig.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()


    # ----------------------------------------------------------------
    # EXPORTACIÓN A PDF con ambas figuras + cabecera
    # ----------------------------------------------------------------

    def generate_pdf_report(self, output_filename="Reporte.pdf"):
        """
        Genera un PDF (una sola página) con:
        - cabecera (fecha, # personas detectadas)
        - ambos gráficos (emociones y atención) en la misma página
        Devuelve la ruta completa del PDF generado.
        """
        # --- 1) calcular métricas a partir de self.data y self.attention ---
        # IDs únicos detectados
        person_ids = set(self.data.keys()) | set(self.attention.keys())

        # Conteo de emociones
        emotion_counts = {}
        for entries in self.data.values():
            for emotion, _ in entries:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Conteo de atención (total attentive / not attentive)
        attentive_count = 0
        not_attentive_count = 0
        for entries in self.attention.values():
            for att_flag, _ in entries:
                if att_flag:
                    attentive_count += 1
                else:
                    not_attentive_count += 1

        # --- 2) crear figura con ambos gráficos en una sola figura  ---
        fig, axes = plt.subplots(2,1, figsize=(8,12), facecolor="#ffffff")

        # ---------------- Emociones (barras) ----------------
        ax0 = axes[0]
        if emotion_counts:
            raw_labels = list(emotion_counts.keys())  # etiquetas internas (inglés)
            labels = [self.emotion_map_es.get(em, em) for em in raw_labels]  # etiquetas español
            values = [emotion_counts[l] for l in raw_labels]

            colors_map = {
                "happy": "goldenrod", "sad": "cornflowerblue", "anger": "orangered",
                "neutral": "slategray", "surprise": "mediumpurple",
                "disgust": "lightseagreen", "fear": "crimson"
            }
            bar_colors = [colors_map.get(l, "gray") for l in raw_labels]

            ax0.bar(labels, values, color=bar_colors)
            ax0.set_title("Distribución de Emociones", fontsize=14, fontweight='bold', color='#333333')
            ax0.set_ylabel("Frecuencia de cuadros", color='#333333')
            ax0.tick_params(axis='x', colors='dimgray')
            ax0.tick_params(axis='y', colors='dimgray')
            ax0.set_facecolor("#ffffff")
        else:
            ax0.text(0.5, 0.5, "No hay datos de emociones", ha='center', va='center', color='#333333')
            ax0.set_axis_off()


        # ---------------- ATENCIÓN ----------------
        ax1 = axes[1]
        total_att = attentive_count + not_attentive_count
        if total_att > 0:
            sizes = [attentive_count, not_attentive_count]
            labels_att = ["Atento", "No Atento"]
            colors_att = ["cadetblue", "lightcoral"]

            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=labels_att,
                colors=colors_att,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={'width': 0.4, 'edgecolor': 'white'}
            )

            for text in texts:
                text.set_color("dimgray")
                text.set_fontsize(12)
            for aut in autotexts:
                aut.set_color("dimgray")
                aut.set_fontsize(12)

            ax1.set_title("Atención Total", fontsize=14, fontweight='bold', color='#333333')
        else:
            ax1.text(0.5, 0.5, "No hay datos de atención", ha='center', va='center', color='#333333')
            ax1.set_axis_off()
        fig.tight_layout()

        # --- 3) guardar figura como imagen temporal dentro de save_folder ---
        tmp_img_name = "tmp_report_graphs.png"
        tmp_img_path = os.path.join(self.save_folder, tmp_img_name)
        fig.savefig(tmp_img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # --- 4) construir PDF con reportlab (una sola página) ---
        output_path = os.path.join(self.save_folder, output_filename)
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Cabecera: título + fecha + # personas
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        story.append(Paragraph("<b>REPORTE DE ANÁLISIS EMOCIONAL</b>", styles["Title"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Fecha del informe: {fecha}", styles["Normal"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Personas detectadas (IDs únicos): {len(person_ids)}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Insertar la imagen con los gráficos (ocupando ancho razonable)
        # ajustar tamaño en pts 
        story.append(Image(tmp_img_path, width=400, height=450))
        story.append(Spacer(1, 12))

        # Agregar resumen numérico opcional: lista de emociones y sus totales
        if emotion_counts:
            total = sum(emotion_counts.values())
            ordered = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            text_lines = "Emociones detectadas (ranking):<br/>" + "<br/>".join([
                f"{i+1}. {self.emotion_map_es.get(k, k)} – {v/total*100:.1f}%"
                for i, (k, v) in enumerate(ordered)
            ])
            story.append(Paragraph(text_lines, styles["Normal"]))
            story.append(Spacer(1, 8))

        # Construir PDF
        doc.build(story)

        # --- 5) eliminar imagen temporal ---
        try:
            os.remove(tmp_img_path)
        except Exception:
            pass

        print(f"[OK] PDF generado en: {output_path}")
        return output_path




    

