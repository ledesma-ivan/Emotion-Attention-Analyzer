# tracker.py
import numpy as np

# Fix para numpy >= 2.0
if not hasattr(np, "float"):
    np.float = float

from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self):
        # Ajustes para bajar ID Switches según versión disponible
        self.tracker = DeepSort(
            max_age=50,                 # tolera oclusiones cortas
            n_init=3,                   # exige 3 detecciones antes de confirmar
            max_cosine_distance=0.3,    # más estricto para evitar duplicados
            nms_max_overlap=0.5         # evita dos cajas muy solapadas
        )

    def update(self, detections, frame=None):
        ds_dets = []
        for det in detections:
            box, conf, cls = det
            x1, y1, x2, y2 = [int(v) for v in box]
            w = x2 - x1
            h = y2 - y1
            ltwh = [x1, y1, w, h]
            ds_dets.append((ltwh, float(conf), cls))

        tracks = self.tracker.update_tracks(ds_dets, frame=frame)

        tracked = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x1, y1, x2, y2 = tr.to_ltrb()
            tracked.append({
                "track_id": tr.track_id,
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            })
        return tracked
