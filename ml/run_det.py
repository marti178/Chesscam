"""Detector de piezas det_v1 (YOLO11n de Ultralytics, AGPL-3.0) en CPU, reanudable.

  .venv-yolo\\Scripts\\python.exe run_det.py
Si se corta, se relanza igual: sigue desde runs/det_v1/weights/last.pt.
"""
from pathlib import Path

from ultralytics import YOLO

RUN = Path("runs/det_v1")
LAST = RUN / "weights" / "last.pt"

if __name__ == "__main__":
    if LAST.exists():
        YOLO(str(LAST)).train(resume=True)
    else:
        YOLO("yolo11n.pt").train(
            data=str(Path("data/det/data.yaml").resolve()), epochs=14, imgsz=512, batch=16, workers=6, device="cpu",
            seed=1234, project=str(RUN.parent.resolve()), name=RUN.name, exist_ok=True, plots=False,
            fliplr=0.0,  # espejar cambia la retícula pero no la clase; se deja en 0 por las piezas asimétricas (caballo)
            hsv_h=0.03, hsv_s=0.6, hsv_v=0.5, degrees=5, mosaic=0.5, close_mosaic=3, patience=50)
    (RUN / "DONE").write_text("ok\n")
