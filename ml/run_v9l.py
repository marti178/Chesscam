"""v9l: ajuste fino del v8l con el tablero nuevo del usuario (nogal y arce), en serie y reanudable.

Train: los mismos datos del v8l + video7 (WhatsApp 21.54.59, autoetiquetado con `autolabel_detector`).
Test del tablero nuevo: video8 (21.55.05, misma posición desde otros ángulos; nunca se entrena con él).
  .venv\\Scripts\\python.exe run_v9l.py
"""
import subprocess
import sys
from pathlib import Path

from run_v8l import EXTRA, FIJO, PY

RUN = Path("runs/v9l")
TRAIN = [PY, "-m", "chessscan.pipeline", "--run", str(RUN), "--arch", "mnv3l", "--data", "data/prepared",
         "--init", "runs/v8l/model.keras", "--main-weight", "0.6", "--epochs", "4", "--steps", "1000", "--lr", "5e-4",
         "--seed", "1234", "--workers", "8", "--retex", "0.3", "--modes", "fp16", "--extra", *EXTRA, "data/real/video7:0.8",
         "--val-extra", "data/prepared2d_tex/val", "data/prepared_chessred/val", "data/preparedbook/val", "data/prepared_chesssight/val",
         "--eval-real", "data/prepared_acapitani/test", "data/prepared_dopaul/test", "data/prepared_francesco/test"]
M = RUN / "chess_square_classifier_fp16.tflite"
STEPS = [
    (RUN / "metrics.json", TRAIN),
    (RUN / "chess_square_classifier_large_emb.tflite",
     [PY, "-m", "chessscan.export_embed", "--model", str(RUN / "model_calibrated.keras"), "--out", str(RUN / "chess_square_classifier_large_emb.tflite")]),
    (RUN / "eval_video8.json", [PY, "-m", "chessscan.eval_video", "--model", str(M), "--data", "data/real/video8", "--out", str(RUN / "eval_video8.json")]),
    (RUN / "eval_video7.json", [PY, "-m", "chessscan.eval_video", "--model", str(M), "--data", "data/real/video7", "--out", str(RUN / "eval_video7.json")]),
    (RUN / "eval_fijo.json", [PY, "-m", "chessscan.evaluate", "--model", str(M), "--data", *FIJO, "--out", str(RUN / "eval_fijo.json")]),
    (RUN / "eval_nuevos.json", [PY, "-m", "chessscan.evaluate", "--model", str(M), "--data", "data/prepared_acapitani/test",
                                "data/prepared_dopaul/test", "data/prepared_francesco/test", "data/prepared_chesssight/val", "--out", str(RUN / "eval_nuevos.json")]),
    (RUN / "eval_video4.json", [PY, "-m", "chessscan.eval_video", "--model", str(M), "--data", "data/real/video4", "--out", str(RUN / "eval_video4.json")]),
    (RUN / "eval_video5.json", [PY, "-m", "chessscan.eval_video", "--model", str(M), "--data", "data/real/video5", "--out", str(RUN / "eval_video5.json")]),
    (RUN / "eval_video6book.json", [PY, "-m", "chessscan.eval_video", "--model", str(M), "--data", "data/real/video6book",
                                    "--frames", "1000", "1412", "--out", str(RUN / "eval_video6book.json")]),
]

if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    for out, cmd in STEPS:
        if out.exists():
            print("ya está", out, flush=True)
            continue
        print("corriendo", " ".join(cmd[2:5]), flush=True)
        subprocess.run(cmd, check=True)
    (RUN / "DONE").write_text("ok\n")
