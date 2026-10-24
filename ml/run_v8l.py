"""v8l de punta a punta, en serie y reanudable: entrenamiento → export con embedding (v1.1) → test fijo.

  .venv\\Scripts\\python.exe run_v8l.py
Si se corta, se relanza igual: el entrenamiento sigue desde la última época (BackupAndRestore) y los
pasos que ya dejaron su archivo se saltean.
"""
import subprocess
import sys
from pathlib import Path

PY = sys.executable
RUN = Path("runs/v8l")
EXTRA = ["data/real/video1:0.5", "data/real/video3:0.8", "data/real/video6book_train:0.45",
         "data/preparedbook/train:0.5", "data/preparedbook_b/train:0.45",
         "data/prepared2d_tex/train:0.45", "data/prepared2d_tex_b/train:0.3", "data/prepared2d/train:0.25",
         "data/prepared_chessred/train:0.5", "data/prepared_chessred_auto/train:0.3", "data/prepared_samryan/train:0.3",
         "data/prepared_chesssight/train:0.8", "data/prepared_acapitani/train:0.8", "data/prepared_dopaul/train:0.4",
         "data/prepared_francesco/train:0.3"]
TRAIN = [PY, "-m", "chessscan.pipeline", "--run", str(RUN), "--arch", "mnv3l", "--data", "data/prepared",
         "--main-weight", "0.6", "--epochs", "24", "--steps", "1000", "--lr", "2e-3", "--seed", "1234", "--workers", "8",
         "--retex", "0.3", "--modes", "fp16", "--extra", *EXTRA,
         "--val-extra", "data/prepared2d_tex/val", "data/prepared_chessred/val", "data/preparedbook/val", "data/prepared_chesssight/val",
         "--eval-real", "data/prepared_acapitani/test", "data/prepared_dopaul/test", "data/prepared_francesco/test"]
M = RUN / "chess_square_classifier_fp16.tflite"
FIJO = ["data/real/video2", "data/prepared_book_real/test", "data/preparedbook/test", "data/preparedbook/test_heldout",
        "data/prepared_real/test", "data/prepared/test", "data/prepared_chessred/test", "data/prepared2d_tex/test_heldout"]
STEPS = [
    (RUN / "metrics.json", TRAIN),
    (RUN / "chess_square_classifier_large_emb.tflite",
     [PY, "-m", "chessscan.export_embed", "--model", str(RUN / "model_calibrated.keras"), "--out", str(RUN / "chess_square_classifier_large_emb.tflite")]),
    (RUN / "eval_fijo.json", [PY, "-m", "chessscan.evaluate", "--model", str(M), "--data", *FIJO, "--out", str(RUN / "eval_fijo.json")]),
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
