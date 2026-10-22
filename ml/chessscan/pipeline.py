"""train → eval → export, reproducible.

Uso típico (desde ml/):
  python -m chessscan.pipeline --run runs/v1 --data data/prepared --epochs 12

Fuentes extra (por ejemplo, recortes reales del autoetiquetado) con peso:
  python -m chessscan.pipeline ... --extra data/real/miTablero:0.5

Salida en --run: model.keras, history.json, metrics.json, *.tflite, labels.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras

from . import geometry as g
from .data import Boards, TrainBatches, all_crops


# --- modelo ------------------------------------------------------------------

def build_model(arch: str = "mnv3s", dropout: float = 0.2) -> keras.Model:
    inp = keras.Input((g.CROP_H, g.CROP_W, 3), name="crops")  # RGB 0..255
    if arch == "mnv3s":
        base = keras.applications.MobileNetV3Small(
            input_shape=(g.CROP_H, g.CROP_W, 3), include_top=False, weights="imagenet",
            include_preprocessing=True, minimalistic=False)
    elif arch == "mnv3l":
        base = keras.applications.MobileNetV3Large(
            input_shape=(g.CROP_H, g.CROP_W, 3), include_top=False, weights="imagenet",
            include_preprocessing=True, minimalistic=False)
    elif arch == "mnv3s_min":
        base = keras.applications.MobileNetV3Small(
            input_shape=(g.CROP_H, g.CROP_W, 3), include_top=False, weights="imagenet",
            include_preprocessing=True, minimalistic=True)
    else:
        raise ValueError(arch)
    x = base(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(len(g.CLASSES), name="logits")(x)
    return keras.Model(inp, out, name=f"chess_square_{arch}")


class Seq(keras.utils.PyDataset):
    def __init__(self, tb: TrainBatches, workers: int):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=16)
        self.tb = tb

    def __len__(self):
        return len(self.tb)

    def __getitem__(self, k):
        return self.tb.batch(k)

    def on_epoch_end(self):
        self.tb.epoch += 1


# --- evaluación ----------------------------------------------------------------

def predict_logits(predict, x: np.ndarray, bs: int = 256) -> np.ndarray:
    return np.concatenate([np.asarray(predict(x[i:i + bs].astype(np.float32))) for i in range(0, len(x), bs)])


def metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """Métricas por casilla y por tablero (de a 64). Las etiquetas -1 (no visibles) se ignoran."""
    pred = logits.argmax(1)
    vis = y >= 0
    ok = (pred == y) & vis
    n_cls = len(g.CLASSES)
    conf = np.zeros((n_cls, n_cls), int)
    np.add.at(conf, (y[vis], pred[vis]), 1)
    err_b = (vis & ~ok).reshape(-1, 64).sum(1)
    yv, okv = y[vis], ok[vis]
    return {
        "squares": int(vis.sum()),
        "square_acc": float(okv.mean()),
        "occupied_acc": float(okv[yv != 0].mean()) if (yv != 0).any() else None,
        "empty_acc": float(okv[yv == 0].mean()) if (yv == 0).any() else None,
        "boards": int(len(err_b)),
        "board_acc_all64": float((err_b == 0).mean()),  # con máscara: todas las casillas visibles bien
        "mean_errors_per_board": float(err_b.mean()),
        "per_class_recall": {c: (float(conf[i, i] / conf[i].sum()) if conf[i].sum() else None)
                             for i, c in enumerate(g.CLASSES)},
        "confusion": conf.tolist(),
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    logits, y = logits[y >= 0], y[y >= 0]
    best = (1e9, 1.0)
    for t in np.exp(np.linspace(np.log(0.3), np.log(5), 200)):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
        best = min(best, (nll, float(t)))
    return best[1]


# --- export ----------------------------------------------------------------------

def export_tflite(model: keras.Model, out: Path, rep: np.ndarray, mode: str) -> Path:
    # SavedModel con lote fijo de 64; from_concrete_functions deja los pesos como
    # variables (READ_VARIABLE) con Keras 3, así que se pasa por un SavedModel.
    sm = out / "saved_model"
    if not sm.exists():
        arch = keras.export.ExportArchive()
        arch.track(model)
        arch.add_endpoint(
            name="serve", fn=lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec([64, g.CROP_H, g.CROP_W, 3], tf.float32, name="crops")])
        arch.write_out(str(sm))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm), signature_keys=["serve"])
    if mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_data():
            for i in range(0, len(rep) - 63, 64):
                yield [rep[i:i + 64].astype(np.float32)]

        conv.representative_dataset = rep_data
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.float32
        conv.inference_output_type = tf.float32
    elif mode == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif mode != "fp32":
        raise ValueError(mode)
    path = out / f"chess_square_classifier_{mode}.tflite"
    path.write_bytes(conv.convert())
    return path


def tflite_predict(path: Path, x: np.ndarray, threads: int = 8) -> np.ndarray:
    it = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    outs = []
    for k in range(0, len(x), 64):
        chunk = x[k:k + 64].astype(np.float32)
        pad = 64 - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.zeros((pad, *chunk.shape[1:]), np.float32)])
        it.set_tensor(i_in, chunk)
        it.invoke()
        outs.append(it.get_tensor(i_out)[: 64 - pad].copy())
    return np.concatenate(outs)


# --- main --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--extra", nargs="*", default=[], help="carpeta:peso de datos preparados extra (train)")
    ap.add_argument("--arch", default="mnv3s")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1500, help="lotes por época")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit-train", type=int, default=0, help="usar solo N tableros (pruebas)")
    ap.add_argument("--eval-only", action="store_true", help="saltear el entrenamiento y usar run/model.keras")
    ap.add_argument("--init", type=Path, help="ajuste fino: partir de este model.keras en vez de ImageNet")
    ap.add_argument("--eval-real", type=Path, nargs="*", default=[], help="otros sets preparados para evaluar (reales, 2D…)")
    ap.add_argument("--val-extra", type=Path, nargs="*", default=[], help="sets extra que se suman a la validación")
    a = ap.parse_args()

    a.run.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(a.seed)
    (a.run / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))

    train = Boards.load(a.data / "train")
    if a.limit_train:
        train = Boards(train.canon[: a.limit_train], train.labels[: a.limit_train])
    val, test = Boards.load(a.data / "val"), Boards.load(a.data / "test")
    sources = [(train, 1.0)]
    for spec in a.extra:
        d, w = spec.rsplit(":", 1)
        sources.append((Boards.load(Path(d)), float(w)))

    xv, yv = all_crops(val)
    for d in a.val_extra:
        xe, ye = all_crops(Boards.load(d))
        xv, yv = np.concatenate([xv, xe]), np.concatenate([yv, ye])
    keep = yv >= 0  # casillas no visibles (-1) fuera de la validación
    xv, yv = xv[keep], yv[keep]

    if not a.eval_only:
        model = keras.models.load_model(a.init) if a.init else build_model(a.arch)
        sched = keras.optimizers.schedules.CosineDecay(
            1e-5, a.epochs * a.steps, warmup_target=a.lr, warmup_steps=min(500, a.steps))
        model.compile(
            optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        tb = TrainBatches(sources, a.batch, a.steps, a.seed)
        t0 = time.time()
        hist = model.fit(
            Seq(tb, a.workers), epochs=a.epochs, validation_data=(xv.astype(np.float32), yv),
            validation_batch_size=256,
            callbacks=[keras.callbacks.BackupAndRestore(str(a.run / "backup")),  # reanuda si se corta
                       keras.callbacks.ModelCheckpoint(a.run / "model.keras", monitor="val_accuracy",
                                                       save_best_only=True),
                       keras.callbacks.CSVLogger(a.run / "history.csv", append=True)],
            verbose=2)
        (a.run / "train_time_s.txt").write_text(f"{time.time() - t0:.0f}\n")
    model = keras.models.load_model(a.run / "model.keras")

    # Calibración por temperatura en val, plegada en la capa final.
    predict = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    pv = predict_logits(predict, xv)
    T = fit_temperature(pv, yv)
    dense = model.get_layer("logits")
    W, b = dense.get_weights()
    dense.set_weights([W / T, b / T])
    model.save(a.run / "model_calibrated.keras")
    import shutil
    shutil.rmtree(a.run / "saved_model", ignore_errors=True)

    xt, yt = all_crops(test)
    res = {"temperature": T, "keras": {"val": metrics(predict_logits(predict, xv), yv),
                                       "test": metrics(predict_logits(predict, xt), yt)}}
    print("keras test square_acc", res["keras"]["test"]["square_acc"])
    real = {str(d): all_crops(Boards.load(d)) for d in a.eval_real}
    for d, (xr, yr) in real.items():
        res["keras"][d] = metrics(predict_logits(predict, xr), yr)
        print("keras", d, res["keras"][d]["square_acc"])

    for mode in ["fp32", "fp16", "int8"]:
        p = export_tflite(model, a.run, xv[: 64 * 40], mode)
        lt = tflite_predict(p, xt)
        res[f"tflite_{mode}"] = {"bytes": p.stat().st_size, "test": metrics(lt, yt)}
        for d, (xr, yr) in real.items():
            res[f"tflite_{mode}"][d] = metrics(tflite_predict(p, xr), yr)
        print(mode, p.stat().st_size, res[f"tflite_{mode}"]["test"]["square_acc"])
    (a.run / "labels.txt").write_text("\n".join(g.CLASSES) + "\n")
    (a.run / "metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
