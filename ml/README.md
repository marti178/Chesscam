# ml/ — clasificador de casillas

Todo lo del modelo que reconoce qué hay en cada casilla: contrato con Android, implementación de referencia, datos, entrenamiento, export a LiteRT y evaluación.

| Archivo | Qué es |
|---|---|
| `CONTRACT.md` | **Contrato con Android**: coordenadas, rectificación, recorte, entrada/salida del modelo, orden de clases |
| `chessscan/geometry.py` | Implementación de referencia del contrato |
| `golden/` | Vectores dorados para verificar el port a Kotlin |
| `export/` | Modelo entregado (`.tflite`), `labels.txt`, `SHA256SUMS` |
| `RESULTS.md` | Métricas medidas |
| `LICENSES.md` | Licencias de datos, pesos y dependencias |
| `notebooks/train_kaggle_colab.ipynb` | El mismo entrenamiento en Kaggle/Colab con GPU |

## Instalación (Windows, Python 3.11)

```bash
py -3.11 -m venv ml/.venv
ml/.venv/Scripts/python -m pip install -r ml/requirements.txt
```

`requirements.txt` fija las versiones directas; `requirements-lock.txt` es el `pip freeze` completo con el que se entrenó. Todos los comandos de abajo se corren **desde `ml/`** con ese intérprete (`.venv/Scripts/python`).

## 1. Datos sintéticos (chesscog)

```bash
mkdir -p data/raw && cd data/raw
curl -L -o val.zip   https://osf.io/download/ydtcs/
curl -L -o test.zip  https://osf.io/download/7p69s/
curl -L -o train.zip https://osf.io/download/z5vgm/
curl -L -o train.z01 https://osf.io/download/np56u/      # train viene partido en 2 volúmenes (~6,7 GB)
"/c/Program Files/7-Zip/7z.exe" x -oextracted train.zip   # 7-Zip une train.z01 + train.zip
unzip -q val.zip -d extracted && unzip -q test.zip -d extracted
cd ../..
python -m chessscan.prepare_chesscog --raw data/raw/extracted --out data/prepared
```

`prepare_chesscog` rectifica cada render a la imagen canónica del contrato y guarda `canon.npy` + `labels.npy` por split (4400 / 146 / 342 tableros). Los recortes se sacan en el entrenamiento, después de aumentar.

## 2. Datos reales propios (autoetiquetado)

1. Armá una posición conocida en tu tablero (la inicial sirve; mejor varias posiciones con todas las piezas en distintas casillas, por ejemplo cargadas desde un FEN).
2. Filmá 20–40 s moviendo el teléfono despacio alrededor del tablero, desde distintos ángulos y alturas, con distintas luces. Que se vean siempre las 4 esquinas del área de juego. Se puede dar la vuelta al tablero: la herramienta reordena las esquinas y rota las etiquetas en cada fotograma.
3. Corré:

```bash
python -m chessscan.autolabel mi_video.mp4 --out data/real/tablero1_inicial \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --a1 BL
```

Se abre el primer fotograma: hacé clic en las 4 esquinas del área de juego en orden lejana‑izquierda, lejana‑derecha, cercana‑derecha, cercana‑izquierda. `--a1` dice dónde quedó a1 en el **primer fotograma**: `BL` si las blancas están del lado de la cámara, `TR` si están enfrente, `TL`/`BR` si están a la izquierda/derecha. El resto de los fotogramas se sigue solo (SIFT + homografía); se descartan los movidos o con el tablero fuera de cuadro. **Mirá `preview.jpg`**: la retícula verde tiene que calzar con las casillas y las letras tienen que estar sobre las piezas correctas. Si no calza, descartá esa carpeta.

Fotos sueltas con FEN conocido: anotá las esquinas en un JSON (ver `eval_annotations/`) y usá `python -m chessscan.prepare_real`. `chessscan.corners.refine` ajusta esquinas aproximadas a los bordes de las casillas.

## 3. Entrenar → evaluar → exportar

```bash
python -m chessscan.pipeline --run runs/v1 --data data/prepared --epochs 10 --steps 1000 --seed 1234
```

Con datos reales propios (recomendado; mezcla 1 tablero real cada ~3 sintéticos):

```bash
python -m chessscan.pipeline --run runs/v2 --data data/prepared --extra data/real/tablero1_inicial:0.3 data/real/tablero1_pos2:0.3
```

Qué hace (`chessscan/pipeline.py`):

- **Modelo:** MobileNetV3‑Small (pesos ImageNet de Keras) + pooling global + dropout + capa densa de 13 logits. La normalización está dentro del modelo (entra RGB 0–255).
- **Lotes:** 8 tableros aumentados por lote → 64 recortes cada uno → se eligen 128 con probabilidad ∝ frecuencia_de_clase^‑0,5 (para que «vacía» no domine). Aumentación (`chessscan/augment.py`): espejo del tablero entero, error de esquinas de hasta ±4 px en la canónica, color/brillo/contraste/gamma/sombra, menos resolución, desenfoque o movimiento, ruido y JPEG.
- **Reproducibilidad:** `keras.utils.set_random_seed(seed)`; cada lote depende solo de `(seed, época, índice)`, así que no importa el orden de los hilos. Versiones fijadas en `requirements-lock.txt`. En CPU el resultado es repetible salvo diferencias de redondeo de oneDNN; en GPU puede variar levemente.
- **Optimización:** AdamW, warmup + coseno, se guarda el mejor por `val_accuracy`.
- **Calibración:** temperatura ajustada en val y plegada en la capa final (la salida ya son logits calibrados).
- **Export:** `chess_square_classifier_{fp32,fp16,int8}.tflite` con lote fijo 64 y E/S float32. int8 usa 40 lotes de val como dataset representativo.
- **Evaluación:** precisión por casilla, de vacías y ocupadas, por clase, tableros perfectos y matriz de confusión en `metrics.json`, para Keras y para cada `.tflite`.

En esta PC (i5‑13420H, sin GPU) un paso de 128 recortes tarda ~0,3 s; 10 épocas × 1000 pasos ≈ 1,5 h. Con GPU: `notebooks/train_kaggle_colab.ipynb`.

Para evaluar un `.tflite` cualquiera y medir latencia:

```bash
python -m chessscan.evaluate --model export/chess_square_classifier.tflite \
    --data data/prepared/test data/prepared_real/test --bench
```

## 4. Publicar un modelo nuevo

```bash
cp runs/vN/chess_square_classifier_int8.tflite export/chess_square_classifier.tflite   # o fp16, según RESULTS.md
cp runs/vN/labels.txt export/
cd export && sha256sum chess_square_classifier.tflite labels.txt > SHA256SUMS && cd ..
python -m chessscan.make_golden --render data/raw/extracted/val/0418.png --model export/chess_square_classifier.tflite --out golden
python -m pytest -q tests
```

Regenerar los vectores dorados cada vez que cambia el modelo (las salidas esperadas dependen de él) y actualizar `RESULTS.md`. Si cambia algo del recorte o de la entrada, es un **contrato nuevo**: subir la versión en `CONTRACT.md` y `geometry.CONTRACT_VERSION`, y avisar a Android.
