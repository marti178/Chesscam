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

## 1b. Otras fuentes del modelo v2

Licencias de cada una en `LICENSES.md`; el modelo v2 queda CC BY‑NC‑SA 4.0.

**Sintético 2D (tableros en pantalla fotografiados):**

```bash
# sets de piezas y texturas de tablero de lichess-org/lila (lista exacta en LICENSES.md)
mkdir -p data/raw/pieces data/raw/boards
for s in cburnett merida mpchess fantasy spatial celtic rhosgfx california caliente maestro fresca cardinal icpieces gioco tatiana dubrovny anarcandy cooke minimal-warmth totoy papercut horsey chessnut staunty; do
  mkdir -p data/raw/pieces/$s
  for p in wK wQ wR wB wN wP bK bQ bR bB bN bP; do
    curl -sf -o data/raw/pieces/$s/$p.svg https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/$s/$p.svg
  done
done
for f in blue-marble.jpg blue.png blue2.jpg blue3.jpg brown.png canvas2.jpg green-plastic.png green.png grey.jpg horsey.jpg ic.png leather.jpg maple.jpg maple2.jpg marble.jpg metal.jpg olive.jpg pink-pyramid.png purple-diag.png purple.png wood.jpg wood2.jpg wood3.jpg wood4.jpg; do
  curl -sf -o data/raw/boards/$f https://raw.githubusercontent.com/lichess-org/lila/master/public/images/board/$f
done
python -m chessscan.pieces2d --svg data/raw/pieces --out data/pieces_png
python -m chessscan.synth2d --out data/prepared2d_tex   --seed 8 --n 2000 200 300 300
python -m chessscan.synth2d --out data/prepared2d_tex_b --seed 9 --n 2000 0 0 0 --splits train
python -m chessscan.synth2d --out data/prepared2d --seed 7 --n 4000 200 300 300
```

`data/prepared2d` del v2 se generó con la primera versión del generador (sin texturas, selección ni premove): para reproducirlo exacto, correr ese último comando con `chessscan/synth2d.py` del commit `3876848`.

`chessnut` y `staunty` quedan reservados para `test_heldout` (sets nunca vistos).

**ChessReD (real, CC BY‑NC‑SA):** `annotations.json` (MD5 `d34bca5ad46ec7a8df96a1d3c36784f3`) de 4TU (o de su copia en el Wayback Machine) y las imágenes preprocesadas de 1024 px del repo de los autores (Google Drive id `1jxmFxjOy0qefdCZ_x3DMNtsvAK4LojEw`, 2 521 777 844 bytes). Se usa el subconjunto `chessred2k`, que es el que trae esquinas:

```bash
python -m chessscan.prepare_chessred --ann annotations.json --images data/raw/chessred --out data/prepared_chessred
```

**samryan18/chess-dataset (real, MIT):**

```bash
git clone --depth 1 https://github.com/samryan18/chess-dataset.git data/raw/samryan18
python -m chessscan.prepare_samryan --raw data/raw/samryan18 --out data/prepared_samryan
```

## 2. Datos reales propios (autoetiquetado)

1. Armá una posición conocida en tu tablero (la inicial sirve; mejor varias posiciones con todas las piezas en distintas casillas, por ejemplo cargadas desde un FEN).
2. Filmá 20–40 s moviendo el teléfono despacio alrededor del tablero, desde distintos ángulos y alturas, con distintas luces. Que se vean siempre las 4 esquinas del área de juego. Se puede dar la vuelta al tablero: la herramienta reordena las esquinas y rota las etiquetas en cada fotograma.
3. Corré:

```bash
python -m chessscan.autolabel mi_video.mp4 --out data/real/tablero1_inicial \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --a1 BL
```

Se abre el primer fotograma: hacé clic en las 4 esquinas del área de juego en orden lejana‑izquierda, lejana‑derecha, cercana‑derecha, cercana‑izquierda. `--a1` dice dónde quedó a1 en el **primer fotograma**: `BL` si las blancas están del lado de la cámara, `TR` si están enfrente, `TL`/`BR` si están a la izquierda/derecha. El resto de los fotogramas se sigue solo (SIFT + homografía); se descartan los movidos o con el tablero fuera de cuadro. **Mirá `preview.jpg`**: la retícula verde tiene que calzar con las casillas y las letras tienen que estar sobre las piezas correctas. Si no calza, descartá esa carpeta.

**Videos donde el tablero no entra entero en el primer cuadro** (lo normal filmando de cerca): `chessscan.autolabel_video`. Se marcan las esquinas en un cuadro cualquiera (pueden quedar afuera de la imagen), se sigue el tablero hacia adelante y hacia atrás, y con `--min-squares 40` se usan solo las casillas que se ven enteras. Así se generó el v3:

```bash
python -m chessscan.autolabel_video VID_20261022_104414.mp4 --cache data/raw/user_video/frames_720.npy     --anchor 1100 --anchor-corners "86.9,326.1 655.3,249.7 933.3,688.8 173.4,848.0" --a1 BL     --fen "rr4k1/R1Q2ppp/4pq2/8/7n/3P4/2P2NPP/5R1K" --out data/real/video1 --min-squares 40
# separación temporal: train 0–679, test 740–1135 (ver RESULTS.md), y después:
python -m chessscan.eval_video --model export/chess_square_classifier.tflite --data data/real/video1 --frames 740 1135 --windows 1 30 60 0
```

Fotos sueltas con FEN conocido: anotá las esquinas en un JSON (ver `eval_annotations/`) y usá `python -m chessscan.prepare_real`. `chessscan.corners.refine` ajusta esquinas aproximadas a los bordes de las casillas.

## 3. Entrenar → evaluar → exportar

```bash
python -m chessscan.pipeline --run runs/v1 --data data/prepared --epochs 10 --steps 1000 --seed 1234
```

**Modelo v2 entregado** (ajuste fino desde v1; ~52 min en CPU):

```bash
python -m chessscan.pipeline --run runs/v2 --data data/prepared --init runs/v1_cpu/model.keras --epochs 10 --steps 1000 --lr 1e-3 --seed 1234 \
  --extra data/prepared2d/train:0.35 data/prepared2d_tex/train:0.35 data/prepared2d_tex_b/train:0.3 data/prepared_chessred/train:1.0 data/prepared_samryan/train:0.3 \
  --val-extra data/prepared2d_tex/val data/prepared_chessred/val \
  --eval-real data/prepared_real/test data/prepared_chessred/test data/prepared_samryan/test data/prepared2d_tex/test data/prepared2d_tex/test_heldout
```

**Modelo v3 entregado** (ajuste fino desde v2 con el video del usuario; ~15 min):

```bash
python -m chessscan.pipeline --run runs/v3 --data data/prepared --init runs/v2/model.keras --epochs 4 --steps 500 --lr 3e-4 --seed 1234   --extra data/real/video1_train:1.2 data/prepared2d_tex/train:0.3 data/prepared2d_tex_b/train:0.2 data/prepared2d/train:0.2 data/prepared_chessred/train:0.8 data/prepared_samryan/train:0.2   --val-extra data/prepared2d_tex/val data/prepared_chessred/val   --eval-real data/real/video1_test data/prepared_real/test data/prepared_chessred/test data/prepared_samryan/test data/prepared2d_tex/test data/prepared2d_tex/test_heldout
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
