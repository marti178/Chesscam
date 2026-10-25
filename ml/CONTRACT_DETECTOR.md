# Contrato del detector de piezas — v1

Alternativa al clasificador de casillas (`CONTRACT.md`): detecta las piezas sobre la foto **sin rectificar** y después las asigna a casillas con la homografía. Usa las mismas 4 esquinas (del detector de esquinas o del OpenCV de la app) en el orden de `CONTRACT.md` §1. Implementación de referencia: `chessscan/det_eval.py` (`prepare`, `decode`, `assign`); caso dorado en `golden/detector_full_1200x800/`.

## Modelo

| | |
|---|---|
| Archivo | `ml/export/piece_detector.tflite`: YOLO11n de Ultralytics, fp16 internamente, E/S float32, 5,4 MB |
| Entrada | índice 0, `[1, 512, 512, 3]` NHWC float32, **RGB / 255** (0–1) |
| Salida | índice 0, `[1, 16, 5376]` float32, sin NMS: filas 0–3 = `cx, cy, w, h` **normalizados a la entrada** (×512 → píxeles), filas 4–15 = puntaje de cada clase (ya en 0–1) |
| Clases (filas 4–15) | `P N B R Q K p n b r q k`, es decir el orden de `labels.txt` sin `empty` (clase de la casilla = fila − 3) |

## Armado de la entrada (`prepare`)

1. Con las 4 esquinas de la imagen original (W×H): `lo`/`hi` = mínimo y máximo de x e y; `bw, bh = hi − lo`.
2. Recorte: `x0 = lo.x − 0,08·bw`, `x1 = hi.x + 0,08·bw`, `y0 = lo.y − 0,35·bh` (margen para las piezas altas de la fila de atrás), `y1 = hi.y + 0,05·bh`; recortar a la imagen y truncar a entero.
3. Si el lado mayor del recorte supera 640, reducirlo a 640 (factor `s`, INTER_AREA); si no, `s = 1`.
4. *Letterbox* a 512×512: `r = min(512/h, 512/w)`, redimensionar a `round(w·r) × round(h·r)` (INTER_LINEAR), pegar en un lienzo gris (114, 114, 114) en `pad_x = round((512 − nw)/2 − 0,1)`, `pad_y = round((512 − nh)/2 − 0,1)`.

## Decodificación (`decode`)

1. Para cada una de las 5376 columnas: clase = argmax de las filas 4–15, `p` = su puntaje. Quedarse con `p ≥ 0,1`.
2. Caja en la entrada: `cx, cy, w, h` × 512 → `x0 = cx − w/2`, etc.
3. **NMS agnóstica a la clase**, IoU 0,6, hasta 300 cajas.
4. A píxeles originales: `x = (x_in − pad_x) / (r·s) + x0_recorte`, ídem `y`.

## Asignación a casillas (`assign`)

1. Punto de apoyo de cada caja: `((x0 + x1)/2, y1 − 0,35·min(w, h))`, abajo al centro y un poco hacia adentro. En vista lateral cae cerca de la base de la pieza y en vista cenital, cerca del centro.
2. Proyectarlo con la homografía de `CONTRACT.md` §2: `col = (u − 32)/32`, `fila = (v − 96)/32`. Si no cae en 0 ≤ col, fila < 8, se descarta (pieza capturada fuera del tablero).
3. Cada casilla se queda con la detección de **mayor `p`**. Las casillas sin detección son `empty`.
4. Umbral de decisión: con `t = 0,35`, la casilla es la pieza si `p ≥ t`; si no, `empty`. Para la fusión temporal, usar `q = sigmoid(logit(p) − logit(t))` para la clase detectada y `1 − q` repartido (95 % a `empty`). Sin detección: `empty` 0,95. Así se suman log‑probabilidades por casilla física, como con el clasificador.

El resultado es la misma grilla de 64 casillas en el orden de imagen (fila 0 = la lejana), así que la orientación (a1) y lo que sigue no cambian.

## Rendimiento medido

Latencia en esta PC: 25 ms con 4 hilos, 46 ms con 1. Una sola inferencia por cuadro, contra 64 recortes en el clasificador. **En el teléfono no está medido.** El int8 no se exportó. Números en `RESULTS.md`.
