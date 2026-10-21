# Contrato ML ↔ Android — v1

Este documento fija **exactamente** cómo se pasa de un fotograma con las 4 esquinas del tablero a la salida del clasificador. La implementación de referencia está en `ml/chessscan/geometry.py` y los vectores dorados en `ml/golden/`. Si algo de acá contradice al código de referencia, es un bug: avisar.

Todo está pensado para portarse a Kotlin con OpenCV usando solo `getPerspectiveTransform`, `warpPerspective`, `submat`, `flip` y `copyTo`.

## 1. Sistema de coordenadas

- Coordenadas de imagen: las de OpenCV. `x` hacia la derecha, `y` hacia abajo, en píxeles; el centro del píxel `(i, j)` está en `(i, j)` (enteros = centros).
- **Retícula del tablero:** 8×8 casillas, `fila` 0–7 y `col` 0–7.
  - **fila 0 = la fila más lejana a la cámara** (arriba en la imagen derecha), fila 7 = la más cercana.
  - **col 0 = la columna de la izquierda de la imagen**, col 7 = la de la derecha.
- **Esquinas:** las 4 esquinas **exteriores del área de juego** (el borde de las 64 casillas, no el marco de madera), en float, en este orden:
  1. `TL` — esquina lejana izquierda (fila 0, col 0)
  2. `TR` — esquina lejana derecha (fila 0, col 7)
  3. `BR` — esquina cercana derecha (fila 7, col 7)
  4. `BL` — esquina cercana izquierda (fila 7, col 0)
- **El borde TL→TR tiene que ser el más lejano a la cámara.** Es lo que hace que las piezas "se inclinen hacia arriba" en la imagen rectificada, y el recorte depende de eso. Recomendación en Android: rotar el fotograma según `imageInfo.rotationDegrees` para que quede derecho y ordenar las esquinas por posición en esa imagen derecha. Si el teléfono se sostiene de costado o cabeza abajo respecto del tablero, el orden se elige igual: el borde de arriba de la imagen derecha es TL→TR.
- La orientación de ajedrez (qué esquina es a1) **no** forma parte de este contrato: se resuelve después del clasificador (SPEC, «Orientación»).
- **Índice de casilla en el lote:** `i = fila * 8 + col` (0 = TL, 7 = fila 0 col 7, 63 = BR).

## 2. Rectificación (imagen canónica)

Constantes (enteros):

| Nombre | Valor | Significado |
|---|---|---|
| `S` | 32 | lado de una casilla en la imagen canónica, en px |
| `MX` | 32 (= 1·S) | margen izquierdo y derecho |
| `MY` | 96 (= 3·S) | margen superior (espacio para piezas altas de la fila 0) |
| `CW` | 320 (= 8·S + 2·MX) | ancho de la imagen canónica |
| `CH` | 352 (= 8·S + MY) | alto de la imagen canónica (sin margen inferior) |

Puntos destino de las esquinas (float):

| Esquina | destino (x, y) |
|---|---|
| TL | (32, 96) |
| TR | (288, 96) |
| BR | (288, 352) |
| BL | (32, 352) |

Es decir: el punto del tablero en `(u, v)` casillas (u = columnas desde la izquierda, v = filas desde la fila 0, ambos en [0, 8]) cae en el píxel canónico `(MX + S·u, MY + S·v)`.

Operaciones:

```
H = getPerspectiveTransform(src = [TL, TR, BR, BL], dst = [(32,96), (288,96), (288,352), (32,352)])   // 3×3, CV_64F, imagen → canónica
canon = warpPerspective(img, H, dsize = (CW=320, CH=352),
                        flags = INTER_LINEAR,            // sin WARP_INVERSE_MAP
                        borderMode = BORDER_CONSTANT, borderValue = (0,0,0))
```

- `img` es **RGB uint8** de 3 canales (no BGR, no RGBA). En Android, convertir el YUV de CameraX a RGB antes del warp, o warpear en el espacio que sea y convertir a RGB antes del recorte; lo que importa es que `canon` sea RGB uint8.
- Si Android obtiene la homografía de otra forma (por ejemplo RANSAC sobre los 81 cruces de la retícula), vale igual siempre que respete el mapeo `(u, v) → (MX + S·u, MY + S·v)`. Para comparar contra los vectores dorados hay que usar `getPerspectiveTransform` con las 4 esquinas dadas.
- Las zonas fuera del fotograma quedan negras (0). Es lo esperado: el modelo se entrena así.

## 3. Recorte de cada casilla

El recorte es **separable**: los límites verticales dependen solo de la fila y los horizontales solo de la columna. Todos son enteros; no hay redondeos que portar.

### Vertical (según la fila)

`arriba(fila) = S + (2·S·(7 − fila)) div 7` (división entera) — cuánto se extiende hacia arriba desde el borde superior de la casilla. Las filas lejanas tienen más espacio porque ahí las piezas se proyectan más largas.

`y0 = MY + S·fila − arriba(fila)`, `y1 = MY + S·(fila + 1)` (exclusivo), `h = y1 − y0`.

| fila | arriba | y0 | y1 | h |
|---|---|---|---|---|
| 0 | 96 | 0 | 128 | 128 |
| 1 | 86 | 42 | 160 | 118 |
| 2 | 77 | 83 | 192 | 109 |
| 3 | 68 | 124 | 224 | 100 |
| 4 | 59 | 165 | 256 | 91 |
| 5 | 50 | 206 | 288 | 82 |
| 6 | 41 | 247 | 320 | 73 |
| 7 | 32 | 288 | 352 | 64 |

### Horizontal (según la columna)

`d = 3 − col` si `col < 4`, si no `d = col − 4` (distancia al centro, 0–3). `extra(col) = (S/4)·(d + 1)` = 8, 16, 24 o 32 px, hacia **afuera** del tablero.

- `col < 4` (mitad izquierda): `x0 = MX + S·col − extra`, `x1 = MX + S·(col + 1)`.
- `col ≥ 4` (mitad derecha): `x0 = MX + S·col`, `x1 = MX + S·(col + 1) + extra`.
- `w = x1 − x0` (x1 exclusivo).

| col | extra | x0 | x1 | w | espejar |
|---|---|---|---|---|---|
| 0 | 32 | 0 | 64 | 64 | sí |
| 1 | 24 | 40 | 96 | 56 | sí |
| 2 | 16 | 80 | 128 | 48 | sí |
| 3 | 8 | 120 | 160 | 40 | sí |
| 4 | 8 | 160 | 200 | 40 | no |
| 5 | 16 | 192 | 240 | 48 | no |
| 6 | 24 | 224 | 280 | 56 | no |
| 7 | 32 | 256 | 320 | 64 | no |

### Armado del recorte

```
parche = canon[y0:y1, x0:x1]                 // h × w × 3
si col < 4: parche = flip(parche, 1)         // espejo horizontal (flipCode = 1)
recorte = ceros(alto = 128, ancho = 64, 3 canales, uint8)
recorte[128 − h : 128, 0 : w] = parche       // pegado abajo a la izquierda
```

Resultado: recorte RGB uint8 de **128 de alto × 64 de ancho**. La casilla propiamente dicha siempre ocupa los 32×32 px de abajo a la izquierda (`recorte[96:128, 0:32]`), y las piezas siempre se inclinan hacia arriba y a la derecha. El resto es contexto o relleno negro.

## 4. Entrada del modelo

| | |
|---|---|
| Archivo | `ml/export/chess_square_classifier.tflite` (+ `labels.txt`, `SHA256SUMS`) |
| Tensor de entrada | 1, nombre irrelevante (usar el índice 0) |
| Forma | `[64, 128, 64, 3]` = `[lote, alto, ancho, canales]` (**NHWC**) |
| dtype | **float32** |
| Valores | los píxeles RGB del recorte tal cual, **0.0–255.0**, sin normalizar. La normalización está dentro del modelo. |
| Orden del lote | `i = fila * 8 + col` |

El lote es fijo de 64. Si alguna vez hace falta otro tamaño, `Interpreter.resizeInput(0, [n, 128, 64, 3])` funciona, pero lo único verificado es 64.

La cuantización (int8 o fp16 internos) es un detalle interno del `.tflite`: **la entrada y la salida son siempre float32**, así que Android no tiene que manejar escalas ni puntos cero. Qué cuantización tiene el archivo entregado está en `ml/RESULTS.md`.

## 5. Salida del modelo

| | |
|---|---|
| Tensor de salida | 1 (índice 0) |
| Forma | `[64, 13]` |
| dtype | float32 |
| Contenido | **logits** (sin softmax), ya calibrados por temperatura. Android aplica `log_softmax` para la fusión y `softmax` para la confianza. |

### Orden de las 13 clases (igual a `labels.txt`)

| índice | etiqueta | pieza |
|---|---|---|
| 0 | `empty` | casilla vacía |
| 1 | `P` | peón blanco |
| 2 | `N` | caballo blanco |
| 3 | `B` | alfil blanco |
| 4 | `R` | torre blanca |
| 5 | `Q` | dama blanca |
| 6 | `K` | rey blanco |
| 7 | `p` | peón negro |
| 8 | `n` | caballo negro |
| 9 | `b` | alfil negro |
| 10 | `r` | torre negra |
| 11 | `q` | dama negra |
| 12 | `k` | rey negro |

Las etiquetas de pieza son los caracteres FEN, así que se pueden usar directamente al armar el FEN (una vez resuelta la orientación).

## 6. Vectores dorados (`ml/golden/`)

Cada caso `ml/golden/<caso>/` trae:

| archivo | contenido |
|---|---|
| `image.png` | fotograma de entrada (PNG RGB) |
| `case.json` | esquinas `[[x,y]×4]` en orden TL, TR, BR, BL; la homografía `H` (3×3, fila mayor); constantes; FEN de la imagen si se conoce; top‑1 esperado por casilla |
| `canonical.png` | imagen canónica esperada (320×352) |
| `crops/r{fila}c{col}.png` | los 64 recortes esperados (64×128) |
| `input_f32.bin` | tensor de entrada `[64,128,64,3]` float32 little‑endian, sin cabecera |
| `output_logits_f32.bin` | salida esperada del `.tflite` entregado, `[64,13]` float32 little‑endian |

Tolerancias para el port:

- `H`: diferencia relativa < 1e‑9 por elemento.
- `canonical.png` y recortes: OpenCV usa aritmética de punto fijo para `INTER_LINEAR` en uint8, así que con la misma versión debería dar idéntico. Se acepta **diferencia máxima ≤ 2 niveles y media < 0.05** (por posibles diferencias de versión o SIMD). La referencia usa `opencv-python-headless==4.12.0.88`.
- Salida del modelo con la entrada dorada exacta: mismo argmax en las 64 casillas y diferencia máxima de logits ≤ 0.1 (los kernels XNNPACK del teléfono no son bit‑idénticos a los de escritorio).

## 7. Esbozo en Kotlin

```kotlin
val S = 32; val MX = 32; val MY = 96; val CW = 320; val CH = 352
val dst = MatOfPoint2f(Point(32.0, 96.0), Point(288.0, 96.0), Point(288.0, 352.0), Point(32.0, 352.0))
val H = Imgproc.getPerspectiveTransform(MatOfPoint2f(tl, tr, br, bl), dst)
val canon = Mat()
Imgproc.warpPerspective(rgb, canon, H, Size(CW.toDouble(), CH.toDouble()),
    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))

fun up(row: Int) = S + (2 * S * (7 - row)) / 7
for (row in 0 until 8) for (col in 0 until 8) {
    val y0 = MY + S * row - up(row); val y1 = MY + S * (row + 1)
    val d = if (col < 4) 3 - col else col - 4
    val extra = (S / 4) * (d + 1)
    val x0 = if (col < 4) MX + S * col - extra else MX + S * col
    val x1 = if (col < 4) MX + S * (col + 1) else MX + S * (col + 1) + extra
    var patch = canon.submat(y0, y1, x0, x1)
    if (col < 4) { val f = Mat(); Core.flip(patch, f, 1); patch = f }
    val crop = Mat.zeros(128, 64, CvType.CV_8UC3)
    patch.copyTo(crop.submat(128 - (y1 - y0), 128, 0, x1 - x0))
    // copiar crop a la entrada float32 en la posición (row*8+col), canal por canal RGB, valor = byte & 0xFF
}
```

## Historial

- **v1** (2026‑10‑21): versión inicial.
