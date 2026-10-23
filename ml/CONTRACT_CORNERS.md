# Contrato del detector de esquinas — v1

Complementa `CONTRACT.md` (que no cambia). El detector **solo reemplaza el paso «imagen → 4 esquinas»**; lo que entra al clasificador sigue siendo exactamente lo mismo. Implementación de referencia: `chessscan/corner_net.py` (`predict_full` y `decode`).

## Entrada

| | |
|---|---|
| Archivo | `ml/export/corner_detector.tflite` (fp16 internamente; E/S float32) |
| Tensor de entrada | índice 0, forma `[1, 256, 256, 3]` NHWC, float32, **RGB 0–255 sin normalizar** |
| Armado | *letterbox*: la imagen entera, de W×H, se escala por `s = 256 / max(W, H)` (INTER_AREA) y se centra; el resto queda negro. Matriz imagen → entrada: `[[s, 0, (256 − W·s)/2], [0, s, (256 − H·s)/2]]` |

## Salidas

Los nombres `heat` y `off` están en la firma `serve` (`getSignatureRunner`), pero los tensores se llaman `StatefulPartitionedCall:N` y **el índice 0 es `off`**. Con `Interpreter` común, identificarlas **por forma**: la de último eje 1 es `heat`, la de último eje 2 es `off`.

| Nombre | Forma | Contenido |
|---|---|---|
| `heat` | `[1, 64, 64, 1]` | **logit** de «en esta celda hay una esquina exterior del área de juego» (una sola clase para las 4 esquinas) |
| `off` | `[1, 64, 64, 2]` | desplazamiento (dx, dy) en celdas, en [0, 1), dentro de la celda |

Stride 4: la celda (x, y) del mapa corresponde al píxel `(x + dx, y + dy) × 4` de la entrada de 256.

## Decodificación (igual que `decode`)

1. `p = sigmoid(heat)`.
2. Picos: celdas donde `p` es máximo en su vecindad de 5×5 y `p > 0,15`.
3. Quedarse con los **4 de mayor `p`**. Si hay menos de 4, no hay tablero (o se ve incompleto): mostrar «Buscando tablero» o «Alejate, falta una esquina».
4. Para cada pico: `(x + dx, y + dy) × 4` → coordenadas de la entrada → aplicar la inversa del letterbox → píxeles de la imagen original.
5. Ordenar las 4 como en `CONTRACT.md` §1 (TL, TR, BR, BL de la imagen derecha) y seguir igual que hoy (homografía, recortes, clasificador).

## Segunda pasada con zoom (recomendada; `predict_zoom`)

6. Con las 4 esquinas de la 1.ª pasada: `lo`, `hi` = mínimo y máximo de sus x e y; `L = max(hi − lo)`; centro `c = (lo + hi)/2`; `s = 256 / (1,3·L)`.
7. Armar la entrada con la afín `[[s, 0, 128 − s·cx], [0, s, 128 − s·cy]]` sobre la **imagen original a resolución completa** (INTER_AREA, borde negro) y correr el modelo otra vez.
8. Decodificar igual (pasos 1–4, con la inversa de esta afín). Cada esquina de la 1.ª pasada se reemplaza por el pico de la 2.ª más cercano, si está a menos de `L/8` píxeles; si no, queda la de la 1.ª.

Cuesta una inferencia más (~2 ms en la PC) y, en los videos del usuario, sube las esquinas con error < 0,25 casillas de 17–33 % a 70 % (ver `RESULTS.md`). En video se puede hacer la 1.ª pasada cada tanto y la 2.ª cada cuadro, con el recorte del cuadro anterior.

Recomendado: tomar como confianza del tablero el **mínimo de los 4 `p`**, y refinar las esquinas con el detector OpenCV actual o con un ajuste de retícula cerca de ellas, si mejora en la medición.

## Qué no cambia

El orden de las esquinas, la canónica, los recortes y la entrada y salida del clasificador.
