# Contrato del detector de esquinas — v1

Complementa `CONTRACT.md` (que no cambia). El detector **solo reemplaza el paso «imagen → 4 esquinas»**; lo que entra al clasificador sigue siendo exactamente lo mismo. Implementación de referencia: `chessscan/corner_net.py` (`predict_full` y `decode`).

## Entrada

| | |
|---|---|
| Archivo | `ml/export/corner_detector.tflite` (fp16 internamente; E/S float32) |
| Tensor de entrada | índice 0, forma `[1, 256, 256, 3]` NHWC, float32, **RGB 0–255 sin normalizar** |
| Armado | *letterbox*: la imagen entera, de W×H, se escala por `s = 256 / max(W, H)` (INTER_AREA) y se centra; el resto queda negro. Matriz imagen → entrada: `[[s, 0, (256 − W·s)/2], [0, s, (256 − H·s)/2]]` |

## Salidas (índices por nombre de tensor: `heat` y `off`)

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

Recomendado: tomar como confianza del tablero el **mínimo de los 4 `p`**, y refinar las esquinas con el detector OpenCV actual o con un ajuste de retícula cerca de ellas, si mejora en la medición.

## Qué no cambia

El orden de las esquinas, la canónica, los recortes y la entrada y salida del clasificador.
