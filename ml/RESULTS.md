# Resultados — clasificador de casillas

Fecha: 22‑10‑2026. Todo lo de esta página se midió en esta PC, salvo lo que se marca como medido por otro. Si algo no se midió, se dice.

## Modelo entregado: v2

| | |
|---|---|
| Archivo | `ml/export/chess_square_classifier.tflite` (SHA‑256 en `export/SHA256SUMS`), métricas completas en `export/metrics_v2.json` |
| Contrato | **v1, sin cambios**: entrada `[64,128,64,3]` float32 RGB 0–255, salida `[64,13]` logits calibrados (T = 1,182) |
| Red | MobileNetV3‑Small + GAP + dropout + densa de 13 (~0,95 M parámetros), **fp16**, 1 916 920 bytes |
| Licencia del modelo | **CC BY‑NC‑SA 4.0** (por ChessReD; uso no comercial, ver `LICENSES.md`) |
| Entrenamiento | ajuste fino desde v1, 10 épocas × 1000 lotes × 128 recortes, semilla 1234, 52 min en CPU (i5‑13420H). Comando en `README.md` |

Mezcla de entrenamiento (proporción de tableros por lote): chesscog sintético 3D (4400 tableros) ~30 %, sintético 2D en pantalla (8000 tableros, 22 sets de piezas y 24 texturas de Lichess) ~30 %, ChessReD real (1442 fotos) ~30 %, samryan18 real (394 fotos) ~9 %.

## Precisión por casilla, un solo fotograma (sin fusión)

Precisión por casilla = casillas bien clasificadas / casillas totales (64 por tablero). Todos los tests están separados del entrenamiento: chesscog por split oficial, ChessReD por partida (split oficial de chessred2k), samryan18 por posición, y el sintético 2D por semilla, con dos sets de piezas que nunca se vieron en el entrenamiento.

| Dominio | Set de test | Tableros | **v1** | **v2** | v2 ocupadas | v2 vacías | v2 errores por tablero |
|---|---|---|---|---|---|---|---|
| 3D sintético | chesscog test | 342 | 99,33 % | **99,05 %** | 97,85 % | 99,66 % | 0,61 |
| 3D real | ChessReD test (celulares, ángulos y luces variados) | 306 | 81,6 % | **97,04 %** | 91,58 % | 99,62 % | 1,90 |
| 3D real | samryan18 test (cenitales, tablero de vinilo) | 105 | 75,2 % | **99,40 %** | 98,27 % | 99,93 % | 0,38 |
| 3D real | fotos de muestra de chesscog (un tablero de madera, nunca visto) | 27 | 83,0 % | **89,0 %** | 77,4 % | 99,2 % | 7,04 |
| 2D sintético | sets de piezas vistos | 300 | 73,2 % | **98,74 %** | 95,72 % | 99,78 % | 0,80 |
| 2D sintético | sets **no vistos** (chessnut, staunty) | 300 | 73,5 % | **98,27 %** | 93,75 % | 99,88 % | 1,11 |
| 2D real | captura del Xiaomi de Lichess en el monitor (**1 imagen, solo prueba de humo**) | 1 | 56/64 (0/8 piezas) | **64/64 (8/8 piezas)** | | | |

Todo con el `.tflite` fp16 entregado (Keras da lo mismo ±0,1 punto).

Cómo leerlo:
- **El 3D sintético empeoró un poco** (99,33 → 99,05 %, es decir 0,4 → 0,6 errores por tablero). Es el costo de compartir la red con 2D y fotos reales. Los demás dominios mejoraron mucho.
- **Las fotos reales mejoraron mucho donde hay datos parecidos en el entrenamiento** (ChessReD 97 %, samryan18 99,4 %). En un tablero y un juego de piezas que el modelo nunca vio (las 27 fotos de chesscog) llega a 89 %: generaliza mejor que el v1, pero lejos de la meta. Esa es la cifra más honesta de lo que puede pasar con el tablero del usuario.
- **2D:** 98,3 % con sets de piezas que el modelo nunca vio. Pero todo es sintético, salvo una sola captura real, que además es un screencap de la app con la retícula dibujada encima. No hay un set de validación 2D real.
- **La meta de la SPEC (≥ 98 % después de la fusión en el set real)** se cumple en un fotograma para samryan18, se acerca en ChessReD, y no se cumple en un tablero nuevo (89 %). La fusión de varios fotogramas no está medida.
- **Clases más débiles** (recall del v2 fp16; matrices completas en `export/metrics_v2.json`):
  - ChessReD: damas (Q 0,72 · q 0,76) y alfiles blancos (B 0,81). Confusiones más frecuentes: P→B (59 casillas), K↔Q (39 + 34), B→Q (32).
  - 2D con sets no vistos: damas (Q 0,61 · q 0,66), confundidas casi siempre con el rey (q→k 50, Q→K 46).
  - Tablero nunca visto (chesscog real): alfil negro 0,35, rey negro 0,37, torre negra 0,50. La confusión más frecuente es p→n (31).
  - Dama/rey es el par más difícil en todos los dominios. Conviene que la app marque como dudosa una casilla cuando esas dos clases están parejas.

### int8 (no entregado)

int8 pierde entre 1 y 6 puntos según el dominio (3D sintético 97,0 %, ChessReD 96,1 %, 2D 96,2 %, chesscog real 83,1 %). Por eso se entrega fp16.

## Latencia

**Medido acá**, en una CPU de escritorio (`python -m chessscan.evaluate --bench`): TFLite 2.19.1 + XNNPACK, i5‑13420H, lote de 64, 50 corridas.

| Modelo | 1 hilo, mediana (p90) | 4 hilos, mediana (p90) |
|---|---|---|
| v2 fp16 | 16,4 ms (17,1) | 5,9 ms (7,0) |
| v1 fp16 | 18,6 ms (20,7) | 9,9 ms (12,1) |

**En el teléfono, medido por otro:** la app en el Xiaomi del usuario (LiteRT, CPU, 4 hilos) mostraba «clasif. 30 ms» con el v1, según la captura `xiaomi_lichess_01.png` de la sesión Android. El v2 tiene la misma arquitectura y el mismo tamaño, así que debería tardar lo mismo, pero **no lo medí**.

## Historial

| Versión | Datos | 3D sint. | ChessReD | samryan18 | chesscog real | 2D sint. (no vistos) |
|---|---|---|---|---|---|---|
| v1 | solo chesscog sintético | 99,33 % | 81,6 % | 75,2 % | 83,0 % | 73,5 % |
| v1 + 2 fotos reales (experimento, no entregado) | + 2 fotos de chesscog real | 99,11 % | — | — | 94,7 % | — |
| **v2** | + 2D sintético, ChessReD, samryan18 | 99,05 % | 97,0 % | 99,4 % | 89,0 % | 98,3 % |

## Qué falta medir o mejorar

1. **Validación real 2D:** el circuito automático propuesto (la PC muestra posiciones y el celular, apoyado, filma) daría cientos de posiciones reales etiquetadas. Hoy hay una sola captura.
2. **Tablero propio del usuario:** el 89 % en un tablero nunca visto indica que el ajuste fino con fotos del tablero real (`chessscan.autolabel`, o fotos sueltas) sigue valiendo la pena.
3. Precisión con fusión temporal y el set `eval/` de la SPEC.
4. Latencia del v2 en el teléfono.
5. Recuperar el 0,3 punto del 3D sintético (más épocas o más peso al 3D) y arreglar int8 si en ARM resulta más rápido.
