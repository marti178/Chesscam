# Resultados — clasificador de casillas

Fecha: 22‑10‑2026. Todo lo de esta página se midió en esta PC, salvo lo que se marca como medido por otro. Si algo no se midió, se dice.

## Modelo entregado: v5

Mismo contrato v1, misma arquitectura, fp16, 1 916 920 bytes (SHA‑256 en `export/SHA256SUMS`). Métricas completas en `export/metrics_v5.json` y `export/video*_v5.json`.

- **v4** = v3 + diagramas de libro sintéticos (`chessscan.synthbook`: 8000 diagramas, 15 sets tipo figurine, rayado, punteado o gris, foto de página y escaneo). 4 épocas × 750 lotes desde v3, 17 min.
- **v5** = v4 + videos reales del usuario: el físico `VID_..._104414` (video1, completo) y `VID_..._132028` (video3), y la primera mitad del libro en el monitor (`VID_..._134155`, cuadros 0–899). 5 épocas × 750 lotes desde v4, 21 min.

### Separación por video (sin la salvedad de «misma posición»)

Los videos físicos `VID_..._132356` (video4, 1928 cuadros) y `VID_..._133425` (video5, 1031 cuadros) **no se usaron para entrenar**: son posiciones nuevas en el mismo tablero del usuario. Sus FEN se verificaron con el modelo antes de usarlos (`chessscan.verify_fen`). El de `132356` estaba escrito desde el lado de las negras; se usó el FEN girado 180° (`2k2b1r/1bp2p2/7p/6p1/Q7/2P1B2P/PPB2PPK/RN2q3`), que es el único que coincide con la imagen. Los tres videos físicos nuevos tienen el tablero entero en todos los cuadros.

| Set (real, del usuario) | Modo | v4 | **v5** |
|---|---|---|---|
| Físico video4 (posición nueva) | 1 cuadro | 76,7 % (piezas 33,2 %) | **82,7 % (51,6 %)** |
| | fusión 60 cuadros (1 s) | 78,4 % (37,0 %) | **84,2 % (54,6 %)** |
| | fusión de todo el video | 82,8 % (47,6 %) | **90,6 % (71,4 %)** |
| Físico video5 (posición nueva) | 1 cuadro | 77,0 % (34,6 %) | **82,8 % (52,2 %)** |
| | fusión 60 cuadros | 78,1 % (37,2 %) | **84,2 % (55,6 %)** |
| | fusión de todo el video | 82,8 % (47,6 %) | **89,1 % (66,7 %)** |
| Libro en el monitor, cuadros 1000–1412 (misma posición que el train) | 1 cuadro | 94,4 % (87,9 %) | 94,5 % (88,4 %) |
| | fusión de todo el tramo | 100 % | 100 % |
| Lichess en el monitor (video2, no usado para entrenar) | 1 cuadro | 99,51 % | **99,95 %** |
| Muestra de libro Alekhine (1 imagen, solo evaluación) | 1 cuadro | 62/64 | **64/64** |

**Lectura honesta:**
- **El tablero físico del usuario con posiciones nuevas está lejos de la meta.** Un cuadro da 83 % de casillas, pero solo la mitad de las piezas. Fusionando todo el video llega a ~90 %. El 98 % que daba el v3 era sobre la misma posición con la que se entrenó y **no se traslada** a posiciones nuevas (el v4, equivalente al v3 en esto, daba 77 % en video4/5).
- Estos videos se filmaron desde ángulos muy rasantes: las piezas se ven muy alargadas y tapan las casillas vecinas. Es el caso más difícil para un clasificador por casilla.
- Clases más débiles en video4/5 (recall por cuadro, v5): alfil negro 0,11, rey negro 0,11–0,23, peón negro 0,29–0,46, dama negra 0–0,62, alfil blanco 0,35. Las piezas **negras (marrones)** de este juego son las que peor se reconocen.
- **2D y libros:** muy bien en lo real disponible, pero son pocas posiciones (una por video).
- Lo que empeoró: el tablero ajeno de chesscog (27 fotos) baja de 86,5 % (v4) a 82,9 % (v5). El modelo se especializa en el tablero del usuario.

### Resto de los sets (v3 → v4 → v5, fp16, 1 cuadro)

| Set | v3 | v4 | v5 |
|---|---|---|---|
| 3D sintético (chesscog test) | 99,02 % | 99,06 % | 99,05 % |
| ChessReD test | 97,08 % | 96,90 % | 96,84 % |
| samryan18 test | 99,33 % | 99,35 % | 99,33 % |
| Tablero ajeno (chesscog real) | 86,1 % | 86,5 % | 82,9 % |
| 2D sintético, sets vistos / no vistos | 98,6 / 98,1 % | 98,7 / 98,1 % | 98,7 / 98,3 % |
| Libro sintético, sets vistos | 90,7 % | 95,9 % | 96,2 % |
| Libro sintético, sets **no vistos** (chessnut, staunty) | 88,1 % | 94,2 % | 94,6 % |

### Detector con libros

El clasificador ya no es el cuello de botella en los libros: en la muestra real da 64/64 con las esquinas correctas. Lo que falla en la app es el **detector**. Con rayado a ±45°, el histograma de ángulos elige la dirección del rayado y no la de la retícula. Diagnóstico y arreglo propuesto (apertura morfológica antes de Canny) en la nota a Android y en `testdata/books/`.

---

## Modelo anterior: v3


v3 = v2 + ajuste fino con el **video del tablero físico del usuario** (`VID_20261022_104414`, una sola posición, 19 s, la cámara da la vuelta al tablero). Mismo contrato v1, misma arquitectura, fp16, 1 916 920 bytes, SHA‑256 en `export/SHA256SUMS`. Ajuste fino: 4 épocas × 500 lotes, lr 3e‑4, semilla 1234, desde `runs/v2/model.keras`. La mezcla de datos es ~39 % de cuadros del video del usuario y el resto, los datos del v2 (para no olvidarlos). Métricas en `export/metrics_v3.json` y `export/video1_v3.json`.

### Videos reales del usuario

Autoetiquetados con `chessscan.autolabel_video` (ancla con esquinas marcadas, seguimiento hacia adelante y hacia atrás con refinamiento por cuadro). Las etiquetas se revisaron a ojo en cuadros de todo el video, desde todos los lados.

- **Video físico:** en **ningún** cuadro entran las 4 esquinas del tablero en la imagen (en el mejor queda una esquina 0,6 casillas afuera), así que la regla «solo tablero completo» dejaba 0 cuadros. Se usó en cambio una máscara por casilla: cuenta una casilla solo si está entera dentro de la imagen, y un cuadro solo si tiene ≥ 40 casillas visibles. Quedaron 1017 de 1136 cuadros. **Separación temporal:** entrenamiento con los cuadros 0–679, hueco de 680–739 (1 s), test con 740–1135 (396 cuadros, 19 219 casillas visibles).
- **Video 2D** (Lichess en el monitor, tema marrón, piezas cburnett): 542 de 641 cuadros con las 4 esquinas adentro. **No se usó para entrenar**: es la validación 2D real.

| Set | Modo | v2 | **v3** |
|---|---|---|---|
| Tablero físico del usuario, test 740–1135 | 1 fotograma | 82,8 % (ocupadas 38,0 %) | **97,8 % (ocupadas 91,8 %)** |
| | fusión de 30 cuadros (0,5 s) | 82,9 % (39,1 %) | **97,9 % (92,4 %)** |
| | fusión de 60 cuadros (1 s) | 83,4 % (40,4 %) | **97,8 % (92,0 %)** |
| | fusión de todo el tramo | 85,9 % (50,0 %) | **98,4 % (94,4 %; 63/64 casillas)** |
| 2D real, Lichess en el monitor (542 cuadros) | 1 fotograma | 99,92 % (ocupadas 99,74 %) | 99,86 % (99,54 %) |
| | fusión de 30 / 60 / todo | 100 % | 100 % |
| Captura del Xiaomi (prueba de humo) | 1 fotograma | 64/64 | 64/64 |

Salvedades importantes:
- En el video físico, **entrenamiento y test son la misma posición** (un solo FEN), vista desde otros ángulos y otros momentos. Mide la adaptación al tablero, las piezas y la luz del usuario, **no** a posiciones nuevas. Puede estar inflado si el modelo aprendió algo de la disposición. Con un segundo video de otra posición se mediría bien.
- Fusión = suma de log‑probabilidades por casilla física, sin pesos ni decaimiento. Ayuda poco porque los errores que quedan son sistemáticos (la misma casilla mal desde ángulos parecidos), no ruido.
- Clases más débiles del v3 en el tablero del usuario (recall por cuadro): torre negra 0,30, rey blanco 0,67, caballo blanco 0,89; el resto, ≥ 0,98. No hay alfiles en esa posición, así que en este tablero no están medidos.
- El 2D real es un solo tema, un solo set de piezas y una sola posición (sin caballos ni torres).

### El resto de los sets (v2 → v3, fp16, un fotograma)

| Set | v2 | v3 |
|---|---|---|
| 3D sintético (chesscog test) | 99,05 % | 99,02 % |
| ChessReD test | 97,04 % | 97,08 % |
| samryan18 test | 99,40 % | 99,33 % |
| Fotos chesscog real (otro tablero, nunca visto) | 89,0 % | **86,1 %** (empeoró) |
| 2D sintético, sets vistos | 98,74 % | 98,63 % |
| 2D sintético, sets no vistos | 98,27 % | 98,05 % |

El v3 se especializa en el tablero del usuario: en un tablero ajeno que no vio pierde 3 puntos. Si la app se va a usar con otros tableros, el v2 generaliza un poco mejor. Los dos tienen el mismo contrato, así que se pueden intercambiar sin tocar Android.

---

## Modelo anterior: v2


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
| v3 fp16 (otra corrida, PC más cargada: v2 dio lo mismo en esa corrida) | 25,4 ms (36,1) | 11,3 ms (13,0) |
| v1 fp16 | 18,6 ms (20,7) | 9,9 ms (12,1) |

**En el teléfono, medido por otro:** la app en el Xiaomi del usuario (LiteRT, CPU, 4 hilos) mostraba «clasif. 30 ms» con el v1, según la captura `xiaomi_lichess_01.png` de la sesión Android. El v2 tiene la misma arquitectura y el mismo tamaño, así que debería tardar lo mismo, pero **no lo medí**.

## Historial

| Versión | Datos | 3D sint. | ChessReD | samryan18 | chesscog real | 2D sint. (no vistos) |
|---|---|---|---|---|---|---|
| v1 | solo chesscog sintético | 99,33 % | 81,6 % | 75,2 % | 83,0 % | 73,5 % |
| v1 + 2 fotos reales (experimento, no entregado) | + 2 fotos de chesscog real | 99,11 % | — | — | 94,7 % | — |
| v2 | + 2D sintético, ChessReD, samryan18 | 99,05 % | 97,0 % | 99,4 % | 89,0 % | 98,3 % |
| v3 | + video del tablero del usuario | 99,02 % | 97,1 % | 99,3 % | 86,1 % | 98,1 % |
| v4 | + libros sintéticos | 99,06 % | 96,9 % | 99,35 % | 86,5 % | 98,1 % |
| **v5** | + 2 videos físicos y 1 de libro del usuario | 99,05 % | 96,8 % | 99,33 % | 82,9 % | 98,3 % |

## Qué falta medir o mejorar

0. **Más posiciones en el tablero del usuario, con piezas negras variadas (alfiles, damas, reyes negros) y desde ángulos menos rasantes.** Es lo que más falta. Antes: grabar un segundo video con otra posición (todas las piezas, incluidos los alfiles) para medir sin la salvedad de «misma posición en train y test» y sumar datos.

1. **Validación real 2D:** el circuito automático propuesto (la PC muestra posiciones y el celular, apoyado, filma) daría cientos de posiciones reales etiquetadas. Hoy hay una sola captura.
2. **Tablero propio del usuario:** el 89 % en un tablero nunca visto indica que el ajuste fino con fotos del tablero real (`chessscan.autolabel`, o fotos sueltas) sigue valiendo la pena.
3. Precisión con fusión temporal y el set `eval/` de la SPEC.
4. Latencia del v2 en el teléfono.
5. Recuperar el 0,3 punto del 3D sintético (más épocas o más peso al 3D) y arreglar int8 si en ARM resulta más rápido.
