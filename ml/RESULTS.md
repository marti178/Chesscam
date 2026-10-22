# Resultados — clasificador de casillas v1

Fecha: 22‑10‑2026. Todo lo de esta página se midió en esta PC. Si algo no se midió, se dice.

## Modelo entregado

| | |
|---|---|
| Archivo | `ml/export/chess_square_classifier.tflite` (SHA‑256 en `export/SHA256SUMS`) |
| Red | MobileNetV3‑Small (pesos ImageNet) + GAP + dropout 0,2 + densa de 13 logits, ~0,95 M parámetros |
| Cuantización | **fp16** (pesos en float16; entrada y salida float32 según el contrato) |
| Tamaño | 1 916 920 bytes (1,9 MB) |
| E/S | entrada `[64,128,64,3]` float32 RGB 0–255 → salida `[64,13]` logits float32 calibrados (T = 1,149) |
| Entrenado con | solo el dataset sintético de chesscog (train, 4400 tableros) |
| Entrenamiento | CPU i5‑13420H, 10 épocas × 1000 lotes × 128 recortes, semilla 1234, 51 min (`runs/v1_cpu`, comando en `README.md`) |

Se eligió fp16 y no int8 porque int8 pierde 1,1 puntos en el test sintético y, en esta CPU x86, encima es más lento (ver latencia).

## Precisión

Precisión por casilla = casillas bien clasificadas / casillas totales (64 por tablero), sin fusión temporal: un solo fotograma por tablero.

### Test sintético (chesscog `test`, 342 tableros, 21 888 casillas)

| Variante | Por casilla | Ocupadas | Vacías | Tableros perfectos (64/64) | Errores por tablero |
|---|---|---|---|---|---|
| Keras fp32 | 99,33 % | 98,57 % | 99,72 % | 68,1 % | 0,43 |
| **TFLite fp16 (entregado)** | **99,33 %** | **98,57 %** | **99,72 %** | **68,7 %** | **0,43** |
| TFLite int8 | 98,27 % | 96,09 % | 99,39 % | 39,2 % | 1,11 |

En val (146 tableros, Keras): 99,26 % por casilla. El render de los vectores dorados (`val/0418`) da 63/64 en los tres casos.

Recall por clase (test sintético, fp16): vacía 1,00 · P 0,99 · N 0,97 · B 0,97 · R 0,99 · Q 0,95 · K 0,99 · p 0,99 · n 0,99 · b 0,97 · r 0,99 · q 0,98 · k 0,99.

### Fotos reales (27 fotos de un tablero real, set de muestra de chesscog, 1728 casillas)

**El modelo entrenado solo con datos sintéticos no transfiere bien a fotos reales.**

| Modelo | Por casilla | Ocupadas | Vacías | Errores por tablero |
|---|---|---|---|---|
| **v1 entregado (solo sintético), fp16** | **83,0 %** | **64,6 %** | 99,4 % | 10,9 |
| v1 + ajuste fino con 2 fotos reales de la posición inicial (experimento, **no** entregado) | 94,7 % | 89,0 % | 99,8 % | 3,4 |

Recall por clase en las fotos reales:

| | vacía | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (solo sintético) | 0,99 | 0,66 | 0,56 | 0,56 | 0,69 | 0,44 | 0,07 | 0,94 | 0,98 | 0,51 | 0,15 | 0,52 | 0,00 |
| v1 + 2 fotos reales | 1,00 | 0,98 | 0,87 | 0,56 | 0,94 | 0,63 | 0,70 | 1,00 | 0,85 | 0,86 | 0,78 | 0,89 | 0,56 |

Cómo leerlo:
- «Vacía / ocupada» transfiere bien: más de 99 %.
- El tipo de pieza en fotos reales es el problema: sin datos reales se confunden mucho, sobre todo reyes, damas y torres negras. Con un paso temprano del entrenamiento (época 2) la precisión era 64 %, casi todo predicho como peón negro; con el modelo final, 83 %.
- Con solo **2 fotos** de ese tablero en la posición inicial (las mismas 2 que usa chesscog para su ajuste fino), sube a 94,7 %. Ese es exactamente el flujo de `chessscan.autolabel`, que con un video da cientos de fotogramas, así que es esperable mejorar más con videos propios. Esto **no está medido** todavía.
- La meta de la SPEC (≥ 98 % después de la fusión en el set real) **no se alcanza con el modelo v1**, ni siquiera con el ajuste fino de 2 fotos medido acá, en un solo fotograma. La fusión de varios fotogramas va a ayudar, pero tampoco está medida.
- Salvedades del set real: es chico (27 fotos), es de un solo tablero y un solo juego de piezas, todas las fotos se sacaron desde arriba con una inclinación parecida, y las esquinas las anoté yo (a ojo y refinadas automáticamente, verificadas con el patrón del damero). Las fotos no tienen licencia explícita: se usan solo para evaluar, salvo las 2 del experimento de ajuste fino, y ese modelo no se distribuye (ver `LICENSES.md`).

## Latencia (CPU de escritorio, referencia)

`python -m chessscan.evaluate --bench`, TensorFlow Lite 2.19.1 con XNNPACK, Intel i5‑13420H (Windows 11), un lote de 64 recortes, 50 corridas después de 5 de calentamiento.

| Variante | 1 hilo, mediana (p90) | 4 hilos, mediana (p90) |
|---|---|---|
| **fp16 (entregado)** | **18,6 ms (20,7)** | **9,9 ms (12,1)** |
| fp32 | 18,8 ms (20,3) | 10,2 ms (11,9) |
| int8 | 73,7 ms (77,0) | 69,8 ms (72,3) |

**No medido:** la latencia en un teléfono. La SPEC pide < 30 ms por lote de 64 en uno de gama media. Una CPU ARM de gama media suele ser varias veces más lenta que esta, así que con 1 hilo probablemente no llegue; con 4 hilos es posible, pero hay que medirlo en el teléfono. En ARM el int8 puede ser más rápido que el fp16 (en x86 no lo fue); también hay que medirlo allá. Si no alcanza, las opciones son la variante `minimalistic` de MobileNetV3 (cuantiza mejor a int8), clasificar menos casillas por fotograma o bajar la frecuencia.

## Qué falta medir o mejorar

1. Datos reales propios: grabar videos con `chessscan.autolabel` (instrucciones en `README.md`) y reentrenar con `--extra`. Es la mejora más grande disponible.
2. Precisión con fusión temporal y en el set `eval/` de la SPEC (no existe todavía).
3. Latencia, fps y temperatura en el teléfono.
4. Corregir la pérdida de int8 (calibración con más datos, cuantización por canal, variante `minimalistic`) si en ARM int8 resulta más rápido.
