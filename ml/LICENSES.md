# Licencias — parte ML

Sección ML del `LICENSES.md` del repo (se deja en `ml/` para no chocar con la rama Android; al integrar, copiarla al `LICENSES.md` de la raíz).

## Datos

| Dato | Licencia | Verificado | Uso |
|---|---|---|---|
| «Dataset of Rendered Chess Game State Images», Wölflein y Arandjelović, OSF `xf3ka` (https://osf.io/xf3ka/) | **CC‑BY 4.0** | 21‑10‑2026, en la API de OSF (`/v2/nodes/xf3ka/?embed=license` → «CC‑By Attribution 4.0 International») | Entrenamiento, validación y test. Los vectores dorados (`ml/golden/`) incluyen un render derivado (`val/0418.png`). Requiere atribución, que va en `ml/golden/README.md` y acá. |
| Fotos reales de muestra de chesscog (`transfer_learning`, ZIP en Google Drive id `1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W`, enlazado desde `chesscog/transfer_learning/download_dataset.py`) | **Sin licencia explícita** para los datos (el código del repo es MIT, pero eso no cubre las fotos) | 21‑10‑2026 | **Solo evaluación local.** No se entrena con ellas, no se suben al repo ni van en el APK. Sí se suben nuestras anotaciones de esquinas (`ml/eval_annotations/`), que son solo coordenadas. |
| Roboflow Universe | — | — | **No se usó ningún dataset de Roboflow.** Si se agrega uno, verificar la licencia de ese dataset en particular (varía) y anotarlo acá antes de usarlo. |
| Videos propios grabados con `chessscan.autolabel` | del usuario | — | Para adaptar el modelo al tablero real. |

## Pesos y código

| Componente | Licencia | Dónde | Va en el APK |
|---|---|---|---|
| Pesos ImageNet de MobileNetV3‑Small (Keras Applications) | Apache 2.0 | inicialización del entrenamiento | sí, transformados en `chess_square_classifier.tflite` |
| chesscog (https://github.com/georg-wolflein/chesscog) | MIT | idea del recorte (espacio arriba, espejo por mitad). El código de `ml/` es propio y no copia código de chesscog | no |
| TensorFlow / LiteRT 2.19.1 | Apache 2.0 | entrenamiento y export | runtime LiteRT: sí (lado Android) |
| OpenCV 4.12 (`opencv-python-headless`) | Apache 2.0 | referencia Python | OpenCV Android: sí (lado Android) |
| NumPy 2.1.3 | BSD‑3 | herramientas | no |
| python‑chess (`chess`) 1.11.2 | **GPL‑3.0** | solo herramientas de escritorio (FEN → etiquetas) | **no**; no enlazar ni portar su código a la app |
| tqdm 4.67.1 | MPL‑2.0 / MIT | herramientas | no |
| matplotlib 3.9.4 | licencia PSF/matplotlib (tipo BSD) | clics en `autolabel` | no |
| pytest 8.3.4 | MIT | pruebas | no |

## Modelo entregado

`ml/export/chess_square_classifier.tflite` es obra derivada de los pesos de ImageNet (Apache 2.0) entrenada con datos CC‑BY 4.0. Al distribuir el APK hay que incluir la atribución del dataset:

> Entrenado con «Dataset of Rendered Chess Game State Images» de Georg Wölflein y Ognjen Arandjelović (2021), https://osf.io/xf3ka/, CC BY 4.0.
