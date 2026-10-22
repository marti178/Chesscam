# Licencias — parte ML

Sección ML del `LICENSES.md` del repo (se deja en `ml/` para no chocar con la rama Android; al integrar, copiarla al `LICENSES.md` de la raíz).

## Datos

| Dato | Licencia | Verificado | Uso |
|---|---|---|---|
| «Dataset of Rendered Chess Game State Images», Wölflein y Arandjelović, OSF `xf3ka` (https://osf.io/xf3ka/) | **CC‑BY 4.0** | 21‑10‑2026, en la API de OSF (`/v2/nodes/xf3ka/?embed=license` → «CC‑By Attribution 4.0 International») | Entrenamiento, validación y test. Los vectores dorados (`ml/golden/`) incluyen un render derivado (`val/0418.png`). Requiere atribución, que va en `ml/golden/README.md` y acá. |
| Fotos reales de muestra de chesscog (`transfer_learning`, ZIP en Google Drive id `1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W`, enlazado desde `chesscog/transfer_learning/download_dataset.py`) | **Sin licencia explícita** para los datos (el código del repo es MIT, pero eso no cubre las fotos) | 21‑10‑2026 | **Solo evaluación local.** No se entrena con ellas, no se suben al repo ni van en el APK. Sí se suben nuestras anotaciones de esquinas (`ml/eval_annotations/`), que son solo coordenadas. |
| Chess Recognition Dataset (ChessReD), Masouris y van Gemert, 4TU.ResearchData, DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f | **CC BY‑NC‑SA 4.0** | 22‑10‑2026, en DataCite (`rightsList`) y en `cfg/chessred.yaml` del repo de los autores | Entrenamiento y evaluación del modelo v2. **El modelo que lo usa queda bajo CC BY‑NC‑SA 4.0: uso no comercial**. El usuario lo aceptó para uso personal y gratuito (22‑10‑2026). |
| samryan18/chess-dataset (https://github.com/samryan18/chess-dataset), Ryan, Venkateswaran, Convey y Deng | **MIT** (`LICENSE.md` del repo, © 2019) | 22‑10‑2026 | Entrenamiento (≈80 %) y test (≈20 %, separado por posición). 500 fotos reales cenitales con FEN; las esquinas se calculan con `chessscan.prepare_samryan`. Requiere incluir el aviso de copyright MIT si se redistribuyen las fotos (no se redistribuyen). |
| Muestra de libro `libro_alekhine_01.png` (captura de un libro con copyright, aportada por el usuario) | **con copyright** | 22‑10‑2026 | **Solo evaluación local.** No se entrena con ella ni se sube al repo; en `eval_annotations/` solo están sus esquinas (coordenadas). |
| `ml/testdata/books/*.jpg` (diagramas sintéticos generados acá) | derivan del set **chessnut** (Alexis Luengas, **Apache 2.0**) | 22‑10‑2026 | Commiteados como imágenes de prueba del detector. Se cumple la atribución de Apache 2.0 en `testdata/books/README.md`. |
| CVChess (arXiv 2511.11522) | — | 22‑10‑2026 | No aporta datos propios: usa ChessReD. |
| Roboflow Universe «Chess Pieces» (joseph‑nelson/chess‑pieces‑new) | Dominio público (según la página del dataset) | 22‑10‑2026 | **No usado** (requiere cuenta para bajarlo; un solo tablero y ángulo). |
| Roboflow Universe, otros | varía | — | No usados. Verificar la licencia de cada dataset antes de usarlo. |

### Sets de piezas 2D (sintético en pantalla)

Fuente: `lichess-org/lila`, `public/piece/<set>/*.svg`; licencias según `lila/COPYING.md` (verificado 22‑10‑2026). Se usan **solo como imágenes de entrenamiento** (rasterizadas por `chessscan.pieces2d`); no se suben al repo ni van en el APK.

| Set | Autor | Licencia | Uso |
|---|---|---|---|
| cburnett | Colin M.L. Burnett | GPL‑2.0+ | entrenamiento |
| merida | Armando Hernández Marroquín | GPL‑2.0+ | entrenamiento |
| mpchess | Maxime Chupin | GPL‑3.0+ | entrenamiento |
| fantasy, spatial, celtic | Maurizio Monge | MIT | entrenamiento |
| rhosgfx | RhosGFX | CC0 1.0 | entrenamiento |
| totoy | Kosal Sen | CC BY 4.0 | entrenamiento |
| papercut | Nikolay Anzarov | CC BY 4.0 | entrenamiento |
| california | Jerry S. | CC BY‑NC‑SA 4.0 | entrenamiento |
| caliente | avi | CC BY‑NC‑SA 4.0 | entrenamiento |
| maestro, fresca, cardinal, icpieces, gioco, tatiana, dubrovny | sadsnake1 | CC BY‑NC‑SA 4.0 | entrenamiento |
| anarcandy | caderek | CC BY‑NC‑SA 4.0 | entrenamiento |
| cooke | fejfar | CC BY‑NC‑SA 4.0 | entrenamiento |
| minimal‑warmth | blunder_reign | CC BY‑NC‑SA 4.0 | entrenamiento |
| horsey | cham, michael1241 | CC BY‑NC‑SA 4.0 | entrenamiento |
| chessnut | Alexis Luengas | Apache 2.0 | **solo test 2D** (reservado) |
| staunty | sadsnake1 | CC BY‑NC‑SA 4.0 | **solo test 2D** (reservado) |

Texturas de tablero: `lila/public/images/board/*` (blue-marble, blue, blue2, blue3, brown, canvas2, green-plastic, green, grey, horsey, ic, leather, maple, maple2, marble, metal, olive, pink-pyramid, purple-diag, purple, wood, wood2, wood3, wood4), de «the lila authors and pirouetti», **AGPL‑3.0+**. Se marcan como **uso personal**: solo imágenes de entrenamiento, no se redistribuyen ni van en el APK.

No usados: los sets de piezas AGPL (letter, pirouetti, pixel), los «freeware» o «solo uso personal» sin texto de licencia claro (alpha, chess7, companion, leipzig), los que no tienen licencia indicada (reillycraig, riohacha), shahi (prohíbe derivados), disguised (piezas disfrazadas, confunde las etiquetas), mono (ambos colores iguales), shapes, kiwen‑suwi y xkcd (no figurativos), y monarchy y Firi (no están en SVG). Los sets de chess.com son propietarios y no se usan. Que el uso de imágenes GPL como datos de entrenamiento haga del modelo una obra derivada es jurídicamente dudoso; como el modelo ya queda CC BY‑NC‑SA por ChessReD y no se vende, se aceptó. Si algún día se quiere un modelo comercial, hay que reentrenar sin ChessReD y sin los sets GPL/NC.
| Videos propios grabados con `chessscan.autolabel` | del usuario | — | Para adaptar el modelo al tablero real. |
| `VID_20261022_132028`, `_132356`, `_133425` (tablero físico, otras posiciones) | del usuario | 22‑10‑2026 | 132028 en el entrenamiento del v5; 132356 y 133425 solo como test. No se suben. |
| `VID_20261022_134155.mp4` (PDF del «Tratado general de ajedrez» en el monitor) | **con copyright** (libro) | 22‑10‑2026 | Uso personal local: cuadros 0–899 en el entrenamiento del v5 y 1000–1412 como test. No se suben cuadros. |
| `VID_20261022_104414.mp4` (tablero físico) y `VID_20261022_105447.mp4` (Lichess en el monitor), grabados por el usuario | del usuario | 22‑10‑2026 | El físico se usa en el entrenamiento del v3 (cuadros 0–679) y en el test (740–1135); el 2D solo para validar. No se suben al repo. |

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
| resvg‑py 0.5.0 | MIT (envuelve resvg, MIT/Apache 2.0) | rasterizar SVG de piezas | no |
| matplotlib 3.9.4 | licencia PSF/matplotlib (tipo BSD) | clics en `autolabel` | no |
| pytest 8.3.4 | MIT | pruebas | no |

## Modelo entregado

`ml/export/chess_square_classifier.tflite` **v2** está entrenado con pesos ImageNet (Apache 2.0), chesscog (CC BY 4.0), ChessReD (**CC BY‑NC‑SA 4.0**), samryan18 (MIT) e imágenes generadas con sets de piezas y texturas de Lichess (licencias en la tabla de arriba, incluidas GPL, AGPL y CC BY‑NC‑SA). **El modelo se distribuye bajo CC BY‑NC‑SA 4.0: solo uso no comercial**, y los derivados con la misma licencia. El usuario aceptó esto para una app personal y gratuita (22‑10‑2026). Para una versión comercial hay que reentrenar solo con fuentes libres (chesscog, samryan18, sets MIT/Apache/CC0/CC BY) y volver a medir.


**Fuentes de entrenamiento por versión** (los modelos v2 en adelante se publican bajo CC BY‑NC‑SA 4.0):

| Modelo | Fuentes |
|---|---|
| v1 | chesscog sintético (CC BY 4.0) |
| v2 | v1 + sintético 2D con sets y texturas de Lichess (GPL, AGPL, MIT, Apache, CC0, CC BY, CC BY‑NC‑SA, según el set) + ChessReD chessred2k (CC BY‑NC‑SA 4.0) + samryan18 (MIT) |
| v3 | v2 + video del tablero del usuario `VID_20261022_104414` (propio) |
| v4 | v3 + diagramas de libro sintéticos (mismos sets de Lichess) |
| v5 | v4 + videos del usuario `VID_20261022_132028` (propio) y **`VID_20261022_134155`, cuadros 0–899: video de un PDF del libro «Tratado general de ajedrez» (con copyright)** |
| v6 | se agrega acá cada fuente nueva al publicarlo |

Atribución que tiene que mostrar la app:

> Modelo entrenado con: «Dataset of Rendered Chess Game State Images» de G. Wölflein y O. Arandjelović (2021, https://osf.io/xf3ka/, CC BY 4.0); «Chess Recognition Dataset (ChessReD)» de A. Masouris y J. van Gemert (2023, 4TU.ResearchData, DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f, CC BY‑NC‑SA 4.0); «chess-dataset» de S. Ryan, M. Venkateswaran, K. Convey y M. Deng (2019, MIT); y sets de piezas y tableros de Lichess (lila, lichess-org; autores y licencias en LICENSES.md). Este modelo se distribuye bajo CC BY‑NC‑SA 4.0.
