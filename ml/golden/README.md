# Vectores dorados (contrato v1)

Generados con `python -m chessscan.make_golden` (ver `ml/CONTRACT.md` §6 para el formato y las tolerancias).

| caso | qué prueba |
|---|---|
| `full_1200x800` | render original, esquinas enteras |
| `small_640x427` | el mismo reducido a ~640 px con `INTER_AREA`, esquinas no enteras |
| `cut_corner` | recortado 300 px a la izquierda: la esquina BL queda fuera del fotograma y aparecen bordes negros |

La imagen de origen es el render `val/0418.png` del dataset sintético de chesscog
(Wölflein y Arandjelović, «Dataset of Rendered Chess Game State Images», OSF, https://osf.io/xf3ka/, CC‑BY 4.0).
FEN `8/8/2Pk1p1p/p2B4/P4PP1/6K1/8/2b5`, juegan negras; la cámara está del lado de las negras, así que a1 queda en la esquina TR de la retícula.

`output_logits_f32.bin` y `expected_top1` en `case.json` aparecen cuando se genera con `--model`.
