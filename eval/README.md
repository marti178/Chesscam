# eval/

Set de evaluación fijo (SPEC «Pruebas»). Por ahora tiene solo etiquetas de tablero; las imágenes con
licencia que permita commitearlas van en `eval/images/` (por defecto de la herramienta).

## `boards.tsv`

Una imagen por línea, separada por tabs: `archivo  TLx TLy  TRx TRy  BRx BRy  BLx BLy  [FEN]`, esquinas del
área de juego en píxeles de la imagen original, en el orden de `ml/CONTRACT.md` (TL→TR = borde de arriba).
Con `archivo  -` se indica una imagen sin tablero completo (el detector debería fallar).

Las líneas `test-*.jpeg` corresponden a `Dynamic-Chess-Board-Piece-Extraction/test-images/` (carpeta del proyecto
principal, **sin licencia: no se commitean**). Sus esquinas salieron del detector y se revisaron a ojo: sirven
para detectar regresiones, no como anotación independiente.

## Correr el detector

Desde `android/`:

```bash
./gradlew :core:evalBoards -Pimages=C:/Users/MartinM/Documents/Claude/Chessapp/Dynamic-Chess-Board-Piece-Extraction/test-images
```

Imprime, por imagen, si encontró el tablero, el error máximo de esquina en casillas (bien si ≤ 0,35), el tiempo
y el soporte de la retícula; al final, cuántas detecta bien y una prueba de consistencia (la misma foto girada
0/90/180/270° y achicada a 640 px tiene que dar el mismo tablero). Deja en `android/core/build/eval/` cada imagen
con la retícula dibujada y `_mosaico.jpg` con todas juntas. Si hay imágenes sin etiqueta, escribe sus esquinas
en `boards_propuestas.tsv` para revisarlas y agregarlas.

Resultado el 21‑10‑2026: 19/19 encontradas y consistencia 76/76. (El «19/19 bien» contra `boards.tsv` no dice
mucho todavía, porque esas etiquetas salieron del mismo detector; lo que sí se verificó a ojo es que la retícula
cae sobre el área de juego en las 19.)
