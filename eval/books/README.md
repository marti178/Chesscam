# eval/books

Diagramas de libro sintéticos con esquinas y FEN exactos, copiados de `ml/testdata/books` (rama
`claude/ml-modelo-de93a1`, commit 069dadb). Generados con `python -m chessscan.synthbook` usando solo el set de piezas
**chessnut** (Alexis Luengas, Apache 2.0, https://github.com/LexLuengas/chessnut-pieces); no hay texto ni imágenes de
libros reales. `boards.tsv` sale de los `.json`.

Varían el rayado de las casillas oscuras (±45°, cruzado, punteado, gris liso), el marco simple o doble, el texto
alrededor, el papel, la curvatura de la página, la perspectiva, la sombra y el desenfoque.

Los usa `BooksDetectorTest` (test JVM de `android/core`) y se pueden correr con la herramienta:

```bash
./gradlew :core:evalBoards -Pimages=../eval/books -Plabels=../eval/books/boards.tsv
```
