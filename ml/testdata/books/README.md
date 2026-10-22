# Diagramas de libro sintéticos para probar el detector

`book_NN.jpg` + `book_NN.json` (FEN, esquina de a1 en la retícula, esquinas exactas TL, TR, BR, BL del área de juego, tamaño de la imagen). Generados con `python -m chessscan.synthbook --testdata testdata/books --count 12 --seed 11`, solo con el set de piezas **chessnut** (Alexis Luengas, Apache 2.0, https://github.com/LexLuengas/chessnut-pieces) para que se puedan redistribuir. No hay texto ni imágenes de libros reales.

Cada uno varía: rayado (±45°, cruzado, punteado, gris liso, blanco), marco simple o doble, coordenadas, texto con figurines alrededor, papel, tinta, curvatura de página, foto inclinada o escaneo, sombra, desenfoque y JPEG. Diagnóstico con `python -m chessscan.detector_probe` (ver la nota a Android en `ml/RESULTS.md` y en la carpeta de notas).
