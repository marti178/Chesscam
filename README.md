# Chesscam

![Android](https://img.shields.io/badge/Android-8.0%2B-3DDC84?logo=android&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-2.4-7F52FF?logo=kotlin&logoColor=white)
![offline](https://img.shields.io/badge/funciona-sin%20conexi%C3%B3n-555)
![modelo](https://img.shields.io/badge/modelo-CC%20BY--NC--SA%204.0-lightgrey)

**Apuntás la cámara del teléfono a un tablero de ajedrez —físico, en una pantalla o en un libro— y la app te da la posición
en FEN y la abre en el editor de Lichess.** Todo el reconocimiento corre en el teléfono, sin conexión; el único permiso es
la cámara.

<!-- CAPTURA PENDIENTE: guardar una captura de la app reconociendo un tablero como docs/captura.png
     (adb exec-out screencap -p > docs/captura.png) y reemplazar este comentario por:
<p align="center"><img src="docs/captura.png" alt="Chesscam reconociendo un tablero" width="300"></p>
-->
> 📷 *Captura de la app: pendiente.*

## Cómo se usa

1. Abrís la app y apuntás al tablero. Arriba aparece qué hacer: «Buscando tablero», «Acercate», «Alejate, falta una
   esquina», «Mové el teléfono para ver las casillas marcadas», «Posición lista».
2. Sobre la imagen se dibuja la retícula (verde si la imagen sirve, ámbar si no) y abajo se va armando el tablero digital.
   Las casillas dudosas quedan en ámbar: moviendo el teléfono se resuelven, porque la app suma lo que ve en cada fotograma.
3. Tocás **Congelar**, corregís las casillas que hagan falta, elegís el turno y los enroques, y tocás **Abrir en Lichess** o
   **Copiar FEN**.

## Cómo funciona

```
cámara (CameraX, ~640×480)
   │
   ▼
detector de tablero ── OpenCV: bordes → líneas → retícula 9×9 → homografía (RANSAC)
   │                    + seguimiento entre fotogramas y control de calidad
   ▼
64 recortes ────────── una imagen por casilla, con lugar arriba para las piezas altas
   │
   ▼
clasificador ───────── red chica (MobileNetV3) en LiteRT: vacía o una de las 12 piezas, con su confianza
   │
   ▼
fusión ─────────────── acumula la evidencia de cada casilla a lo largo de los fotogramas
   │
   ▼
orientación ────────── a1 oscura + dónde están las piezas blancas; «Girar tablero» si se equivoca
   │
   ▼
FEN ────────────────── validado (reyes, peones, cantidad de piezas) → editor de Lichess
```

La geometría la resuelve OpenCV y una red chica clasifica cada casilla. Así cada pieza cae en su casilla y cada casilla
tiene una confianza. El detector y el clasificador se pueden probar en la PC con el mismo código que usa la app.

## Estado

Versión de desarrollo (APK debug), probada en un solo teléfono, de gama alta: Xiaomi 24122RKC7G, Snapdragon 8 Elite,
Android 16. El modelo incluido es el **v5**.

Precisión del clasificador por casilla (casillas bien clasificadas sobre el total), según
[`ml/RESULTS.md`](ml/RESULTS.md):

| Set | Un fotograma | Con fusión |
|---|---|---|
| Tablero físico del usuario, **posiciones nuevas** (2 videos no usados para entrenar) | 82,7–82,8 % | 89,1–90,6 % (todo el video) |
| Lichess en un monitor, filmado (no usado para entrenar) | 99,95 % | — |
| Diagrama de libro real (Alekhine, 1 imagen) | 64/64 | — |
| Diagramas de libro sintéticos con sets de piezas no vistos | 94,6 % | — |
| ChessReD (fotos reales de celulares) | 96,8 % | — |
| Fotos de un tablero de madera nunca visto (chesscog) | 82,9 % | — |

En el tablero físico la mitad de las piezas todavía se clasifica mal en un fotograma (las negras son las peores), así
que hay que revisar la posición al congelar. Las pantallas y los libros andan mucho mejor.

Detección del tablero (ver [`android/README.md`](android/README.md)): 19/19 fotos de prueba y 10/12 diagramas de libro
sintéticos.

Tiempos en el Xiaomi: unos 29 ms por fotograma clasificando (23 ms el modelo), y unos 30 fps de análisis.

## Instalar y compilar

- **Instalar el APK:** compilarlo (abajo) e instalarlo con
  `adb install -r android/app/build/outputs/apk/debug/app-debug.apk`. En Xiaomi/HyperOS hay que activar «Instalar vía USB»
  en Opciones de desarrollador.
- **App Android** (Kotlin, Jetpack Compose, CameraX, OpenCV, LiteRT): cómo armar el entorno, compilar, correr los tests y
  medir en el teléfono, en [`android/README.md`](android/README.md).
  ```bash
  cd android && ./gradlew assembleDebug
  ```
- **Modelo** (entrenamiento, evaluación y export a LiteRT, en Python): [`ml/README.md`](ml/README.md). El contrato entre el
  modelo y la app (formato de los recortes, entrada y salida) está en [`ml/CONTRACT.md`](ml/CONTRACT.md).
- La especificación completa del proyecto está en [`SPEC.md`](SPEC.md).

## Licencias

- **Modelo** (`chess_square_classifier.tflite`): **CC BY‑NC‑SA 4.0, solo uso no comercial**, porque se entrenó con
  ChessReD (Masouris y van Gemert, 2023, CC BY‑NC‑SA 4.0). También usa el dataset sintético de chesscog (Wölflein y
  Arandjelović, 2021, https://osf.io/xf3ka/, CC BY 4.0), chess-dataset de samryan18 (MIT) y sets de piezas y tableros de
  Lichess. La atribución completa está en la app (botón «Créditos»).
- **Dependencias de la app:** OpenCV, LiteRT, CameraX y Jetpack Compose, todas Apache 2.0.
- **Código de este repositorio:** [GNU GPL v3.0](LICENSE). Podés usarlo, modificarlo y redistribuirlo; si
  distribuís una versión modificada, tiene que ser también GPL‑3.0 y con su código fuente. El modelo y los datos
  tienen sus propias licencias, indicadas arriba.
- Detalle en [`android/LICENSES.md`](android/LICENSES.md) y [`ml/LICENSES.md`](ml/LICENSES.md).

## Limitaciones conocidas

- Probada en un solo teléfono, de gama alta; en uno de gama media va a ser más lenta.
- En tableros físicos la precisión todavía es baja: ~83 % por casilla en un fotograma (solo la mitad de las piezas)
  en el tablero del usuario con posiciones nuevas, ~90 % fusionando, y 83 % en un tablero que el modelo nunca vio.
  Conviene revisar siempre la posición al congelar.
- La confusión más común es dama ↔ rey del mismo color (la app marca esas casillas como dudosas).
- La orientación (de qué lado juegan las blancas) es heurística; con pocas piezas puede equivocarse, y se corrige con
  «Girar tablero».
- El detector necesita ver buena parte de las líneas del tablero; tableros muy tapados, muy de costado o con poco contraste
  entre casillas no se detectan.
- No reconoce al paso ni los contadores de jugadas (el FEN sale con `-` y `0 1`), y no sigue partidas.
- El modelo es de uso no comercial (ver Licencias).
