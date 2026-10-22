# Chess Live Scanner — app Android

Implementación de la parte Android de [`SPEC.md`](../SPEC.md): cámara en vivo → tablero → 64 casillas →
clasificador → fusión temporal → orientación → FEN → Lichess. Todo offline; el único permiso es `CAMERA`.

> **Estado (22‑10‑2026):** compila, tests verdes, APK debug generado con el modelo v1 de `ml/`.
> **No se probó en un teléfono** (no había ninguno conectado): no hay medición de latencia, fps ni
> temperatura, y no se corrió el test instrumentado que compara los logits del modelo en Android con
> los vectores dorados. Nada de lo que sigue garantiza que la app funcione bien en un dispositivo real.

## Estructura

Dos módulos Gradle (Kotlin, version catalog en `gradle/libs.versions.toml`):

| módulo / paquete | qué hace |
|---|---|
| `core` (JVM puro, testeable en la PC) | |
| `core/.../board` | `BoardDetector`: Canny → Hough → dos familias de líneas → cruces → RANSAC de retícula 9×9 → `findHomography(RANSAC)` → ventana 8×8 elegida por el patrón de damero. `BoardTracker`: sigue los 81 cruces con flujo óptico + RANSAC, redetecta cada 15 fotogramas, mantiene la identidad de las esquinas (coordenadas del tablero) y avisa saltos. `FrameQuality`: esquinas fuera, tamaño, ángulo, movimiento, nitidez (varianza del Laplaciano), retícula tapada, y pesos por casilla. |
| `core/.../classify` | `SquareClassifier` (interfaz), `FakeSquareClassifier`, `SquareCrops` (rectificación y recorte **idénticos a `ml/CONTRACT.md` v1**, verificados con los vectores dorados). |
| `core/.../fusion` | `FusionEngine`: log‑probabilidades por casilla con decaimiento, pesos por fotograma y por casilla, racha mínima, cambio consistente (pieza movida), estabilidad ≥ 1 s. `BoardFrame`: pasa celdas de la imagen a celdas del tablero según qué esquina está arriba. |
| `core/.../orientation` | `GridMapping` (retícula ↔ casillas para las 4 rotaciones), `Orientation`: paridad de casillas oscuras por luminancia (a1 oscura), puntaje por filas, fila media y peones enfrentados; marca dudosa. |
| `core/.../fen` | FEN ↔ tablero, validación con motivos en español, enroques posibles, `EditablePosition` (posición congelada editable). |
| `core/.../pipeline` | `ScanPipeline`: un fotograma completo, mismo código en la app y en las herramientas de escritorio. |
| `core/.../eval` | Herramienta de escritorio `evalBoards`. |
| `app` (Android) | |
| `app/.../camera` | `YuvFrame` (YUV_420_888 → `Mat` gris directo del plano Y y RGB vía NV21 solo al clasificar, rotado según el sensor), `FrameAnalyzer` (CameraX `ImageAnalysis`, `KEEP_ONLY_LATEST`, ~640×480). |
| `app/.../classify` | `LiteRtSquareClassifier` (LiteRT `Interpreter`, CPU/XNNPACK, 4 hilos; verifica formas `[64,128,64,3]` → `[64,13]`). |
| `app/.../ui` | Pantalla única en Compose Material 3: cámara con la retícula (verde/ámbar) y casillas dudosas, mensaje de estado, tablero digital en vivo, «Girar tablero», «Congelar»; congelado: editor de casilla, girar (180° y 90°), turno, enroques, FEN, «Copiar FEN», «Abrir en Lichess» (`ACTION_VIEW` a `https://lichess.org/editor/<FEN con _>`, con advertencia si no es estable/válida), «Volver a escanear». Baja la frecuencia de clasificación según `PowerManager.thermalStatus`. |

## Reproducir el entorno (Windows 11)

1. **JDK 17**
   ```bash
   winget install --id Microsoft.OpenJDK.17 -e
   ```
   Queda en `C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot`. Definir `JAVA_HOME` con esa ruta (o tenerlo en el PATH).
2. **Android SDK** (command‑line tools oficiales de Google, sin Android Studio) en `%LOCALAPPDATA%\Android\Sdk`:
   - Bajar `commandlinetools-win-13114758_latest.zip` de `https://dl.google.com/android/repository/` y descomprimirlo como `%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\`.
   - Aceptar licencias e instalar:
     ```bash
     sdkmanager --licenses
     ```
     ```bash
     sdkmanager "platform-tools" "platforms;android-37.0" "build-tools;37.0.0"
     ```
     (también se instalaron `platforms;android-36` y `build-tools;36.0.0`, no hacen falta.)
3. **`android/local.properties`** (no se commitea): `sdk.dir=C:/Users/<usuario>/AppData/Local/Android/Sdk`, o definir `ANDROID_HOME`.
4. Gradle no hace falta instalarlo: el wrapper baja Gradle 9.8.2.

Versiones: AGP 9.4.1, Kotlin 2.4.21, compileSdk/targetSdk 37, minSdk 26, CameraX 1.6.2, Compose BOM 2026.09.00,
OpenCV 4.12.0 (`org.opencv:opencv`, la misma versión que la referencia de `ml/`), LiteRT 1.4.2.

## Comandos

Desde `android/`:

```bash
./gradlew assembleDebug
```
APK en `app/build/outputs/apk/debug/app-debug.apk` (≈ 114 MB: incluye OpenCV para arm64 y x86_64, sin minificar).

```bash
./gradlew test
```
Tests JVM de `core`: FEN y validación, orientación, fusión con secuencias sintéticas, posición editable, y
visión sobre tableros sintéticos con perspectiva (detector, seguimiento, calidad, recortes, pipeline completo
con el clasificador falso). OpenCV de escritorio viene del jar `org.openpnp:opencv` (nativos incluidos).

Vectores dorados de la sesión ML (rama `claude/ml-modelo-de93a1`, `ml/golden/`):
```bash
./gradlew :core:test -PgoldenDir=<ruta a ml/golden>
```
Compara homografía (< 1e‑9), imagen canónica, los 64 recortes y el tensor de entrada (máx ≤ 2 niveles, media < 0.05),
y comprueba FEN y orientación con las etiquetas reales del caso. Sin `-PgoldenDir` esos tests se saltean.

Detector sobre imágenes reales (`eval/`, ver [`../eval/README.md`](../eval/README.md)):
```bash
./gradlew :core:evalBoards -Pimages=<carpeta> [-Plabels=<tsv>] [-Pout=<carpeta>]
```

Instalar en un teléfono con depuración USB:
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Modelo

`app/src/main/assets/chess_square_classifier.tflite` + `labels.txt` son el modelo v1 de la rama
`claude/ml-modelo-de93a1` (commit c5a677b, fp16, 1,9 MB; SHA‑256 `e88d238c…c6aebd`, igual a `ml/export/SHA256SUMS`).
`LiteRtSquareClassifier` lo carga sin comprimir y comprueba que `labels.txt` tenga el orden de clases del contrato
y que los tensores sean `[64,128,64,3]` → `[64,13]`. Si algo falla usa `FakeSquareClassifier` (solo textura:
vacía / «peón» claro / «peón» oscuro) y la pantalla dice «Sin modelo: clasificador de prueba».
Para actualizarlo: copiar `ml/export/chess_square_classifier.tflite` y `labels.txt` a `app/src/main/assets/`.

Según `ml/RESULTS.md`: 99,3 % por casilla en test sintético pero **83 % en fotos reales** (vacía/ocupada > 99 %;
el tipo de pieza es flojo hasta reentrenar con datos del tablero del usuario). La atribución CC BY 4.0 del
dataset está en la app (botón «Créditos»).

Test instrumentado (necesita teléfono o emulador): compara la salida del modelo en Android con
`output_logits_f32.bin` (mismo argmax en las 64 casillas, |Δlogit| ≤ 0,1), desde `input_u8.bin` y también por el
camino completo PNG → recorte con OpenCV Android → modelo. Los vectores van solo en el APK de test:
```bash
./gradlew connectedDebugAndroidTest -PgoldenDir=<ruta a ml/golden>
```

## Resultados medidos (en la PC)

- Tests: 38/38 verdes con `-PgoldenDir` (35 sin él, que saltea los 3 dorados).
- Vectores dorados (`full_1200x800`, `small_640x427`, `cut_corner`): homografía, canónica, recortes y tensor dentro de tolerancia.
- Detector en los 3 renders dorados: error de esquina 0,01–0,03 casillas (incluso con la esquina BL fuera del cuadro, que se extrapola y el control de calidad marca «Alejate, falta una esquina»).
- Detector en las 19 fotos de `Dynamic-Chess-Board-Piece-Extraction/test-images` (solo prueba local, sin licencia):
  encuentra el tablero en **19/19** y, girando cada foto 0/90/180/270° y achicándola a 640 px como el fotograma
  de la cámara, da el mismo tablero en **76/76**. Ojo: las etiquetas de esas fotos salieron del propio detector
  y se revisaron a ojo sobre la retícula dibujada; no son una anotación independiente.
- Tiempo del detector en la PC: ~10–20 ms por imagen de 640 px (sin medir en el teléfono).

## Limitaciones conocidas

- **No probado en dispositivo.** Latencia, fps, temperatura y la alineación de la retícula sobre la vista previa no están verificados.
- **Modelo v1 flojo en piezas reales** (83 % por casilla en fotos reales, según ML). La confianza está calibrada con datos sintéticos, así que «Posición lista» puede aparecer con piezas mal clasificadas: conviene revisar siempre al congelar.
- **Test instrumentado de logits sin correr** (no hubo dispositivo): no está verificado que LiteRT en el teléfono dé los mismos logits que la referencia.
- Umbrales del control de calidad (nitidez 40, tamaño 35 % del lado menor, ángulo 0,45, movimiento 0,25 casillas/fotograma) puestos a ojo; hay que ajustarlos con fotogramas reales del teléfono.
- La retícula se dibuja suponiendo que la vista previa (`FIT_CENTER`) y el análisis 4:3 tienen el mismo campo visual; en algunos teléfonos puede quedar corrida unos píxeles.
- El detector necesita ver una buena parte de las líneas del damero; tableros sin contraste entre casillas, muy tapados o muy de costado (cociente de lados < 0,45) no se detectan o se rechazan.
- Si el tablero se pierde más de 15 fotogramas seguidos, o la nueva detección cae a más de 1,5 casillas de la anterior, se reinicia la fusión.
- La orientación usa heurísticas (a1 oscura por luminancia, filas, peones). Con pocas piezas, o casillas de poco contraste, queda «dudosa» y hay que usar «Girar».
- Al paso siempre `-` y contadores `0 1` (decidido en la SPEC).
- LiteRT 1.4.2 en vez de 2.x: la 2.x arrastra Play AI Delivery, WorkManager y Room y agrega permisos de servicio en primer plano. El APK pide solo `CAMERA` (más el permiso interno de firma `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION` de androidx.core, que no es un permiso de usuario).
- APK debug grande (≈ 114 MB) por OpenCV completo en dos ABIs; un release con `abiFilters` solo arm64 y R8 debería bajar mucho.
- Licencias de las dependencias de Android en [`LICENSES.md`](LICENSES.md).
