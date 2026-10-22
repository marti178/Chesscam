# Chess Live Scanner — app Android

Implementación de la parte Android de [`SPEC.md`](../SPEC.md): cámara en vivo → tablero → 64 casillas →
clasificador → fusión temporal → orientación → FEN → Lichess. Todo offline; el único permiso es `CAMERA`.

> **Estado (22‑10‑2026):** compila, tests verdes, APK debug con el modelo **v2** de `ml/` (licencia **CC BY‑NC‑SA 4.0,
> solo uso no comercial**), instalado y medido en
> **un** teléfono (Xiaomi 24122RKC7G, Snapdragon 8 Elite, gama alta: **no representa la gama media de la SPEC**).
> Todavía no se midió la precisión con el tablero real del usuario.

## Medido en el teléfono (22‑10‑2026)

Xiaomi 24122RKC7G, Android 16 (HyperOS), SoC SM8750, arm64‑v8a. Gama alta: en un teléfono de gama media hay que
esperar varias veces estos tiempos.

- **Vectores dorados en el teléfono** (`GoldenModelTest`): 2/2 OK en los 3 casos, con el modelo v2 y sus vectores
  regenerados (a7e4bb4). |Δlogit| máx = 1,6·10⁻⁴ (tolerancia 0,1), tanto desde `input_u8.bin` como por el camino
  completo PNG → recorte con OpenCV Android → modelo. (Con v1 había dado 6·10⁻⁵.)
- La tabla de latencias es del v1; con el v2 (misma arquitectura y tamaño) se repitió y da lo mismo: modelo 23,2 ms,
  pipeline completo 29,4 ms.
- **Latencia por etapa** (`PipelineBenchmarkTest`, fotograma 640×427, mediana / p90):

  | etapa | mediana | p90 |
  |---|---|---|
  | detector completo (sin seguimiento) | 11,5 ms | 13,4 ms |
  | seguimiento (flujo óptico + RANSAC) | 0,8 ms | 1,2 ms |
  | control de calidad | 0,5 ms | 0,5 ms |
  | rectificar + 64 recortes + tensor | 3,8 ms | 4,1 ms (era 48 ms antes de optimizar el copiado) |
  | clasificador fp16, XNNPACK | 23,2 ms | 23,8 ms |
  | clasificador fp16 sin XNNPACK | 30,3 ms | 33,4 ms |
  | pipeline completo clasificando | 29,4 ms | 31,0 ms |

  El clasificador da lo mismo con 1, 2, 4 o 6 hilos (23,2 ms), incluso con XNNPACK explícito. Puede ser que el
  proceso de test corra con CPUs restringidas, pero en la app real también da ~31 ms (ver abajo), así que por ahora
  los hilos no ayudan. No hay variante int8 para comparar (ML entregó solo fp16).
- **En la app, 5 minutos abierta con la cámara** (logcat `SCAN`, 240 muestras cada 30 fotogramas): mediana 30 fps de
  análisis (14–30 fps según la luz: la cámara baja a ~14 fps con poca luz). Con tablero y clasificando en cada
  fotograma: ~30 fps, 33,5 ms por fotograma (clasificación 30,9 ms con recorte incluido, detección/seguimiento 1,7 ms).
  Sin tablero: 2–10 ms por fotograma.
- **Temperatura:** batería de 29,4 °C a 34,1 °C en 5 minutos; estado térmico 0 (`NONE`) todo el tiempo, así que la
  reducción por temperatura nunca se activó. Ojo: el teléfono estaba cargando por USB, lo que también calienta.
- Para medir de nuevo: `adb logcat -s SCAN` con la app abierta, y el benchmark con
  `adb shell am instrument -w -e class com.chessscanner.app.PipelineBenchmarkTest com.chessscanner.live.test/androidx.test.runner.AndroidJUnitRunner`
  (resultados con la etiqueta `BENCH`).
- HyperOS: hay que activar «Instalar vía USB» en Opciones de desarrollador, y cada instalación pide confirmación en
  el teléfono. `connectedDebugAndroidTest` de Gradle falla porque instala con `-g` (permiso que HyperOS no da);
  hay que instalar los dos APK con `adb install -r` y lanzar `am instrument` a mano.

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

`app/src/main/assets/chess_square_classifier.tflite` + `labels.txt` son el modelo **v2** de la rama
`claude/ml-modelo-de93a1` (commit a7e4bb4, fp16, 1 916 920 bytes; SHA‑256 `48ba3677…2f0e82c`, igual a
`ml/export/SHA256SUMS`, verificado también dentro del APK instalado).
`LiteRtSquareClassifier` lo carga sin comprimir y comprueba que `labels.txt` tenga el orden de clases del contrato
y que los tensores sean `[64,128,64,3]` → `[64,13]`. Si algo falla usa `FakeSquareClassifier` (solo textura:
vacía / «peón» claro / «peón» oscuro) y la pantalla dice «Sin modelo: clasificador de prueba».
Para actualizarlo: copiar `ml/export/chess_square_classifier.tflite` y `labels.txt` a `app/src/main/assets/`.

Precisión por casilla en un solo fotograma según `ml/RESULTS.md` (v1 → v2): ChessReD real 81,6 → 97,0 %,
samryan18 real 75,2 → 99,4 %, tablero real nunca visto (chesscog, 27 fotos) 83,0 → 89,0 %, 2D sintético 73,5 → 98,3 %,
captura de Lichess en el Xiaomi 0/8 → 8/8 piezas. La confusión más común es dama ↔ rey del mismo color: la fusión deja
esa casilla dudosa si el otro de la pareja tiene ≥ 3 % de probabilidad (`FusionConfig.queenKingMinOther`).

**Licencia: el v2 es CC BY‑NC‑SA 4.0 (por ChessReD), solo uso no comercial** (ver [`LICENSES.md`](LICENSES.md)).
La atribución completa está en la app (botón «Créditos»).

Test instrumentado (necesita teléfono o emulador): compara la salida del modelo en Android con
`output_logits_f32.bin` (mismo argmax en las 64 casillas, |Δlogit| ≤ 0,1), desde `input_u8.bin` y también por el
camino completo PNG → recorte con OpenCV Android → modelo. Los vectores van solo en el APK de test:
```bash
./gradlew connectedDebugAndroidTest -PgoldenDir=<ruta a ml/golden>
```

## Resultados medidos (en la PC)

- Tests: 41/41 verdes con `-PgoldenDir` (38 sin él, que saltea los 3 dorados).
- Vectores dorados (`full_1200x800`, `small_640x427`, `cut_corner`): homografía, canónica, recortes y tensor dentro de tolerancia.
- Detector en los 3 renders dorados: error de esquina 0,01–0,03 casillas (incluso con la esquina BL fuera del cuadro, que se extrapola y el control de calidad marca «Alejate, falta una esquina»).
- Detector en las 19 fotos de `Dynamic-Chess-Board-Piece-Extraction/test-images` (solo prueba local, sin licencia):
  encuentra el tablero en **19/19** y, girando cada foto 0/90/180/270° y achicándola a 640 px como el fotograma
  de la cámara, da el mismo tablero en **76/76**. Ojo: las etiquetas de esas fotos salieron del propio detector
  y se revisaron a ojo sobre la retícula dibujada; no son una anotación independiente.
- Tiempo del detector en la PC: ~10–20 ms por imagen de 640 px (en el Xiaomi, 11,5 ms).

## Limitaciones conocidas

- **Probado en un solo teléfono, de gama alta.** Falta medir en gama media y verificar a ojo, con el tablero real, que la retícula quede alineada sobre la vista previa en distintos teléfonos.
- **Modelo v2: 89 % por casilla en un tablero real nunca visto** (según ML, un fotograma). Todavía no se midió con el tablero del usuario ni después de la fusión. «Posición lista» puede aparecer con piezas mal clasificadas: conviene revisar siempre al congelar.
- **Licencia del modelo CC BY‑NC‑SA 4.0:** el APK con este modelo no puede usarse con fines comerciales.
- Sin tablero detectado, el tablero digital se muestra vacío y atenuado («Sin posición»); la fusión se conserva hasta 15 fotogramas por si el tablero vuelve, y después se reinicia.
- Umbrales del control de calidad (nitidez 40, tamaño 35 % del lado menor, ángulo 0,45, movimiento 0,25 casillas/fotograma) puestos a ojo; hay que ajustarlos con fotogramas reales del teléfono.
- La retícula se dibuja suponiendo que la vista previa (`FIT_CENTER`) y el análisis 4:3 tienen el mismo campo visual; en algunos teléfonos puede quedar corrida unos píxeles.
- El detector necesita ver una buena parte de las líneas del damero; tableros sin contraste entre casillas, muy tapados o muy de costado (cociente de lados < 0,45) no se detectan o se rechazan.
- Si el tablero se pierde más de 15 fotogramas seguidos, o la nueva detección cae a más de 1,5 casillas de la anterior, se reinicia la fusión.
- La orientación usa heurísticas (a1 oscura por luminancia, filas, peones). Con pocas piezas, o casillas de poco contraste, queda «dudosa» y hay que usar «Girar».
- Al paso siempre `-` y contadores `0 1` (decidido en la SPEC).
- LiteRT 1.4.2 en vez de 2.x: la 2.x arrastra Play AI Delivery, WorkManager y Room y agrega permisos de servicio en primer plano. El APK pide solo `CAMERA` (más el permiso interno de firma `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION` de androidx.core, que no es un permiso de usuario).
- APK debug grande (≈ 114 MB) por OpenCV completo en dos ABIs; un release con `abiFilters` solo arm64 y R8 debería bajar mucho.
- Licencias de las dependencias de Android en [`LICENSES.md`](LICENSES.md).
