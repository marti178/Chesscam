# Chess Live Scanner — especificación

App Android que abre la cámara, reconoce la posición de un tablero físico de ajedrez mientras el usuario mueve el teléfono, genera el FEN y lo abre en Lichess. Todo el reconocimiento corre en el teléfono, sin conexión.

## Alcance

**Entra:**
- Cámara en vivo al abrir la app. No hay foto, galería ni subida de imágenes.
- Detección del tablero, corrección de perspectiva y las 64 casillas.
- Reconocimiento de las 12 piezas más "vacía" en cada casilla, con su confianza.
- Fusión de varios fotogramas: si una casilla queda tapada o dudosa desde un ángulo, el usuario mueve el teléfono y la evidencia nueva se suma a la anterior.
- Tablero digital en vivo; las casillas dudosas se marcan.
- Botón para congelar, corrección manual de piezas, turno y enroques.
- FEN validado, botones «Copiar FEN» y «Abrir en Lichess».

**No entra (decidido):** Stockfish o cualquier motor, seguimiento de partida o PGN, captura al paso, contadores de jugadas editables (se fijan en `0 1`), cuentas o servidores, modo foto.

## Arquitectura

Pipeline híbrido: la geometría la resuelve OpenCV y una red chica clasifica cada casilla. Así cada pieza ya cae en su casilla, se obtiene una confianza por casilla para la interfaz y se evita la licencia AGPL de Ultralytics/YOLO.

```
CameraX (ImageAnalysis, STRATEGY_KEEP_ONLY_LATEST, ~640 px de lado)
  → detector de tablero (OpenCV): bordes → líneas → retícula 9×9 → homografía (RANSAC)
  → seguimiento: si la homografía del fotograma anterior sigue valiendo, se refina en vez de redetectar
  → control de calidad: tablero completo, tamaño mínimo, nitidez (varianza del Laplaciano), ángulo
  → 64 recortes (formato fijado en ml/CONTRACT.md)
  → clasificador LiteRT (13 clases), un solo lote de 64
  → fusión temporal por casilla (en coordenadas del tablero, no de la imagen)
  → orientación (qué esquina es a1)
  → tablero digital + FEN
```

### Clasificador (sesión ML)
- Base: el enfoque de chesscog (Wölflein y Arandjelović, 2021; código MIT). Recorte por casilla con espacio hacia arriba para piezas altas y espejado para que la pieza siempre "se incline" igual.
- Red móvil (MobileNetV3/EfficientNet‑Lite o similar) exportada a LiteRT `.tflite`, cuantizada int8 o fp16, que corra en CPU con XNNPACK en menos de 30 ms por lote de 64 en un teléfono de gama media. La GPU es opcional.
- Datos: dataset sintético de chesscog (OSF `xf3ka`, CC‑BY 4.0, ~5000 renders) más datos reales. Cada dataset de Roboflow Universe se usa solo si su licencia lo permite, y se anota en `LICENSES.md`.
- Datos reales sin etiquetar a mano: se filman videos del tablero en una posición conocida (posición inicial o posiciones cargadas desde FEN). Como la posición se conoce, las 64 casillas se etiquetan solas. Esa es la herramienta principal para adaptar el modelo a tableros reales.
- Alternativa, si el clasificador por casilla no llega a la precisión pedida: detector RF‑DETR Nano (Apache 2.0, ya existe un port a LiteRT), asignando cada pieza a una casilla por el punto base de su caja proyectado con la homografía. Se decide midiendo, no por intuición.
- Entrenamiento: esta PC no tiene GPU NVIDIA. Se entrena con un notebook para Kaggle o Colab (GPU gratis), o en CPU si el tiempo resulta razonable. Tiene que ser reproducible: semillas, versiones fijadas y un script `train → eval → export`.

### Fusión temporal y multivista
- Por casilla se acumula la log‑probabilidad de cada una de las 13 clases con decaimiento exponencial. El peso de cada fotograma depende de la nitidez y de qué tan de frente se ve la casilla.
- Si la homografía se pierde o el tablero salta más allá de un umbral, se reinicia la acumulación: el tablero cambió de lugar o el usuario apuntó a otro tablero.
- Si una casilla cambia de forma consistente durante N fotogramas, se actualiza: se movió una pieza.
- Rechazo de manos, sombras y reflejos: los fotogramas con mucho movimiento o con la retícula tapada se descartan. Una clase solo cuenta si gana durante varios fotogramas seguidos.
- Una posición es «estable» cuando las 64 casillas superan el umbral de confianza durante al menos ~1 s. Recién entonces se habilita «Abrir en Lichess» sin advertencia.

### Orientación
La cámara no sabe de qué lado juegan las blancas. Por eso:
1. a1 tiene que ser casilla oscura, lo que deja 2 orientaciones de las 4 posibles.
2. Entre esas dos se elige la que deja más piezas blancas en las filas 1–2 y más negras en las 7–8. Si la diferencia es chica, se marca como dudosa.
3. Un botón «Girar tablero» corrige el resultado a mano en un toque.

## FEN y Lichess
- Turno: blancas por defecto, se cambia con un toque.
- Enroques: se proponen los que son posibles según la posición (rey y torre en sus casillas iniciales) y el usuario los puede desactivar. Al paso: `-`. Contadores: `0 1`.
- Validación antes de exportar: sintaxis del FEN, un rey de cada color, ningún peón en las filas 1 u 8, como máximo 16 piezas por color. Las posiciones ilegales se muestran con su motivo; no se corrigen inventando piezas.
- Lichess (verificado el 21‑10‑2026 en el navegador): `https://lichess.org/editor/<FEN con espacios reemplazados por _>` abre el editor con la posición y el turno correctos. El editor permite corregir y pasar a análisis desde ahí, así que es mejor destino que `/analysis`. Se abre con un `Intent.ACTION_VIEW`; si está instalada la app de Lichess, la toma Android. Siempre queda «Copiar FEN» como alternativa.

## Interfaz (español, uso con una mano)
- Pantalla única: cámara arriba con la retícula detectada encima, en verde si está bien y en ámbar si es dudosa. Abajo, el tablero digital en vivo con las casillas dudosas resaltadas.
- Mensaje de estado: «Buscando tablero», «Acercate», «Alejate, falta una esquina», «Imagen movida», «Mové el teléfono para ver las casillas marcadas», «Posición lista».
- Botón grande «Congelar». Con la posición congelada: tocar una casilla abre un selector de pieza (12 piezas más vacía), y aparecen «Girar tablero», el turno, los enroques, el FEN, «Copiar FEN», «Abrir en Lichess» y «Volver a escanear».
- Material 3 con Jetpack Compose.

## Rendimiento y privacidad
- Permisos: solo `CAMERA`. Ni siquiera `INTERNET`, porque Lichess se abre con un Intent.
- La cámara se libera cuando la app pasa a segundo plano (lifecycle de CameraX).
- La detección corre en un hilo de análisis. La clasificación es adaptable: se salta si el tablero no cambió y la confianza ya es alta, y baja la frecuencia si el dispositivo se calienta (`PowerManager.thermalStatus`).
- minSdk 26, targetSdk el último estable. El modelo va dentro del APK y no se descarga nada.

## Tecnologías
Kotlin, Jetpack Compose, CameraX, OpenCV para Android (paquete oficial de Maven Central, Apache 2.0), LiteRT. Gradle con version catalog. Todas las licencias se registran en `LICENSES.md`.

## Pruebas
- **Unitarias (JVM):** FEN ↔ tablero, validación, orientación, fusión temporal con secuencias sintéticas.
- **Pipeline sin cámara:** un test instrumentado y una herramienta de escritorio que pasan videos o imágenes grabados por el mismo código que usa la app, para medir la precisión por casilla en un set de evaluación fijo (`eval/`).
- **Set de evaluación:** fotos y videos reales con FEN conocido que cubran tableros llenos y con pocas piezas, los seis tipos en ambos colores, ángulos, distancias, luz, sombras, oclusiones, tablero cortado y posiciones ilegales.
- **Teléfono real:** precisión por casilla, latencia por fotograma, fps de clasificación y temperatura o batería tras 5 min. Requiere un teléfono del usuario con depuración USB.
- **Meta inicial:** al menos 98 % de casillas correctas en el set real después de la fusión, y menos de 3 s desde apuntar hasta tener la posición estable.

## Estructura del repositorio
```
android/        app Android (Gradle)
ml/             entrenamiento, export, CONTRACT.md, implementación de referencia en Python
eval/           videos e imágenes con FEN conocido, y scripts de medición
SPEC.md         este documento
LICENSES.md     licencias de dependencias, datos y modelo
```

## Contrato entre ML y Android
`ml/CONTRACT.md` define el formato del recorte, el tamaño, la normalización, el orden de las 13 clases, la entrada y salida del modelo y el sistema de coordenadas de la retícula: fila 0 = la más lejana a la cámara, columna 0 = izquierda en la imagen. La orientación a1 se resuelve después del clasificador. La sesión ML entrega además "vectores dorados" (imagen, esquinas, recortes esperados y salidas esperadas) para que Android verifique que su implementación coincide píxel a píxel.

## Etapas
1. ✅ Investigación y licencias base (este documento).
2. **ML:** contrato, dataset, entrenamiento, export a LiteRT y evaluación.
3. **Android:** proyecto, CameraX, detector de tablero, interfaz, FEN y Lichess. Usa un clasificador falso hasta que llegue el modelo.
4. Integración: modelo real, fusión, orientación.
5. Medición en el teléfono, datos reales, reentrenamiento y APK final con documentación de limitaciones.
