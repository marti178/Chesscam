# Licencias — app Android

Dependencias que van dentro del APK o se usan para compilar/probar `android/`. (Datos y modelo: ver `ml/LICENSES.md`.)

| componente | versión | licencia | uso |
|---|---|---|---|
| OpenCV para Android (`org.opencv:opencv`, Maven Central) | 4.12.0 | Apache 2.0 | detección del tablero, homografías, recortes (APK) |
| LiteRT (`com.google.ai.edge.litert:litert`) | 1.4.2 | Apache 2.0 | inferencia del clasificador (APK) |
| AndroidX CameraX | 1.6.2 | Apache 2.0 | cámara (APK) |
| Jetpack Compose / Material 3 (BOM 2026.09.00), AndroidX Activity, Lifecycle, Core | — | Apache 2.0 | interfaz (APK) |
| Kotlin stdlib / kotlinx.coroutines | 2.4.21 | Apache 2.0 | (APK) |
| OpenCV de escritorio empaquetado por OpenPnP (`org.openpnp:opencv`) | 4.9.0-0 | Apache 2.0 (OpenCV) / BSD (empaquetado) | solo tests y herramienta de evaluación en la PC, no va en el APK |
| JUnit | 4.13.2 | EPL 1.0 | solo tests |
| Android Gradle Plugin, Gradle | 9.4.1 / 9.8.2 | Apache 2.0 | compilación |

## Modelo dentro del APK

`app/src/main/assets/chess_square_classifier.tflite` (de `ml/export/`, ver `ml/LICENSES.md`): derivado de pesos
MobileNetV3‑Small de ImageNet (Apache 2.0), entrenado con datos CC BY 4.0. Atribución obligatoria, que la app
muestra en «Créditos»:

> Entrenado con «Dataset of Rendered Chess Game State Images» de Georg Wölflein y Ognjen Arandjelović (2021),
> https://osf.io/xf3ka/, CC BY 4.0.

Los vectores dorados (render derivado del mismo dataset) van solo en el APK de test, y solo si se pasa `-PgoldenDir`.

El detector de tablero está inspirado en el de chesscog (Wölflein y Arandjelović, 2021, MIT); el código es propio.
Las imágenes de `Dynamic-Chess-Board-Piece-Extraction/test-images` no tienen licencia: se usan solo localmente y no se commitean.
