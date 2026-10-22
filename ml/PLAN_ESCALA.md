# Plan para escalar el entrenamiento («hardcore»)

Fecha: 22‑10‑2026. Estado de partida: v5 (MobileNetV3‑Small, 0,95 M parámetros, fp16, contrato v1). Se lee junto con `RESULTS.md`.

**Diagnóstico de dónde estamos.** Tablero físico del usuario con posiciones nuevas: 83 % de casillas y 52 % de piezas por cuadro, ~90 % fusionando. 2D en pantalla: ≥ 99,5 % en lo real disponible. Libro real: 64/64 en la única muestra, y 94–100 % en el monitor. Lo que falla es la **variedad de piezas físicas**: solo hay chesscog (un juego sintético), ChessReD (un juego real) y el tablero del usuario. Por eso el plan pone el grueso del esfuerzo en el 3D.

Leyenda de lo que está verificado: ✅ verificado hoy · ⚠️ a verificar antes de usar · ❌ descartado.

---

## 1. 3D físico a escala (render sintético)

### 1.1 Herramienta
- **Blender no está instalado** en esta PC. Lo que hace falta: Blender 4.x LTS (GPL, gratis, se baja el zip portable de blender.org sin cuenta, ~350 MB) **o** el paquete `bpy` de PyPI (Blender como módulo de Python, para scripts sin interfaz). **No lo instalé**: necesito el OK del usuario para bajar e instalar ese software.
- Motor: **Cycles en CPU** acá (Eevee necesita una GPU decente; la Intel UHD integrada sirve para probar, no para renders masivos). En la nube, Cycles con CUDA/OptiX.

### 1.2 Modelos de piezas
| Fuente | Licencia | Estado | Uso |
|---|---|---|---|
| **Generador paramétrico propio** (Staunton por revolución: perfiles de peón, torre, alfil, dama y rey con proporciones al azar, coronas, collares, bases; caballo con silueta extruida y deformada) | propio | se puede empezar ya, sin Blender (genera mallas OBJ con numpy) | la fuente principal: cientos de «juegos» distintos, sin problemas de licencia |
| Poly Haven «Chess Set» (Riley Queen) | CC0 ✅ (API de Poly Haven) | descargable sin cuenta | juego de mármol vintage; **reservado para test** |
| Sketchfab, juegos con licencia CC0/CC‑BY | ⚠️ por modelo | la descarga pide **cuenta** (la tiene que bajar el usuario) | 10–20 juegos reales (Staunton, modernos, de viaje, temáticos); cada uno anotado en LICENSES.md |
| Thingiverse/Printables, juegos imprimibles (STL) CC0/CC‑BY | ⚠️ por modelo; muchos son CC‑BY‑NC, que sirve porque el modelo ya es NC | descarga con o sin cuenta según el sitio | más formas (sin texturas: se les asigna el material) |
| Juego de chesscog | ⚠️ no está en el repo ni en el dataset | — | no |

Materiales, por juego y al azar: madera (arce, nogal, ébano, boj, teñidos), plástico mate o brillante, resina, metal, vidrio esmerilado, mármol, pintados. Dos colores por juego, que no tienen por qué ser blanco y negro: crema/marrón como el del usuario, rojo/negro, natural/teñido. Texturas de madera y mármol de Poly Haven (CC0 ✅).

### 1.3 Tableros, escena y cámara
- Tableros: madera plegable con bisagra (como el del usuario), vinilo de torneo, cartón, vidrio, mármol, tableros con bordes y coordenadas, sin bordes, colores de casilla variados (crema/marrón, verde/blanco, gris).
- Luz: **HDRI de interiores de Poly Haven (301, CC0 ✅)** más focos al azar, sombras duras, contraluz y reflejos.
- Cámara: altura y ángulo de 20° (muy rasante, como los videos del usuario) a 90° (cenital), distancia variable, desde cualquier lado y desde las esquinas, distancia focal de celular (24–28 mm equivalentes), desenfoque leve y ruido.
- Oclusiones: manos y brazos (mallas simples o modelos CC0), tazas, vasos, piezas capturadas al costado del tablero (como en los videos del usuario), cables.
- Posiciones: de partidas reales (como chesscog con las de Carlsen) más posiciones al azar con densidad variable.
- Etiquetas: las esquinas exactas salen de la proyección de la cámara y el FEN de la escena. Se rectifica con el contrato v1, igual que todo lo demás.

### 1.4 Volumen y tiempo de render (estimaciones; se miden con un piloto de 20 imágenes antes de lanzar)
Resolución 960×720 (alcanza: el contrato usa 32 px por casilla), Cycles con 32–64 muestras y denoiser OIDN.

| Dónde | s/imagen (estimado) | 10 000 imágenes | 30 000 |
|---|---|---|---|
| Esta PC, i5‑13420H en CPU (12 hilos) | 15–40 s | **40–110 h** | 125–330 h |
| Kaggle, GPU T4 (gratis, 30 h/semana) | 1,5–3 s | 4–8 h | 12–25 h (≈1 semana de cuota) |
| GPU alquilada, RTX 4090 (vast.ai, RunPod) | 0,3–0,8 s | 1–2 h | 3–7 h |

**Recomendación:** generar acá (gratis) las mallas paramétricas y las escenas como `.blend` o scripts, hacer el piloto de 20 renders en esta PC para medir, y **renderizar en Kaggle** (gratis, con `bpy` y CUDA). Si Kaggle no alcanza o no deja, una RTX 4090 alquilada lo hace en un par de horas (costo estimado más abajo).

---

## 2. Datos reales

| Dataset | Licencia | Estado | Uso propuesto |
|---|---|---|---|
| ChessReD (Masouris y van Gemert): 10 800 fotos, 2078 con esquinas | CC BY‑NC‑SA 4.0 ✅ | usado (chessred2k) | entrenar; **sumar las 8700 fotos sin esquinas** detectándolas con el detector de Android (o el de chesscog) y verificando con el FEN, que da 4× más fotos reales del mismo juego desde muchos ángulos |
| samryan18/chess-dataset: 500 fotos cenitales | MIT ✅ | usado | entrenar y testear |
| Fotos de muestra de chesscog: 29 de un juego | sin licencia ✅ (por eso solo para evaluar) | solo test | test de «juego nunca visto» |
| Roboflow «Chess Pieces» (joseph‑nelson): 289 fotos con cajas | dominio público según la página ✅ | requiere cuenta para bajar | test de otro juego (hay que anotar esquinas una vez) |
| Otros datasets de Roboflow Universe (cajas de piezas, decenas de juegos) | ⚠️ varía por dataset (CC BY 4.0 es común) | requiere cuenta | revisar uno por uno; útiles como **test** de variedad de juegos aunque haya que anotar esquinas |
| Kaggle «Chess Positions» y similares (renders 2D con FEN) | ⚠️ | — | probablemente redundante con nuestro 2D sintético |
| **Videos del usuario** | propios ✅ | 5 videos físicos | seguir, con el protocolo de abajo |
| **Correcciones del usuario en la app** (pedido a Android) | propios | por armar | ver 2.1 |

### 2.1 Correcciones del usuario como datos
- Cuando el usuario corrige una casilla en «Congelar», la app guarda: la **imagen canónica** de ese cuadro (320×352, contrato v1), la etiqueta corregida de **las 64 casillas** (no solo la tocada: una corrección también confirma las otras 63), el FEN final, las esquinas, la orientación y la versión del modelo. Formato propuesto: un `.png` y un `.json` por cuadro congelado, en `Android/data/<app>/files/corrections/`, que se bajan por adb.
- Del lado ML: `chessscan.import_corrections` convierte eso al formato preparado (canon/labels) y lo suma con `--extra`. Las casillas que el usuario no tocó cuentan con menos peso (pueden ser errores no vistos).
- Las correcciones de una partida van **todas a train o todas a test**: nunca se reparten.

### 2.2 Protocolo de videos (para el usuario)
Por cada posición: 20–30 s, celular apoyado o en la mano, a 40–60 cm, con las 4 esquinas visibles todo el tiempo, dando media vuelta al tablero. Variar la luz entre videos. Prioridad: posiciones con **piezas negras variadas** (alfiles, damas y reyes negros, que hoy son lo peor).

---

## 3. 2D y libros

- **Sets de Lichess:** ya están los 24 con licencia clara (ver LICENSES.md). Faltan los «freeware» o «solo uso personal» (alpha, leipzig, chess7, companion): **como el modelo ya es NC y de uso personal, se podrían sumar** marcándolos así. Decisión del usuario.
- **chess.com:** sus sets y tableros son propietarios ❌. No se usan. Se cubren con sets parecidos y con el aumento de datos.
- **Otras apps y sitios:** chessgames, ChessBase, Lichess móvil: capturas propias del usuario como test.
- **Tipografías de ajedrez libres** (para libros y para la notación figurine del texto):
  | Fuente | Licencia | Estado |
  |---|---|---|
  | Noto Sans Symbols 2 (Google), con piezas U+2654–265F | SIL OFL 1.1 ⚠️ (es OFL, falta bajarla y anotarla) | libre |
  | DejaVu Sans / Serif, con piezas Unicode | licencia Bitstream Vera/DejaVu (libre) ⚠️ | libre |
  | GNU FreeSerif / FreeSans, con piezas Unicode | GPL con excepción de fuentes ⚠️ | libre |
  | Chess Merida, Alpha, Leipzig, Cases, Berlin (Enpassant / Armando Marroquín / Eric Bentzen) | «freeware» o «personal non‑commercial» ⚠️ | solo si el usuario acepta el uso personal |
  | Tipografías comerciales de libros (ChessBase Figurine, Chess Berlin, etc.) | propietarias ❌ | no |
- Variantes de impresión que faltan en el generador: tramado de offset (medios tonos), páginas escaneadas torcidas con sombra del lomo fuerte, papel muy amarillo, fotocopias, diagramas de revista a color, diagramas con flechas o marcas, e impresión de baja resolución (rayado que se vuelve moiré).

---

## 4. Modelo

| Opción | Parámetros | Costo relativo por lote (vs. v5) | Latencia estimada en el Xiaomi (lote 64, 4 hilos) |
|---|---|---|---|
| MobileNetV3‑Small (v5) | 0,95 M | 1× | **~30 ms, medido por la app** (dato de Android) |
| MobileNetV3‑Large | 3,0 M (sin la cabeza) | ~3–4× | ~90–130 ms (estimado) |
| EfficientNet‑Lite0 | 3,4 M | ~5–6× | ~150 ms (estimado) |
| EfficientNet‑Lite1/2 | 4,2 / 4,8 M | ~8–12× (piden más resolución) | 250 ms o más (estimado) |

- Hoy **lo que más limita son los datos, más que el modelo**. En el v5, la exactitud de entrenamiento (96,8 %, con aumentación fuerte) y la de validación siguen subiendo al final, sin señales de que el modelo chico esté saturado, y los errores grandes son de generalización: posiciones y ángulos nuevos en el tablero del usuario, juegos de piezas nunca vistos. No está medido si un modelo más grande ayudaría: hay que probarlo. Lo esperable es que un modelo más grande, sin más variedad de datos, memorice más el tablero del usuario.
- Plan: (1) con los datos nuevos, entrenar MobileNetV3‑Small y MobileNetV3‑Large con el mismo pipeline (el código ya acepta `--arch`; falta agregar `mnv3l`); (2) medir la latencia real de las dos en el Xiaomi; (3) elegir. La fusión no necesita clasificar 30 veces por segundo: con 8–10 clasificaciones por segundo alcanza, así que **MobileNetV3‑Large a ~100 ms puede ser aceptable** si clasifica uno de cada 3 cuadros. Es decisión de producto, con el dato medido.
- Otras mejoras baratas: entrenar más épocas (con 3–5× los datos), *test‑time augmentation* no (cuesta latencia), y cuantización int8 por canal con más calibración para recuperar velocidad si se elige Large.

### 4.1 Entrenamiento en GPU de nube (estimaciones; precios orientativos que hay que verificar el día que se use; **no abrí cuentas ni gasté nada**)
- Entrenar acá en CPU: ~50 min cada 10 000 lotes con Small. Con 30 000 renders más los reales, y Large, serían ~6–10 h por corrida: lento pero posible de noche.
- **Kaggle** (gratis, T4 o P100, 30 h de GPU por semana): entrenamiento ~3–5× más rápido. El cuello de botella es la aumentación en CPU (Kaggle da 4 vCPU), así que hay que precalcular los recortes aumentados o mover la aumentación a `tf.data` en GPU. Recomendado: render + entrenamiento en Kaggle.
- **Colab gratis:** T4 con límites variables. Colab Pro ~US$10/mes (unidades de cómputo para ~50 h de T4).
- **GPU alquilada** (vast.ai o RunPod): RTX 4090 ~US$0,3–0,7/h. Render de 30 000 imágenes (~5 h) más 3 corridas de entrenamiento (~3 h) ≈ 8–10 h ≈ **US$3–7 en total**. Una A100 no hace falta.

---

## 5. Cómo medimos «generaliza»

Un **set de test fijo** (`ml/TESTSET.md` con las sumas SHA‑256 de cada archivo), que **nunca entra al entrenamiento** y se congela antes de generar los datos nuevos:

| Bloque | Contenido | Por qué |
|---|---|---|
| T1 Físico, juego del usuario, posiciones nuevas | videos 4 y 5 del usuario (ya separados) y los próximos videos de test | lo que el usuario va a usar |
| T2 Físico, juegos nunca vistos | 27 fotos de chesscog, Roboflow «Chess Pieces» (si el usuario lo baja), el juego CC0 de Poly Haven renderizado, **2–3 juegos de amigos o de un club filmados una vez** | generalización real a otros juegos |
| T3 Físico, sintético con juegos, tableros y HDRI reservados | ~20 % de los juegos paramétricos y de Sketchfab, y ~20 % de las HDRI, nunca usados en train | mide el render sin mezclar con lo visto |
| T4 2D | sets reservados (chessnut, staunty), videos reales de pantalla de Lichess y chess.com con temas variados | 2D |
| T5 Libros | tipografías y sets reservados, y **5–10 fotos de diagramas de 2–3 libros reales del usuario** (solo test) | libros |

- Métricas por bloque: precisión por casilla y de piezas, por cuadro y fusionada (1 s y todo el video), y recall por clase. Reporte por bloque, **nunca un promedio único**.
- Regla: un modelo nuevo solo reemplaza al anterior si no empeora más de 0,5 puntos en ningún bloque y mejora en T1 o T2.

---

## 6. Orden propuesto y costo

| Paso | Qué | Dónde | Costo | Necesita del usuario |
|---|---|---|---|---|
| 0 | Congelar el set de test (TESTSET.md) | acá | gratis | — |
| 1 | Generador paramétrico de piezas (mallas OBJ) y descarga de HDRI y texturas de Poly Haven (CC0) | acá | gratis | — (en curso) |
| 2 | Instalar Blender (o `bpy`) y piloto de 20 renders para medir tiempos | acá | gratis | **OK para instalar Blender** |
| 3 | Render masivo (10 000–30 000) | Kaggle, o GPU alquilada | gratis, o ~US$2–5 | cuenta de Kaggle (la abre el usuario) o decisión de alquilar |
| 4 | ChessReD completo (8700 fotos sin esquinas) con detector y verificación por FEN | acá | gratis | — |
| 5 | Juegos 3D de Sketchfab/Thingiverse | — | gratis | bajarlos con su cuenta |
| 6 | Entrenar Small y Large con todo | acá (de noche) o Kaggle | gratis | — |
| 7 | Medir latencia de Large en el Xiaomi | Android | — | teléfono conectado |
| 8 | Correcciones del usuario como datos | Android y ML | gratis | usar la app |
