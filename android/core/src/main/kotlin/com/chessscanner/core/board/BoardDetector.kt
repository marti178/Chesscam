package com.chessscanner.core.board

import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Tablero detectado. [corners] en orden del contrato (TL, TR, BR, BL; TL→TR = borde de arriba de la
 * imagen) y [lattice] = los 81 cruces, fila mayor (fila 0 = borde TL→TR), en coordenadas de la imagen
 * de entrada.
 */
class BoardDetection(
    val corners: List<Pt>,
    val lattice: List<Pt>,
    /** Fracción de cruces de la retícula confirmados por líneas detectadas (0–1). */
    val support: Double,
    /** Qué tan marcado es el damero dentro de la ventana elegida (diferencia de luminancia media). */
    val checker: Double,
) {
    /** Homografía retícula (col, fila en [0, 8]) → imagen. */
    val gridToImage: Homography by lazy {
        Homography.fromPoints(
            listOf(Pt(0.0, 0.0), Pt(8.0, 0.0), Pt(8.0, 8.0), Pt(0.0, 8.0)), corners,
        )!!
    }

    companion object {
        fun fromCorners(corners: List<Pt>, support: Double = 1.0, checker: Double = 0.0): BoardDetection {
            val h = Homography.fromPoints(listOf(Pt(0.0, 0.0), Pt(8.0, 0.0), Pt(8.0, 8.0), Pt(0.0, 8.0)), corners)!!
            val lattice = List(81) { h.apply((it % 9).toDouble(), (it / 9).toDouble()) }
            return BoardDetection(corners, lattice, support, checker)
        }
    }
}

data class DetectorParams(
    /** Lado mayor de la imagen de trabajo; las más grandes se achican. */
    val workSize: Int = 640,
    val cannyLow: Double = 40.0,
    val cannyHigh: Double = 120.0,
    /** Votos mínimos de Hough como fracción del lado menor de la imagen de trabajo. */
    val houghFraction: Double = 0.16,
    val maxLines: Int = 120,
    val maxLinesPerFamily: Int = 24,
    /** Tolerancia angular para asignar una línea a una familia. */
    val familyToleranceDeg: Double = 28.0,
    /** Distancia mínima entre líneas de la misma familia, como fracción de la diagonal. */
    val mergeFraction: Double = 0.012,
    /** Residuo máximo (en casillas) para que un cruce cuente como inlier de la retícula. */
    val inlierTolerance: Double = 0.12,
    val minInliers: Int = 16,
    /** Contraste mínimo del damero (niveles de gris) para aceptar el tablero. */
    val minChecker: Double = 6.0,
    /** Picos del histograma de ángulos que se combinan de a pares. */
    val anglePeaks: Int = 3,
    /** Con este soporte (fracción de los 81 cruces) no se prueban más variantes. */
    val goodSupport: Double = 0.6,
    /** Lados de la apertura morfológica para los reintentos (diagramas rayados), a la resolución de trabajo. */
    val openKernels: List<Int> = listOf(3, 5, 7),
    /** Contraste del damero a partir del cual no se castiga a un candidato. */
    val fullChecker: Double = 40.0,
    /** Fracción mínima de los 81 cruces confirmada por líneas para aceptar un tablero. */
    val minSupport: Double = 0.35,
)

/**
 * Detector de tablero con OpenCV: bordes → líneas (Hough) → dos familias de líneas → cruces →
 * hipótesis RANSAC de retícula → homografía refinada con `findHomography(RANSAC)` → elección de la
 * ventana de 8×8 casillas por el patrón de damero. Inspirado en el detector de chesscog (MIT).
 */
class BoardDetector(val params: DetectorParams = DetectorParams()) {

    private class Line(val rho: Double, val theta: Double, val rank: Int) {
        val nx = cos(theta)
        val ny = sin(theta)
        fun intersect(o: Line): Pt? {
            val det = nx * o.ny - ny * o.nx
            if (abs(det) < 1e-6) return null
            return Pt((rho * o.ny - ny * o.rho) / det, (nx * o.rho - rho * o.nx) / det)
        }
    }

    /** Detecta el tablero en una imagen de un canal (gris, uint8). Devuelve null si no lo encuentra. */
    fun detect(gray: Mat): BoardDetection? {
        val scale = min(1.0, params.workSize.toDouble() / max(gray.cols(), gray.rows()))
        val g = Mat()
        if (scale < 1.0) Imgproc.resize(gray, g, Size(), scale, scale, Imgproc.INTER_AREA) else gray.copyTo(g)
        try {
            val result = detectWorking(g) ?: return null
            val inv = 1.0 / scale
            return BoardDetection.fromCorners(result.corners.map { it * inv }, result.support, result.checker)
        } finally {
            g.release()
        }
    }

    /**
     * Primero el camino normal; si no da una retícula bien apoyada, reintenta con una apertura
     * morfológica antes de los bordes (rellena el rayado de las casillas oscuras de los diagramas de
     * libro, que si no forma rectas diagonales largas que le ganan a la retícula). Gana el candidato con
     * más cruces de la retícula confirmados.
     */
    private fun detectWorking(g: Mat): BoardDetection? {
        val first = detectWith(g, 0)
        if (first != null && isGood(first)) return first
        var best = first
        for (k in params.openKernels) {
            val c = detectWith(g, k) ?: continue
            if (best == null || quality(c) > quality(best) + 1e-9) best = c
            if (isGood(c)) break
        }
        // Con tantas variantes alguna retícula espuria puede pasar: se exige un mínimo de cruces confirmados.
        return best?.takeIf { it.support >= params.minSupport }
    }

    /**
     * Puntaje para elegir entre candidatos: cruces confirmados, castigando el damero poco marcado (las
     * líneas de texto alrededor de un diagrama pueden formar una «retícula» sin casillas alternadas).
     */
    private fun quality(d: BoardDetection) = d.support * min(1.0, d.checker / params.fullChecker)

    private fun isGood(d: BoardDetection) = d.support >= params.goodSupport && d.checker >= params.fullChecker / 2

    private fun detectWith(g: Mat, openKernel: Int): BoardDetection? {
        val w = g.cols()
        val h = g.rows()
        val diag = hypot(w.toDouble(), h.toDouble())
        val blurred = Mat()
        val edges = Mat()
        val linesMat = Mat()
        val lines: List<Line>
        try {
            if (openKernel > 0) {
                val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(openKernel.toDouble(), openKernel.toDouble()))
                Imgproc.morphologyEx(g, blurred, Imgproc.MORPH_OPEN, kernel)
                kernel.release()
                Imgproc.GaussianBlur(blurred, blurred, Size(5.0, 5.0), 0.0)
            } else {
                Imgproc.GaussianBlur(g, blurred, Size(5.0, 5.0), 0.0)
            }
            Imgproc.Canny(blurred, edges, params.cannyLow, params.cannyHigh)
            val threshold = max(30, (params.houghFraction * min(w, h)).roundToInt())
            Imgproc.HoughLines(edges, linesMat, 1.0, PI / 360, threshold)
            lines = (0 until min(linesMat.rows(), params.maxLines)).map {
                val v = linesMat.get(it, 0)
                var rho = v[0]
                var theta = v[1]
                if (rho < 0) { rho = -rho; theta -= PI }
                Line(rho, theta, it)
            }
        } finally {
            blurred.release(); edges.release(); linesMat.release()
        }
        if (lines.size < 6) return null

        // Se prueban los pares entre los picos de ángulo más altos: en los diagramas rayados uno de los
        // dos picos principales puede ser la dirección del rayado.
        var best: BoardDetection? = null
        for ((famA, famB) in splitFamilies(lines)) {
            val c = fitFamilies(g, famA, famB, w, h, diag) ?: continue
            if (best == null || quality(c) > quality(best) + 1e-9) best = c
        }
        return best
    }

    private fun fitFamilies(g: Mat, famA: List<Line>, famB: List<Line>, w: Int, h: Int, diag: Double): BoardDetection? {
        val a = dedupe(famA, famB, w, h, params.mergeFraction * diag)
        val b = dedupe(famB, famA, w, h, params.mergeFraction * diag)
        if (a.size < 3 || b.size < 3) return null

        // Cruces de todas las líneas A × B.
        val margin = 0.25 * diag
        val pts = Array(a.size) { i ->
            Array(b.size) { j ->
                a[i].intersect(b[j])?.takeIf { it.x > -margin && it.y > -margin && it.x < w + margin && it.y < h + margin }
            }
        }
        val all = ArrayList<Pt>()
        for (row in pts) for (p in row) if (p != null) all += p

        val best = ransac(pts, all) ?: return null
        // Refinamiento: homografía retícula → imagen con todos los inliers (RANSAC de OpenCV).
        var model = best
        repeat(2) { model = refine(model, all) ?: model }
        val inliers = inliersOf(model, all)
        if (inliers.size < params.minInliers) return null

        val window = chooseWindow(g, model, inliers) ?: return null
        val (u0, v0, checker) = window
        if (checker < params.minChecker) return null
        val corners = listOf(
            model.apply(u0.toDouble(), v0.toDouble()), model.apply(u0 + 8.0, v0.toDouble()),
            model.apply(u0 + 8.0, v0 + 8.0), model.apply(u0.toDouble(), v0 + 8.0),
        )
        if (!Quad.isConvex(corners)) return null
        val inWindow = inliers.count { (u, v) -> u in u0..u0 + 8 && v in v0..v0 + 8 }
        return BoardDetection.fromCorners(Quad.contractOrder(corners), inWindow / 81.0, checker)
    }

    /**
     * Pares de familias de orientación: picos del histograma de ángulos (hasta [DetectorParams.anglePeaks],
     * separados al menos 12°), combinados de a dos si están a ≥ 35°. El primer par es el de los dos picos
     * más altos, como antes.
     */
    private fun splitFamilies(lines: List<Line>): List<Pair<List<Line>, List<Line>>> {
        val bins = 180
        val hist = DoubleArray(bins)
        for (l in lines) {
            val deg = (((l.theta * 180 / PI) % 180) + 180) % 180
            val weight = 1.0 / (1 + l.rank / 20.0)
            for (d in -3..3) hist[((deg.toInt() + d) % bins + bins) % bins] += weight * (4 - abs(d))
        }
        val peaks = ArrayList<Int>()
        for (bin in hist.indices.sortedByDescending { hist[it] }) {
            if (hist[bin] <= 0 || peaks.size >= params.anglePeaks) break
            if (peaks.all { angDist(bin.toDouble(), it.toDouble()) >= 12 }) peaks += bin
        }
        val pairs = ArrayList<Pair<Int, Int>>()
        for (i in peaks.indices) for (j in i + 1 until peaks.size) {
            if (angDist(peaks[i].toDouble(), peaks[j].toDouble()) >= 35) pairs += peaks[i] to peaks[j]
        }
        val tol = params.familyToleranceDeg
        return pairs.map { (p1, p2) ->
            val fa = ArrayList<Line>()
            val fb = ArrayList<Line>()
            for (l in lines) {
                val deg = (((l.theta * 180 / PI) % 180) + 180) % 180
                val d1 = angDist(deg, p1.toDouble())
                val d2 = angDist(deg, p2.toDouble())
                if (d1 <= d2 && d1 <= tol) fa += l else if (d2 < d1 && d2 <= tol) fb += l
            }
            fa to fb
        }
    }

    private fun angDist(a: Double, b: Double): Double {
        val d = abs(a - b) % 180
        return min(d, 180 - d)
    }

    /**
     * Ordena una familia por su posición a lo largo de una recta de referencia de la otra familia
     * (por el centro de la imagen) y une las líneas casi coincidentes, quedándose con la más votada.
     */
    private fun dedupe(fam: List<Line>, other: List<Line>, w: Int, h: Int, minGap: Double): List<Line> {
        val sx = other.sumOf { cos(2 * it.theta) }
        val sy = other.sumOf { sin(2 * it.theta) }
        val theta = atan2(sy, sx) / 2
        val cx = w / 2.0
        val cy = h / 2.0
        val ref = Line(cx * cos(theta) + cy * sin(theta), theta, 0)
        // Dirección de la recta de referencia.
        val dx = -sin(theta)
        val dy = cos(theta)
        val withT = fam.mapNotNull { l -> l.intersect(ref)?.let { p -> l to ((p.x - cx) * dx + (p.y - cy) * dy) } }
            .sortedBy { it.second }
        val kept = ArrayList<Pair<Line, Double>>()
        for (item in withT) {
            val last = kept.lastOrNull()
            if (last != null && item.second - last.second < minGap) {
                if (item.first.rank < last.first.rank) kept[kept.size - 1] = item
            } else {
                kept += item
            }
        }
        // Si hay demasiadas, quedarse con las más votadas, conservando el orden.
        val limit = params.maxLinesPerFamily
        val result = if (kept.size <= limit) kept else {
            val top = kept.sortedBy { it.first.rank }.take(limit).toSet()
            kept.filter { it in top }
        }
        return result.map { it.first }
    }

    /** Modelo de retícula: homografía retícula (u, v) → imagen. */
    private fun ransac(pts: Array<Array<Pt?>>, all: List<Pt>): Homography? {
        val na = pts.size
        val nb = pts[0].size
        var best: Homography? = null
        var bestScore = 0
        val unit = listOf(Pt(0.0, 0.0), Pt(1.0, 0.0), Pt(1.0, 1.0), Pt(0.0, 1.0))
        for (i in 0 until na - 1) for (i2 in i + 1..min(i + 2, na - 1)) {
            for (j in 0 until nb - 1) for (j2 in j + 1..min(j + 2, nb - 1)) {
                val q = listOf(pts[i][j] ?: continue, pts[i][j2] ?: continue, pts[i2][j2] ?: continue, pts[i2][j] ?: continue)
                if (!Quad.isConvex(q)) continue
                val toGrid = Homography.fromPoints(q, unit) ?: continue
                val score = scoreModel(toGrid, all)
                if (score > bestScore) {
                    bestScore = score
                    best = toGrid
                }
            }
        }
        if (bestScore < params.minInliers) return null
        return best?.inverse()
    }

    /** Cantidad de cruces distintos que caen cerca de un entero de la retícula; 0 si la retícula es imposible. */
    private fun scoreModel(toGrid: Homography, all: List<Pt>): Int {
        val seen = HashSet<Int>()
        var umin = Int.MAX_VALUE; var umax = Int.MIN_VALUE; var vmin = Int.MAX_VALUE; var vmax = Int.MIN_VALUE
        val tol = params.inlierTolerance
        for (p in all) {
            val g = toGrid.apply(p)
            if (g.x.isNaN() || abs(g.x) > 30 || abs(g.y) > 30) continue
            val u = g.x.roundToInt()
            val v = g.y.roundToInt()
            if (abs(g.x - u) < tol && abs(g.y - v) < tol) {
                if (seen.add((u + 64) * 256 + (v + 64))) {
                    umin = min(umin, u); umax = max(umax, u); vmin = min(vmin, v); vmax = max(vmax, v)
                }
            }
        }
        // Un tablero tiene 9 líneas por lado; se toleran hasta 2 de más (marco) que se recortan después.
        if (seen.isEmpty() || umax - umin > 10 || vmax - vmin > 10) return 0
        return seen.size
    }

    /** Cruces que caen en enteros de la retícula, con sus coordenadas enteras (u, v). */
    private fun inliersOf(gridToImage: Homography, all: List<Pt>): List<Triple<Int, Int, Pt>> {
        val toGrid = gridToImage.inverse()
        val tol = params.inlierTolerance * 1.25
        val byCell = HashMap<Int, Triple<Int, Int, Pt>>()
        val err = HashMap<Int, Double>()
        for (p in all) {
            val g = toGrid.apply(p)
            if (g.x.isNaN() || abs(g.x) > 30 || abs(g.y) > 30) continue
            val u = g.x.roundToInt()
            val v = g.y.roundToInt()
            val e = max(abs(g.x - u), abs(g.y - v))
            if (e < tol) {
                val key = (u + 64) * 256 + (v + 64)
                if (e < (err[key] ?: Double.MAX_VALUE)) { err[key] = e; byCell[key] = Triple(u, v, p) }
            }
        }
        return byCell.values.toList()
    }

    private fun refine(model: Homography, all: List<Pt>): Homography? {
        val inl = inliersOf(model, all)
        if (inl.size < 8) return null
        val src = MatOfPoint2f(*inl.map { Point(it.first.toDouble(), it.second.toDouble()) }.toTypedArray())
        val dst = MatOfPoint2f(*inl.map { Point(it.third.x, it.third.y) }.toTypedArray())
        val mask = Mat()
        val hm = Calib3d.findHomography(src, dst, Calib3d.RANSAC, 3.0, mask)
        src.release(); dst.release(); mask.release()
        if (hm.empty()) return null
        val m = DoubleArray(9)
        for (r in 0..2) for (c in 0..2) m[r * 3 + c] = hm.get(r, c)[0]
        hm.release()
        return Homography(m)
    }

    /**
     * Elige la ventana de 8×8 casillas dentro de la retícula encontrada: la que tiene el damero más
     * marcado adentro y que no sigue alternando justo afuera (ahí debería estar el marco).
     */
    private fun chooseWindow(g: Mat, model: Homography, inliers: List<Triple<Int, Int, Pt>>): Triple<Int, Int, Double>? {
        val umin = inliers.minOf { it.first }
        val umax = inliers.maxOf { it.first }
        val vmin = inliers.minOf { it.second }
        val vmax = inliers.maxOf { it.second }
        val w = g.cols()
        val h = g.rows()
        val data = ByteArray(w * h)
        g.get(0, 0, data)

        val lumaCache = HashMap<Int, Double?>()
        fun cellLuma(u: Int, v: Int): Double? = lumaCache.getOrPut((u + 64) * 256 + (v + 64)) {
            var sum = 0.0
            var n = 0
            // Muestras en el 60 % central de la casilla.
            for (sy in 0 until 5) for (sx in 0 until 5) {
                val p = model.apply(u + 0.2 + 0.15 * sx, v + 0.2 + 0.15 * sy)
                val x = floor(p.x).toInt()
                val y = floor(p.y).toInt()
                if (x < 0 || y < 0 || x >= w || y >= h) continue
                sum += (data[y * w + x].toInt() and 0xFF)
                n++
            }
            if (n < 13) null else sum / n
        }

        // Se deja correr la ventana hasta 2 casillas más allá de lo observado: con casillas rayadas o
        // tapadas puede no haber cruces en las primeras columnas o filas.
        fun uRange(lo: Int, hi: Int): IntRange = min(lo, hi - 8) - EXT..max(lo, hi - 8) + EXT
        var best: Triple<Int, Int, Double>? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (u0 in uRange(umin, umax)) for (v0 in uRange(vmin, vmax)) {
            val inside = ArrayList<Pair<Int, Double>>()
            for (v in v0 until v0 + 8) for (u in u0 until u0 + 8) {
                val l = cellLuma(u, v) ?: continue
                inside += ((u + v) and 1) to l
            }
            if (inside.size < 40) continue
            val mean = inside.sumOf { it.second } / inside.size
            val sIn = inside.sumOf { (par, l) -> if (par == 0) l - mean else mean - l } / inside.size
            var sRing = 0.0
            var nRing = 0
            for (v in v0 - 1..v0 + 8) for (u in u0 - 1..u0 + 8) {
                if (u in u0 until u0 + 8 && v in v0 until v0 + 8) continue
                val l = cellLuma(u, v) ?: continue
                sRing += if (((u + v) and 1) == 0) l - mean else mean - l
                nRing++
            }
            val ring = if (nRing > 0) sRing / nRing else 0.0
            // Signo: el ring "continúa" el damero si alterna igual que adentro.
            val cont = if (sIn >= 0) ring else -ring
            // Suma sobre las 64 casillas (las que no se ven cuentan 0): una ventana que se sale de la imagen pierde.
            // Y pesa cuántos cruces confirmados caen adentro: una ventana corrida sobre el piso o el texto no tiene.
            val crossings = inliers.count { (u, v, _) -> u in u0..u0 + 8 && v in v0..v0 + 8 } / 81.0
            val score = (abs(sIn) * inside.size / 64.0 - 0.7 * max(0.0, cont)) * (0.5 + 0.5 * crossings)
            if (score > bestScore) {
                bestScore = score
                best = Triple(u0, v0, abs(sIn) * 2) // diferencia media clara − oscura
            }
        }
        return best
    }
}

private const val EXT = 1
