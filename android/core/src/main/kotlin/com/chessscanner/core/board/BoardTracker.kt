package com.chessscanner.core.board

import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.video.Video
import kotlin.math.max

data class TrackerParams(
    /** Cada cuántos fotogramas se vuelve a detectar aunque el seguimiento ande bien (corrige deriva). */
    val redetectEvery: Int = 15,
    /** Cruces seguidos mínimos para aceptar el seguimiento. */
    val minTracked: Int = 30,
    /** Una detección a más de esta distancia (en casillas) de la anterior es otro lugar u otro tablero. */
    val jumpSquares: Double = 1.5,
    /** Fotogramas sin tablero tolerados antes de darlo por perdido (manos, reflejos). */
    val maxLostFrames: Int = 15,
)

/** Resultado de [BoardTracker.update]. */
class TrackUpdate(
    /** Tablero en este fotograma, esquinas en orden del contrato; null si no hay. */
    val board: BoardDetection?,
    /** Qué esquina del tablero persistente (0..3, horario) aparece como TL de la imagen. */
    val shift: Int,
    /** El tablero cambió de lugar, es otro, o se perdió del todo: hay que reiniciar la fusión. */
    val jumped: Boolean,
    /** Desplazamiento máximo de una esquina desde el fotograma anterior, en casillas. */
    val motionSquares: Double,
    /** Fracción de cruces de la retícula visibles y coherentes (bajo = tapado). */
    val support: Double,
    val tracked: Boolean,
)

/**
 * Seguimiento entre fotogramas. Mantiene las 4 esquinas del tablero con identidad persistente (para
 * que la fusión acumule en coordenadas del tablero aunque el teléfono gire): sigue los 81 cruces con
 * flujo óptico y refina la homografía con RANSAC; si se pierde, o cada [TrackerParams.redetectEvery]
 * fotogramas, vuelve a detectar y empareja la detección con las esquinas conocidas.
 */
class BoardTracker(
    private val detector: BoardDetector = BoardDetector(),
    private val params: TrackerParams = TrackerParams(),
) {
    private var prevGray: Mat? = null
    /** Esquinas persistentes P0..P3 (orden horario) en la imagen del último fotograma. */
    private var corners: List<Pt>? = null
    private var shift = 0
    private var sinceDetect = 0
    private var lostFrames = 0

    fun reset() {
        prevGray?.release()
        prevGray = null
        corners = null
        shift = 0
        sinceDetect = 0
        lostFrames = 0
    }

    fun update(gray: Mat): TrackUpdate {
        val previous = corners
        var tracked: Pair<List<Pt>, Double>? = null
        val prev = prevGray
        if (previous != null && prev != null && lostFrames == 0 && prev.size() == gray.size()) {
            tracked = track(prev, gray, previous)
        }

        var next: List<Pt>? = tracked?.first
        var support = tracked?.second ?: 0.0
        var jumped = false
        sinceDetect++
        if (tracked == null || sinceDetect >= params.redetectEvery) {
            val det = detector.detect(gray)
            if (det != null) {
                sinceDetect = 0
                val reference = next ?: previous
                if (reference == null) {
                    next = det.corners
                    support = det.support
                    shift = 0
                    jumped = true
                } else {
                    val (k, err) = bestShift(det.corners, reference)
                    if (err <= params.jumpSquares) {
                        // Misma posición: la detección corrige la deriva del seguimiento.
                        next = List(4) { det.corners[(it - k + 4) % 4] }
                        support = max(support, det.support)
                    } else if (tracked == null) {
                        next = det.corners
                        support = det.support
                        shift = 0
                        jumped = true
                    }
                    // Si el seguimiento anda y la detección no coincide, se confía en el seguimiento.
                }
            }
        }

        prevGray?.release()
        prevGray = gray.clone()

        if (next == null) {
            lostFrames++
            // Perdido del todo: lo que se haya acumulado ya no vale (el tablero pudo cambiar).
            val dropped = lostFrames > params.maxLostFrames && corners != null
            if (lostFrames > params.maxLostFrames) corners = null
            return TrackUpdate(null, shift, dropped, 0.0, 0.0, false)
        }
        lostFrames = 0
        val square = Quad.meanSide(next) / 8
        val motion = if (previous != null && !jumped) (0 until 4).maxOf { next[it].dist(previous[it]) } / square else 0.0
        corners = next
        shift = chooseShift(next, shift, square)
        val ordered = List(4) { next[(shift + it) % 4] }
        return TrackUpdate(BoardDetection.fromCorners(ordered, support), shift, jumped, motion, support, tracked != null)
    }

    /** Rotación k del orden tal que det[i] ≈ ref[(i + k) % 4], y el error máximo en casillas. */
    private fun bestShift(det: List<Pt>, ref: List<Pt>): Pair<Int, Double> {
        val square = Quad.meanSide(ref) / 8
        return (0 until 4).map { k -> k to (0 until 4).maxOf { det[it].dist(ref[(it + k) % 4]) } / square }
            .minBy { it.second }
    }

    /** La esquina que va como TL es la que abre el borde de más arriba; con histéresis para no saltar. */
    private fun chooseShift(p: List<Pt>, current: Int, square: Double): Int {
        fun edgeY(k: Int) = (p[k].y + p[(k + 1) % 4].y) / 2
        val best = Quad.topEdgeIndex(p)
        return if (edgeY(current) - edgeY(best) < 0.3 * square) current else best
    }

    private fun track(prev: Mat, gray: Mat, corners: List<Pt>): Pair<List<Pt>, Double>? {
        val h = Homography.fromPoints(GRID_CORNERS, corners) ?: return null
        val grid = List(81) { Pt((it % 9).toDouble(), (it / 9).toDouble()) }
        val prevPts = MatOfPoint2f(*grid.map { h.apply(it).let { p -> Point(p.x, p.y) } }.toTypedArray())
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        try {
            Video.calcOpticalFlowPyrLK(prev, gray, prevPts, nextPts, status, err, Size(21.0, 21.0), 3)
            val st = status.toArray()
            val np = nextPts.toArray()
            val ok = st.indices.filter { st[it].toInt() == 1 }
            if (ok.size < params.minTracked) return null
            val src = MatOfPoint2f(*ok.map { Point(grid[it].x, grid[it].y) }.toTypedArray())
            val dst = MatOfPoint2f(*ok.map { np[it] }.toTypedArray())
            val mask = Mat()
            val hm = Calib3d.findHomography(src, dst, Calib3d.RANSAC, 3.0, mask)
            val inliers = if (mask.empty()) 0 else org.opencv.core.Core.countNonZero(mask)
            src.release(); dst.release(); mask.release()
            if (hm.empty() || inliers < params.minTracked) { hm.release(); return null }
            val m = DoubleArray(9)
            for (r in 0..2) for (c in 0..2) m[r * 3 + c] = hm.get(r, c)[0]
            hm.release()
            val nh = Homography(m)
            val next = GRID_CORNERS.map { nh.apply(it) }
            if (!Quad.isConvex(next) || Quad.signedArea(next) <= 0) return null
            return next to inliers / 81.0
        } finally {
            prevPts.release(); nextPts.release(); status.release(); err.release()
        }
    }

    private companion object {
        val GRID_CORNERS = listOf(Pt(0.0, 0.0), Pt(8.0, 0.0), Pt(8.0, 8.0), Pt(0.0, 8.0))
    }
}
