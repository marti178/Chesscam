package com.chessscanner.core.board

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class QualityParams(
    /** Lado mínimo del tablero como fracción del lado menor del fotograma. */
    val minSideFraction: Double = 0.35,
    /** Varianza mínima del Laplaciano dentro del tablero (fotograma de ~640 px). */
    val minSharpness: Double = 40.0,
    /** Movimiento máximo de una esquina entre fotogramas, en casillas. */
    val maxMotionSquares: Double = 0.25,
    /** Cociente mínimo entre lados opuestos (qué tan de costado se ve el tablero). */
    val minSideRatio: Double = 0.45,
    /** Fracción mínima de la retícula visible (por debajo, algo la tapa: manos, reflejos). */
    val minSupport: Double = 0.35,
)

enum class QualityIssue { NONE, CORNER_OUT, TOO_SMALL, TOO_OBLIQUE, BLURRY, MOVING, OCCLUDED }

class QualityReport(
    val issue: QualityIssue,
    val sharpness: Double,
    /** Peso del fotograma para la fusión, en [0, 1]. */
    val frameWeight: Double,
    /** Peso por celda de la imagen (fila * 8 + col): qué tan de frente y grande se ve; 0 = fuera de cuadro. */
    val cellWeights: DoubleArray,
) {
    val ok get() = issue == QualityIssue.NONE
}

/** Control de calidad del fotograma (SPEC: tablero completo, tamaño, nitidez, ángulo, movimiento). */
class FrameQuality(val params: QualityParams = QualityParams()) {

    fun assess(gray: Mat, board: BoardDetection, motionSquares: Double, support: Double): QualityReport {
        val w = gray.cols()
        val h = gray.rows()
        val c = board.corners
        val cells = cellWeights(board, w, h)
        val margin = 2.0
        val inside = c.all { it.x >= margin && it.y >= margin && it.x <= w - 1 - margin && it.y <= h - 1 - margin }
        if (!inside) return QualityReport(QualityIssue.CORNER_OUT, 0.0, 0.0, cells)

        val sides = (0 until 4).map { c[it].dist(c[(it + 1) % 4]) }
        if (sides.min() < params.minSideFraction * min(w, h)) return QualityReport(QualityIssue.TOO_SMALL, 0.0, 0.0, cells)
        val ratio = min(min(sides[0], sides[2]) / max(sides[0], sides[2]), min(sides[1], sides[3]) / max(sides[1], sides[3]))
        if (ratio < params.minSideRatio) return QualityReport(QualityIssue.TOO_OBLIQUE, 0.0, 0.0, cells)
        if (motionSquares > params.maxMotionSquares) return QualityReport(QualityIssue.MOVING, 0.0, 0.0, cells)

        val sharp = sharpness(gray, board)
        if (sharp < params.minSharpness) return QualityReport(QualityIssue.BLURRY, sharp, 0.0, cells)
        if (support < params.minSupport) return QualityReport(QualityIssue.OCCLUDED, sharp, 0.0, cells)

        val weight = min(1.0, sharp / (2 * params.minSharpness)) * (0.5 + 0.5 * min(1.0, support / 0.8))
        return QualityReport(QualityIssue.NONE, sharp, weight, cells)
    }

    /** Varianza del Laplaciano dentro del rectángulo que contiene al tablero. */
    fun sharpness(gray: Mat, board: BoardDetection): Double {
        val xs = board.corners.map { it.x }
        val ys = board.corners.map { it.y }
        val x0 = max(0, xs.min().toInt()); val y0 = max(0, ys.min().toInt())
        val x1 = min(gray.cols(), xs.max().toInt() + 1); val y1 = min(gray.rows(), ys.max().toInt() + 1)
        if (x1 - x0 < 8 || y1 - y0 < 8) return 0.0
        val roi = gray.submat(Rect(x0, y0, x1 - x0, y1 - y0))
        val lap = Mat()
        Imgproc.Laplacian(roi, lap, CvType.CV_16S)
        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std)
        val s = std.toArray()[0]
        roi.release(); lap.release(); mean.release(); std.release()
        return s * s
    }

    companion object {
        /**
         * Peso por celda de la imagen: raíz del área de la celda relativa a la más grande (las casillas
         * lejanas o muy de costado aportan menos). 0 si alguna esquina de la celda queda fuera de cuadro.
         */
        fun cellWeights(board: BoardDetection, w: Int, h: Int): DoubleArray {
            val l = board.lattice
            val area = DoubleArray(64)
            for (r in 0 until 8) for (c in 0 until 8) {
                val q = listOf(l[r * 9 + c], l[r * 9 + c + 1], l[(r + 1) * 9 + c + 1], l[(r + 1) * 9 + c])
                val visible = q.all { it.x >= 0 && it.y >= 0 && it.x < w && it.y < h }
                area[r * 8 + c] = if (visible) kotlin.math.abs(Quad.signedArea(q)) else 0.0
            }
            val maxArea = area.max()
            if (maxArea <= 0) return area
            return DoubleArray(64) { sqrt(area[it] / maxArea) }
        }
    }
}
