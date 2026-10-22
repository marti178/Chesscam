package com.chessscanner.core.board

import kotlin.math.abs
import kotlin.math.hypot

/** Punto en float (coordenadas de imagen de OpenCV: x a la derecha, y hacia abajo). */
data class Pt(val x: Double, val y: Double) {
    operator fun minus(o: Pt) = Pt(x - o.x, y - o.y)
    operator fun plus(o: Pt) = Pt(x + o.x, y + o.y)
    operator fun times(k: Double) = Pt(x * k, y * k)
    fun dist(o: Pt) = hypot(x - o.x, y - o.y)
}

/** Homografía 3×3 en fila mayor, en Kotlin puro (rápida para miles de hipótesis RANSAC). */
class Homography(val m: DoubleArray) {
    init { require(m.size == 9) }

    fun apply(x: Double, y: Double): Pt {
        val w = m[6] * x + m[7] * y + m[8]
        return Pt((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }

    fun apply(p: Pt) = apply(p.x, p.y)

    fun inverse(): Homography {
        val a = m
        val c00 = a[4] * a[8] - a[5] * a[7]
        val c01 = a[5] * a[6] - a[3] * a[8]
        val c02 = a[3] * a[7] - a[4] * a[6]
        val det = a[0] * c00 + a[1] * c01 + a[2] * c02
        val inv = doubleArrayOf(
            c00, a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
            c01, a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
            c02, a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3],
        )
        for (i in 0 until 9) inv[i] /= det
        return Homography(inv)
    }

    /** Compone: primero [first], después this. */
    fun after(first: Homography): Homography {
        val a = m
        val b = first.m
        val r = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += a[i * 3 + k] * b[k * 3 + j]
            r[i * 3 + j] = s
        }
        return Homography(r)
    }

    companion object {
        /** Homografía exacta de 4 correspondencias (src[i] → dst[i]), o null si es degenerada. */
        fun fromPoints(src: List<Pt>, dst: List<Pt>): Homography? {
            require(src.size == 4 && dst.size == 4)
            val a = Array(8) { DoubleArray(9) }
            for (i in 0 until 4) {
                val (x, y) = src[i]
                val (u, v) = dst[i]
                a[2 * i] = doubleArrayOf(x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u)
                a[2 * i + 1] = doubleArrayOf(0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v)
            }
            val h = solve(a) ?: return null
            return Homography(doubleArrayOf(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0))
        }

        /** Gauss con pivoteo parcial sobre una matriz aumentada n × (n+1). */
        private fun solve(a: Array<DoubleArray>): DoubleArray? {
            val n = a.size
            for (col in 0 until n) {
                var piv = col
                for (r in col + 1 until n) if (abs(a[r][col]) > abs(a[piv][col])) piv = r
                if (abs(a[piv][col]) < 1e-12) return null
                val t = a[piv]; a[piv] = a[col]; a[col] = t
                for (r in 0 until n) if (r != col) {
                    val f = a[r][col] / a[col][col]
                    if (f != 0.0) for (c in col..n) a[r][c] -= f * a[col][c]
                }
            }
            return DoubleArray(n) { a[it][n] / a[it][it] }
        }
    }
}

object Quad {
    /** Área con signo (positiva si el orden es horario en coordenadas con y hacia abajo). */
    fun signedArea(p: List<Pt>): Double {
        var s = 0.0
        for (i in p.indices) {
            val a = p[i]
            val b = p[(i + 1) % p.size]
            s += a.x * b.y - b.x * a.y
        }
        return s / 2
    }

    fun isConvex(p: List<Pt>): Boolean {
        var sign = 0
        for (i in p.indices) {
            val a = p[i]
            val b = p[(i + 1) % p.size]
            val c = p[(i + 2) % p.size]
            val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            val s = if (cross > 0) 1 else if (cross < 0) -1 else 0
            if (s != 0) {
                if (sign != 0 && s != sign) return false
                sign = s
            }
        }
        return sign != 0
    }

    fun meanSide(p: List<Pt>): Double = p.indices.sumOf { p[it].dist(p[(it + 1) % p.size]) } / p.size

    /**
     * Ordena 4 esquinas según ml/CONTRACT.md §1: orden horario TL, TR, BR, BL donde TL→TR es el borde
     * de más arriba en la imagen. Devuelve también cuánto se rotó la lista ([first] = índice de la
     * esquina de entrada que quedó como TL, tras poner la entrada en orden horario).
     */
    fun contractOrder(corners: List<Pt>): List<Pt> {
        val cw = if (signedArea(corners) >= 0) corners else corners.reversed()
        val k = topEdgeIndex(cw)
        return List(4) { cw[(k + it) % 4] }
    }

    /** Índice k tal que el borde (k, k+1) es el de menor y media. */
    fun topEdgeIndex(cw: List<Pt>): Int = (0 until 4).minBy { (cw[it].y + cw[(it + 1) % 4].y) / 2 }
}
