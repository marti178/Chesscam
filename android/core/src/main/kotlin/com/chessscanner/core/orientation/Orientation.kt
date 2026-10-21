package com.chessscanner.core.orientation

import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece
import kotlin.math.abs

/**
 * Pasaje de la retícula (fila 0 = lejana, col 0 = izquierda; ml/CONTRACT.md §1) a casillas de ajedrez.
 *
 * [rotation] dice en qué esquina de la retícula está a1:
 *  - 0: a1 abajo a la izquierda (fila 7, col 0) — blancas del lado de la cámara.
 *  - 1: a1 arriba a la izquierda (fila 0, col 0).
 *  - 2: a1 arriba a la derecha (fila 0, col 7) — blancas del lado opuesto.
 *  - 3: a1 abajo a la derecha (fila 7, col 7).
 * Cada paso es un giro de 90° en sentido horario de la imagen.
 */
object GridMapping {
    /** Casilla (rank * 8 + file) que corresponde a la celda de la retícula. */
    fun square(row: Int, col: Int, rotation: Int): Int {
        val (file, rank) = when (rotation and 3) {
            0 -> col to 7 - row
            1 -> row to col
            2 -> 7 - col to row
            else -> 7 - row to 7 - col
        }
        return rank * 8 + file
    }

    /** Celda de la retícula (fila * 8 + col) donde cae la casilla. */
    fun cell(square: Int, rotation: Int): Int {
        for (i in 0 until 64) if (square(i / 8, i % 8, rotation) == square) return i
        error("sin celda para $square")
    }

    /** Paridad `(fila + col) % 2` de la celda que ocupa a1 con esa rotación. */
    fun a1Parity(rotation: Int): Int = when (rotation and 3) {
        0 -> 1   // (7, 0)
        1 -> 0   // (0, 0)
        2 -> 1   // (0, 7)
        else -> 0 // (7, 7)
    }

    fun toBoard(grid: Array<Piece>, rotation: Int): Board {
        val cells = Array(64) { Piece.EMPTY }
        for (i in 0 until 64) cells[square(i / 8, i % 8, rotation)] = grid[i]
        return Board(cells)
    }
}

/** Resultado de [Orientation.resolve]. */
data class OrientationResult(val rotation: Int, val doubtful: Boolean, val reason: String?)

object Orientation {
    /**
     * Estima qué paridad de celdas `(fila + col) % 2` es la de casillas oscuras a partir de la
     * luminancia media de cada celda. Si se da [empty], se usan solo las celdas vacías (las piezas
     * ensucian el promedio). Devuelve null si el contraste es muy bajo para decidir.
     */
    fun darkParity(luma: DoubleArray, empty: BooleanArray? = null, minContrast: Double = 8.0): Int? {
        require(luma.size == 64)
        fun mean(parity: Int, onlyEmpty: Boolean): Double? {
            val v = (0 until 64).filter { (it / 8 + it % 8) % 2 == parity && (!onlyEmpty || empty!![it]) }
                .map { luma[it] }
            return if (v.size < 3) null else v.sorted()[v.size / 2]
        }
        val useEmpty = empty != null && (mean(0, true) != null && mean(1, true) != null)
        val m0 = mean(0, useEmpty) ?: return null
        val m1 = mean(1, useEmpty) ?: return null
        if (abs(m0 - m1) < minContrast) return null
        return if (m0 < m1) 0 else 1
    }

    /**
     * Elige la rotación (SPEC «Orientación»): a1 oscura deja 2 candidatas; entre ellas gana la que
     * deja más piezas blancas en las filas 1–2 y negras en las 7–8. Si la diferencia es chica, o no se
     * pudo saber cuáles casillas son oscuras, se marca dudosa.
     */
    fun resolve(grid: Array<Piece>, darkParity: Int?, minMargin: Int = 3): OrientationResult {
        val candidates = (0..3).filter { darkParity == null || GridMapping.a1Parity(it) == darkParity }
        val scored = candidates.map { it to score(grid, it) }.sortedByDescending { it.second }
        val (best, bestScore) = scored[0]
        val margin = bestScore - scored[1].second
        val reason = when {
            darkParity == null -> "No se distinguen las casillas claras de las oscuras"
            margin < minMargin -> "Pocas piezas para saber de qué lado juegan las blancas"
            else -> null
        }
        return OrientationResult(best, reason != null, reason)
    }

    /** Piezas blancas en filas 1–2 y negras en 7–8 suman; al revés restan. */
    fun score(grid: Array<Piece>, rotation: Int): Int {
        var s = 0
        for (i in 0 until 64) {
            val p = grid[i]
            val white = p.isWhite ?: continue
            val rank = GridMapping.square(i / 8, i % 8, rotation) / 8
            val near = rank <= 1
            val far = rank >= 6
            if (near) s += if (white) 1 else -1
            if (far) s += if (white) -1 else 1
        }
        return s
    }
}
