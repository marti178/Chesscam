package com.chessscanner.core.fen

import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece
import com.chessscanner.core.orientation.GridMapping

/**
 * Posición congelada que el usuario corrige antes de exportar. Guarda las piezas en coordenadas del
 * tablero físico ([grid], como la fusión) para que «Girar tablero» no mezcle las correcciones.
 */
data class EditablePosition(
    val grid: List<Piece>,
    /** Celdas (del tablero físico) que la fusión no dio por seguras y el usuario no tocó. */
    val doubtfulCells: Set<Int>,
    val rotation: Int,
    val whiteToMove: Boolean = true,
    /** Enroques que el usuario apagó (K, Q, k, q). Los demás se proponen si la posición los permite. */
    val castlingOff: Set<Char> = emptySet(),
    /** La posición estaba estable (64 casillas confiables ≥ 1 s) al congelar. */
    val stable: Boolean = false,
) {
    init { require(grid.size == 64) }

    val board: Board get() = GridMapping.toBoard(grid.toTypedArray(), rotation)

    /** Casillas de ajedrez dudosas. */
    val doubtfulSquares: Set<Int> get() = doubtfulCells.map { GridMapping.square(it / 8, it % 8, rotation) }.toSet()

    val possibleCastling: Castling get() = Castling.possible(board)

    val castling: Castling get() = possibleCastling.let { p ->
        Castling(
            p.whiteKing && 'K' !in castlingOff, p.whiteQueen && 'Q' !in castlingOff,
            p.blackKing && 'k' !in castlingOff, p.blackQueen && 'q' !in castlingOff,
        )
    }

    val fen: String get() = Fen.build(board, whiteToMove, castling)

    val problems: List<String> get() = Validation.problems(board)

    /** Lista para exportar sin advertencia: estable, sin dudas y válida. */
    val clean: Boolean get() = stable && doubtfulCells.isEmpty() && problems.isEmpty()

    /** Cambia la pieza de una casilla de ajedrez; la casilla deja de ser dudosa. */
    fun edit(square: Int, piece: Piece): EditablePosition {
        val cell = GridMapping.cell(square, rotation)
        return copy(grid = grid.toMutableList().also { it[cell] = piece }, doubtfulCells = doubtfulCells - cell)
    }

    /** Gira la vista [quarters] cuartos de vuelta (2 = «Girar tablero», el otro lado con a1 oscura). */
    fun rotate(quarters: Int) = copy(rotation = (rotation + quarters) and 3)

    fun toggleCastling(right: Char) = copy(castlingOff = if (right in castlingOff) castlingOff - right else castlingOff + right)
}
