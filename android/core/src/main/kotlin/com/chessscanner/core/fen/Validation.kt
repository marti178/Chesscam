package com.chessscanner.core.fen

import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece

/**
 * Validación antes de exportar (SPEC «FEN y Lichess»). No corrige nada: solo da los motivos,
 * que van tal cual a la interfaz.
 */
object Validation {
    fun problems(board: Board): List<String> {
        val out = mutableListOf<String>()
        val all = board.toArray()
        val wk = all.count { it == Piece.WK }
        val bk = all.count { it == Piece.BK }
        if (wk == 0) out += "Falta el rey blanco"
        if (wk > 1) out += "Hay $wk reyes blancos"
        if (bk == 0) out += "Falta el rey negro"
        if (bk > 1) out += "Hay $bk reyes negros"
        val badPawns = (0..7).flatMap { f -> listOf(f, 56 + f) }
            .filter { board[it] == Piece.WP || board[it] == Piece.BP }
        if (badPawns.isNotEmpty())
            out += "Peón en la fila 1 u 8: " + badPawns.joinToString(", ") { Board.squareName(it) }
        val white = all.count { it.isWhite == true }
        val black = all.count { it.isWhite == false }
        if (white > 16) out += "Hay $white piezas blancas (máximo 16)"
        if (black > 16) out += "Hay $black piezas negras (máximo 16)"
        return out
    }

    /** Problemas de un FEN en texto: primero sintaxis, después la posición. */
    fun problems(fen: String): List<String> = try {
        problems(Fen.parse(fen).board)
    } catch (e: FenSyntaxException) {
        listOf("FEN mal formado: ${e.message}")
    }
}
