package com.chessscanner.core.fen

import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece

/** Derechos de enroque (K, Q, k, q). */
data class Castling(
    val whiteKing: Boolean = false,
    val whiteQueen: Boolean = false,
    val blackKing: Boolean = false,
    val blackQueen: Boolean = false,
) {
    fun toFen(): String = buildString {
        if (whiteKing) append('K')
        if (whiteQueen) append('Q')
        if (blackKing) append('k')
        if (blackQueen) append('q')
    }.ifEmpty { "-" }

    /** Deja solo los derechos que la posición permite. */
    fun restrictTo(possible: Castling) = Castling(
        whiteKing && possible.whiteKing, whiteQueen && possible.whiteQueen,
        blackKing && possible.blackKing, blackQueen && possible.blackQueen,
    )

    companion object {
        val NONE = Castling()

        /** Enroques posibles según la posición: rey y torre correspondiente en sus casillas iniciales. */
        fun possible(board: Board): Castling {
            fun at(name: String) = board[Board.square(name)]
            val wk = at("e1") == Piece.WK
            val bk = at("e8") == Piece.BK
            return Castling(
                whiteKing = wk && at("h1") == Piece.WR,
                whiteQueen = wk && at("a1") == Piece.WR,
                blackKing = bk && at("h8") == Piece.BR,
                blackQueen = bk && at("a8") == Piece.BR,
            )
        }
    }
}

object Fen {
    /** Campo de colocación de piezas (primer campo del FEN). */
    fun placement(board: Board): String = (7 downTo 0).joinToString("/") { rank ->
        buildString {
            var empty = 0
            for (file in 0..7) {
                val c = board[file, rank].fen
                if (c == null) {
                    empty++
                } else {
                    if (empty > 0) append(empty)
                    empty = 0
                    append(c)
                }
            }
            if (empty > 0) append(empty)
        }
    }

    /** FEN completo. Al paso siempre `-` y contadores fijos `0 1` (SPEC). */
    fun build(board: Board, whiteToMove: Boolean, castling: Castling): String =
        "${placement(board)} ${if (whiteToMove) 'w' else 'b'} ${castling.toFen()} - 0 1"

    /** URL del editor de Lichess: espacios reemplazados por `_` (verificado el 21-10-2026). */
    fun lichessEditorUrl(fen: String): String = "https://lichess.org/editor/" + fen.trim().replace(' ', '_')

    /** Interpreta un FEN (al menos el campo de piezas). Lanza [FenSyntaxException] con el motivo. */
    fun parse(fen: String): Parsed {
        val fields = fen.trim().split(Regex("\\s+"))
        if (fields.isEmpty() || fields[0].isEmpty()) throw FenSyntaxException("FEN vacío")
        val ranks = fields[0].split('/')
        if (ranks.size != 8) throw FenSyntaxException("debe tener 8 filas y tiene ${ranks.size}")
        val cells = Array(64) { Piece.EMPTY }
        ranks.forEachIndexed { i, text ->
            val rank = 7 - i
            var file = 0
            for (ch in text) {
                if (ch in '1'..'8') {
                    file += ch - '0'
                } else {
                    val p = Piece.fromFen(ch)
                        ?: throw FenSyntaxException("carácter inválido '$ch' en la fila ${rank + 1}")
                    if (file > 7) throw FenSyntaxException("la fila ${rank + 1} tiene más de 8 casillas")
                    cells[rank * 8 + file] = p
                    file++
                }
                if (file > 8) throw FenSyntaxException("la fila ${rank + 1} tiene más de 8 casillas")
            }
            if (file != 8) throw FenSyntaxException("la fila ${rank + 1} tiene $file casillas en vez de 8")
        }
        val turn = fields.getOrElse(1) { "w" }
        if (turn != "w" && turn != "b") throw FenSyntaxException("turno inválido '$turn'")
        val castlingText = fields.getOrElse(2) { "-" }
        if (!Regex("-|K?Q?k?q?").matches(castlingText) || castlingText.isEmpty())
            throw FenSyntaxException("enroques inválidos '$castlingText'")
        val castling = Castling('K' in castlingText, 'Q' in castlingText, 'k' in castlingText, 'q' in castlingText)
        return Parsed(Board(cells), turn == "w", castling)
    }

    data class Parsed(val board: Board, val whiteToMove: Boolean, val castling: Castling)
}

class FenSyntaxException(message: String) : IllegalArgumentException(message)
