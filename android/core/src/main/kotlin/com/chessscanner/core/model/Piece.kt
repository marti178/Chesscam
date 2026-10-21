package com.chessscanner.core.model

/**
 * Las 13 clases del clasificador, en el orden de `ml/CONTRACT.md` §5 (índice = salida del modelo).
 * [fen] es el carácter FEN; para [EMPTY] no hay carácter.
 */
enum class Piece(val fen: Char?, val isWhite: Boolean?) {
    EMPTY(null, null),
    WP('P', true), WN('N', true), WB('B', true), WR('R', true), WQ('Q', true), WK('K', true),
    BP('p', false), BN('n', false), BB('b', false), BR('r', false), BQ('q', false), BK('k', false);

    val isPiece: Boolean get() = this != EMPTY

    companion object {
        const val COUNT = 13
        private val byChar = entries.filter { it.fen != null }.associateBy { it.fen!! }
        fun fromFen(c: Char): Piece? = byChar[c]
        fun fromIndex(i: Int): Piece = entries[i]
    }
}
