package com.chessscanner.core.model

/**
 * Posición en coordenadas de ajedrez. Índice = `rank * 8 + file`, con file 0 = a, rank 0 = fila 1
 * (a1 = 0, h1 = 7, a8 = 56, h8 = 63).
 */
class Board(private val cells: Array<Piece> = Array(64) { Piece.EMPTY }) {
    init { require(cells.size == 64) }

    operator fun get(file: Int, rank: Int): Piece = cells[rank * 8 + file]
    operator fun get(square: Int): Piece = cells[square]
    fun with(square: Int, piece: Piece): Board = Board(cells.copyOf().also { it[square] = piece })
    fun toArray(): Array<Piece> = cells.copyOf()

    override fun equals(other: Any?) = other is Board && cells.contentEquals(other.cells)
    override fun hashCode() = cells.contentHashCode()
    override fun toString() = (7 downTo 0).joinToString("\n") { r ->
        (0..7).joinToString("") { f -> (this[f, r].fen ?: '.').toString() }
    }

    companion object {
        fun squareName(square: Int): String = "${'a' + square % 8}${square / 8 + 1}"
        fun square(name: String): Int {
            require(name.length == 2 && name[0] in 'a'..'h' && name[1] in '1'..'8') { name }
            return (name[1] - '1') * 8 + (name[0] - 'a')
        }
    }
}
