package com.chessscanner.core

import com.chessscanner.core.fen.EditablePosition
import com.chessscanner.core.fen.Fen
import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece
import com.chessscanner.core.orientation.GridMapping
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EditablePositionTest {
    private val start = Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").board
    private fun grid(board: Board, rot: Int) = List(64) { board[GridMapping.square(it / 8, it % 8, rot)] }

    @Test fun fenAndCastlingFromFrozenGrid() {
        val p = EditablePosition(grid(start, 2), emptySet(), rotation = 2, stable = true)
        assertEquals("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", p.fen)
        assertTrue(p.clean)
        assertEquals("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b Kkq - 0 1",
            p.copy(whiteToMove = false).toggleCastling('Q').fen)
    }

    @Test fun rotateKeepsEditsOnThePhysicalSquare() {
        var p = EditablePosition(grid(start, 0), setOf(GridMapping.cell(Board.square("e2"), 0)), rotation = 0)
        assertEquals(setOf(Board.square("e2")), p.doubtfulSquares)
        p = p.edit(Board.square("e2"), Piece.EMPTY).edit(Board.square("e4"), Piece.WP)
        assertTrue(p.doubtfulCells.isEmpty())
        val turned = p.rotate(2)
        // La misma celda física ahora se llama distinto, pero no se perdió.
        assertEquals(Piece.WP, turned.board[Board.square("d5")])
        assertEquals(Piece.WP, turned.rotate(2).board[Board.square("e4")])
    }

    @Test fun castlingOnlyWhenPossibleAndProblemsBlockClean() {
        val p = EditablePosition(grid(start, 0), emptySet(), 0, stable = true).edit(Board.square("h1"), Piece.EMPTY)
        assertEquals("Qkq", p.castling.toFen())
        val bad = p.edit(Board.square("e1"), Piece.EMPTY)
        assertEquals(listOf("Falta el rey blanco"), bad.problems)
        assertFalse(bad.clean)
        assertFalse(EditablePosition(grid(start, 0), emptySet(), 0, stable = false).clean)
    }
}
