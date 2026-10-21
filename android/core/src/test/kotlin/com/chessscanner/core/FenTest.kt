package com.chessscanner.core

import com.chessscanner.core.fen.Castling
import com.chessscanner.core.fen.Fen
import com.chessscanner.core.fen.FenSyntaxException
import com.chessscanner.core.fen.Validation
import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FenTest {
    private val start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    @Test fun roundTripStartPosition() {
        val parsed = Fen.parse("$start w KQkq - 0 1")
        assertEquals(start, Fen.placement(parsed.board))
        assertEquals(Piece.WR, parsed.board[Board.square("a1")])
        assertEquals(Piece.BK, parsed.board[Board.square("e8")])
        assertEquals(Piece.WP, parsed.board[Board.square("e2")])
        assertEquals("$start w KQkq - 0 1", Fen.build(parsed.board, true, parsed.castling))
    }

    @Test fun roundTripVariousPositions() {
        for (p in listOf(
            "8/8/8/8/8/8/8/8",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8",
            "4k3/8/8/8/8/8/8/4K2R",
        )) assertEquals(p, Fen.placement(Fen.parse(p).board))
    }

    @Test fun buildUsesFixedFieldsAndTurn() {
        val b = Fen.parse("4k3/8/8/8/8/8/8/4K3").board
        assertEquals("4k3/8/8/8/8/8/8/4K3 b - - 0 1", Fen.build(b, false, Castling.NONE))
    }

    @Test fun lichessUrlReplacesSpaces() {
        assertEquals(
            "https://lichess.org/editor/$start" + "_w_KQkq_-_0_1",
            Fen.lichessEditorUrl("$start w KQkq - 0 1"),
        )
    }

    @Test fun syntaxErrorsHaveReasons() {
        fun reason(fen: String) = try { Fen.parse(fen); null } catch (e: FenSyntaxException) { e.message }
        assertTrue(reason("8/8/8")!!.contains("8 filas"))
        assertTrue(reason("p8/8/8/8/8/8/8/8")!!.contains("más de 8"))
        assertTrue(reason("9/8/8/8/8/8/8/8")!!.contains("'9'"))
        assertTrue(reason("7/8/8/8/8/8/8/8")!!.contains("7 casillas"))
        assertTrue(reason("x7/8/8/8/8/8/8/8")!!.contains("'x'"))
        assertTrue(reason("$start x")!!.contains("turno"))
        assertTrue(reason("$start w KK")!!.contains("enroques"))
    }

    @Test fun castlingSuggestedOnlyWhenKingAndRookHome() {
        assertEquals(Castling(true, true, true, true), Castling.possible(Fen.parse(start).board))
        val c = Castling.possible(Fen.parse("r3k3/8/8/8/8/8/8/4K2R").board)
        assertEquals(Castling(whiteKing = true, whiteQueen = false, blackKing = false, blackQueen = true), c)
        // Rey negro en e8 pero torres corridas, rey blanco fuera de e1: nada.
        assertEquals(Castling.NONE, Castling.possible(Fen.parse("1r2k1r1/8/8/8/8/8/8/3K3R").board))
        assertEquals("Kq", c.toFen())
        assertEquals("-", Castling.NONE.toFen())
        assertEquals("K", Castling(true, true, false, false).restrictTo(Castling(true, false, true, true)).toFen())
    }

    @Test fun validStartHasNoProblems() {
        assertTrue(Validation.problems("$start w KQkq - 0 1").isEmpty())
    }

    @Test fun validationReasons() {
        assertEquals(listOf("Falta el rey blanco", "Falta el rey negro"), Validation.problems("8/8/8/8/8/8/8/8"))
        assertTrue(Validation.problems("4k3/8/8/8/8/8/8/3KK3").contains("Hay 2 reyes blancos"))
        val pawn = Validation.problems("P3k3/8/8/8/8/8/8/4K2p")
        assertEquals(listOf("Peón en la fila 1 u 8: a8, h1"), pawn)
        val many = Validation.problems("4k3/8/8/8/8/QQQQQQQQ/QQQQQQQQ/QQQQKQQQ")
        assertTrue(many.contains("Hay 24 piezas blancas (máximo 16)"))
        assertFalse(many.any { it.contains("negras") })
        assertEquals(listOf("FEN mal formado: debe tener 8 filas y tiene 1"), Validation.problems("8"))
    }
}
