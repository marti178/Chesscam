package com.chessscanner.core

import com.chessscanner.core.fen.Fen
import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece
import com.chessscanner.core.orientation.GridMapping
import com.chessscanner.core.orientation.Orientation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OrientationTest {
    private val start = Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").board

    /** Retícula (fila * 8 + col) que se vería con el tablero puesto con esa rotación. */
    private fun gridOf(board: Board, rotation: Int): Array<Piece> =
        Array(64) { board[GridMapping.square(it / 8, it % 8, rotation)] }

    @Test fun mappingIsBijectiveAndPutsA1InTheRightCorner() {
        val corners = mapOf(0 to (7 to 0), 1 to (0 to 0), 2 to (0 to 7), 3 to (7 to 7))
        for (rot in 0..3) {
            val squares = (0 until 64).map { GridMapping.square(it / 8, it % 8, rot) }.toSet()
            assertEquals(64, squares.size)
            val (r, c) = corners.getValue(rot)
            assertEquals(Board.square("a1"), GridMapping.square(r, c, rot))
            assertEquals((r + c) % 2, GridMapping.a1Parity(rot))
            assertEquals(r * 8 + c, GridMapping.cell(Board.square("a1"), rot))
        }
    }

    @Test fun eachRotationIsNinetyDegreesClockwiseFromThePrevious() {
        // Girar la imagen 90° horario lleva la celda (r, c) a (c, 7 - r).
        for (rot in 0..3) for (i in 0 until 64) {
            val r = i / 8; val c = i % 8
            assertEquals(GridMapping.square(r, c, rot), GridMapping.square(c, 7 - r, rot + 1))
        }
    }

    @Test fun toBoardInvertsTheView() {
        for (rot in 0..3) assertEquals(start, GridMapping.toBoard(gridOf(start, rot), rot))
    }

    @Test fun resolvesStartPositionFromAnySide() {
        for (rot in 0..3) {
            val grid = gridOf(start, rot)
            val parity = GridMapping.a1Parity(rot)
            val r = Orientation.resolve(grid, parity)
            assertEquals("rotación $rot", rot, r.rotation)
            assertFalse(r.doubtful)
            assertEquals(start, GridMapping.toBoard(grid, r.rotation))
        }
    }

    @Test fun darkSquaresRuleOutTwoOfFourRotations() {
        // Posición de medio juego con blancas del lado de la cámara.
        val b = Fen.parse("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R").board
        val grid = gridOf(b, 0)
        assertEquals(0, Orientation.resolve(grid, 1).rotation)
        // Con la otra paridad, la única opción coherente gira 90°.
        assertTrue(Orientation.resolve(grid, 0).rotation in listOf(1, 3))
    }

    @Test fun fewPiecesIsDoubtful() {
        val b = Fen.parse("8/8/3k4/8/8/4K3/8/8").board
        val r = Orientation.resolve(gridOf(b, 0), 1)
        assertTrue(r.doubtful)
        assertTrue(Orientation.resolve(gridOf(start, 0), null).doubtful)
    }

    @Test fun darkParityFromLuminance() {
        // Celdas con (fila + col) par oscuras.
        val luma = DoubleArray(64) { if ((it / 8 + it % 8) % 2 == 0) 60.0 else 180.0 }
        assertEquals(0, Orientation.darkParity(luma))
        val inverted = DoubleArray(64) { 240.0 - luma[it] }
        assertEquals(1, Orientation.darkParity(inverted))
        assertNull(Orientation.darkParity(DoubleArray(64) { 120.0 }))
        // Piezas blancas sobre casillas oscuras no la engañan si se sabe cuáles están vacías.
        val noisy = luma.copyOf()
        val empty = BooleanArray(64) { true }
        for (i in 0 until 64) if ((i / 8 + i % 8) % 2 == 0 && i < 40) { noisy[i] = 250.0; empty[i] = false }
        assertEquals(0, Orientation.darkParity(noisy, empty))
    }
}
