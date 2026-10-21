package com.chessscanner.core

import com.chessscanner.core.fusion.BoardFrame
import com.chessscanner.core.fusion.FusionConfig
import com.chessscanner.core.fusion.FusionEngine
import com.chessscanner.core.model.Piece
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class FusionTest {
    /** Logits sintéticos: la clase verdadera con probabilidad ~[p], el resto repartido. */
    private fun frame(truth: Array<Piece>, p: Double = 0.8, noise: Random? = null, override: Map<Int, Piece> = emptyMap()): Array<FloatArray> =
        Array(64) { i ->
            val cls = override[i] ?: truth[i]
            val other = (1 - p) / 12
            FloatArray(Piece.COUNT) { c ->
                val base = kotlin.math.ln(if (c == cls.ordinal) p else other)
                (base + (noise?.nextDouble(-0.3, 0.3) ?: 0.0)).toFloat()
            }
        }

    private val truth = Array(64) { i ->
        when (i) { 0 -> Piece.BR; 4 -> Piece.BK; 60 -> Piece.WK; in 48..55 -> Piece.WP; else -> Piece.EMPTY }
    }

    @Test fun convergesAndBecomesStableAfterOneSecond() {
        val f = FusionEngine()
        var t = 0L
        repeat(10) { f.addFrame(frame(truth, noise = Random(it)), 1.0, null, t); t += 50 }
        val s = f.snapshot(t)
        assertEquals(truth.toList(), s.labels.toList())
        assertEquals(0, s.doubtfulCount)
        assertFalse("no pasó 1 s con todo confiable", s.stable)
        repeat(25) { f.addFrame(frame(truth, noise = Random(100 + it)), 1.0, null, t); t += 50 }
        assertTrue(f.snapshot(t).stable)
    }

    @Test fun oneFrameIsNotEnough() {
        val f = FusionEngine()
        f.addFrame(frame(truth, p = 0.99), 1.0, null, 0)
        val s = f.snapshot(0)
        assertTrue(s.labels.all { it == null })
        assertEquals(64, s.doubtfulCount)
    }

    @Test fun occludedSquareStaysDoubtfulUntilSeen() {
        val f = FusionEngine()
        val weights = DoubleArray(64) { if (it == 10) 0.0 else 1.0 }
        repeat(20) { f.addFrame(frame(truth), 1.0, weights, it * 50L) }
        val s = f.snapshot()
        assertNull(s.labels[10])
        assertFalse(s.confident[10])
        assertEquals(1, s.doubtfulCount)
        // El usuario mueve el teléfono y la casilla aparece: se suma a lo anterior.
        repeat(6) { f.addFrame(frame(truth), 1.0, null, 1000 + it * 50L) }
        assertEquals(0, f.snapshot().doubtfulCount)
    }

    @Test fun shortGlitchDoesNotFlipASquare() {
        val f = FusionEngine()
        repeat(15) { f.addFrame(frame(truth), 1.0, null, it * 50L) }
        // Una mano tapa e4 dos fotogramas y el clasificador dice "peón negro" con mucha seguridad.
        repeat(2) { f.addFrame(frame(truth, p = 0.95, override = mapOf(36 to Piece.BP)), 1.0, null, 800L + it * 50) }
        assertEquals(Piece.EMPTY, f.snapshot().labels[36])
    }

    @Test fun consistentChangeUpdatesTheSquare() {
        val f = FusionEngine(FusionConfig(changeFrames = 8))
        repeat(30) { f.addFrame(frame(truth), 1.0, null, it * 50L) }
        val moved = mapOf(52 to Piece.EMPTY, 36 to Piece.WP) // e2-e4
        repeat(10) { f.addFrame(frame(truth, override = moved), 1.0, null, 1500L + it * 50) }
        val s = f.snapshot()
        assertEquals(Piece.EMPTY, s.labels[52])
        assertEquals(Piece.WP, s.labels[36])
    }

    @Test fun lowConfidenceFramesNeedMoreEvidence() {
        val f = FusionEngine()
        repeat(3) { f.addFrame(frame(truth, p = 0.3), 0.2, null, it * 50L) }
        assertTrue(f.snapshot().doubtfulCount > 0)
    }

    @Test fun resetForgetsEverything() {
        val f = FusionEngine()
        repeat(10) { f.addFrame(frame(truth), 1.0, null, it * 50L) }
        f.reset()
        val s = f.snapshot()
        assertEquals(0, s.frames)
        assertTrue(s.labels.all { it == null })
    }

    @Test fun boardFrameRotatesWithThePhone() {
        // Con shift 0 es la identidad.
        for (i in 0 until 64) assertEquals(i, BoardFrame.boardCell(i / 8, i % 8, 0))
        // Shift 1: la esquina 1 del tablero (TR) aparece como TL de la imagen.
        assertEquals(7, BoardFrame.boardCell(0, 0, 1))
        assertEquals(63, BoardFrame.boardCell(0, 7, 1))
        // Cuatro giros vuelven al principio, y cada shift es una biyección.
        for (k in 0..3) assertEquals(64, (0 until 64).map { BoardFrame.boardCell(it / 8, it % 8, k) }.toSet().size)
        // La fusión acumula lo mismo aunque el teléfono gire alrededor del tablero.
        val f = FusionEngine()
        repeat(20) { n ->
            val shift = n % 4
            val imageTruth = Array(64) { Piece.EMPTY }
            for (i in 0 until 64) imageTruth[i] = truth[BoardFrame.boardCell(i / 8, i % 8, shift)]
            val boardLogits = Array(64) { FloatArray(0) }
            BoardFrame.toBoard(frame(imageTruth), shift, boardLogits)
            f.addFrame(boardLogits, 1.0, null, n * 50L)
        }
        assertEquals(truth.toList(), f.snapshot().labels.toList())
    }
}
