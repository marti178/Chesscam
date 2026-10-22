package com.chessscanner.core

import com.chessscanner.core.board.BoardDetector
import com.chessscanner.core.eval.BoardLabels
import com.chessscanner.core.eval.BoardMetrics
import com.chessscanner.core.eval.TOLERANCE_SQUARES
import com.chessscanner.core.eval.loadDesktopOpenCv
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Regresión del detector con diagramas de libro (casillas rayadas, texto alrededor, marco doble):
 * `eval/books`, esquinas exactas. El 22‑10‑2026 detecta bien 10 de 12 (antes de la apertura
 * morfológica y los pares de picos de ángulo, 1 de 12).
 */
class BooksDetectorTest {
    companion object {
        @BeforeClass @JvmStatic fun load() = loadDesktopOpenCv()
    }

    @Test fun detectsMostBookDiagrams() {
        val dir = System.getProperty("eval.dir")?.let { File(it, "books") }
        assumeTrue("sin eval/books", dir?.isDirectory == true)
        val labels = BoardLabels.read(File(dir, "boards.tsv"))
        val detector = BoardDetector()
        val results = labels.entries.sortedBy { it.key }.map { (name, label) ->
            val img = Imgcodecs.imread(File(dir, name).path)
            val gray = Mat().also { Imgproc.cvtColor(img, it, Imgproc.COLOR_BGR2GRAY) }
            val det = detector.detect(gray)
            val err = if (det == null || label == null) null else BoardMetrics.cornerErrorSquares(det.corners, label.corners)
            println("$name: ${err?.let { "%.2f casillas".format(it) } ?: "no detectado"}")
            err != null && err <= TOLERANCE_SQUARES
        }
        assertTrue("diagramas bien detectados: ${results.count { it }} / ${results.size}", results.count { it } >= 10)
    }
}
