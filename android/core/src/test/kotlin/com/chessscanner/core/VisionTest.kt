package com.chessscanner.core

import com.chessscanner.core.board.BoardDetector
import com.chessscanner.core.board.BoardTracker
import com.chessscanner.core.board.FrameQuality
import com.chessscanner.core.board.Pt
import com.chessscanner.core.board.QualityIssue
import com.chessscanner.core.classify.FakeSquareClassifier
import com.chessscanner.core.classify.SquareCrops
import com.chessscanner.core.eval.BoardMetrics
import com.chessscanner.core.eval.loadDesktopOpenCv
import com.chessscanner.core.model.Piece
import com.chessscanner.core.pipeline.ScanPipeline
import com.chessscanner.core.pipeline.ScanStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** Detector, seguimiento, calidad y recortes sobre tableros sintéticos (sin imágenes de terceros). */
class VisionTest {
    companion object {
        @BeforeClass @JvmStatic fun load() = loadDesktopOpenCv()

        const val SQ = 60
        const val FRAME = 40

        /** Tablero plano (RGB) con marco, casillas claras/oscuras (a1 oscura abajo a la izquierda) y algunas "piezas". */
        fun flatBoard(pieces: Boolean): Mat {
            val n = 8 * SQ + 2 * FRAME
            val m = Mat(n, n, CvType.CV_8UC3, Scalar(90.0, 60.0, 40.0))
            for (r in 0 until 8) for (c in 0 until 8) {
                val dark = (r + c) % 2 == 1
                val color = if (dark) Scalar(110.0, 70.0, 50.0) else Scalar(225.0, 210.0, 170.0)
                val x = FRAME + c * SQ
                val y = FRAME + r * SQ
                Imgproc.rectangle(m, Point(x.toDouble(), y.toDouble()), Point((x + SQ - 1).toDouble(), (y + SQ - 1).toDouble()), color, -1)
                if (pieces && (r <= 1 || r >= 6)) {
                    val pc = if (r >= 6) Scalar(245.0, 245.0, 240.0) else Scalar(25.0, 25.0, 25.0)
                    Imgproc.circle(m, Point(x + SQ / 2.0, y + SQ / 2.0), SQ / 3, pc, -1)
                }
            }
            return m
        }

        /** Coloca el tablero plano en una escena con perspectiva; devuelve la escena RGB y las esquinas del área de juego. */
        fun scene(flat: Mat, corners: List<Pt>, w: Int = 640, h: Int = 480): Pair<Mat, List<Pt>> {
            val n = flat.cols().toDouble()
            val src = MatOfPoint2f(Point(0.0, 0.0), Point(n, 0.0), Point(n, n), Point(0.0, n))
            // Las esquinas pedidas son las del área de juego: se extrapola el marco.
            val f = FRAME.toDouble()
            val p = 8.0 * SQ
            val play = MatOfPoint2f(Point(f, f), Point(f + p, f), Point(f + p, f + p), Point(f, f + p))
            val dst = MatOfPoint2f(*corners.map { Point(it.x, it.y) }.toTypedArray())
            val hPlay = Imgproc.getPerspectiveTransform(play, dst)
            val out = Mat(h, w, CvType.CV_8UC3, Scalar(170.0, 165.0, 160.0))
            // Fondo con algo de textura.
            val noise = Mat(h, w, CvType.CV_8UC3)
            Core.randn(noise, 0.0, 6.0)
            Core.add(out, noise, out)
            Imgproc.warpPerspective(flat, out, hPlay, Size(w.toDouble(), h.toDouble()), Imgproc.INTER_LINEAR, Core.BORDER_TRANSPARENT)
            src.release(); play.release(); dst.release(); hPlay.release(); noise.release()
            return out to corners
        }

        fun gray(rgb: Mat): Mat = Mat().also { Imgproc.cvtColor(rgb, it, Imgproc.COLOR_RGB2GRAY) }

        val TILTED = listOf(Pt(190.0, 70.0), Pt(470.0, 85.0), Pt(560.0, 430.0), Pt(95.0, 410.0))
    }

    @Test fun detectsSyntheticBoardsWithPerspective() {
        val cases = listOf(
            TILTED,
            listOf(Pt(160.0, 60.0), Pt(480.0, 60.0), Pt(500.0, 420.0), Pt(140.0, 420.0)),
            // Girado ~30°.
            listOf(Pt(300.0, 30.0), Pt(560.0, 180.0), Pt(420.0, 450.0), Pt(140.0, 290.0)),
        )
        for (pieces in listOf(false, true)) for (c in cases) {
            val (rgb, truth) = scene(flatBoard(pieces), c)
            val det = BoardDetector().detect(gray(rgb))
            assertNotNull("no detectó $c (piezas=$pieces)", det)
            val err = BoardMetrics.cornerErrorSquares(det!!.corners, truth)
            assertTrue("error $err casillas en $c", err < 0.15)
            // Orden del contrato: horario y TL→TR arriba.
            val tl = det.corners[0]; val tr = det.corners[1]
            assertTrue(det.corners.indices.all { k ->
                (det.corners[k].y + det.corners[(k + 1) % 4].y) / 2 >= (tl.y + tr.y) / 2 - 1e-9
            })
        }
    }

    @Test fun noBoardNoDetection() {
        val img = Mat(480, 640, CvType.CV_8UC1, Scalar(128.0))
        val noise = Mat(480, 640, CvType.CV_8UC1)
        Core.randn(noise, 0.0, 20.0)
        Core.add(img, noise, img)
        Imgproc.rectangle(img, Point(100.0, 100.0), Point(300.0, 250.0), Scalar(30.0), 4)
        assertNull(BoardDetector().detect(img))
    }

    @Test fun qualityFlagsCutBoardAndSmallBoard() {
        val cut = listOf(Pt(-40.0, 40.0), Pt(420.0, 50.0), Pt(450.0, 440.0), Pt(-20.0, 430.0))
        val (rgb, _) = scene(flatBoard(false), cut)
        val g = gray(rgb)
        val det = BoardDetector().detect(g)
        if (det != null) {
            assertEquals(QualityIssue.CORNER_OUT, FrameQuality().assess(g, det, 0.0, 1.0).issue)
        }
        val small = listOf(Pt(280.0, 200.0), Pt(380.0, 200.0), Pt(385.0, 300.0), Pt(275.0, 300.0))
        val (rgb2, _) = scene(flatBoard(false), small)
        val g2 = gray(rgb2)
        val det2 = BoardDetector().detect(g2)
        if (det2 != null) assertEquals(QualityIssue.TOO_SMALL, FrameQuality().assess(g2, det2, 0.0, 1.0).issue)
        val (rgb3, _) = scene(flatBoard(false), TILTED)
        val g3 = gray(rgb3)
        assertEquals(QualityIssue.NONE, FrameQuality().assess(g3, BoardDetector().detect(g3)!!, 0.0, 1.0).issue)
    }

    @Test fun trackerFollowsSmallMovesAndKeepsCornerIdentity() {
        val tracker = BoardTracker()
        val flat = flatBoard(true)
        var first: List<Pt>? = null
        for (step in 0 until 12) {
            val d = step * 4.0
            val c = TILTED.map { Pt(it.x + d, it.y - d / 2) }
            val (rgb, truth) = scene(flat, c)
            val upd = tracker.update(gray(rgb))
            assertNotNull(upd.board)
            assertEquals(step == 0, upd.jumped)
            assertTrue(BoardMetrics.cornerErrorSquares(upd.board!!.corners, truth) < 0.2)
            if (step > 0) assertTrue("paso $step: movimiento ${upd.motionSquares}", upd.motionSquares < 0.3)
            if (first == null) first = upd.board.corners
        }
        // Otro tablero en otro lugar: salto.
        val far = listOf(Pt(20.0, 20.0), Pt(250.0, 25.0), Pt(260.0, 250.0), Pt(15.0, 240.0))
        val (rgb, _) = scene(flat, far, 640, 480)
        var jumped = false
        repeat(3) { jumped = jumped || tracker.update(gray(rgb)).jumped }
        assertTrue(jumped)
    }

    @Test fun cropTablesMatchContract() {
        assertEquals(listOf(0, 42, 83, 124, 165, 206, 247, 288), (0..7).map { SquareCrops.y0(it) })
        assertEquals(listOf(128, 160, 192, 224, 256, 288, 320, 352), (0..7).map { SquareCrops.y1(it) })
        assertEquals(listOf(0, 40, 80, 120, 160, 192, 224, 256), (0..7).map { SquareCrops.x0(it) })
        assertEquals(listOf(64, 96, 128, 160, 200, 240, 280, 320), (0..7).map { SquareCrops.x1(it) })
    }

    @Test fun cropIsPastedBottomLeftAndMirroredOnTheLeftHalf() {
        val crops = SquareCrops()
        // Canónica con un degradé horizontal: x en el canal R.
        val canon = Mat(SquareCrops.CH, SquareCrops.CW, CvType.CV_8UC3)
        for (y in 0 until SquareCrops.CH) for (x in 0 until SquareCrops.CW) canon.put(y, x, byteArrayOf((x % 256).toByte(), y.toByte(), 7))
        // col 5 (mitad derecha): sin espejo. El píxel (127, 0) del recorte = canon(y1-1, x0).
        var c = crops.crop(canon, 7, 5)
        assertEquals(SquareCrops.x0(5), c.get(127, 0)[0].toInt())
        assertEquals(0.0, c.get(0, 0)[0], 0.0) // relleno negro arriba (fila 7: h = 64)
        // col 2 (mitad izquierda): espejado, el píxel (127, 0) = canon(y1-1, x1-1).
        c = crops.crop(canon, 7, 2)
        assertEquals(SquareCrops.x1(2) - 1, c.get(127, 0)[0].toInt())
        // Relleno a la derecha del parche (w = 48 < 64).
        assertEquals(0.0, c.get(127, 50)[0], 0.0)
        // La casilla propiamente dicha cae en recorte[96:128, 0:32].
        c = crops.crop(canon, 0, 4)
        assertEquals(SquareCrops.MY + 0, c.get(96, 0)[1].toInt())
    }

    @Test fun rectifyMapsCornersToCanonicalPoints() {
        val (rgb, truth) = scene(flatBoard(false), TILTED)
        val crops = SquareCrops()
        val canon = crops.rectify(rgb, truth)
        assertEquals(SquareCrops.CW, canon.cols())
        assertEquals(SquareCrops.CH, canon.rows())
        // a1 (fila 7, col 0) oscura y h1 (fila 7, col 7) clara en el plano sintético.
        val luma = crops.cellLuma(canon)
        assertTrue(luma[56] < luma[63])
        assertTrue(luma[0] > luma[1])
    }

    @Test fun losingTheBoardNeverShowsStalePieces() {
        val pipeline = ScanPipeline(FakeSquareClassifier())
        val (rgb, _) = scene(flatBoard(true), TILTED)
        val g = gray(rgb)
        var t = 0L
        repeat(30) { pipeline.process(g, { rgb }, t); t += 50 }
        assertTrue(pipeline.fusion.snapshot().labels.any { it != null && it != Piece.EMPTY })

        // Sin tablero: no se devuelve ninguna posición, ni siquiera en el primer fotograma.
        val blank = Mat(480, 640, CvType.CV_8UC3, Scalar(150.0, 150.0, 150.0))
        val gBlank = gray(blank)
        val first = pipeline.process(gBlank, { blank }, t)
        assertEquals(ScanStatus.SEARCHING, first.status)
        assertNull(first.fusion)
        assertNull(first.orientation)
        // Perdido del todo (más de 15 fotogramas): la fusión se reinicia.
        repeat(20) { t += 50; assertNull(pipeline.process(gBlank, { blank }, t).fusion) }
        assertEquals(0, pipeline.fusion.snapshot().frames)

        // Un tablero vacío en otro lugar: arranca de cero, sin las "piezas" de antes.
        val other = listOf(Pt(150.0, 60.0), Pt(500.0, 70.0), Pt(520.0, 420.0), Pt(130.0, 410.0))
        val (rgb2, _) = scene(flatBoard(false), other)
        val g2 = gray(rgb2)
        t += 50
        val r = pipeline.process(g2, { rgb2 }, t)
        assertTrue(r.fusion == null || r.fusion!!.labels.all { it == null || it == Piece.EMPTY })
        repeat(10) { t += 50; pipeline.process(g2, { rgb2 }, t) }
        assertTrue(pipeline.fusion.snapshot().labels.all { it == null || it == Piece.EMPTY })
    }

    @Test fun jumpToAnotherBoardResetsFusion() {
        val pipeline = ScanPipeline(FakeSquareClassifier())
        val (rgb, _) = scene(flatBoard(true), TILTED)
        val g = gray(rgb)
        var t = 0L
        repeat(20) { pipeline.process(g, { rgb }, t); t += 50 }
        // Salto directo (sin fotogramas en blanco) a otro tablero, vacío, lejos del anterior.
        val far = listOf(Pt(20.0, 20.0), Pt(300.0, 25.0), Pt(310.0, 300.0), Pt(15.0, 290.0))
        val (rgb2, _) = scene(flatBoard(false), far)
        val g2 = gray(rgb2)
        var sawReset = false
        repeat(20) {
            t += 50
            val r = pipeline.process(g2, { rgb2 }, t)
            if (r.fusion == null || r.fusion!!.frames <= 1) sawReset = true
        }
        assertTrue("no se reinició la fusión al cambiar de tablero", sawReset)
        assertTrue(pipeline.fusion.snapshot().labels.all { it == null || it == Piece.EMPTY })
    }

    @Test fun pipelineRunsEndToEndWithFakeClassifier() {
        val pipeline = ScanPipeline(FakeSquareClassifier())
        val (rgb, _) = scene(flatBoard(true), TILTED)
        val g = gray(rgb)
        var last = pipeline.process(g, { rgb }, 0)
        for (i in 1..40) last = pipeline.process(g, { rgb }, i * 50L)
        assertTrue(last.boardGood)
        val labels = last.fusion!!.labelsOrEmpty()
        // Las filas del medio están vacías y las de los costados tienen "piezas".
        assertTrue((16 until 48).count { labels[it] == Piece.EMPTY } >= 28)
        assertTrue((0 until 16).count { labels[it] != Piece.EMPTY } >= 12)
        assertEquals(ScanStatus.READY, last.status)
        // Blancas abajo en el plano sintético y a1 oscura: rotación 0.
        assertEquals(0, last.orientation!!.rotation)
    }
}
