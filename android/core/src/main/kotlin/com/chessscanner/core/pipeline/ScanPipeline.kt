package com.chessscanner.core.pipeline

import com.chessscanner.core.board.BoardDetection
import com.chessscanner.core.board.BoardTracker
import com.chessscanner.core.board.FrameQuality
import com.chessscanner.core.board.QualityIssue
import com.chessscanner.core.classify.SquareClassifier
import com.chessscanner.core.classify.SquareCrops
import com.chessscanner.core.fusion.BoardFrame
import com.chessscanner.core.fusion.FusionEngine
import com.chessscanner.core.fusion.FusionSnapshot
import com.chessscanner.core.model.Piece
import com.chessscanner.core.orientation.Orientation
import com.chessscanner.core.orientation.OrientationResult
import org.opencv.core.Mat

/** Mensajes de estado de la SPEC («Interfaz»), en el orden en que se evalúan. */
enum class ScanStatus(val message: String) {
    SEARCHING("Buscando tablero"),
    CORNER_OUT("Alejate, falta una esquina"),
    TOO_SMALL("Acercate"),
    TOO_OBLIQUE("Poné el teléfono más de frente"),
    BLURRY("Imagen movida"),
    OCCLUDED("Algo tapa el tablero"),
    ANALYZING("Analizando…"),
    MOVE_FOR_DOUBTFUL("Mové el teléfono para ver las casillas marcadas"),
    READY("Posición lista"),
}

class ScanResult(
    val status: ScanStatus,
    /** Tablero en la imagen de este fotograma (para dibujar la retícula), o null. */
    val board: BoardDetection?,
    /** true si el fotograma pasó el control de calidad (retícula verde); false = ámbar. */
    val boardGood: Boolean,
    /** Esquina del tablero persistente que está en TL de la imagen (para ubicar casillas en la imagen). */
    val shift: Int,
    /** Estado fusionado en coordenadas del tablero; null si todavía no hay tablero. */
    val fusion: FusionSnapshot?,
    val orientation: OrientationResult?,
    val classified: Boolean,
    val detectMs: Double,
    val classifyMs: Double,
    val imageWidth: Int,
    val imageHeight: Int,
)

/**
 * El pipeline completo de un fotograma, sin nada de Android: seguimiento/detección → control de
 * calidad → recortes (contrato ML) → clasificador → fusión → orientación. Lo usan la app y las
 * herramientas de escritorio. No es thread-safe: un solo hilo de análisis.
 */
class ScanPipeline(var classifier: SquareClassifier) {
    private val tracker = BoardTracker()
    private val quality = FrameQuality()
    private val crops = SquareCrops()
    private val input = SquareCrops.newInputBuffer()
    val fusion = FusionEngine()
    private val luma = DoubleArray(64)
    private var lumaFrames = 0
    private var lastClassifyMs = Long.MIN_VALUE / 2

    /** Intervalo mínimo entre clasificaciones (la app lo sube si el teléfono se calienta). */
    @Volatile var minClassifyIntervalMs: Long = 0

    /** Si las 64 casillas ya son confiables y el tablero está quieto, se clasifica como mucho cada tanto. */
    var idleClassifyIntervalMs: Long = 500

    fun reset() {
        tracker.reset()
        fusion.reset()
        luma.fill(0.0)
        lumaFrames = 0
    }

    /**
     * Procesa un fotograma.
     * @param gray imagen en gris (uint8), derecha (ya rotada según el sensor).
     * @param rgb se llama solo si hace falta clasificar: la misma imagen en RGB uint8.
     */
    fun process(gray: Mat, rgb: () -> Mat, timestampMs: Long): ScanResult {
        val t0 = System.nanoTime()
        val upd = tracker.update(gray)
        if (upd.jumped) {
            fusion.reset()
            luma.fill(0.0)
            lumaFrames = 0
        }
        val board = upd.board
        val detectMs = (System.nanoTime() - t0) / 1e6
        val w = gray.cols()
        val h = gray.rows()
        if (board == null) {
            val snap = if (fusion.snapshot().frames > 0) fusion.snapshot(timestampMs) else null
            return ScanResult(ScanStatus.SEARCHING, null, false, upd.shift, snap, snap?.let(::orient), false, detectMs, 0.0, w, h)
        }

        val q = quality.assess(gray, board, upd.motionSquares, upd.support)
        val before = fusion.snapshot(timestampMs)
        if (!q.ok) {
            val status = when (q.issue) {
                QualityIssue.CORNER_OUT -> ScanStatus.CORNER_OUT
                QualityIssue.TOO_SMALL -> ScanStatus.TOO_SMALL
                QualityIssue.TOO_OBLIQUE -> ScanStatus.TOO_OBLIQUE
                QualityIssue.OCCLUDED -> ScanStatus.OCCLUDED
                else -> ScanStatus.BLURRY
            }
            return ScanResult(status, board, false, upd.shift, before.takeIf { it.frames > 0 }, orient(before), false, detectMs, 0.0, w, h)
        }

        val allConfident = before.frames > 0 && before.doubtfulCount == 0
        val sinceLast = timestampMs - lastClassifyMs
        val skip = sinceLast < minClassifyIntervalMs ||
            (allConfident && upd.motionSquares < 0.05 && sinceLast < idleClassifyIntervalMs)
        var classifyMs = 0.0
        if (!skip) {
            val t1 = System.nanoTime()
            val canon = crops.rectify(rgb(), board.corners)
            crops.fillInput(canon, input)
            val logits = classifier.classify(input)
            classifyMs = (System.nanoTime() - t1) / 1e6
            lastClassifyMs = timestampMs

            val boardLogits = Array(64) { FloatArray(0) }
            BoardFrame.toBoard(logits, upd.shift, boardLogits)
            val weights = BoardFrame.toBoard(q.cellWeights, upd.shift)
            fusion.addFrame(boardLogits, q.frameWeight, weights, timestampMs)

            val l = BoardFrame.toBoard(crops.cellLuma(canon), upd.shift)
            lumaFrames++
            val k = 1.0 / minOf(lumaFrames, 20)
            for (i in 0 until 64) luma[i] += k * (l[i] - luma[i])
        }
        val snap = fusion.snapshot(timestampMs)
        val status = when {
            snap.stable -> ScanStatus.READY
            snap.frames < 5 -> ScanStatus.ANALYZING
            snap.doubtfulCount > 0 -> ScanStatus.MOVE_FOR_DOUBTFUL
            else -> ScanStatus.ANALYZING
        }
        return ScanResult(status, board, true, upd.shift, snap, orient(snap), !skip, detectMs, classifyMs, w, h)
    }

    private fun orient(snap: FusionSnapshot): OrientationResult {
        val labels = snap.labelsOrEmpty()
        val empty = BooleanArray(64) { labels[it] == Piece.EMPTY && snap.confident[it] }
        val parity = if (lumaFrames > 0) Orientation.darkParity(luma, empty) else null
        return Orientation.resolve(labels, parity)
    }
}
