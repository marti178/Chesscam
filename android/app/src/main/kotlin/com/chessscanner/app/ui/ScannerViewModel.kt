package com.chessscanner.app.ui

import android.app.Application
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import com.chessscanner.app.camera.FrameAnalyzer
import com.chessscanner.app.classify.LiteRtSquareClassifier
import com.chessscanner.core.board.Pt
import com.chessscanner.core.fen.EditablePosition
import com.chessscanner.core.fusion.BoardFrame
import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece
import com.chessscanner.core.orientation.GridMapping
import com.chessscanner.core.pipeline.ScanPipeline
import com.chessscanner.core.pipeline.ScanResult
import com.chessscanner.core.pipeline.ScanStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Lo que muestra la pantalla mientras escanea. */
data class LiveUi(
    val status: ScanStatus = ScanStatus.SEARCHING,
    /** Los 81 cruces en la imagen de análisis (derecha), o null. */
    val lattice: List<Pt>? = null,
    val boardGood: Boolean = false,
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    /** Celdas de la imagen (fila * 8 + col) dudosas, para marcarlas sobre la cámara. */
    val doubtfulImageCells: Set<Int> = emptySet(),
    /** Piezas en coordenadas del tablero físico y si cada una es confiable. */
    val grid: List<Piece> = List(64) { Piece.EMPTY },
    val confident: List<Boolean> = List(64) { false },
    val hasFusion: Boolean = false,
    val autoRotation: Int = 0,
    val orientationDoubtful: Boolean = true,
    val orientationReason: String? = null,
    val stable: Boolean = false,
    val fps: Double = 0.0,
    val frameMs: Double = 0.0,
    val classifyMs: Double = 0.0,
)

class ScannerViewModel(app: Application) : AndroidViewModel(app) {
    val classifier = LiteRtSquareClassifier.createOrFake(app)
    private val pipeline = ScanPipeline(classifier)
    val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private val _live = MutableStateFlow(LiveUi())
    val live: StateFlow<LiveUi> = _live.asStateFlow()

    private val _frozen = MutableStateFlow<EditablePosition?>(null)
    val frozen: StateFlow<EditablePosition?> = _frozen.asStateFlow()

    /** Giro manual mientras escanea («Girar tablero»), sumado a la rotación automática. */
    private val _rotationOffset = MutableStateFlow(0)
    val rotationOffset: StateFlow<Int> = _rotationOffset.asStateFlow()

    @Volatile private var resetRequested = false
    private var lastFrameAt = 0L
    private var fps = 0.0
    private var frameCount = 0

    val analyzer = FrameAnalyzer(pipeline, paused = { _frozen.value != null }) { result, ms ->
        if (resetRequested) {
            resetRequested = false
            pipeline.reset()
            return@FrameAnalyzer
        }
        publish(result, ms)
    }

    private val powerManager = app.getSystemService(PowerManager::class.java)
    private val thermalListener: Any? = if (Build.VERSION.SDK_INT >= 29) {
        PowerManager.OnThermalStatusChangedListener { applyThermal(it) }.also {
            powerManager.addThermalStatusListener(it)
            applyThermal(powerManager.currentThermalStatus)
        }
    } else null

    /** Baja la frecuencia de clasificación si el teléfono se calienta (SPEC «Rendimiento»). */
    private fun applyThermal(status: Int) {
        pipeline.minClassifyIntervalMs = when {
            status >= PowerManager.THERMAL_STATUS_SEVERE -> 1000
            status >= PowerManager.THERMAL_STATUS_MODERATE -> 300
            else -> 0
        }
    }

    private fun publish(r: ScanResult, frameMs: Double) {
        val now = System.nanoTime()
        if (lastFrameAt != 0L) {
            val inst = 1e9 / (now - lastFrameAt)
            fps = if (fps == 0.0) inst else 0.9 * fps + 0.1 * inst
        }
        lastFrameAt = now
        if (++frameCount % 30 == 0) {
            // Para medir en el teléfono: adb logcat -s SCAN
            Log.i(
                "SCAN",
                "fps=%.1f frame=%.1fms detect=%.1fms classify=%.1fms clasificó=%b estado=%s img=%dx%d intervaloTérmico=%dms".format(
                    fps, frameMs, r.detectMs, r.classifyMs, r.classified, r.status, r.imageWidth, r.imageHeight,
                    pipeline.minClassifyIntervalMs,
                ),
            )
        }
        val f = r.fusion
        val doubtfulImage = if (f == null || r.board == null) emptySet() else
            (0 until 64).filter { !f.confident[BoardFrame.boardCell(it / 8, it % 8, r.shift)] }.toSet()
        _live.update { old ->
            LiveUi(
                status = r.status,
                lattice = r.board?.lattice,
                boardGood = r.boardGood,
                imageWidth = r.imageWidth,
                imageHeight = r.imageHeight,
                doubtfulImageCells = doubtfulImage,
                grid = f?.labelsOrEmpty()?.toList() ?: List(64) { Piece.EMPTY },
                confident = f?.confident?.toList() ?: List(64) { false },
                hasFusion = f != null && f.frames > 0,
                autoRotation = r.orientation?.rotation ?: old.autoRotation,
                orientationDoubtful = r.orientation?.doubtful ?: true,
                orientationReason = r.orientation?.reason,
                stable = f?.stable == true,
                fps = fps,
                frameMs = frameMs,
                classifyMs = if (r.classified) r.classifyMs else old.classifyMs,
            )
        }
    }

    /** Rotación que se muestra ahora (automática + giro manual). */
    fun liveRotation(l: LiveUi = _live.value) = (l.autoRotation + _rotationOffset.value) and 3

    fun liveBoard(l: LiveUi): Board = GridMapping.toBoard(l.grid.toTypedArray(), liveRotation(l))

    fun liveDoubtfulSquares(l: LiveUi): Set<Int> {
        if (!l.hasFusion) return emptySet()
        val rot = liveRotation(l)
        return (0 until 64).filter { !l.confident[it] }.map { GridMapping.square(it / 8, it % 8, rot) }.toSet()
    }

    fun freeze() {
        val l = _live.value
        _frozen.value = EditablePosition(
            grid = l.grid,
            doubtfulCells = if (l.hasFusion) (0 until 64).filter { !l.confident[it] }.toSet() else (0 until 64).toSet(),
            rotation = liveRotation(l),
            stable = l.stable,
        )
    }

    fun rescan() {
        _frozen.value = null
        _rotationOffset.value = 0
        resetRequested = true
        _live.value = LiveUi()
    }

    fun rotateLive(quarters: Int) = _rotationOffset.update { (it + quarters) and 3 }
    fun edit(square: Int, piece: Piece) = _frozen.update { it?.edit(square, piece) }
    fun rotateFrozen(quarters: Int) = _frozen.update { it?.rotate(quarters) }
    fun setWhiteToMove(white: Boolean) = _frozen.update { it?.copy(whiteToMove = white) }
    fun toggleCastling(right: Char) = _frozen.update { it?.toggleCastling(right) }

    override fun onCleared() {
        if (Build.VERSION.SDK_INT >= 29 && thermalListener != null) {
            powerManager.removeThermalStatusListener(thermalListener as PowerManager.OnThermalStatusChangedListener)
        }
        // Se cierra en el hilo de análisis, después del último fotograma en curso.
        analysisExecutor.execute { classifier.close() }
        analysisExecutor.shutdown()
    }
}
