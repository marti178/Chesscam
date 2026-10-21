package com.chessscanner.app.camera

import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.chessscanner.core.pipeline.ScanPipeline
import com.chessscanner.core.pipeline.ScanResult

/**
 * Analizador de CameraX (estrategia KEEP_ONLY_LATEST): convierte el fotograma y lo pasa por el
 * pipeline en el hilo de análisis. [onResult] recibe cada resultado y el tiempo total del fotograma.
 */
class FrameAnalyzer(
    private val pipeline: ScanPipeline,
    private val paused: () -> Boolean,
    private val onResult: (ScanResult, Double) -> Unit,
) : ImageAnalysis.Analyzer {
    private val frame = YuvFrame()

    override fun analyze(image: ImageProxy) {
        try {
            if (paused()) return
            val t0 = System.nanoTime()
            frame.load(image)
            val result = pipeline.process(frame.gray, { frame.rgb() }, SystemClock.elapsedRealtime())
            onResult(result, (System.nanoTime() - t0) / 1e6)
        } finally {
            image.close()
        }
    }
}
