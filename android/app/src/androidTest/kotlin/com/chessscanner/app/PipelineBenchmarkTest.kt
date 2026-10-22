package com.chessscanner.app

import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.chessscanner.app.classify.LiteRtSquareClassifier
import com.chessscanner.core.board.BoardDetector
import com.chessscanner.core.board.BoardTracker
import com.chessscanner.core.board.FrameQuality
import com.chessscanner.core.classify.SquareCrops
import com.chessscanner.core.pipeline.ScanPipeline
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * Latencia por etapa en el teléfono, sobre la imagen dorada achicada a 640 px (tamaño del fotograma de
 * análisis). Resultados en logcat con la etiqueta `BENCH`. Necesita `-PgoldenDir`.
 */
@RunWith(AndroidJUnit4::class)
class PipelineBenchmarkTest {
    companion object {
        @BeforeClass @JvmStatic fun load() { check(OpenCVLoader.initLocal()) }
        private const val TAG = "BENCH"
    }

    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    private fun report(name: String, warmup: Int = 5, runs: Int = 40, block: () -> Unit) {
        repeat(warmup) { block() }
        val t = DoubleArray(runs) { val t0 = System.nanoTime(); block(); (System.nanoTime() - t0) / 1e6 }
        t.sort()
        val msg = "%-28s mediana %6.1f ms  p90 %6.1f ms  mín %6.1f ms".format(name, t[runs / 2], t[runs * 9 / 10], t[0])
        Log.i(TAG, msg)
        println("$TAG $msg")
    }

    @Test fun stageLatencies() {
        val case = "full_1200x800"
        assumeTrue("sin vectores dorados (-PgoldenDir)", testAssets.list("golden")?.contains(case) == true)
        Log.i(TAG, "dispositivo ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE}, ${Build.SOC_MODEL}")

        val bmp = BitmapFactory.decodeStream(testAssets.open("golden/$case/image.png"))
        val rgba = Mat().also { Utils.bitmapToMat(bmp, it) }
        val big = Mat().also { Imgproc.cvtColor(rgba, it, Imgproc.COLOR_RGBA2RGB) }
        val k = 640.0 / max(big.cols(), big.rows())
        val rgb = Mat().also { Imgproc.resize(big, it, Size(), k, k, Imgproc.INTER_AREA) }
        val gray = Mat().also { Imgproc.cvtColor(rgb, it, Imgproc.COLOR_RGB2GRAY) }
        Log.i(TAG, "fotograma ${rgb.cols()}×${rgb.rows()}")

        val detector = BoardDetector()
        val det = checkNotNull(detector.detect(gray)) { "no detectó el tablero" }
        report("detector (completo)") { detector.detect(gray) }

        val tracker = BoardTracker()
        tracker.update(gray)
        report("seguimiento (flujo óptico)") { tracker.update(gray) }

        val quality = FrameQuality()
        report("control de calidad") { quality.assess(gray, det, 0.0, 1.0) }

        val crops = SquareCrops()
        val input = SquareCrops.newInputBuffer()
        report("rectificar + 64 recortes") { crops.fillInput(crops.rectify(rgb, det.corners), input) }

        for (threads in listOf(1, 2, 4, 6)) {
            val clf = LiteRtSquareClassifier.create(appContext, threads)
            report("clasificador fp16 $threads hilo(s)", warmup = 5, runs = 30) { clf.classify(input) }
            clf.close()
        }
        // Variantes de opciones del intérprete, para entender de qué depende.
        val fd = appContext.assets.openFd(LiteRtSquareClassifier.MODEL_ASSET)
        val model = java.io.FileInputStream(fd.fileDescriptor).channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val out = Array(64) { FloatArray(13) }
        for ((label, opts) in listOf(
            "sin XNNPACK 4 hilos" to org.tensorflow.lite.Interpreter.Options().setNumThreads(4).setUseXNNPACK(false),
            "XNNPACK explícito 4 hilos" to org.tensorflow.lite.Interpreter.Options().setNumThreads(4).setUseXNNPACK(true),
            "XNNPACK explícito 1 hilo" to org.tensorflow.lite.Interpreter.Options().setNumThreads(1).setUseXNNPACK(true),
        )) {
            val itp = org.tensorflow.lite.Interpreter(model, opts)
            report(label, warmup = 5, runs = 20) { input.rewind(); itp.run(input, out) }
            itp.close()
        }

        val clf = LiteRtSquareClassifier.create(appContext, 4)
        val pipeline = ScanPipeline(clf)
        pipeline.idleClassifyIntervalMs = 0
        var t = 0L
        report("pipeline completo (clasifica)", warmup = 5, runs = 30) {
            pipeline.process(gray, { rgb }, t); t += 33
        }
        clf.close()
    }
}
