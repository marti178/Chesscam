package com.chessscanner.app.classify

import android.content.Context
import android.util.Log
import com.chessscanner.core.classify.FakeSquareClassifier
import com.chessscanner.core.classify.SquareClassifier
import com.chessscanner.core.model.Piece
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Clasificador real con LiteRT (CPU, XNNPACK por defecto) según ml/CONTRACT.md §4–5:
 * entrada float32 [64, 128, 64, 3], salida float32 [64, 13] logits.
 */
class LiteRtSquareClassifier(model: MappedByteBuffer, threads: Int = 4) : SquareClassifier {
    override val name = "LiteRT v1"
    override val isFake = false
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(threads))
    private val output = Array(64) { FloatArray(Piece.COUNT) }

    init {
        val inShape = interpreter.getInputTensor(0).shape().toList()
        val outShape = interpreter.getOutputTensor(0).shape().toList()
        require(inShape == listOf(64, 128, 64, 3)) { "Entrada inesperada del modelo: $inShape" }
        require(outShape == listOf(64, Piece.COUNT)) { "Salida inesperada del modelo: $outShape" }
    }

    override fun classify(input: ByteBuffer): Array<FloatArray> {
        input.rewind()
        interpreter.run(input, output)
        return Array(64) { output[it].copyOf() }
    }

    override fun close() = interpreter.close()

    companion object {
        const val MODEL_ASSET = "chess_square_classifier.tflite"
        const val LABELS_ASSET = "labels.txt"
        private const val TAG = "Classifier"

        /** El modelo de `assets/` si está; si no, el clasificador falso (y la UI lo avisa). */
        fun createOrFake(context: Context): SquareClassifier = try {
            create(context)
        } catch (e: Exception) {
            Log.w(TAG, "Sin modelo LiteRT, uso el clasificador de prueba: ${e.message}")
            FakeSquareClassifier()
        }

        /** Carga el modelo de `assets/` y comprueba que `labels.txt` tenga el orden de clases de [Piece]. */
        fun create(context: Context, threads: Int = 4): LiteRtSquareClassifier {
            val labels = context.assets.open(LABELS_ASSET).bufferedReader().readLines().map { it.trim() }.filter { it.isNotEmpty() }
            val expected = Piece.entries.map { it.fen?.toString() ?: "empty" }
            require(labels == expected) { "labels.txt no coincide con el contrato: $labels" }
            val fd = context.assets.openFd(MODEL_ASSET)
            val model = FileInputStream(fd.fileDescriptor).channel.use { ch ->
                ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
            return LiteRtSquareClassifier(model, threads)
        }
    }
}
