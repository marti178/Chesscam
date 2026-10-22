package com.chessscanner.core.classify

import com.chessscanner.core.model.Piece
import java.nio.ByteBuffer

/**
 * Clasificador de las 64 casillas (ml/CONTRACT.md §4–5).
 * Entrada: float32 [64, 128, 64, 3] NHWC, RGB 0–255, en un ByteBuffer directo con orden nativo
 * (ver [SquareCrops.fillInput]). Salida: 64 × 13 logits, clases en el orden de [Piece].
 */
interface SquareClassifier : AutoCloseable {
    /** Nombre para mostrar (y para saber si es el falso). */
    val name: String
    val isFake: Boolean
    fun classify(input: ByteBuffer): Array<FloatArray>
    override fun close() {}
}

/**
 * Clasificador falso hasta que llegue el modelo: mira la textura de la casilla (abajo a la izquierda
 * del recorte) y dice «vacía» si es pareja; si no, un peón del color que mejor contrasta con la
 * casilla. Sirve para probar la interfaz y la fusión, no para reconocer posiciones.
 */
class FakeSquareClassifier : SquareClassifier {
    override val name = "clasificador de prueba (sin modelo)"
    override val isFake = true

    override fun classify(input: ByteBuffer): Array<FloatArray> {
        val fb = input.duplicate().order(input.order()).asFloatBuffer()
        val out = Array(64) { FloatArray(Piece.COUNT) }
        val h = SquareCrops.CROP_H
        val w = SquareCrops.CROP_W
        for (i in 0 until 64) {
            val base = i * SquareCrops.CROP_BYTES
            var sum = 0.0
            var sum2 = 0.0
            var n = 0
            // Centro de la casilla: recorte[100:124, 4:28].
            for (y in h - 28 until h - 4) for (x in 4 until 28) {
                val p = base + (y * w + x) * 3
                val l = 0.299 * fb.get(p) + 0.587 * fb.get(p + 1) + 0.114 * fb.get(p + 2)
                sum += l; sum2 += l * l; n++
            }
            val mean = sum / n
            val std = kotlin.math.sqrt(kotlin.math.max(0.0, sum2 / n - mean * mean))
            val cls = when {
                std < 22 -> Piece.EMPTY
                mean > 110 -> Piece.WP
                else -> Piece.BP
            }
            out[i][cls.ordinal] = 4f
        }
        return out
    }
}
