package com.chessscanner.core.classify

import com.chessscanner.core.board.Pt
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Rectificación y recorte de las 64 casillas, idénticos a `ml/CONTRACT.md` v1 (§2 y §3).
 * No es thread-safe: reutiliza buffers entre fotogramas.
 */
class SquareCrops {
    companion object {
        const val S = 32
        const val MX = 32
        const val MY = 96
        const val CW = 320
        const val CH = 352
        const val CROP_H = 128
        const val CROP_W = 64
        const val BATCH = 64
        const val CROP_BYTES = CROP_H * CROP_W * 3
        /** Tamaño en bytes del tensor de entrada float32 [64, 128, 64, 3]. */
        const val INPUT_BYTES = BATCH * CROP_BYTES * 4

        private val DST = MatOfPoint2f(
            Point(MX.toDouble(), MY.toDouble()), Point((MX + 8 * S).toDouble(), MY.toDouble()),
            Point((MX + 8 * S).toDouble(), (MY + 8 * S).toDouble()), Point(MX.toDouble(), (MY + 8 * S).toDouble()),
        )

        fun up(row: Int) = S + (2 * S * (7 - row)) / 7
        fun y0(row: Int) = MY + S * row - up(row)
        fun y1(row: Int) = MY + S * (row + 1)
        private fun extra(col: Int) = (S / 4) * ((if (col < 4) 3 - col else col - 4) + 1)
        fun x0(col: Int) = if (col < 4) MX + S * col - extra(col) else MX + S * col
        fun x1(col: Int) = if (col < 4) MX + S * (col + 1) else MX + S * (col + 1) + extra(col)

        fun newInputBuffer(): ByteBuffer = ByteBuffer.allocateDirect(INPUT_BYTES).order(ByteOrder.nativeOrder())
    }

    /** Imagen canónica del último [rectify] (RGB uint8, 320×352). */
    val canonical = Mat()
    private val crop = Mat(CROP_H, CROP_W, CvType.CV_8UC3)
    private val flipped = Mat()
    private val bytes = ByteArray(CROP_BYTES)

    /** Homografía imagen → canónica (3×3, CV_64F) a partir de las esquinas TL, TR, BR, BL. */
    fun homography(corners: List<Pt>): Mat {
        val src = MatOfPoint2f(*corners.map { Point(it.x, it.y) }.toTypedArray())
        return Imgproc.getPerspectiveTransform(src, DST).also { src.release() }
    }

    /** Rectifica [rgb] (RGB uint8) a la imagen canónica. */
    fun rectify(rgb: Mat, corners: List<Pt>): Mat {
        val h = homography(corners)
        Imgproc.warpPerspective(
            rgb, canonical, h, Size(CW.toDouble(), CH.toDouble()),
            Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0),
        )
        h.release()
        return canonical
    }

    /** Recorte de la casilla (fila, col) de la imagen canónica, 128×64 RGB. El Mat se reutiliza. */
    fun crop(canon: Mat, row: Int, col: Int): Mat {
        val y0 = y0(row); val y1 = y1(row); val x0 = x0(col); val x1 = x1(col)
        crop.setTo(Scalar(0.0, 0.0, 0.0))
        val patch = canon.submat(y0, y1, x0, x1)
        val src = if (col < 4) { Core.flip(patch, flipped, 1); flipped } else patch
        val target = crop.submat(CROP_H - (y1 - y0), CROP_H, 0, x1 - x0)
        src.copyTo(target)
        target.release(); patch.release()
        return crop
    }

    /** Llena [out] con el tensor float32 [64, 128, 64, 3] (valores 0–255, orden i = fila * 8 + col). */
    fun fillInput(canon: Mat, out: ByteBuffer) {
        out.clear()
        val fb = out.asFloatBuffer()
        for (row in 0 until 8) for (col in 0 until 8) {
            crop(canon, row, col).get(0, 0, bytes)
            for (b in bytes) fb.put((b.toInt() and 0xFF).toFloat())
        }
        out.rewind()
    }

    /** Luminancia media del 50 % central de cada casilla en la imagen canónica (para la orientación). */
    fun cellLuma(canon: Mat): DoubleArray {
        val gray = Mat()
        Imgproc.cvtColor(canon, gray, Imgproc.COLOR_RGB2GRAY)
        val out = DoubleArray(64)
        for (row in 0 until 8) for (col in 0 until 8) {
            val x = MX + S * col + S / 4
            val y = MY + S * row + S / 4
            val roi = gray.submat(y, y + S / 2, x, x + S / 2)
            out[row * 8 + col] = Core.mean(roi).`val`[0]
            roi.release()
        }
        gray.release()
        return out
    }
}
