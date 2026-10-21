package com.chessscanner.app.camera

import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * Conversión de un fotograma YUV_420_888 de CameraX a Mats de OpenCV, ya derechos según
 * `imageInfo.rotationDegrees` (ml/CONTRACT.md §1 pide la imagen derecha).
 *
 * El gris sale directo del plano Y (barato, se usa siempre); el RGB se arma con NV21 → RGB solo
 * cuando el pipeline va a clasificar. Reutiliza buffers entre fotogramas; un solo hilo.
 */
class YuvFrame {
    private var yBytes = ByteArray(0)
    private var nv21 = ByteArray(0)
    private val grayRaw = Mat()
    private val yuvRaw = Mat()
    private val rgbRaw = Mat()
    val gray = Mat()
    val rgb = Mat()
    private var rotation = 0
    private var image: ImageProxy? = null

    /** Carga el plano Y del fotograma y deja el gris derecho en [gray]. */
    fun load(image: ImageProxy) {
        this.image = image
        rotation = image.imageInfo.rotationDegrees
        val w = image.width
        val h = image.height
        val plane = image.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        if (yBytes.size != w * h) yBytes = ByteArray(w * h)
        buf.rewind()
        if (rowStride == w) {
            buf.get(yBytes, 0, w * h)
        } else {
            for (r in 0 until h) {
                buf.position(r * rowStride)
                buf.get(yBytes, r * w, w)
            }
        }
        if (grayRaw.rows() != h || grayRaw.cols() != w) grayRaw.create(h, w, CvType.CV_8UC1)
        grayRaw.put(0, 0, yBytes)
        rotate(grayRaw, gray)
    }

    /** RGB uint8 derecho del último fotograma cargado (válido mientras el ImageProxy siga abierto). */
    fun rgb(): Mat {
        val img = checkNotNull(image)
        val w = img.width
        val h = img.height
        val size = w * h
        if (nv21.size != size * 3 / 2) nv21 = ByteArray(size * 3 / 2)
        System.arraycopy(yBytes, 0, nv21, 0, size)
        val u = img.planes[1]
        val v = img.planes[2]
        val ub = u.buffer
        val vb = v.buffer
        val cw = w / 2
        val ch = h / 2
        var o = size
        for (r in 0 until ch) {
            val ur = r * u.rowStride
            val vr = r * v.rowStride
            for (c in 0 until cw) {
                nv21[o++] = vb.get(vr + c * v.pixelStride)
                nv21[o++] = ub.get(ur + c * u.pixelStride)
            }
        }
        if (yuvRaw.rows() != h * 3 / 2 || yuvRaw.cols() != w) yuvRaw.create(h * 3 / 2, w, CvType.CV_8UC1)
        yuvRaw.put(0, 0, nv21)
        Imgproc.cvtColor(yuvRaw, rgbRaw, Imgproc.COLOR_YUV2RGB_NV21)
        rotate(rgbRaw, rgb)
        return rgb
    }

    private fun rotate(src: Mat, dst: Mat) {
        when (rotation) {
            90 -> Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(src, dst, Core.ROTATE_180)
            270 -> Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> src.copyTo(dst)
        }
    }

    fun release() {
        listOf(grayRaw, yuvRaw, rgbRaw, gray, rgb).forEach { it.release() }
    }
}
