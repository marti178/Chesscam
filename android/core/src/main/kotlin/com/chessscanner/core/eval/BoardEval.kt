package com.chessscanner.core.eval

import com.chessscanner.core.board.BoardDetection
import com.chessscanner.core.board.BoardDetector
import com.chessscanner.core.board.Pt
import com.chessscanner.core.board.Quad
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.Locale
import kotlin.math.max

/** Etiqueta de `eval/boards.tsv`: esquinas del área de juego en orden del contrato (TL, TR, BR, BL). */
data class BoardLabel(val file: String, val corners: List<Pt>, val fen: String?)

object BoardLabels {
    /**
     * Formato: una imagen por línea, separada por tabs:
     * `archivo  tlx tly  trx try  brx bry  blx bly  [fen]`. Las líneas con `#` son comentarios.
     * Una línea con solo el archivo y `-` significa «no hay tablero completo» (el detector debería fallar).
     */
    fun read(file: File): Map<String, BoardLabel?> {
        if (!file.exists()) return emptyMap()
        return file.readLines().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }.associate { line ->
            val f = line.split(Regex("\\s+"))
            if (f.size >= 2 && f[1] == "-") {
                f[0] to null
            } else {
                require(f.size >= 9) { "Línea mal formada: $line" }
                val c = (0 until 4).map { Pt(f[1 + 2 * it].toDouble(), f[2 + 2 * it].toDouble()) }
                f[0] to BoardLabel(f[0], c, f.drop(9).joinToString(" ").ifBlank { null })
            }
        }
    }
}

object BoardMetrics {
    /** Error máximo de esquina, en casillas (lado medio / 8), tomando el mejor giro del orden. */
    fun cornerErrorSquares(detected: List<Pt>, label: List<Pt>): Double {
        val square = Quad.meanSide(label) / 8
        return (0 until 4).minOf { k -> (0 until 4).maxOf { detected[(it + k) % 4].dist(label[it]) } } / square
    }
}

/**
 * Herramienta de escritorio: corre [BoardDetector] sobre cada imagen de una carpeta, guarda la imagen
 * con la retícula dibujada y, si hay etiquetas, cuenta cuántas detecta bien (todas las esquinas a menos
 * de [TOLERANCE_SQUARES] casillas).
 */
fun main(args: Array<String>) {
    loadDesktopOpenCv()
    val images = File(args.getOrElse(0) { "eval/images" })
    val labels = BoardLabels.read(File(args.getOrElse(1) { "eval/boards.tsv" }))
    val out = File(args.getOrElse(2) { "build/eval" }).apply { mkdirs() }
    val files = images.listFiles { f -> f.extension.lowercase() in setOf("jpg", "jpeg", "png") }
        ?.sortedWith(compareBy({ it.nameWithoutExtension.length }, { it.name })).orEmpty()
    require(files.isNotEmpty()) { "No hay imágenes en $images" }

    val detector = BoardDetector()
    var ok = 0; var labeled = 0; var found = 0
    val thumbs = ArrayList<Mat>()
    val proposals = ArrayList<String>()
    var consistencyOk = 0; var consistencyTotal = 0
    println("imagen\tdetectado\terror(casillas)\tresultado\tms\tsoporte\tdamero")
    for (f in files) {
        val img = Imgcodecs.imread(f.path)
        val gray = Mat()
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
        val t0 = System.nanoTime()
        val det = detector.detect(gray)
        val ms = (System.nanoTime() - t0) / 1e6
        if (det != null) found++
        val label = labels[f.name]
        val hasLabel = labels.containsKey(f.name)
        val (err, verdict) = when {
            !hasLabel -> null to "sin etiqueta"
            label == null -> null to (if (det == null) "OK (sin tablero)" else "FALSO POSITIVO")
            det == null -> null to "NO DETECTADO"
            else -> BoardMetrics.cornerErrorSquares(det.corners, label.corners).let { e ->
                e to (if (e <= TOLERANCE_SQUARES) "OK" else "MAL")
            }
        }
        if (hasLabel) {
            labeled++
            if (verdict.startsWith("OK")) ok++
        }
        println(
            "${f.name}\t${det != null}\t${err?.let { "%.2f".format(it) } ?: "-"}\t$verdict\t${"%.0f".format(ms)}\t" +
                (det?.let { "%.2f\t%.0f".format(it.support, it.checker) } ?: "-\t-"),
        )
        if (det != null && !hasLabel) {
            proposals += f.name + "\t" + det.corners.joinToString("\t") { "%.1f\t%.1f".format(Locale.ROOT, it.x, it.y) }
        }
        // Consistencia: la misma imagen girada (como el teléfono en otra posición) y achicada a 640 px
        // (como el fotograma de la cámara) tiene que dar el mismo tablero.
        if (det != null) for (rot in 0..3) {
            val turned = rotated(gray, rot, 640.0)
            val d2 = detector.detect(turned.first)
            turned.first.release()
            consistencyTotal++
            val back = d2?.corners?.map(turned.second)
            if (back != null && BoardMetrics.cornerErrorSquares(back, det.corners) <= TOLERANCE_SQUARES) consistencyOk++
            else println("  inconsistente: ${f.name} girada ${rot * 90}° a 640 px")
        }
        drawOverlay(img, det, label)
        Imgcodecs.imwrite(File(out, f.nameWithoutExtension + "_det.jpg").path, img)
        val thumb = Mat()
        Imgproc.resize(img, thumb, Size(THUMB.toDouble(), THUMB * img.rows().toDouble() / img.cols()))
        Imgproc.putText(thumb, f.nameWithoutExtension, Point(6.0, 24.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 0.0, 255.0), 2)
        thumbs += thumb
        img.release(); gray.release()
    }
    writeMontage(thumbs, File(out, "_mosaico.jpg"))
    println()
    println("Tableros encontrados: $found / ${files.size}")
    if (labeled > 0) println("Detectados bien (error ≤ $TOLERANCE_SQUARES casillas): $ok / $labeled etiquetadas")
    println("Consistencia girada 0/90/180/270° a 640 px: $consistencyOk / $consistencyTotal")
    if (proposals.isNotEmpty()) {
        File(out, "boards_propuestas.tsv").writeText(proposals.joinToString("\n", postfix = "\n"))
        println("Esquinas detectadas de imágenes sin etiqueta (revisar a ojo antes de usarlas): boards_propuestas.tsv")
    }
    println("Imágenes con la retícula dibujada en: ${out.absolutePath}")
}

const val TOLERANCE_SQUARES = 0.35
private const val THUMB = 300

/** Mosaico de miniaturas (5 por fila) para revisar el resultado de un vistazo. */
private fun writeMontage(thumbs: List<Mat>, file: File) {
    if (thumbs.isEmpty()) return
    val cols = 5
    val cellH = thumbs.maxOf { it.rows() }
    val rows = (thumbs.size + cols - 1) / cols
    val canvas = Mat(rows * cellH, cols * THUMB, thumbs[0].type(), Scalar(40.0, 40.0, 40.0))
    thumbs.forEachIndexed { i, t ->
        val x = (i % cols) * THUMB
        val y = (i / cols) * cellH
        t.copyTo(canvas.submat(y, y + t.rows(), x, x + t.cols()))
        t.release()
    }
    Imgcodecs.imwrite(file.path, canvas)
    canvas.release()
}

fun drawOverlay(img: Mat, det: BoardDetection?, label: BoardLabel?) {
    val t = max(2, img.cols() / 400)
    label?.corners?.let { c ->
        for (i in 0 until 4) Imgproc.line(img, c[i].cv(), c[(i + 1) % 4].cv(), Scalar(255.0, 0.0, 255.0), t)
    }
    det ?: return
    val l = det.lattice
    for (i in 0..8) {
        Imgproc.line(img, l[i * 9].cv(), l[i * 9 + 8].cv(), Scalar(0.0, 220.0, 0.0), t)
        Imgproc.line(img, l[i].cv(), l[72 + i].cv(), Scalar(0.0, 220.0, 0.0), t)
    }
    Imgproc.circle(img, det.corners[0].cv(), 6 * t, Scalar(0.0, 0.0, 255.0), -1)
    Imgproc.putText(img, "TL", det.corners[0].cv(), Imgproc.FONT_HERSHEY_SIMPLEX, t * 0.8, Scalar(0.0, 0.0, 255.0), t)
}

private fun Pt.cv() = Point(x, y)

/**
 * Gira la imagen [quarter] cuartos de vuelta en sentido horario y la achica a [size] de lado mayor.
 * Devuelve también la función que lleva un punto de la imagen girada a la original.
 */
private fun rotated(gray: Mat, quarter: Int, size: Double): Pair<Mat, (Pt) -> Pt> {
    val w = gray.cols().toDouble()
    val h = gray.rows().toDouble()
    val k = size / max(w, h)
    val small = Mat()
    Imgproc.resize(gray, small, Size(), k, k, Imgproc.INTER_AREA)
    val out = Mat()
    when (quarter) {
        1 -> Core.rotate(small, out, Core.ROTATE_90_CLOCKWISE)
        2 -> Core.rotate(small, out, Core.ROTATE_180)
        3 -> Core.rotate(small, out, Core.ROTATE_90_COUNTERCLOCKWISE)
        else -> small.copyTo(out)
    }
    small.release()
    val sw = w * k
    val sh = h * k
    val back: (Pt) -> Pt = { p ->
        val q = when (quarter) {
            1 -> Pt(p.y, sh - 1 - p.x)
            2 -> Pt(sw - 1 - p.x, sh - 1 - p.y)
            3 -> Pt(sw - 1 - p.y, p.x)
            else -> p
        }
        q * (1 / k)
    }
    return out to back
}

/** Carga los nativos de OpenCV de escritorio (jar de openpnp) sin depender de él al compilar la app. */
fun loadDesktopOpenCv() {
    Class.forName("nu.pattern.OpenCV").getMethod("loadLocally").invoke(null)
}
