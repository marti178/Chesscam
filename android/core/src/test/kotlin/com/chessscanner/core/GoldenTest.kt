package com.chessscanner.core

import com.chessscanner.core.board.Pt
import com.chessscanner.core.classify.SquareCrops
import com.chessscanner.core.eval.loadDesktopOpenCv
import com.chessscanner.core.fen.Fen
import com.chessscanner.core.model.Piece
import com.chessscanner.core.orientation.GridMapping
import com.chessscanner.core.orientation.Orientation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max

/**
 * Vectores dorados de ml/golden (ml/CONTRACT.md §6). Se corre con
 * `./gradlew :core:test -PgoldenDir=<carpeta ml/golden>`; sin esa propiedad se saltea.
 */
class GoldenTest {
    companion object {
        @BeforeClass @JvmStatic fun load() = loadDesktopOpenCv()
        private val dir: File? = System.getProperty("golden.dir")?.let(::File)?.takeIf { it.isDirectory }
        private fun cases(): List<File> = dir?.listFiles { f -> File(f, "case.json").exists() }?.sortedBy { it.name }.orEmpty()

        /** Diferencia máxima y media entre dos buffers uint8. */
        fun diff(a: ByteArray, b: ByteArray): Pair<Int, Double> {
            require(a.size == b.size) { "tamaños ${a.size} vs ${b.size}" }
            var mx = 0
            var sum = 0L
            for (i in a.indices) {
                val d = abs((a[i].toInt() and 0xFF) - (b[i].toInt() and 0xFF))
                mx = max(mx, d); sum += d
            }
            return mx to sum.toDouble() / a.size
        }

        fun readRgb(f: File): Mat {
            val bgr = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_COLOR)
            check(!bgr.empty()) { "no se pudo leer $f" }
            return Mat().also { Imgproc.cvtColor(bgr, it, Imgproc.COLOR_BGR2RGB); bgr.release() }
        }

        fun bytes(m: Mat): ByteArray = ByteArray((m.total() * m.channels()).toInt()).also { m.get(0, 0, it) }
    }

    private fun assertClose(what: String, d: Pair<Int, Double>) {
        // Tolerancia del contrato §6 (la referencia usa OpenCV 4.12, el jar de escritorio 4.9).
        assertTrue("$what: diferencia máxima ${d.first}", d.first <= 2)
        assertTrue("$what: diferencia media ${d.second}", d.second < 0.05)
    }

    @Test fun geometryMatchesReference() {
        assumeTrue("sin -PgoldenDir", dir != null)
        val cases = cases()
        assertTrue("no hay casos en $dir", cases.isNotEmpty())
        for (case in cases) {
            val json = Json.parse(File(case, "case.json").readText()) as Map<*, *>
            val corners = (json["corners_TL_TR_BR_BL"] as List<*>).map { p -> (p as List<*>).let { Pt((it[0] as Number).toDouble(), (it[1] as Number).toDouble()) } }
            val href = (json["H_row_major"] as List<*>).flatMap { r -> (r as List<*>).map { (it as Number).toDouble() } }

            val crops = SquareCrops()
            val h = crops.homography(corners)
            for (i in 0 until 9) {
                val v = h.get(i / 3, i % 3)[0]
                assertTrue("${case.name}: H[$i] = $v vs ${href[i]}", abs(v - href[i]) <= 1e-9 * max(1.0, abs(href[i])))
            }

            val rgb = readRgb(File(case, "image.png"))
            val canon = crops.rectify(rgb, corners)
            assertEquals(CvType.CV_8UC3, canon.type())
            assertClose("${case.name} canonical", diff(bytes(canon), bytes(readRgb(File(case, "canonical.png")))))

            File(case, "crops").takeIf { it.isDirectory }?.let { cropDir ->
                for (r in 0 until 8) for (c in 0 until 8) {
                    val expected = readRgb(File(cropDir, "r${r}c$c.png"))
                    assertClose("${case.name} r${r}c$c", diff(bytes(crops.crop(canon, r, c)), bytes(expected)))
                }
            }

            val input = SquareCrops.newInputBuffer()
            crops.fillInput(canon, input)
            val got = ByteArray(SquareCrops.BATCH * SquareCrops.CROP_BYTES)
            val fb = input.order(ByteOrder.nativeOrder()).asFloatBuffer()
            for (i in got.indices) got[i] = fb.get(i).toInt().toByte()
            val expectedInput = File(case, "input_u8.bin").takeIf { it.exists() }?.readBytes()
                ?: File(case, "input_f32.bin").readBytes().let { raw ->
                    val f = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    ByteArray(f.remaining()) { f.get(it).toInt().toByte() }
                }
            assertClose("${case.name} tensor de entrada", diff(got, expectedInput))
        }
    }

    @Test fun orientationAndFenFromTrueLabels() {
        assumeTrue("sin -PgoldenDir", dir != null)
        for (case in cases()) {
            val json = Json.parse(File(case, "case.json").readText()) as Map<*, *>
            val labels = json["true_labels"] as? List<*> ?: continue
            val fen = json["fen"] as? String ?: continue
            val corner = json["a1_corner"] as? String ?: continue
            val grid = labels.map { l -> if (l == "empty") Piece.EMPTY else Piece.fromFen((l as String)[0])!! }.toTypedArray()
            val expectedRotation = mapOf("BL" to 0, "TL" to 1, "TR" to 2, "BR" to 3).getValue(corner)
            assertEquals(fen, Fen.placement(GridMapping.toBoard(grid, expectedRotation)))

            // La paridad de las casillas oscuras sale de la imagen canónica real.
            val corners = (json["corners_TL_TR_BR_BL"] as List<*>).map { p -> (p as List<*>).let { Pt((it[0] as Number).toDouble(), (it[1] as Number).toDouble()) } }
            val crops = SquareCrops()
            val canon = crops.rectify(readRgb(File(case, "image.png")), corners)
            val empty = BooleanArray(64) { grid[it] == Piece.EMPTY }
            val parity = Orientation.darkParity(crops.cellLuma(canon), empty)
            assertEquals("${case.name}: paridad oscura", GridMapping.a1Parity(expectedRotation), parity)
            val r = Orientation.resolve(grid, parity)
            println("${case.name}: rotación ${r.rotation} (esperada $expectedRotation), dudosa=${r.doubtful} ${r.reason ?: ""}")
            if (!r.doubtful) assertEquals(expectedRotation, r.rotation)
        }
    }

    @Test fun modelOutputFileHasExpectedShape() {
        assumeTrue("sin -PgoldenDir", dir != null)
        for (case in cases()) {
            val f = File(case, "output_logits_f32.bin")
            if (f.exists()) assertEquals(64 * Piece.COUNT * 4L, f.length())
        }
    }
}

/** JSON mínimo para leer case.json en los tests (objetos, listas, números, strings, bool, null). */
object Json {
    fun parse(s: String): Any? = P(s).run { val v = value(); ws(); require(i == s.length) { "basura al final" }; v }

    private class P(val s: String) {
        var i = 0
        fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
        fun value(): Any? {
            ws()
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> { i += 4; true }
                'f' -> { i += 5; false }
                'n' -> { i += 4; null }
                else -> if (c == '-' || c.isDigit()) num() else error("JSON inválido en $i: '$c'")
            }
        }
        fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++; ws()
            if (s[i] == '}') { i++; return m }
            while (true) {
                ws(); val k = str(); ws(); require(s[i] == ':'); i++
                m[k] = value(); ws()
                if (s[i] == ',') { i++; continue }
                require(s[i] == '}'); i++; return m
            }
        }
        fun arr(): List<Any?> {
            val l = ArrayList<Any?>()
            i++; ws()
            if (s[i] == ']') { i++; return l }
            while (true) {
                l += value(); ws()
                if (s[i] == ',') { i++; continue }
                require(s[i] == ']'); i++; return l
            }
        }
        fun str(): String {
            require(s[i] == '"'); i++
            val b = StringBuilder()
            while (s[i] != '"') {
                if (s[i] == '\\') {
                    i++
                    when (s[i]) {
                        'n' -> b.append('\n'); 't' -> b.append('\t')
                        'u' -> { b.append(s.substring(i + 1, i + 5).toInt(16).toChar()); i += 4 }
                        else -> b.append(s[i])
                    }
                } else b.append(s[i])
                i++
            }
            i++
            return b.toString()
        }
        fun num(): Double {
            val st = i
            while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
            return s.substring(st, i).toDouble()
        }
    }
}
