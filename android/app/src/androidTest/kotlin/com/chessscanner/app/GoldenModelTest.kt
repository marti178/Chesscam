package com.chessscanner.app

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.chessscanner.app.classify.LiteRtSquareClassifier
import com.chessscanner.core.board.Pt
import com.chessscanner.core.classify.SquareCrops
import com.chessscanner.core.model.Piece
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max

/**
 * Vectores dorados de ml/golden con el modelo real en el teléfono (ml/CONTRACT.md §6): mismo argmax en
 * las 64 casillas y |Δlogit| ≤ 0,1. Los casos se empaquetan en el APK de test con
 * `./gradlew connectedDebugAndroidTest -PgoldenDir=<ml/golden>`; sin eso, se saltea.
 */
@RunWith(AndroidJUnit4::class)
class GoldenModelTest {
    companion object {
        @BeforeClass @JvmStatic fun load() { check(OpenCVLoader.initLocal()) }
    }

    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    private fun cases(): List<String> = testAssets.list("golden")?.filter { c ->
        testAssets.list("golden/$c")?.contains("output_logits_f32.bin") == true
    }.orEmpty().sorted()

    private fun floats(path: String): FloatArray {
        val b = ByteBuffer.wrap(testAssets.open(path).readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.remaining()) { b.get(it) }
    }

    private fun check(case: String, logits: Array<FloatArray>, what: String) {
        val expected = floats("golden/$case/output_logits_f32.bin")
        var maxDiff = 0.0
        for (i in 0 until 64) {
            val e = FloatArray(Piece.COUNT) { expected[i * Piece.COUNT + it] }
            assertEquals("$case $what casilla $i argmax", e.indices.maxBy { e[it] }, logits[i].indices.maxBy { logits[i][it] })
            for (c in 0 until Piece.COUNT) maxDiff = max(maxDiff, abs((e[c] - logits[i][c]).toDouble()))
        }
        println("GOLDEN $case $what: |Δlogit| máx = $maxDiff")
        assertTrue("$case $what: |Δlogit| máx $maxDiff", maxDiff <= 0.1)
    }

    @Test fun modelMatchesGoldenOutputsFromGoldenInput() {
        val cases = cases()
        assumeTrue("sin vectores dorados en el APK de test (-PgoldenDir)", cases.isNotEmpty())
        val clf = LiteRtSquareClassifier.create(appContext)
        val input = SquareCrops.newInputBuffer()
        for (case in cases) {
            val u8 = testAssets.open("golden/$case/input_u8.bin").readBytes()
            input.clear()
            val fb = input.asFloatBuffer()
            for (b in u8) fb.put((b.toInt() and 0xFF).toFloat())
            val t0 = System.nanoTime()
            val logits = clf.classify(input)
            println("GOLDEN $case: inferencia ${(System.nanoTime() - t0) / 1e6} ms")
            check(case, logits, "entrada dorada")
        }
        clf.close()
    }

    /** Camino completo en el teléfono: PNG → esquinas → recorte con OpenCV Android → modelo. */
    @Test fun fullPathFromImageMatchesGoldenOutputs() {
        val cases = cases()
        assumeTrue("sin vectores dorados en el APK de test (-PgoldenDir)", cases.isNotEmpty())
        val clf = LiteRtSquareClassifier.create(appContext)
        val crops = SquareCrops()
        val input = SquareCrops.newInputBuffer()
        for (case in cases) {
            val json = JSONObject(testAssets.open("golden/$case/case.json").bufferedReader().readText())
            val c = json.getJSONArray("corners_TL_TR_BR_BL")
            val corners = (0 until 4).map { Pt(c.getJSONArray(it).getDouble(0), c.getJSONArray(it).getDouble(1)) }
            val bmp = BitmapFactory.decodeStream(testAssets.open("golden/$case/image.png"))
            val rgba = Mat()
            Utils.bitmapToMat(bmp, rgba)
            val rgb = Mat()
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            crops.fillInput(crops.rectify(rgb, corners), input)
            check(case, clf.classify(input), "camino completo")
        }
        clf.close()
    }
}
