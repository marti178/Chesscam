package com.chessscanner.app.ui

import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlin.math.min

/**
 * Vista de cámara con la retícula detectada encima: verde si el fotograma pasó el control de calidad,
 * ámbar si es dudoso; las casillas dudosas se rellenan en ámbar.
 */
@Composable
fun CameraPreview(viewModel: ScannerViewModel, live: LiveUi, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            // FIT_CENTER: la imagen de análisis (4:3) y la vista se corresponden con una escala uniforme.
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    LaunchedEffect(lifecycleOwner) {
        val provider = ProcessCameraProvider.awaitInstance(context)
        val selector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build()
        val preview = Preview.Builder()
            .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build())
            .build()
            .also { it.surfaceProvider = previewView.surfaceProvider }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(selector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { it.setAnalyzer(viewModel.analysisExecutor, viewModel.analyzer) }
        provider.unbindAll()
        // Atado al ciclo de vida: la cámara se libera cuando la app pasa a segundo plano.
        provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }

    Box(modifier) {
        AndroidView({ previewView }, Modifier.fillMaxSize())
        Canvas(Modifier.fillMaxSize()) {
            val lattice = live.lattice ?: return@Canvas
            if (live.imageWidth == 0) return@Canvas
            val s = min(size.width / live.imageWidth, size.height / live.imageHeight)
            val ox = (size.width - live.imageWidth * s) / 2
            val oy = (size.height - live.imageHeight * s) / 2
            fun p(i: Int) = Offset(ox + lattice[i].x.toFloat() * s, oy + lattice[i].y.toFloat() * s)
            val amber = Color(0xFFFFB300)
            val color = if (live.boardGood) Color(0xFF00E676) else amber
            for (cell in live.doubtfulImageCells) {
                val r = cell / 8
                val c = cell % 8
                val path = Path().apply {
                    moveTo(p(r * 9 + c).x, p(r * 9 + c).y)
                    lineTo(p(r * 9 + c + 1).x, p(r * 9 + c + 1).y)
                    lineTo(p((r + 1) * 9 + c + 1).x, p((r + 1) * 9 + c + 1).y)
                    lineTo(p((r + 1) * 9 + c).x, p((r + 1) * 9 + c).y)
                    close()
                }
                drawPath(path, amber.copy(alpha = 0.35f))
            }
            val stroke = 2.dp.toPx()
            for (i in 0..8) {
                drawLine(color, p(i * 9), p(i * 9 + 8), stroke)
                drawLine(color, p(i), p(72 + i), stroke)
            }
            drawCircle(color, 5.dp.toPx(), p(0), style = Stroke(stroke))
        }
    }
}
