package com.chessscanner.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.chessscanner.app.ui.ScannerScreen
import com.chessscanner.app.ui.ScannerViewModel
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    private val viewModel: ScannerViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        if (!OpenCVLoader.initLocal()) Log.e("MainActivity", "No se pudo cargar OpenCV")
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AppTheme {
                var granted by remember {
                    mutableStateOf(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
                }
                val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
                LaunchedEffect(Unit) { if (!granted) launcher.launch(Manifest.permission.CAMERA) }
                if (granted) ScannerScreen(viewModel) else PermissionScreen { launcher.launch(Manifest.permission.CAMERA) }
            }
        }
    }
}

@Composable
private fun PermissionScreen(onRequest: () -> Unit) {
    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier.padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "La app necesita la cámara para ver el tablero. Todo se procesa en el teléfono: no se guarda ni se envía ninguna imagen.",
                style = MaterialTheme.typography.bodyLarge,
            )
            Button(onClick = onRequest) { Text("Permitir cámara") }
        }
    }
}

@Composable
private fun AppTheme(content: @Composable () -> Unit) {
    val green = Color(0xFF3E5C3A)
    val scheme = if (isSystemInDarkTheme()) darkColorScheme(primary = Color(0xFF9CCC8F), secondary = Color(0xFFBCAAA4))
    else lightColorScheme(primary = green, secondary = Color(0xFF6D4C41))
    MaterialTheme(colorScheme = scheme, content = content)
}
