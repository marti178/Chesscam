package com.chessscanner.app.ui

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.chessscanner.core.fen.EditablePosition
import com.chessscanner.core.fen.Fen
import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece
import com.chessscanner.core.pipeline.ScanStatus

/** Pantalla única (SPEC «Interfaz»): escaneo en vivo o posición congelada para corregir y exportar. */
@Composable
fun ScannerScreen(viewModel: ScannerViewModel) {
    val live by viewModel.live.collectAsStateWithLifecycle()
    val frozen by viewModel.frozen.collectAsStateWithLifecycle()
    // Se lee para recomponer cuando cambia el giro manual.
    viewModel.rotationOffset.collectAsStateWithLifecycle().value
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Box(Modifier.safeDrawingPadding()) {
            val f = frozen
            if (f == null) LiveScreen(viewModel, live) else FrozenScreen(viewModel, f)
        }
    }
}

@Composable
private fun LiveScreen(viewModel: ScannerViewModel, live: LiveUi) {
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth().background(Color.Black)) {
            CameraPreview(viewModel, live, Modifier.fillMaxSize())
            StatusBanner(live, Modifier.align(Alignment.TopCenter).padding(12.dp))
            if (viewModel.classifier.isFake) Text(
                "Sin modelo: clasificador de prueba",
                Modifier.align(Alignment.BottomStart).padding(8.dp)
                    .background(Color(0xAA000000), RoundedCornerShape(6.dp)).padding(horizontal = 8.dp, vertical = 4.dp),
                color = Color(0xFFFFCC80), fontSize = 12.sp,
            )
            Text(
                "%.0f fps · %.0f ms · clasif. %.0f ms".format(live.fps, live.frameMs, live.classifyMs),
                Modifier.align(Alignment.BottomEnd).padding(8.dp)
                    .background(Color(0xAA000000), RoundedCornerShape(6.dp)).padding(horizontal = 8.dp, vertical = 4.dp),
                color = Color.White, fontSize = 12.sp,
            )
        }
        Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            BoardView(
                viewModel.liveBoard(live), viewModel.liveDoubtfulSquares(live),
                Modifier.width(184.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (live.hasFusion && live.orientationDoubtful) Text(
                    (live.orientationReason ?: "Orientación dudosa") + ". Revisá con «Girar».",
                    fontSize = 13.sp, color = MaterialTheme.colorScheme.error,
                )
                OutlinedButton(onClick = { viewModel.rotateLive(2) }, Modifier.fillMaxWidth()) { Text("Girar tablero") }
                Button(
                    onClick = viewModel::freeze,
                    modifier = Modifier.fillMaxWidth().height(64.dp),
                    enabled = live.hasFusion,
                ) { Text("Congelar", fontSize = 20.sp, fontWeight = FontWeight.Bold) }
            }
        }
    }
}

@Composable
private fun StatusBanner(live: LiveUi, modifier: Modifier) {
    val color = when (live.status) {
        ScanStatus.READY -> Color(0xFF2E7D32)
        ScanStatus.ANALYZING, ScanStatus.MOVE_FOR_DOUBTFUL -> Color(0xFF455A64)
        else -> Color(0xFFE65100)
    }
    Text(
        live.status.message,
        modifier.background(color, RoundedCornerShape(20.dp)).padding(horizontal = 16.dp, vertical = 8.dp),
        color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.SemiBold,
    )
}

@Composable
private fun FrozenScreen(viewModel: ScannerViewModel, pos: EditablePosition) {
    val context = LocalContext.current
    var editing by remember { mutableStateOf<Int?>(null) }
    var confirmOpen by remember { mutableStateOf(false) }
    val board = pos.board
    val fen = pos.fen
    val problems = pos.problems

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Tocá una casilla para corregirla", style = MaterialTheme.typography.titleMedium)
        BoardView(board, pos.doubtfulSquares, Modifier.fillMaxWidth(), onSquareClick = { editing = it })
        if (pos.doubtfulCells.isNotEmpty()) Text(
            "${pos.doubtfulCells.size} casilla(s) dudosa(s) marcadas en ámbar.",
            color = Color(0xFFE65100),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { viewModel.rotateFrozen(2) }) { Text("Girar tablero") }
            OutlinedButton(onClick = { viewModel.rotateFrozen(1) }) { Text("Girar 90°") }
        }
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            SegmentedButton(pos.whiteToMove, { viewModel.setWhiteToMove(true) }, SegmentedButtonDefaults.itemShape(0, 2)) { Text("Juegan blancas") }
            SegmentedButton(!pos.whiteToMove, { viewModel.setWhiteToMove(false) }, SegmentedButtonDefaults.itemShape(1, 2)) { Text("Juegan negras") }
        }
        val possible = pos.possibleCastling
        val castling = pos.castling
        Text("Enroques", style = MaterialTheme.typography.titleSmall)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            CastleChip("O-O blancas", possible.whiteKing, castling.whiteKing) { viewModel.toggleCastling('K') }
            CastleChip("O-O-O blancas", possible.whiteQueen, castling.whiteQueen) { viewModel.toggleCastling('Q') }
            CastleChip("O-O negras", possible.blackKing, castling.blackKing) { viewModel.toggleCastling('k') }
            CastleChip("O-O-O negras", possible.blackQueen, castling.blackQueen) { viewModel.toggleCastling('q') }
        }
        if (problems.isNotEmpty()) Column {
            Text("Posición ilegal:", color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold)
            problems.forEach { Text("• $it", color = MaterialTheme.colorScheme.error) }
        }
        SelectionContainer {
            Text(fen, fontFamily = FontFamily.Monospace, fontSize = 14.sp)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { copyFen(context, fen) }, Modifier.weight(1f)) { Text("Copiar FEN") }
            Button(
                onClick = { if (pos.clean) openLichess(context, fen) else confirmOpen = true },
                modifier = Modifier.weight(1f),
            ) { Text("Abrir en Lichess") }
        }
        Button(
            onClick = viewModel::rescan, Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
        ) { Text("Volver a escanear") }
    }

    editing?.let { sq ->
        PiecePicker(Board.squareName(sq), onPick = { viewModel.edit(sq, it); editing = null }, onDismiss = { editing = null })
    }
    if (confirmOpen) {
        val reasons = buildList {
            if (!pos.stable) add("La posición no llegó a estar estable.")
            if (pos.doubtfulCells.isNotEmpty()) add("Quedan ${pos.doubtfulCells.size} casillas dudosas.")
            addAll(problems)
        }
        AlertDialog(
            onDismissRequest = { confirmOpen = false },
            title = { Text("¿Abrir igual?") },
            text = { Text(reasons.joinToString("\n")) },
            confirmButton = { TextButton({ confirmOpen = false; openLichess(context, fen) }) { Text("Abrir en Lichess") } },
            dismissButton = { TextButton({ confirmOpen = false }) { Text("Revisar") } },
        )
    }
}

@Composable
private fun CastleChip(label: String, possible: Boolean, selected: Boolean, onClick: () -> Unit) {
    FilterChip(selected = selected, onClick = onClick, enabled = possible, label = { Text(label) })
}

@Composable
private fun PiecePicker(square: String, onPick: (Piece) -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Casilla $square") },
        text = {
            LazyVerticalGrid(GridCells.Fixed(4), Modifier.height(260.dp)) {
                items(listOf(Piece.EMPTY) + Piece.entries.filter { it.isWhite == true } + Piece.entries.filter { it.isWhite == false }) { p ->
                    Box(
                        Modifier.padding(4.dp).size(56.dp)
                            .background(Color(0xFFD7C4A3), RoundedCornerShape(8.dp))
                            .clickable { onPick(p) },
                        contentAlignment = Alignment.Center,
                    ) {
                        if (p == Piece.EMPTY) Text("Vacía", fontSize = 13.sp) else PieceGlyph(p, 36f)
                    }
                }
            }
        },
        confirmButton = { TextButton(onDismiss) { Text("Cancelar") } },
    )
}

private fun copyFen(context: Context, fen: String) {
    context.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("FEN", fen))
    Toast.makeText(context, "FEN copiado", Toast.LENGTH_SHORT).show()
}

/** Abre el editor de Lichess con ACTION_VIEW; si está la app de Lichess, la toma Android. */
private fun openLichess(context: Context, fen: String) {
    try {
        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(Fen.lichessEditorUrl(fen))))
    } catch (e: ActivityNotFoundException) {
        Toast.makeText(context, "No hay navegador para abrir Lichess. Usá «Copiar FEN».", Toast.LENGTH_LONG).show()
    }
}
