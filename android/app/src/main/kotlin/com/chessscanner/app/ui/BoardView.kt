package com.chessscanner.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.draw.alpha
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.chessscanner.core.model.Board
import com.chessscanner.core.model.Piece

/** Glifo de la pieza; se usan los rellenos para ambos colores y se pintan de blanco o negro. */
fun Piece.glyph(): String = when (this) {
    Piece.EMPTY -> ""
    Piece.WK, Piece.BK -> "♚"
    Piece.WQ, Piece.BQ -> "♛"
    Piece.WR, Piece.BR -> "♜"
    Piece.WB, Piece.BB -> "♝"
    Piece.WN, Piece.BN -> "♞"
    Piece.WP, Piece.BP -> "♟"
} + "︎" // forzar presentación de texto (♟ tiene versión emoji)

fun Piece.label(): String = when (this) {
    Piece.EMPTY -> "Vacía"
    Piece.WK -> "Rey blanco"; Piece.WQ -> "Dama blanca"; Piece.WR -> "Torre blanca"
    Piece.WB -> "Alfil blanco"; Piece.WN -> "Caballo blanco"; Piece.WP -> "Peón blanco"
    Piece.BK -> "Rey negro"; Piece.BQ -> "Dama negra"; Piece.BR -> "Torre negra"
    Piece.BB -> "Alfil negro"; Piece.BN -> "Caballo negro"; Piece.BP -> "Peón negro"
}

@Composable
fun PieceGlyph(piece: Piece, sizeSp: Float) {
    if (piece == Piece.EMPTY) return
    val white = piece.isWhite == true
    Text(
        piece.glyph(),
        style = TextStyle(
            fontSize = sizeSp.sp,
            color = if (white) Color(0xFFFDFDFD) else Color(0xFF151515),
            shadow = Shadow(if (white) Color.Black else Color.White.copy(alpha = 0.6f), blurRadius = 3f),
        ),
    )
}

/**
 * Tablero digital desde el lado de las blancas (a1 abajo a la izquierda). Las casillas [doubtful] se
 * marcan en ámbar. Si [onSquareClick] no es null, cada casilla se puede tocar.
 */
@Composable
fun BoardView(
    board: Board,
    doubtful: Set<Int>,
    modifier: Modifier = Modifier,
    /** Si no es null, el tablero se atenúa y muestra este texto encima (no hay posición vigente). */
    placeholder: String? = null,
    onSquareClick: ((Int) -> Unit)? = null,
) {
    Box(modifier.aspectRatio(1f), contentAlignment = Alignment.Center) {
    BoxWithConstraints(Modifier.fillMaxSize().alpha(if (placeholder != null) 0.35f else 1f)) {
        val cell = maxWidth / 8
        val fontSp = cell.value * 0.72f
        Column {
            for (rank in 7 downTo 0) Row {
                for (file in 0 until 8) {
                    val sq = rank * 8 + file
                    val dark = (rank + file) % 2 == 0
                    var m = Modifier.size(cell).background(if (dark) Color(0xFFB58863) else Color(0xFFF0D9B5))
                    if (sq in doubtful) m = m.background(Color(0x99FFB300)).border(2.dp, Color(0xFFFF8F00))
                    if (onSquareClick != null) m = m.clickable { onSquareClick(sq) }
                    Box(m, contentAlignment = Alignment.Center) {
                        PieceGlyph(board[sq], fontSp)
                        if (file == 0) Text(
                            "${rank + 1}", Modifier.align(Alignment.TopStart),
                            fontSize = (cell.value * 0.2f).sp, color = Color(0x99000000),
                        )
                        if (rank == 0) Text(
                            "${'a' + file}", Modifier.align(Alignment.BottomEnd),
                            fontSize = (cell.value * 0.2f).sp, color = Color(0x99000000),
                        )
                    }
                }
            }
        }
    }
    if (placeholder != null) Text(
        placeholder,
        Modifier.background(Color(0xCC000000), RoundedCornerShape(8.dp)).padding(horizontal = 10.dp, vertical = 6.dp),
        color = Color.White, fontSize = 14.sp,
    )
    }
}
