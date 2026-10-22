package com.chessscanner.core.fusion

import com.chessscanner.core.model.Piece
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

data class FusionConfig(
    /** Decaimiento exponencial de la evidencia por fotograma. */
    val decay: Double = 0.92,
    /** Confianza mínima de una casilla para considerarla resuelta. */
    val confidenceThreshold: Double = 0.9,
    /** Evidencia mínima acumulada (suma de pesos con decaimiento) para que una casilla cuente. */
    val minEvidence: Double = 2.0,
    /** Una clase solo cuenta si gana durante esta cantidad de fotogramas seguidos. */
    val minStreak: Int = 3,
    /** Si otra clase gana esta cantidad de fotogramas seguidos, la casilla se actualiza (se movió una pieza). */
    val changeFrames: Int = 8,
    /** Tiempo con las 64 casillas confiables para declarar la posición estable. */
    val stableMs: Long = 1000,
    /**
     * Dama ↔ rey del mismo color es la confusión más común del modelo: si la casilla dice una y la otra
     * tiene al menos esta probabilidad, queda dudosa aunque supere [confidenceThreshold].
     */
    val queenKingMinOther: Double = 0.03,
)

/** Estado fusionado. Índices en coordenadas del tablero (ver [BoardFrame]). */
class FusionSnapshot(
    /** Clase vigente por casilla, o null si todavía no hay ninguna que haya ganado lo suficiente. */
    val labels: Array<Piece?>,
    val confidence: DoubleArray,
    val confident: BooleanArray,
    val stable: Boolean,
    val frames: Int,
) {
    val doubtfulCount: Int get() = confident.count { !it }
    fun labelsOrEmpty(): Array<Piece> = Array(64) { labels[it] ?: Piece.EMPTY }
}

/**
 * Fusión temporal por casilla (SPEC «Fusión temporal y multivista»). Acumula la log‑probabilidad de
 * las 13 clases con decaimiento exponencial, pesada por la calidad del fotograma y por qué tan bien se
 * ve cada casilla. No sabe nada de imágenes: recibe logits ya en coordenadas del tablero.
 */
class FusionEngine(val config: FusionConfig = FusionConfig()) {
    private val acc = Array(64) { DoubleArray(Piece.COUNT) }
    private val evidence = DoubleArray(64)
    private val frameTop = IntArray(64) { -1 }
    private val streak = IntArray(64)
    private val committed = IntArray(64) { -1 }
    private var confidentSince: Long? = null
    private var frames = 0

    fun reset() {
        acc.forEach { it.fill(0.0) }
        evidence.fill(0.0)
        frameTop.fill(-1)
        streak.fill(0)
        committed.fill(-1)
        confidentSince = null
        frames = 0
    }

    /**
     * Suma un fotograma.
     * @param logits 64 × 13, en coordenadas del tablero.
     * @param frameWeight peso del fotograma (nitidez, movimiento), en [0, 1].
     * @param cellWeights peso por casilla (qué tan de frente se ve; 0 = no visible), o null = todas 1.
     */
    fun addFrame(logits: Array<FloatArray>, frameWeight: Double, cellWeights: DoubleArray?, timestampMs: Long) {
        require(logits.size == 64)
        frames++
        val lp = DoubleArray(Piece.COUNT)
        for (i in 0 until 64) {
            val w = frameWeight * (cellWeights?.get(i) ?: 1.0)
            if (w <= 0.0) continue
            logSoftmax(logits[i], lp)
            val top = argmax(lp)
            if (top == frameTop[i]) streak[i]++ else { frameTop[i] = top; streak[i] = 1 }

            val a = acc[i]
            for (c in a.indices) a[c] = config.decay * a[c] + w * lp[c]
            evidence[i] = config.decay * evidence[i] + w

            if (committed[i] >= 0 && top != committed[i] && streak[i] >= config.changeFrames) {
                // Cambio consistente: se olvida lo anterior y se queda con la evidencia reciente.
                val n = config.changeFrames.toDouble()
                for (c in a.indices) a[c] = w * n * lp[c]
                evidence[i] = w * n
                committed[i] = top
            }
            val fused = argmax(a)
            if (fused != committed[i] && frameTop[i] == fused && streak[i] >= config.minStreak) committed[i] = fused
        }
        val all = (0 until 64).all { isConfident(it) }
        confidentSince = if (all) confidentSince ?: timestampMs else null
        lastTimestamp = timestampMs
    }

    private var lastTimestamp = 0L

    private fun posterior(i: Int, out: DoubleArray) {
        // Templado con la raíz de la evidencia: los fotogramas seguidos no son independientes.
        val t = 1.0 / sqrt(max(evidence[i], 1.0))
        val a = acc[i]
        var m = Double.NEGATIVE_INFINITY
        for (c in a.indices) m = max(m, a[c] * t)
        var sum = 0.0
        for (c in a.indices) { out[c] = exp(a[c] * t - m); sum += out[c] }
        for (c in a.indices) out[c] /= sum
    }

    private fun confidenceOf(i: Int): Double {
        val k = committed[i]
        if (k < 0) return 0.0
        val p = DoubleArray(Piece.COUNT)
        posterior(i, p)
        return p[k]
    }

    private fun isConfident(i: Int): Boolean {
        val k = committed[i]
        if (k < 0 || argmax(acc[i]) != k || evidence[i] < config.minEvidence) return false
        val p = DoubleArray(Piece.COUNT)
        posterior(i, p)
        if (p[k] < config.confidenceThreshold) return false
        val sibling = QUEEN_KING[Piece.fromIndex(k)] ?: return true
        return p[sibling.ordinal] < config.queenKingMinOther
    }

    fun snapshot(timestampMs: Long = lastTimestamp): FusionSnapshot {
        val since = confidentSince
        return FusionSnapshot(
            labels = Array(64) { committed[it].takeIf { k -> k >= 0 }?.let(Piece::fromIndex) },
            confidence = DoubleArray(64) { confidenceOf(it) },
            confident = BooleanArray(64) { isConfident(it) },
            stable = since != null && timestampMs - since >= config.stableMs,
            frames = frames,
        )
    }

    companion object {
        private val QUEEN_KING = mapOf(
            Piece.WQ to Piece.WK, Piece.WK to Piece.WQ, Piece.BQ to Piece.BK, Piece.BK to Piece.BQ,
        )

        fun logSoftmax(x: FloatArray, out: DoubleArray) {
            var m = Double.NEGATIVE_INFINITY
            for (v in x) m = max(m, v.toDouble())
            var s = 0.0
            for (v in x) s += exp(v - m)
            val l = m + ln(s)
            for (c in x.indices) out[c] = x[c] - l
        }

        fun softmax(x: FloatArray): DoubleArray {
            val out = DoubleArray(x.size)
            logSoftmax(x, out)
            for (c in out.indices) out[c] = exp(out[c])
            return out
        }

        fun argmax(x: DoubleArray): Int {
            var best = 0
            for (c in 1 until x.size) if (x[c] > x[best]) best = c
            return best
        }
    }
}

/**
 * Coordenadas del tablero persistentes. El seguimiento mantiene la identidad de las 4 esquinas del
 * tablero aunque el teléfono gire alrededor; [shift] = qué esquina del tablero (0..3, horario) aparece
 * hoy como TL de la imagen. Convierte celdas de imagen (contrato ML) a celdas del tablero.
 */
object BoardFrame {
    /** Celda del tablero para la celda de imagen (fila, col) con la esquina [shift] en TL. */
    fun boardCell(row: Int, col: Int, shift: Int): Int {
        var r = row
        var c = col
        repeat(shift and 3) { val nr = c; c = 7 - r; r = nr }
        return r * 8 + c
    }

    fun <T> toBoard(image: Array<T>, shift: Int, out: Array<T>) {
        for (i in 0 until 64) out[boardCell(i / 8, i % 8, shift)] = image[i]
    }

    fun toBoard(image: DoubleArray, shift: Int): DoubleArray {
        val out = DoubleArray(64)
        for (i in 0 until 64) out[boardCell(i / 8, i % 8, shift)] = image[i]
        return out
    }
}
