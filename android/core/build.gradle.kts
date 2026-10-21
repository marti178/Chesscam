plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    // Solo la API Java de OpenCV: en el teléfono la aporta el AAR oficial, en la PC el jar de openpnp.
    compileOnly(libs.opencv.desktop)
    testImplementation(libs.opencv.desktop)
    testImplementation(libs.junit)
}

val evalRuntime by configurations.creating {
    extendsFrom(configurations.runtimeClasspath.get())
}
dependencies { evalRuntime(libs.opencv.desktop) }

// Herramienta de escritorio: corre el detector de tablero (el mismo código que usa la app) sobre eval/.
//   ./gradlew :core:evalBoards -Pimages=<carpeta> [-Plabels=<json>] [-Pout=<carpeta>]
tasks.register<JavaExec>("evalBoards") {
    group = "verification"
    description = "Corre el detector de tablero sobre una carpeta de imágenes y reporta cuántas detecta bien."
    classpath = sourceSets["main"].output + evalRuntime
    mainClass.set("com.chessscanner.core.eval.BoardEvalKt")
    val root = rootProject.projectDir.parentFile
    args(
        (findProperty("images") as String?) ?: root.resolve("eval/images").path,
        (findProperty("labels") as String?) ?: root.resolve("eval/boards.json").path,
        (findProperty("out") as String?) ?: layout.buildDirectory.dir("eval").get().asFile.path,
    )
}

tasks.test {
    // Vectores dorados de ml/ (opcional): -PgoldenDir=<ml/golden>
    (findProperty("goldenDir") as String?)?.let { systemProperty("golden.dir", it) }
}
