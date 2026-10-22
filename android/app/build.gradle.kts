plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.chessscanner.app"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.chessscanner.live"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // Teléfonos reales (arm64) y emulador x86_64; deja afuera ABIs viejas para achicar el APK.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        noCompress += "tflite"
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":core"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    debugImplementation(libs.androidx.compose.ui.tooling)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.opencv.android)
    implementation(libs.litert)
    testImplementation(libs.junit)
    androidTestImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.junit)
}

// Vectores dorados de ml/golden dentro del APK de test (no del de la app):
//   ./gradlew connectedDebugAndroidTest -PgoldenDir=<ml/golden>
val goldenAssets = layout.buildDirectory.dir("golden-assets")
val syncGoldenAssets by tasks.registering(Sync::class) {
    (findProperty("goldenDir") as String?)?.let { dir ->
        from(dir) { include("*/case.json", "*/image.png", "*/image.jpg", "*/input_u8.bin", "*/output_logits_f32.bin") }
    }
    into(goldenAssets.map { it.dir("golden") })
}
android.sourceSets.getByName("androidTest").assets.directories.add(goldenAssets.get().asFile.path)
tasks.matching { it.name == "mergeDebugAndroidTestAssets" }.configureEach { dependsOn(syncGoldenAssets) }
