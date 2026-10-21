// swift-tools-version: 6.0
import PackageDescription

// Voice Changer engine for Sunder (macOS only): RVC v2 voice-model training and
// conversion. The trainable synthesizer runs in MLX; the frozen feature
// extractors (ContentVec, RMVPE) run in CoreML like the rest of the app.
let package = Package(
    name: "VoiceKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "VoiceKit", targets: ["VoiceKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
    ],
    targets: [
        .target(
            name: "VoiceKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Parity checks against PyTorch fixtures and a headless train/convert
        // runner. Build with xcodebuild (MLX needs its Metal library).
        .executableTarget(
            name: "voicekit-cli",
            dependencies: ["VoiceKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
