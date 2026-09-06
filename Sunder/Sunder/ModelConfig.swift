import Foundation

/// Truly universal constants across every model in this app: all of them
/// (Mel-Band-Roformer, BS-Roformer, ...) are trained on 44.1kHz stereo audio.
/// Everything else (STFT params, chunking, stem count/names) varies per
/// model -- see AIModel.spec / ModelSpec.
nonisolated enum ModelConfig {
    static let sampleRate: Double = 44100
    static let channels = 2
}
