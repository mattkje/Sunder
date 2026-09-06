import Foundation

/// Constants from config_deux_becruily.yaml (becruily/mel-band-roformer-deux).
/// Keep in sync with tools/convert/checkpoint/config_deux_becruily.yaml and
/// the CoreML graph produced by tools/convert/build_coreml.py.
nonisolated enum ModelConfig {
    static let sampleRate: Double = 44100
    static let channels = 2

    // STFT
    static let nFFT = 2048
    static let hopLength = 441
    static let winLength = 2048
    static let numFreqBins = nFFT / 2 + 1 // 1025

    // Inference chunking (config's `inference:` section)
    static let chunkSize = 573300 // samples per chunk, per channel
    static let numOverlap = 2
    static let timeFrames = 1301 // STFT frame count for a chunkSize-length chunk (see build_coreml.py)

    static let stemNames = ["Vocals", "Instrumental"]
    static let numStems = stemNames.count
}
