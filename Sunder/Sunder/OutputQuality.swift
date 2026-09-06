import AVFoundation
import Foundation

/// Output encoding presets. Uncompressed 16-bit WAV is ~5x the size of a
/// reasonable AAC encode for the same audio, so "Web" (the default) trades
/// some bitrate for a much smaller file; "Lossless" keeps the original
/// uncompressed behavior for anyone who wants it.
nonisolated enum OutputQuality: Int, CaseIterable, Identifiable {
    case web = 0
    case standard = 1
    case high = 2
    case lossless = 3

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .web: return "Web"
        case .standard: return "Standard"
        case .high: return "High"
        case .lossless: return "Lossless"
        }
    }

    var detail: String {
        switch self {
        case .web: return "AAC, 128 kbps — smallest files"
        case .standard: return "AAC, 192 kbps"
        case .high: return "AAC, 256 kbps"
        case .lossless: return "16-bit WAV — largest files, no compression"
        }
    }

    var fileExtension: String {
        self == .lossless ? "wav" : "m4a"
    }

    private var bitRate: Int? {
        switch self {
        case .web: return 128_000
        case .standard: return 192_000
        case .high: return 256_000
        case .lossless: return nil
        }
    }

    func fileSettings(channels: Int) -> [String: Any] {
        switch self {
        case .lossless:
            return [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: ModelConfig.sampleRate,
                AVNumberOfChannelsKey: channels,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ]
        case .web, .standard, .high:
            return [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: ModelConfig.sampleRate,
                AVNumberOfChannelsKey: channels,
                AVEncoderBitRateKey: bitRate!,
            ]
        }
    }

    static let storageKey = "outputQuality"

    static var current: OutputQuality {
        let raw = UserDefaults.standard.object(forKey: storageKey) as? Int
        return raw.flatMap(OutputQuality.init(rawValue:)) ?? .web
    }
}
