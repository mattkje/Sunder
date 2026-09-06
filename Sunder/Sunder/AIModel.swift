import Foundation

/// Everything about a model that varies between architectures/checkpoints:
/// its STFT params, fixed inference chunk length (and the STFT frame count
/// that length produces -- baked into the CoreML graph's input shape, see
/// tools/convert/build_coreml.py), overlap-add stride, and stem count/names.
nonisolated struct ModelSpec {
    let nFFT: Int
    let hopLength: Int
    let winLength: Int
    let chunkSize: Int // samples per chunk, per channel
    let numOverlap: Int
    let timeFrames: Int // STFT frame count for a chunkSize-length chunk
    let stemNames: [String]

    var numFreqBins: Int { nFFT / 2 + 1 }
    var numStems: Int { stemNames.count }
}

/// The separation model to use. Each model is a compiled .mlmodelc, zipped,
/// hosted as a GitHub Release asset (see tools/convert/README.md for how
/// one is built) and downloaded on demand into Application Support --
/// nothing is bundled in the app itself, so adding a model costs the app's
/// download size nothing until someone actually picks it.
nonisolated enum AIModel: String, CaseIterable, Identifiable {
    case melBandRoformerDeux
    case bsRoformerHyperACE

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .melBandRoformerDeux: return "Mel-Band RoFormer (Deux)"
        case .bsRoformerHyperACE: return "BS-Roformer HyperACE"
        }
    }

    var detail: String {
        switch self {
        case .melBandRoformerDeux: return "Best all-round quality for both vocals and instrumental"
        case .bsRoformerHyperACE: return "Less muddy instrumental (unwa)"
        }
    }

    /// Base name of the .mlmodelc this model unpacks to (without extension).
    var resourceName: String {
        switch self {
        case .melBandRoformerDeux: return "VocalsInstrumental"
        case .bsRoformerHyperACE: return "BSRoformerHyperACE"
        }
    }

    var spec: ModelSpec {
        switch self {
        case .melBandRoformerDeux:
            // config_deux_becruily.yaml
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 573300, numOverlap: 2, timeFrames: 1301,
                stemNames: ["Vocals", "Instrumental"]
            )
        case .bsRoformerHyperACE:
            // pcunwa/BS-Roformer-HyperACE config.yaml -- note its STFT hop
            // (model.stft_hop_length) differs from Deux's, and it outputs a
            // single stem (instrumental only, no separate vocals stem).
            return ModelSpec(
                nFFT: 2048, hopLength: 512, winLength: 2048,
                chunkSize: 960000, numOverlap: 4, timeFrames: 1876,
                stemNames: ["Instrumental"]
            )
        }
    }

    /// Download URL for the zipped, precompiled .mlmodelc.
    var downloadURL: URL {
        let base = "https://github.com/mattkje/Sunder/releases/download/models-v1/"
        return URL(string: base + "\(resourceName).mlmodelc.zip")!
    }

    /// Approximate download size, for display before downloading.
    var approximateSizeMB: Int {
        switch self {
        case .melBandRoformerDeux: return 490
        case .bsRoformerHyperACE: return 270
        }
    }

    static let storageKey = "selectedModel"

    static var current: AIModel {
        let raw = UserDefaults.standard.string(forKey: storageKey)
        return raw.flatMap(AIModel.init(rawValue:)) ?? .melBandRoformerDeux
    }
}
