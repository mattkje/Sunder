import Foundation

/// Everything about a model that varies between architectures/checkpoints:
/// its STFT params, fixed inference chunk length (and the STFT frame count
/// that length produces -- baked into the CoreML graph's input shape, see
/// tools/convert/build_coreml*.py), overlap-add stride, and stem count/names.
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
///
/// BS-Roformer HyperACE (unwa) is not included: its mask estimator embeds an
/// extra CNN + custom "hypergraph attention" head well beyond the other
/// models' plain per-band MLP, and hasn't been converted yet.
nonisolated enum AIModel: String, CaseIterable, Identifiable {
    case melBandRoformerDeux
    case bsRoformerResurrectionInst
    case melRoformerGaboxFv7
    case melRoformerUnwaV1ePlus

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .melBandRoformerDeux: return "Mel-Band RoFormer (Deux)"
        case .bsRoformerResurrectionInst: return "BS-Roformer Resurrection"
        case .melRoformerGaboxFv7: return "Mel-RoFormer Gabox Fv7"
        case .melRoformerUnwaV1ePlus: return "Mel-RoFormer unwa v1e+"
        }
    }

    var detail: String {
        switch self {
        case .melBandRoformerDeux: return "Best all-round quality for both vocals and instrumental"
        case .bsRoformerResurrectionInst: return "Fast, instrumental only (unwa)"
        case .melRoformerGaboxFv7: return "Instrumental only, full-bodied mix (Gabox)"
        case .melRoformerUnwaV1ePlus: return "Instrumental only (unwa)"
        }
    }

    /// Base name of the .mlmodelc this model unpacks to (without extension).
    var resourceName: String {
        switch self {
        case .melBandRoformerDeux: return "VocalsInstrumental"
        case .bsRoformerResurrectionInst: return "BSRoformerInstResurrection"
        case .melRoformerGaboxFv7: return "MelRoformerGaboxFv7"
        case .melRoformerUnwaV1ePlus: return "MelRoformerUnwaV1ePlus"
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
        case .bsRoformerResurrectionInst:
            // config_BandSplit-Roformer_Resurrection_Instrumental_by-Unwa.yaml --
            // single stem (instrumental only).
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 749259, numOverlap: 2, timeFrames: 1700,
                stemNames: ["Instrumental"]
            )
        case .melRoformerGaboxFv7, .melRoformerUnwaV1ePlus:
            // config_melband_roformer_inst_gabox.yaml / config_melband_roformer_inst.yaml --
            // both share identical audio/inference params; both single stem.
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 485100, numOverlap: 2, timeFrames: 1101,
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
        case .bsRoformerResurrectionInst: return 182
        case .melRoformerGaboxFv7, .melRoformerUnwaV1ePlus: return 810
        }
    }

    static let storageKey = "selectedModel"

    static var current: AIModel {
        let raw = UserDefaults.standard.string(forKey: storageKey)
        return raw.flatMap(AIModel.init(rawValue:)) ?? .melBandRoformerDeux
    }
}
