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

/// The separation model to use. Each model is an .mlpackage, zipped, hosted
/// as a GitHub Release asset (see tools/convert/README.md for how one is
/// built) and downloaded on demand into Application Support -- nothing is
/// bundled in the app itself, so adding a model costs the app's download
/// size nothing until someone actually picks it.
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
        case .melBandRoformerDeux:
            // A single chunk of Deux's full T=1301 model was measured
            // exceeding iOS's ~3.3GB jetsam ceiling regardless of compute
            // backend (CPU+GPU, CPU-only, and Neural Engine all hit the same
            // limit) -- the model's actual peak memory per chunk is just too
            // big for a phone. iOS gets a second package converted at half
            // the sequence length (T=651, see tools/convert/build_coreml.py)
            // instead; macOS has no such ceiling and keeps the full model.
            #if os(iOS)
            return "VocalsInstrumentalMobile"
            #else
            return "VocalsInstrumental"
            #endif
        case .bsRoformerResurrectionInst:
            #if os(iOS)
            return "BSRoformerInstResurrectionMobile"
            #else
            return "BSRoformerInstResurrection"
            #endif
        case .melRoformerGaboxFv7:
            #if os(iOS)
            return "MelRoformerGaboxFv7Mobile"
            #else
            return "MelRoformerGaboxFv7"
            #endif
        case .melRoformerUnwaV1ePlus:
            #if os(iOS)
            return "MelRoformerUnwaV1ePlusMobile"
            #else
            return "MelRoformerUnwaV1ePlus"
            #endif
        }
    }

    var spec: ModelSpec {
        switch self {
        case .melBandRoformerDeux:
            // config_deux_becruily.yaml -- see resourceName's doc comment on
            // why iOS uses a shorter-chunk variant of the same checkpoint.
            #if os(iOS)
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 286650, numOverlap: 2, timeFrames: 651,
                stemNames: ["Vocals", "Instrumental"]
            )
            #else
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 573300, numOverlap: 2, timeFrames: 1301,
                stemNames: ["Vocals", "Instrumental"]
            )
            #endif
        case .bsRoformerResurrectionInst:
            // config_BandSplit-Roformer_Resurrection_Instrumental_by-Unwa.yaml --
            // single stem (instrumental only). iOS uses a shorter-chunk
            // variant, same reasoning as Deux above.
            #if os(iOS)
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 374409, numOverlap: 2, timeFrames: 850,
                stemNames: ["Instrumental"]
            )
            #else
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 749259, numOverlap: 2, timeFrames: 1700,
                stemNames: ["Instrumental"]
            )
            #endif
        case .melRoformerGaboxFv7, .melRoformerUnwaV1ePlus:
            // config_melband_roformer_inst_gabox.yaml / config_melband_roformer_inst.yaml --
            // both share identical audio/inference params; both single stem.
            // iOS uses a shorter-chunk variant, same reasoning as Deux above.
            #if os(iOS)
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 242550, numOverlap: 2, timeFrames: 551,
                stemNames: ["Instrumental"]
            )
            #else
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 485100, numOverlap: 2, timeFrames: 1101,
                stemNames: ["Instrumental"]
            )
            #endif
        }
    }

    /// Download URL for the zipped .mlpackage (compiled to .mlmodelc
    /// on-device by ModelDownloader.install -- see its doc comment for why).
    var downloadURL: URL {
        let base = "https://github.com/mattkje/Sunder/releases/download/models-v1/"
        return URL(string: base + "\(resourceName).mlpackage.zip")!
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
