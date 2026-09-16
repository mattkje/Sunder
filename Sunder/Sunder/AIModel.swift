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

/// What a model does with its one audio input: split it into stems
/// (`.separation`, `AIModel.spec.numStems` stems out) or restore/enhance it
/// in place (`.restoration`, always a single output regardless of stem
/// count elsewhere -- see `AppSection.allowedModels`, which routes on this
/// rather than inferring it from `numStems`).
nonisolated enum ModelTask {
    case separation
    case restoration
}

/// The separation/restoration model to use. Each model is an .mlpackage,
/// zipped, hosted as a GitHub Release asset (see tools/convert/README.md
/// for how one is built) and downloaded on demand into Application Support
/// -- nothing is bundled in the app itself, so adding a model costs the
/// app's download size nothing until someone actually picks it.
///
/// BS-Roformer HyperACE (unwa) is not included: its mask estimator embeds an
/// extra CNN + custom "hypergraph attention" head well beyond the other
/// models' plain per-band MLP, and hasn't been converted yet.
nonisolated enum AIModel: String, CaseIterable, Identifiable {
    case melBandRoformerDeux
    case bsRoformerResurrectionInst
    case melRoformerGaboxFv7
    case melRoformerUnwaV1ePlus
    case bsRoformer4Stems
    case bsRoformer6StemsSW
    // Enhance (restoration) is macOS-only: Apollo's per-chunk memory/compute
    // cost wasn't validated against iOS's tight jetsam ceiling the way the
    // separation models were (see resourceName's Deux comment), and no
    // Mobile-chunk variant was built -- these cases don't exist at all in
    // the iOS binary, matching AppSection.enhance's own macOS-only case.
    #if os(macOS)
    case apolloUniversal
    case apolloVocalEnhancerV2
    #endif

    var id: String { rawValue }

    var task: ModelTask {
        switch self {
        case .melBandRoformerDeux, .bsRoformerResurrectionInst, .melRoformerGaboxFv7,
             .melRoformerUnwaV1ePlus, .bsRoformer4Stems, .bsRoformer6StemsSW:
            return .separation
        #if os(macOS)
        case .apolloUniversal, .apolloVocalEnhancerV2:
            return .restoration
        #endif
        }
    }

    var displayName: String {
        switch self {
        case .melBandRoformerDeux: return "Mel-Band RoFormer (Deux)"
        case .bsRoformerResurrectionInst: return "BS-Roformer Resurrection"
        case .melRoformerGaboxFv7: return "Mel-RoFormer Gabox Fv7"
        case .melRoformerUnwaV1ePlus: return "Mel-RoFormer unwa v1e+"
        case .bsRoformer4Stems: return "BS-RoFormer 4-Stem (ZFTurbo)"
        case .bsRoformer6StemsSW: return "BS-RoFormer 6-Stem SW (jarredou)"
        #if os(macOS)
        case .apolloUniversal: return "Apollo Universal (Lew)"
        case .apolloVocalEnhancerV2: return "Vocal Enhancer v2 Beta (Lew)"
        #endif
        }
    }

    var detail: String {
        switch self {
        case .melBandRoformerDeux: return "Best all-round quality for both vocals and instrumental"
        case .bsRoformerResurrectionInst: return "Fast, instrumental only (unwa)"
        case .melRoformerGaboxFv7: return "Instrumental only, full-bodied mix (Gabox)"
        case .melRoformerUnwaV1ePlus: return "Instrumental only (unwa)"
        case .bsRoformer4Stems: return "Splits into vocals, drums, bass, and other (ZFTurbo)"
        case .bsRoformer6StemsSW: return "Splits into vocals, drums, bass, guitar, piano, and other (jarredou)"
        #if os(macOS)
        case .apolloUniversal: return "General restoration and upscaling for any music (Lew)"
        case .apolloVocalEnhancerV2: return "Restores and enhances vocal quality -- v2 beta (Lew)"
        #endif
        }
    }

    /// (author, license-or-nil) for the Credits page and About's disclaimer --
    /// derived here rather than hand-listed on each view, so a newly added
    /// model's credit line appears automatically (same reasoning as
    /// AppSection.allowedModels deriving from AIModel's own properties).
    /// `license` is nil for community checkpoints with no formally named
    /// license attached (most of them) -- not an omission, there just isn't
    /// one to state.
    var attribution: (author: String, license: String?) {
        switch self {
        case .melBandRoformerDeux: return ("becruily", "CC BY-NC 4.0")
        case .bsRoformerResurrectionInst: return ("unwa", nil)
        case .melRoformerGaboxFv7: return ("Gabox", nil)
        case .melRoformerUnwaV1ePlus: return ("unwa", nil)
        case .bsRoformer4Stems: return ("ZFTurbo", "MIT")
        case .bsRoformer6StemsSW: return ("jarredou", nil)
        #if os(macOS)
        case .apolloUniversal, .apolloVocalEnhancerV2: return ("Lew", nil)
        #endif
        }
    }

    /// SF Symbol shown next to this model in the picker and Settings list --
    /// purely cosmetic, distinct per model so the two lists are scannable.
    var icon: String {
        switch self {
        case .melBandRoformerDeux: return "mic.fill"
        case .bsRoformerResurrectionInst: return "waveform"
        case .melRoformerGaboxFv7: return "waveform.path"
        case .melRoformerUnwaV1ePlus: return "waveform.badge.magnifyingglass"
        case .bsRoformer4Stems: return "rectangle.split.2x2"
        case .bsRoformer6StemsSW: return "square.grid.3x2"
        #if os(macOS)
        case .apolloUniversal: return "sparkles"
        case .apolloVocalEnhancerV2: return "person.wave.2.fill"
        #endif
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
        case .bsRoformer4Stems:
            #if os(iOS)
            return "BSRoformer4StemsMobile"
            #else
            return "BSRoformer4Stems"
            #endif
        case .bsRoformer6StemsSW:
            #if os(iOS)
            return "BSRoformer6StemsSWMobile"
            #else
            return "BSRoformer6StemsSW"
            #endif
        #if os(macOS)
        case .apolloUniversal:
            return "ApolloUniversal"
        case .apolloVocalEnhancerV2:
            // Matches the already-uploaded/verified release asset's name
            // (built as LewVocalEnhancerV2.mlpackage) -- not ApolloVocalEnhancerV2,
            // which was a naming mismatch that made downloadURL 404.
            return "LewVocalEnhancerV2"
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
        case .bsRoformer4Stems:
            // config_bs_roformer_384_8_2_485100.yaml (ZFTurbo/Music-Source-Separation-Training
            // release v1.0.12) -- 4 stems. stemNames order matches the
            // checkpoint's own training.instruments list exactly
            // (['drums', 'bass', 'other', 'vocals']); SeparationQueue zips
            // this to the model's output tensors by index, so it isn't just
            // display order. iOS uses a shorter-chunk variant, same
            // reasoning as Deux above.
            #if os(iOS)
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 242550, numOverlap: 2, timeFrames: 551,
                stemNames: ["Drums", "Bass", "Other", "Vocals"]
            )
            #else
            return ModelSpec(
                nFFT: 2048, hopLength: 441, winLength: 2048,
                chunkSize: 485100, numOverlap: 2, timeFrames: 1101,
                stemNames: ["Drums", "Bass", "Other", "Vocals"]
            )
            #endif
        case .bsRoformer6StemsSW:
            // BS-Rofo-SW-Fixed.yaml (jarredou, via openmirlab/bs-roformer-infer's
            // enerjazzer mirror -- original jarredou HF host is gone) -- 6 stems,
            // hop=512 (unlike every other model here, which uses 441; DSP.swift's
            // STFTProcessor takes hopLength as a plain parameter so this needs no
            // special-casing there). stemNames order matches the checkpoint's own
            // training.instruments list exactly (['bass', 'drums', 'other',
            // 'vocals', 'guitar', 'piano']), same rule as bsRoformer4Stems above.
            // iOS uses a shorter-chunk variant, same reasoning as Deux above; its
            // on-device memory footprint hasn't been validated (no iOS Simulator
            // was available when this was converted).
            #if os(iOS)
            return ModelSpec(
                nFFT: 2048, hopLength: 512, winLength: 2048,
                chunkSize: 294400, numOverlap: 2, timeFrames: 576,
                stemNames: ["Bass", "Drums", "Other", "Vocals", "Guitar", "Piano"]
            )
            #else
            return ModelSpec(
                nFFT: 2048, hopLength: 512, winLength: 2048,
                chunkSize: 588800, numOverlap: 2, timeFrames: 1151,
                stemNames: ["Bass", "Drums", "Other", "Vocals", "Guitar", "Piano"]
            )
            #endif
        #if os(macOS)
        case .apolloUniversal:
            // Apollo's window is 20ms @ 44.1kHz: nFFT = 44100*20/1000 = 882
            // (hop = nFFT/2 = 441) -- not a power of two, unlike every other
            // model here, so DSP.swift's STFTProcessor routes this through
            // its Bluestein (arbitrary-length) path instead of vDSP_fft_zrip.
            // Single output stem -- Apollo reconstructs one restored
            // stream, it doesn't separate stems (see AIModel.task /
            // wrapper_apollo.py). macOS-only, no Mobile variant (see the
            // case declarations' doc comment). This checkpoint's
            // feature_dim is 2x the Vocal Enhancer's (384 vs 192), which
            // made converting it at a 10s chunk (matching Vocal Enhancer)
            // impractically slow -- 3s (chunk_size=132300, matching this
            // checkpoint's own config default) converts in line with the
            // other models instead, at the cost of slightly more overlap
            // seams per minute of audio (Demixer's overlap-add already
            // handles this generically).
            return ModelSpec(
                nFFT: 882, hopLength: 441, winLength: 882,
                chunkSize: 132300, numOverlap: 2, timeFrames: 301,
                stemNames: ["Enhanced"]
            )
        case .apolloVocalEnhancerV2:
            // Same architecture as apolloUniversal above (nFFT=882 via the
            // Bluestein path, single output stem) -- lighter feature_dim
            // (192) converts fine at a full 10s chunk.
            return ModelSpec(
                nFFT: 882, hopLength: 441, winLength: 882,
                chunkSize: 441000, numOverlap: 2, timeFrames: 1001,
                stemNames: ["Enhanced"]
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
        case .bsRoformer4Stems: return 504
        case .bsRoformer6StemsSW: return 403
        #if os(macOS)
        case .apolloUniversal: return 140
        case .apolloVocalEnhancerV2: return 36
        #endif
        }
    }

    static let storageKey = "selectedModel"

    static var current: AIModel {
        let raw = UserDefaults.standard.string(forKey: storageKey)
        return raw.flatMap(AIModel.init(rawValue:)) ?? .melBandRoformerDeux
    }
}
