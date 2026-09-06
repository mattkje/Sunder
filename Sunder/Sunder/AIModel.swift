import Foundation

/// The separation model to use. Each model is a compiled .mlmodelc, zipped,
/// hosted as a GitHub Release asset (see tools/convert/README.md for how
/// one is built) and downloaded on demand into Application Support --
/// nothing is bundled in the app itself, so adding a model costs the app's
/// download size nothing until someone actually picks it.
nonisolated enum AIModel: String, CaseIterable, Identifiable {
    case melBandRoformerDeux

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .melBandRoformerDeux: return "Mel-Band RoFormer (Deux)"
        }
    }

    var detail: String {
        switch self {
        case .melBandRoformerDeux: return "Best all-round quality for both vocals and instrumental"
        }
    }

    /// Base name of the .mlmodelc this model unpacks to (without extension).
    var resourceName: String {
        switch self {
        case .melBandRoformerDeux: return "VocalsInstrumental"
        }
    }

    /// Download URL for the zipped, precompiled .mlmodelc.
    var downloadURL: URL {
        let base = "https://github.com/mattkje/Sunder/releases/download/models-v1/"
        switch self {
        case .melBandRoformerDeux:
            return URL(string: base + "VocalsInstrumental.mlmodelc.zip")!
        }
    }

    /// Approximate download size, for display before downloading.
    var approximateSizeMB: Int {
        switch self {
        case .melBandRoformerDeux: return 490
        }
    }

    static let storageKey = "selectedModel"

    static var current: AIModel {
        let raw = UserDefaults.standard.string(forKey: storageKey)
        return raw.flatMap(AIModel.init(rawValue:)) ?? .melBandRoformerDeux
    }
}
