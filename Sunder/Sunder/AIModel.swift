import Foundation

/// The separation model to use. Only one exists today, but the picker in
/// ContentView is built against this list so adding a second model later is
/// just a new case + a bundled .mlpackage, not a UI change.
nonisolated enum AIModel: String, CaseIterable, Identifiable {
    case melBandRoformerDeux

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .melBandRoformerDeux: return "Mel-Band RoFormer (Deux)"
        }
    }

    /// Base name of the bundled .mlmodelc resource (without extension).
    var resourceName: String {
        switch self {
        case .melBandRoformerDeux: return "VocalsInstrumental"
        }
    }

    static let storageKey = "selectedModel"

    static var current: AIModel {
        let raw = UserDefaults.standard.string(forKey: storageKey)
        return raw.flatMap(AIModel.init(rawValue:)) ?? .melBandRoformerDeux
    }
}
