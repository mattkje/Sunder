import Foundation

/// The sidebar's navigation destinations. All three run through the same
/// `SeparationScreen` -- they differ only in which `AIModel`s they offer.
/// Vocal Remover/Splitter are split by what a separation model outputs
/// (`AIModel.spec.numStems`); Enhance is every restoration model
/// (`AIModel.task == .restoration`), which has no stem concept at all, so
/// it's routed on `task` rather than being swept in by a stem count.
///
/// Enhance is macOS-only (see `AIModel`'s `apolloUniversal`/
/// `apolloVocalEnhancerV2` cases, which likewise only exist on macOS) --
/// the case doesn't exist at all on iOS rather than existing with an empty
/// model list, so there's no dead sidebar entry there.
nonisolated enum AppSection: String, CaseIterable, Identifiable {
    case vocalRemover
    case splitter
    #if os(macOS)
    case enhance
    #endif

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .vocalRemover: return "Vocal Remover"
        case .splitter: return "Splitter"
        #if os(macOS)
        case .enhance: return "Enhance"
        #endif
        }
    }

    var icon: String {
        switch self {
        case .vocalRemover: return "mic.slash"
        case .splitter: return "arrow.trianglehead.branch"
        #if os(macOS)
        case .enhance: return "wand.and.stars"
        #endif
        }
    }

    var subtitle: String {
        switch self {
        case .vocalRemover: return "Strip the vocals, keep the instrumental"
        case .splitter: return "Split a song into vocals and instrumental"
        #if os(macOS)
        case .enhance: return "Clean up and enhance audio quality"
        #endif
        }
    }

    /// The verb this screen's action button/progress copy uses --
    /// "Separate" reads wrong on a restoration action.
    var verb: String {
        switch self {
        case .vocalRemover, .splitter: return "Separate"
        #if os(macOS)
        case .enhance: return "Enhance"
        #endif
        }
    }

    /// Models this screen offers, derived from each model's own properties
    /// rather than a hardcoded case list, so a newly added model lands on
    /// the right screen automatically.
    var allowedModels: [AIModel] {
        switch self {
        case .vocalRemover: return AIModel.allCases.filter { $0.task == .separation && $0.spec.numStems == 1 }
        case .splitter: return AIModel.allCases.filter { $0.task == .separation && $0.spec.numStems > 1 }
        #if os(macOS)
        case .enhance: return AIModel.allCases.filter { $0.task == .restoration }
        #endif
        }
    }
}
