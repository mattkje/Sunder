import SwiftUI

/// Shared style tokens so the main screen, Settings, and the About window
/// read as one visual language instead of each rolling its own platform
/// color / gradient logic.
enum PlatformColorToken {
    case windowBackground
    case controlBackground
}

extension Color {
    init(platformColor token: PlatformColorToken) {
        #if os(macOS)
        switch token {
        case .windowBackground: self = Color(nsColor: .windowBackgroundColor)
        case .controlBackground: self = Color(nsColor: .controlBackgroundColor)
        }
        #else
        switch token {
        case .windowBackground: self = Color(uiColor: .systemBackground)
        case .controlBackground: self = Color(uiColor: .secondarySystemBackground)
        }
        #endif
    }
}

enum AppTheme {
    /// The soft window/control-background diagonal gradient used behind
    /// every top-level screen (About, main screen, Settings).
    static func backgroundGradient(for colorScheme: ColorScheme) -> LinearGradient {
        LinearGradient(
            colors: colorScheme == .dark
                ? [Color(platformColor: .windowBackground), Color(platformColor: .controlBackground)]
                : [Color(platformColor: .controlBackground), Color(platformColor: .windowBackground)],
            startPoint: .top,
            endPoint: .bottom
        )
    }
}
