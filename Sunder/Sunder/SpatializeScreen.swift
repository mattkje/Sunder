import SwiftUI

/// The sidebar's "Spatialize" destination -- shows the same session
/// `MixerScreen` shows (via `MixWorkspace.shared`), so adding/aligning a
/// stem in Mixer and then switching here sees the same, up-to-date song.
/// No import UI of its own; that's Mixer's job.
struct SpatializeScreen: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var workspace = MixWorkspace.shared

    var body: some View {
        ZStack {
            AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()

            if let session = workspace.session {
                SpatializeView(session: session)
                    .id(session.id)
            } else {
                emptyState
            }
        }
        .navigationTitle("Spatialize")
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "move.3d")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("Nothing to Spatialize Yet")
                .font(.title3.weight(.medium))
            Text("Separate a file, or import one in the Mixer tab, then come back here to place its stems in 3D.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }
}

#Preview {
    SpatializeScreen()
}
