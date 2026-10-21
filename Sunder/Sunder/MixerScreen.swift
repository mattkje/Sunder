import SwiftUI
import UniformTypeIdentifiers

/// The sidebar's persistent "Mixer" destination. Unlike the sheet that pops
/// up right after a separation finishes (SeparationScreen, driven by
/// `queue.status == .done`), this stays around: it shows whatever
/// `MixWorkspace.shared.session` currently is (a separation result, an
/// import, or either with stems added/aligned on top -- see MixWorkspace),
/// so Auto Mix/Master/Align and re-exporting a mix don't require re-running
/// a separation first, and stays in sync with the Spatialize tab's own view
/// of the same session.
///
/// It also accepts files imported directly here (no separation involved):
/// "Import Files" replaces whatever's showing with the picked files (each
/// becomes one stem, named after itself); "Add Stem" instead appends picked
/// file(s) to whatever's already showing -- e.g. adding an outside vocal
/// track onto a song's already-separated stems, then lining it up with
/// MixerView's "Align" tool.
struct MixerScreen: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var workspace = MixWorkspace.shared
    @State private var showFileImporter = false
    @State private var showAddStemImporter = false

    private var session: SeparationQueue.MixSession? { workspace.session }

    var body: some View {
        ZStack {
            AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()

            if workspace.isImporting {
                ProgressView("Loading files…")
            } else if let session {
                MixerView(session: session, onClose: nil, onUpdateSession: { workspace.updateSession($0) })
                    .id(session.id)
            } else {
                emptyState
            }
        }
        .navigationTitle("Mixer")
        .toolbar {
            if session != nil {
                ToolbarItem {
                    Button {
                        showAddStemImporter = true
                    } label: {
                        Label("Add Stem", systemImage: "plus.circle")
                    }
                }
            }
            ToolbarItem {
                Button {
                    showFileImporter = true
                } label: {
                    Label("Import Files", systemImage: "plus")
                }
            }
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result, !urls.isEmpty {
                workspace.importFiles(urls)
            }
        }
        .fileImporter(
            isPresented: $showAddStemImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result, !urls.isEmpty {
                workspace.addStems(urls)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("No Mix Yet")
                .font(.title3.weight(.medium))
            Text("Separate a file from Vocal Remover or Splitter, or import your own audio files below -- each one becomes a stem you can mix and export.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            if let importError = workspace.importError {
                Label(importError, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            Button("Import Files") { showFileImporter = true }
                .buttonStyle(.borderedProminent)
        }
    }
}

#Preview {
    MixerScreen()
}
