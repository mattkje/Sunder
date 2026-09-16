import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#endif

/// Opens automatically once a separation finishes (SeparationScreen presents
/// this as a sheet whenever `SeparationQueue.shared.status` is `.done`).
/// Nothing from the separation is written to disk until one of the Save
/// actions below runs -- the stems only exist in memory
/// (`SeparationQueue.MixSession`) until then.
///
/// Mute/solo only, no per-stem volume -- that's what was asked for, not a
/// simplification of something bigger.
struct MixerView: View {
    let session: SeparationQueue.MixSession
    let onClose: () -> Void

    @State private var player: MixerPlayer?
    @State private var setupError: String?
    @Environment(\.colorScheme) private var colorScheme
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue

    #if os(iOS)
    @State private var shareItems: [URL]?
    #endif

    private var quality: OutputQuality {
        OutputQuality(rawValue: qualityRaw) ?? .web
    }

    private var stemNames: [String] { session.model.spec.stemNames }
    private var baseName: String { session.inputURL.deletingPathExtension().lastPathComponent }

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()

                if let player {
                    content(player)
                } else if let setupError {
                    Label(setupError, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .padding()
                } else {
                    ProgressView().onAppear { setUp() }
                }
            }
            .navigationTitle("Mixer")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        player?.stop()
                        onClose()
                    }
                }
            }
        }
        #if os(macOS)
        .frame(width: 480, height: 520)
        #endif
        #if os(iOS)
        .sheet(item: Binding(
            get: { shareItems.map { ShareItems(urls: $0) } },
            set: { shareItems = $0?.urls }
        )) { items in
            ShareSheet(items: items.urls)
        }
        #endif
    }

    @ViewBuilder
    private func content(_ player: MixerPlayer) -> some View {
        VStack(spacing: 0) {
            transport(player)
            Divider()
            List {
                ForEach(Array(stemNames.enumerated()), id: \.offset) { i, name in
                    stemRow(player, index: i, name: name)
                }
            }
            #if os(macOS)
            .listStyle(.inset)
            #endif
            Divider()
            saveBar(player)
        }
    }

    private func transport(_ player: MixerPlayer) -> some View {
        VStack(spacing: 8) {
            Slider(
                value: Binding(
                    get: { player.currentTime },
                    set: { player.seek(to: $0) }
                ),
                in: 0...max(player.duration, 0.01)
            )
            HStack {
                Text(timeString(player.currentTime))
                Spacer()
                Button {
                    player.togglePlayPause()
                } label: {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 32))
                }
                .buttonStyle(.plain)
                Spacer()
                Text(timeString(player.duration))
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
        }
        .padding(16)
    }

    private func stemRow(_ player: MixerPlayer, index: Int, name: String) -> some View {
        HStack(spacing: 12) {
            Text(name).font(.body)
            Spacer()
            Toggle("Mute", isOn: Binding(
                get: { player.muted[index] },
                set: { player.muted[index] = $0 }
            ))
            .toggleStyle(.button)
            .tint(.red)

            Toggle("Solo", isOn: Binding(
                get: { player.soloed.contains(index) },
                set: { on in
                    if on { player.soloed.insert(index) } else { player.soloed.remove(index) }
                }
            ))
            .toggleStyle(.button)
            .tint(.yellow)

            Button {
                saveSingle(stemIndex: index, name: name)
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }

    private func saveBar(_ player: MixerPlayer) -> some View {
        HStack(spacing: 12) {
            Button("Save All Separately") { saveAllSeparately() }
            Spacer()
            Button("Save Mix") { saveMix(player) }
                .buttonStyle(.borderedProminent)
        }
        .padding(16)
    }

    private func setUp() {
        do {
            player = try MixerPlayer(stems: session.stems)
            player?.play()
        } catch {
            setupError = error.localizedDescription
        }
    }

    private func timeString(_ t: Double) -> String {
        let total = Int(t.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    // MARK: - Save actions

    private func saveSingle(stemIndex: Int, name: String) {
        let url = tempFile(name: "\(baseName)_\(name)")
        guard (try? AudioIO.writeAudio(channels: session.stems[stemIndex], to: url, quality: quality)) != nil else { return }
        deliver([url])
    }

    private func saveMix(_ player: MixerPlayer) {
        let url = tempFile(name: "\(baseName)_Mix")
        guard (try? AudioIO.writeAudio(channels: player.audibleMix(), to: url, quality: quality)) != nil else { return }
        deliver([url])
    }

    private func saveAllSeparately() {
        var urls: [URL] = []
        for (i, name) in stemNames.enumerated() {
            let url = tempFile(name: "\(baseName)_\(name)")
            guard (try? AudioIO.writeAudio(channels: session.stems[i], to: url, quality: quality)) != nil else { continue }
            urls.append(url)
        }
        deliver(urls)
    }

    /// Creates the (fresh, per-call) scratch directory up front -- AVAudioFile
    /// requires the parent directory to already exist before it will write.
    private func tempFile(name: String) -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("\(name).\(quality.fileExtension)")
    }

    /// Writes land in a fresh temp dir (see `tempFile`) either way; this
    /// step is what actually gets them to a place the user chose.
    private func deliver(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        #if os(macOS)
        if urls.count == 1 {
            savePanel(suggesting: urls[0])
        } else {
            chooseFolderAndCopy(urls)
        }
        #else
        shareItems = urls
        #endif
    }

    #if os(macOS)
    private func savePanel(suggesting sourceURL: URL) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = sourceURL.lastPathComponent
        panel.allowedContentTypes = quality.fileExtension == "wav" ? [.wav] : [.mpeg4Audio]
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        try? FileManager.default.removeItem(at: destination)
        try? FileManager.default.copyItem(at: sourceURL, to: destination)
    }

    private func chooseFolderAndCopy(_ urls: [URL]) {
        let panel = NSOpenPanel()
        panel.title = "Choose where to save the separated tracks"
        panel.prompt = "Choose"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let outputDir = panel.url else { return }
        for url in urls {
            let destination = outputDir.appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.removeItem(at: destination)
            try? FileManager.default.copyItem(at: url, to: destination)
        }
        NSWorkspace.shared.activateFileViewerSelecting(urls.map { outputDir.appendingPathComponent($0.lastPathComponent) })
    }
    #endif
}

#if os(iOS)
private struct ShareItems: Identifiable {
    let urls: [URL]
    var id: String { urls.map(\.absoluteString).joined() }
}

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [URL]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif

#Preview {
    MixerView(
        session: .init(inputURL: URL(fileURLWithPath: "/tmp/test.mp3"), model: .melBandRoformerDeux, stems: []),
        onClose: {}
    )
}
