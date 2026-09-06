import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(\.openSettings) private var openSettings
    @State private var engine = SeparationEngine()
    @State private var downloader = ModelDownloader.shared
    @State private var inputURL: URL?
    @State private var isTargeted = false
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue
    @AppStorage(AIModel.storageKey) private var selectedModelRaw = AIModel.melBandRoformerDeux.rawValue

    private var selectedModel: AIModel {
        AIModel(rawValue: selectedModelRaw) ?? .melBandRoformerDeux
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("Separate vocals from instrumental")
                .font(.title3)
                .foregroundStyle(.secondary)

            modelPicker

            dropZone

            switch engine.state {
            case .idle:
                EmptyView()
            case .separating(let progress):
                ProgressView(value: progress)
                Text("Separating… \(Int(progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Cancel") { engine.cancel() }
            case .done(let outputs):
                VStack(alignment: .leading, spacing: 6) {
                    Label("Done", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    ForEach(outputs, id: \.self) { url in
                        Text(url.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting(outputs)
                    }
                }
            case .cancelled:
                Label("Cancelled", systemImage: "xmark.circle")
                    .foregroundStyle(.secondary)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }

            if let inputURL, isIdleOrFailed, downloader.state(for: selectedModel) == .ready {
                Button("Separate \"\(inputURL.lastPathComponent)\"") {
                    separate(inputURL)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(32)
        .frame(minWidth: 420, minHeight: 320)
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    openSettings()
                } label: {
                    Image(systemName: "gearshape")
                }
                .help("Settings")
            }
        }
    }

    private var modelPicker: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                Text("Model:")
                    .foregroundStyle(.secondary)
                Picker("", selection: $selectedModelRaw) {
                    ForEach(AIModel.allCases) { model in
                        Text(model.displayName).tag(model.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .fixedSize()
            }
            .font(.callout)

            modelStatusRow
        }
    }

    @ViewBuilder
    private var modelStatusRow: some View {
        switch downloader.state(for: selectedModel) {
        case .ready:
            EmptyView()
        case .notDownloaded:
            Button("Download Model (~\(selectedModel.approximateSizeMB) MB)") {
                downloader.download(selectedModel)
            }
            .font(.caption)
        case .downloading(let progress):
            HStack(spacing: 8) {
                ProgressView(value: progress).frame(width: 140)
                Text("Downloading… \(Int(progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Cancel") { downloader.cancelDownload(selectedModel) }
                    .font(.caption)
            }
        case .failed(let message):
            HStack(spacing: 8) {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                Button("Retry") { downloader.download(selectedModel) }
                    .font(.caption)
            }
        }
    }

    private var isIdleOrFailed: Bool {
        switch engine.state {
        case .idle, .failed, .cancelled: return true
        default: return false
        }
    }

    private var dropZone: some View {
        RoundedRectangle(cornerRadius: 12)
            .strokeBorder(isTargeted ? Color.accentColor : Color.secondary.opacity(0.4), style: StrokeStyle(lineWidth: 2, dash: [6]))
            .background(RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.05)))
            .frame(height: 140)
            .overlay {
                VStack(spacing: 8) {
                    Image(systemName: "waveform")
                        .font(.system(size: 28))
                        .foregroundStyle(.secondary)
                    Text(inputURL?.lastPathComponent ?? "Drop an audio file here, or click to choose")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            }
            .onTapGesture { chooseFile() }
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                guard let provider = providers.first else { return false }
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    if let url {
                        DispatchQueue.main.async {
                            inputURL = url
                            engine.state = .idle
                        }
                    }
                }
                return true
            }
    }

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            inputURL = url
            engine.state = .idle
        }
    }

    /// The app is sandboxed with user-selected-file access only, which is
    /// scoped to files/folders the user explicitly picks -- writing new
    /// output files next to an arbitrary opened/dropped input isn't covered,
    /// so ask where to save before running.
    private func separate(_ inputURL: URL) {
        let panel = NSOpenPanel()
        panel.title = "Choose where to save the separated tracks"
        panel.prompt = "Choose"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = inputURL.deletingLastPathComponent()
        guard panel.runModal() == .OK, let outputDir = panel.url else { return }
        let quality = OutputQuality(rawValue: qualityRaw) ?? .web
        engine.run(inputURL: inputURL, outputDir: outputDir, quality: quality, model: selectedModel)
    }
}

#Preview {
    ContentView()
}
