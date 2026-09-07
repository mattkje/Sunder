import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#endif

struct ContentView: View {
    #if os(macOS)
    @Environment(\.openSettings) private var openSettings
    #else
    @State private var showSettings = false
    #endif
    @State private var engine = SeparationEngine()
    @State private var downloader = ModelDownloader.shared
    @State private var inputURL: URL?
    @State private var isTargeted = false
    @State private var showFileImporter = false
    @State private var showURLImporter = false
    @State private var pendingURLText = ""
    @State private var urlImportError: String?
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue
    @AppStorage(AIModel.storageKey) private var selectedModelRaw = AIModel.melBandRoformerDeux.rawValue

    private var selectedModel: AIModel {
        AIModel(rawValue: selectedModelRaw) ?? .melBandRoformerDeux
    }

    var body: some View {
        // .toolbar only has a bar to render into inside a NavigationStack on
        // iOS (macOS window toolbars work without one) -- without this the
        // gear button compiled fine but was never actually visible, so
        // Settings/About were unreachable on iOS.
        #if os(iOS)
        NavigationStack { content }
        #else
        content
        #endif
    }

    private var content: some View {
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
                doneRow(outputs)
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
        #if os(macOS)
        .frame(minWidth: 340, minHeight: 380)
        #endif
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    #if os(macOS)
                    openSettings()
                    #else
                    showSettings = true
                    #endif
                } label: {
                    Image(systemName: "gearshape")
                }
                .help("Settings")
            }
        }
        .fileImporter(isPresented: $showFileImporter, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result {
                inputURL = url
                engine.state = .idle
            }
        }
        .sheet(isPresented: $showURLImporter) { urlImportSheet }
        #if os(iOS)
        .sheet(isPresented: $showSettings) { SettingsView() }
        #endif
    }

    @ViewBuilder
    private func doneRow(_ outputs: [URL]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Done", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
            ForEach(outputs, id: \.self) { url in
                Text(url.lastPathComponent).font(.caption).foregroundStyle(.secondary)
            }
            #if os(macOS)
            Button("Show in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting(outputs)
            }
            #else
            Button("Share / Save to Files") {
                presentShareSheet(for: outputs)
            }
            #endif
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

    @ViewBuilder
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
                    Text(inputURL?.lastPathComponent ?? dropZonePrompt)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            }
            .onTapGesture { showFileImporter = true }
            #if os(macOS)
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
            #endif

        Button("Paste a link…") { showURLImporter = true }
            .font(.caption)
    }

    private var dropZonePrompt: String {
        #if os(macOS)
        "Drop an audio file here, or click to choose"
        #else
        "Tap to choose an audio file"
        #endif
    }

    @ViewBuilder
    private var urlImportSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Import from a link").font(.headline)
            TextField("https://…/song.mp3", text: $pendingURLText)
                #if os(iOS)
                .textInputAutocapitalization(.never)
                .keyboardType(.URL)
                #endif
                .autocorrectionDisabled()
                .textFieldStyle(.roundedBorder)
            if let urlImportError {
                Text(urlImportError).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button("Cancel") { showURLImporter = false }
                Button("Download") { importFromPastedURL() }
                    .buttonStyle(.borderedProminent)
                    .disabled(URL(string: pendingURLText)?.scheme.map { ["http", "https"].contains($0) } != true)
            }
        }
        .padding(24)
        .frame(minWidth: 360)
    }

    private func importFromPastedURL() {
        guard let url = URL(string: pendingURLText) else { return }
        urlImportError = nil
        Task {
            do {
                let localURL = try await RemoteAudioDownloader.download(url)
                inputURL = localURL
                engine.state = .idle
                showURLImporter = false
                pendingURLText = ""
            } catch {
                urlImportError = error.localizedDescription
            }
        }
    }

    #if os(iOS)
    private func presentShareSheet(for outputs: [URL]) {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let root = scene.windows.first?.rootViewController else { return }
        let activity = UIActivityViewController(activityItems: outputs, applicationActivities: nil)
        // On iPad this presents as a popover, not a sheet, and crashes at
        // runtime without a source rect to anchor it to -- iPhone ignores
        // popoverPresentationController entirely, so this is harmless there.
        if let popover = activity.popoverPresentationController {
            popover.sourceView = root.view
            popover.sourceRect = CGRect(x: root.view.bounds.midX, y: root.view.bounds.midY, width: 0, height: 0)
            popover.permittedArrowDirections = []
        }
        root.present(activity, animated: true)
    }
    #endif

    #if os(macOS)
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
    #else
    /// No "choose a save folder" step on iOS -- there's nothing analogous to
    /// a Finder location to pick beforehand. Outputs go to a scratch
    /// directory in the app's own container, then `doneRow` above offers a
    /// share sheet (Save to Files, AirDrop, etc.) once separation finishes.
    private func separate(_ inputURL: URL) {
        let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let quality = OutputQuality(rawValue: qualityRaw) ?? .web
        engine.run(inputURL: inputURL, outputDir: outputDir, quality: quality, model: selectedModel)
    }
    #endif
}

#Preview {
    ContentView()
}
