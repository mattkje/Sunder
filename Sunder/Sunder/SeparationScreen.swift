import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#endif

/// The separation UI shared by the "Vocal Remover" and "Splitter" sidebar
/// destinations -- same `SeparationEngine`/`ModelDownloader` either way,
/// just scoped to `section.allowedModels` (never called for `.restoration`,
/// whose `allowedModels` is empty; see `ContentView`).
struct SeparationScreen: View {
    let section: AppSection

    @Environment(\.colorScheme) private var colorScheme
    @State private var queue = SeparationQueue.shared
    @State private var downloader = ModelDownloader.shared
    @State private var inputURLs: [URL] = []
    @State private var isTargeted = false
    @State private var isDropZoneHovered = false
    @State private var showFileImporter = false
    @State private var showURLImporter = false
    @State private var pendingURLText = ""
    @State private var urlImportError: String?
    @AppStorage(AIModel.storageKey) private var selectedModelRaw = AIModel.melBandRoformerDeux.rawValue

    /// Falls back within this screen's own model set (not a single global
    /// default) since Vocal Remover and Splitter offer disjoint models but
    /// share the one `@AppStorage` selection key with Settings.
    private var selectedModel: AIModel {
        if let stored = AIModel(rawValue: selectedModelRaw), section.allowedModels.contains(stored) {
            return stored
        }
        return section.allowedModels.first!
    }

    var body: some View {
        ZStack {
            AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()

            VStack(spacing: 20) {
                Text(section.subtitle)
                    .font(.title3)
                    .foregroundStyle(.secondary)

                modelPicker
                    .disabled(queue.isRunning)

                dropZone
                    .disabled(queue.isRunning)

                switch queue.status {
                case .idle:
                    EmptyView()
                case .separating(let url, let index, let total, let progress):
                    separatingCard(inputURL: url, index: index, total: total, progress: progress)
                case .done:
                    EmptyView() // MixerView (below) covers this as a sheet
                case .cancelled:
                    Label("Cancelled", systemImage: "xmark.circle")
                        .foregroundStyle(.secondary)
                case .failed(let url, let message):
                    Label("\(url.lastPathComponent): \(message)", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                }

                if !inputURLs.isEmpty, isIdleOrFailed, downloader.state(for: selectedModel) == .ready {
                    Button(separateButtonTitle) {
                        separateAll()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
            }
            .padding(32)
            .animation(.easeInOut(duration: 0.2), value: queue.status)
        }
        .navigationTitle(section.displayName)
        .onAppear { queue.resetIfFinished() }
        .fileImporter(
            isPresented: $showFileImporter, allowedContentTypes: [.audio],
            allowsMultipleSelection: allowsMultipleFiles
        ) { result in
            if case .success(let urls) = result {
                addInputURLs(urls)
            }
        }
        .sheet(isPresented: $showURLImporter) { urlImportSheet }
        .sheet(isPresented: Binding(
            get: { if case .done = queue.status { return true }; return false },
            set: { if !$0 { queue.finishMixing() } }
        )) {
            if case .done(let session) = queue.status {
                MixerView(session: session, onClose: { queue.finishMixing() })
            }
        }
    }

    /// macOS can batch multiple files into one queue; mobile stays strictly
    /// one file at a time (no queue) by never offering multi-select here.
    private var allowsMultipleFiles: Bool {
        #if os(macOS)
        true
        #else
        false
        #endif
    }

    private func addInputURLs(_ urls: [URL]) {
        queue.resetIfFinished()
        inputURLs.append(contentsOf: urls)
    }

    private var separateButtonTitle: String {
        inputURLs.count == 1
            ? "\(section.verb) \"\(inputURLs[0].lastPathComponent)\""
            : "\(section.verb) \(inputURLs.count) Files"
    }

    @ViewBuilder
    private func separatingCard(inputURL: URL, index: Int, total: Int, progress: Double) -> some View {
        VStack(spacing: 8) {
            ProgressCapsule(progress: progress)
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    if total > 1 {
                        Text("File \(index) of \(total)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Text(inputURL.lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                Text("\(Int(progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Cancel") { queue.cancel() }
                    .font(.caption)
            }
        }
    }

    /// A `Menu` rather than `Picker(pickerStyle: .menu)` for two reasons:
    /// on macOS the native menu-style picker draws its own bezel, which
    /// nested inside this row's capsule background showed as a border
    /// inside a border; and its label always reflects `selectedModel`
    /// (which falls back within this screen's own model set), so switching
    /// between Vocal Remover/Splitter/Enhance -- which share the one
    /// `@AppStorage` selection key but offer disjoint models -- never shows
    /// a blank selection the way binding straight to `$selectedModelRaw`
    /// did whenever the stored raw value wasn't one of this screen's tags.
    /// It's one card with the download/progress status below a divider,
    /// rather than a picker capsule with a separate status pill floating
    /// underneath it, since the model and its download state are the same
    /// piece of information and should read as one unit.
    private var modelPicker: some View {
        VStack(spacing: 0) {
            Menu {
                ForEach(section.allowedModels) { model in
                    Button {
                        selectedModelRaw = model.rawValue
                    } label: {
                        if model == selectedModel {
                            Label(model.displayName, systemImage: "checkmark")
                        } else {
                            Text(model.displayName)
                        }
                    }
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: selectedModel.icon)
                        .foregroundStyle(.secondary)
                    Text(selectedModel.displayName)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .font(.callout)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .menuIndicator(.hidden)

            if downloader.state(for: selectedModel) != .ready {
                Divider()
                modelStatusRow
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.secondary.opacity(0.08))
        )
        .fixedSize()
        .animation(.easeInOut(duration: 0.2), value: downloader.state(for: selectedModel))
    }

    @ViewBuilder
    private var modelStatusRow: some View {
        switch downloader.state(for: selectedModel) {
        case .ready:
            EmptyView()
        case .notDownloaded:
            Button {
                downloader.download(selectedModel)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle.fill")
                    Text("Download Model")
                    Text("· ~\(selectedModel.approximateSizeMB) MB")
                        .foregroundStyle(.secondary)
                }
                .font(.caption.weight(.medium))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.primary)
        case .downloading(let progress):
            HStack(spacing: 8) {
                ProgressCapsule(progress: progress).frame(width: 100)
                Text("\(Int(progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Cancel") { downloader.cancelDownload(selectedModel) }
                    .buttonStyle(.plain)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .failed(let message):
            HStack(spacing: 8) {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
                Button("Retry") { downloader.download(selectedModel) }
                    .buttonStyle(.plain)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.primary)
            }
        }
    }

    private var isIdleOrFailed: Bool {
        switch queue.status {
        case .idle, .failed, .cancelled: return true
        default: return false
        }
    }

    private var dropZone: some View {
        VStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(.regularMaterial)
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(
                        isTargeted ? Color.accentColor : Color.secondary.opacity(0.25),
                        style: StrokeStyle(lineWidth: isTargeted ? 2 : 1.5, dash: inputURLs.isEmpty ? [7, 6] : [])
                    )

                if inputURLs.isEmpty {
                    emptyDropPrompt
                } else {
                    selectedFilesList
                }
            }
            .frame(minHeight: 130)
            .shadow(color: .black.opacity(isTargeted ? 0.12 : 0.05), radius: isTargeted ? 16 : 8, x: 0, y: 4)
            .scaleEffect(isTargeted ? 1.015 : 1)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isTargeted)
            .animation(.easeInOut(duration: 0.15), value: isDropZoneHovered)
            .onTapGesture { if inputURLs.isEmpty { showFileImporter = true } }
            #if os(macOS)
            .onHover { isDropZoneHovered = $0 }
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                for provider in providers {
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        if let url {
                            DispatchQueue.main.async {
                                addInputURLs([url])
                            }
                        }
                    }
                }
                return true
            }
            #endif

            HStack(spacing: 10) {
                PillButton(title: "Choose File", systemImage: "folder") {
                    showFileImporter = true
                }
                PillButton(title: "Paste a Link", systemImage: "link") {
                    showURLImporter = true
                }
            }
        }
    }

    private var emptyDropPrompt: some View {
        VStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(.primary)
                    .frame(width: 48, height: 48)
                Image(systemName: "waveform")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.background)
            }
            .scaleEffect(isTargeted || isDropZoneHovered ? 1.1 : 1)

            Text(dropZonePrompt)
                .font(.callout.weight(.medium))
                .foregroundStyle(.primary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
    }

    private var selectedFilesList: some View {
        VStack(spacing: 8) {
            ForEach(inputURLs, id: \.self) { url in
                selectedFileRow(url)
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
    }

    private func selectedFileRow(_ url: URL) -> some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(.primary)
                    .frame(width: 44, height: 44)
                Image(systemName: "waveform")
                    .foregroundStyle(.background)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(url.lastPathComponent)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("Ready to \(section.verb.lowercased())")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button {
                inputURLs.removeAll { $0 == url }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
    }

    private var dropZonePrompt: String {
        #if os(macOS)
        "Drop audio files here, or choose them below"
        #else
        "Choose an audio file below"
        #endif
    }

    @ViewBuilder
    private var urlImportSheet: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 10) {
                ZStack {
                    Circle().fill(.primary).frame(width: 34, height: 34)
                    Image(systemName: "link")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.background)
                }
                Text("Import from a link").font(.headline)
            }

            TextField("https://…/song.mp3", text: $pendingURLText)
                #if os(iOS)
                .textInputAutocapitalization(.never)
                .keyboardType(.URL)
                #endif
                .autocorrectionDisabled()
                .textFieldStyle(.plain)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.secondary.opacity(0.08))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Color.secondary.opacity(0.2))
                )

            if let urlImportError {
                Label(urlImportError, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
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
        .background(AppTheme.backgroundGradient(for: colorScheme))
    }

    private func importFromPastedURL() {
        guard let url = URL(string: pendingURLText) else { return }
        urlImportError = nil
        Task {
            do {
                let localURL = try await RemoteAudioDownloader.download(url)
                addInputURLs([localURL])
                showURLImporter = false
                pendingURLText = ""
            } catch {
                urlImportError = error.localizedDescription
            }
        }
    }

    /// Output location is no longer chosen here -- MixerView asks for it (if
    /// at all) once the user actually saves something, after mixing.
    private func separateAll() {
        let jobs = inputURLs.map { SeparationQueue.Job(inputURL: $0, model: selectedModel) }
        inputURLs = []
        queue.enqueue(jobs)
    }
}

/// Secondary action pill ("Choose File" / "Paste a Link") with a hover
/// highlight -- same rounded/hover visual grammar as AboutView's LinkButton,
/// sized for a full-width two-up row instead of a compact icon link.
private struct PillButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 13, weight: .medium))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(isHovered ? Color.accentColor.opacity(0.15) : Color.secondary.opacity(0.1))
                )
                .foregroundColor(isHovered ? .accentColor : .primary)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .animation(.easeInOut(duration: 0.15), value: isHovered)
    }
}

/// Rounded-capsule progress bar used in place of the default linear
/// `ProgressView` for separation/download progress -- matches the app's
/// capsule/pill visual language (see AboutView's version badge).
private struct ProgressCapsule: View {
    let progress: Double

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.secondary.opacity(0.15))
                Capsule()
                    .fill(Color.accentColor)
                    .frame(width: geo.size.width * max(0, min(1, progress)))
            }
        }
        .frame(height: 6)
        .animation(.easeInOut(duration: 0.2), value: progress)
    }
}

#Preview {
    SeparationScreen(section: .vocalRemover)
}
