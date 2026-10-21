#if os(macOS)
import AVFoundation
import SwiftUI
import UniformTypeIdentifiers
import VoiceKit

/// The sidebar's "Voice Changer" destination (macOS only): create voice
/// models from vocal recordings (on-device RVC fine-tuning, see VoiceKit),
/// then convert a Mixer stem or any file to one of them.
struct VoiceScreen: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var workspace = VoiceWorkspace.shared
    @State private var assets = VoiceAssetDownloader.shared
    @State private var seedVCAssets = VoiceAssetDownloader.seedVC
    @State private var mix = MixWorkspace.shared
    @State private var showNewVoice = false
    @State private var retrainVoice: VoiceProfile?
    @State private var renaming: VoiceProfile?
    @State private var renameText = ""

    // Convert form
    @State private var useMixStem = true
    @State private var stemIndex = 0
    @State private var fileURL: URL?
    @State private var showFileImporter = false
    @State private var voiceID: UUID?
    @State private var settings = ConversionSettings()
    @AppStorage("voiceChangerEnhance") private var enhance = true
    @State private var downloader = ModelDownloader.shared
    @State private var player: AVAudioPlayer?

    var body: some View {
        ZStack {
            AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    Text("Train a voice from vocal recordings, then make any vocal sound like it.")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    if assets.state == .ready {
                        voicesSection
                        convertSection
                    } else {
                        downloadCard
                    }
                    if let error = workspace.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                    }
                }
                .padding(32)
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
            }
        }
        .navigationTitle("Voice Changer")
        .sheet(isPresented: $showNewVoice) { NewVoiceSheet() }
        .sheet(item: $retrainVoice) { NewVoiceSheet(existing: $0) }
        .fileImporter(isPresented: $showFileImporter, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result { fileURL = url }
        }
        .alert("Rename Voice", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $renameText)
            Button("Rename") {
                if let renaming, !renameText.trimmingCharacters(in: .whitespaces).isEmpty {
                    workspace.rename(renaming, to: renameText)
                }
                renaming = nil
            }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
        .onAppear(perform: pickDefaults)
        .onChange(of: workspace.voices) { pickDefaults() }
        .onDisappear { player?.stop() }
    }

    // MARK: Download

    private var downloadCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Voice Changer Models", systemImage: "shippingbox")
                .font(.headline)
            Text("Voice training and conversion need a one-time download: the content and pitch analyzers plus the pretrained voice model every new voice starts from.")
                .font(.callout)
                .foregroundStyle(.secondary)
            switch assets.state {
            case .downloading(let progress):
                HStack {
                    ProgressCapsule(progress: progress)
                    Button("Cancel") { assets.cancelDownload() }
                }
            case .failed(let message):
                Text(message).font(.caption).foregroundStyle(.red)
                Button("Retry") { assets.download() }
            default:
                Button("Download (~\(assets.approximateSizeMB) MB)") { assets.download() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.background.opacity(0.6)))
    }

    // MARK: Voices

    private var voicesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Voices").font(.headline)
                Spacer()
                Button {
                    importVoice()
                } label: {
                    Label("Import…", systemImage: "square.and.arrow.down")
                }
                Button {
                    showNewVoice = true
                } label: {
                    Label("New Voice…", systemImage: "plus")
                }
                .disabled(workspace.job != .idle)
            }
            if workspace.voices.isEmpty {
                Text("No voices yet. Add 10–30 minutes of clean, solo vocals (no music or reverb) to train one.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            ForEach(workspace.voices) { voice in
                voiceRow(voice)
            }
            trainingCard
        }
    }

    private func voiceRow(_ voice: VoiceProfile) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform.and.person.filled")
                .foregroundStyle(.background)
                .frame(width: 28, height: 28)
                .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(.primary))
            VStack(alignment: .leading, spacing: 2) {
                Text(voice.name)
                Text(status(of: voice)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if workspace.job == .idle, !voice.isTrained {
                // Prepared data on disk -> continue from the checkpoint;
                // otherwise (preparation never finished, e.g. it crashed or
                // was stopped) start over from freshly picked recordings.
                if voice.canResume {
                    Button("Resume Training") { workspace.resume(voice) }
                        .font(.caption)
                } else {
                    Button("Train…") { retrainVoice = voice }
                        .font(.caption)
                }
            }
            Menu {
                Button("Rename…") { renameText = voice.name; renaming = voice }
                if voice.isTrained {
                    Button("Export…") { exportVoice(voice) }
                }
                Button("Delete", role: .destructive) { workspace.delete(voice) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .disabled(isTraining(voice))
        }
        .padding(.vertical, 2)
    }

    private var voiceFileType: UTType {
        UTType(filenameExtension: VoiceWorkspace.fileExtension) ?? .zip
    }

    private func exportVoice(_ voice: VoiceProfile) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(voice.name).\(VoiceWorkspace.fileExtension)"
        panel.allowedContentTypes = [voiceFileType]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        workspace.exportVoice(voice, to: url)
    }

    private func importVoice() {
        let panel = NSOpenPanel()
        panel.title = "Import a voice"
        panel.allowedContentTypes = [voiceFileType]
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        workspace.importVoice(from: url)
    }

    private func status(of voice: VoiceProfile) -> String {
        let minutes = String(format: "%.1f min of audio", voice.trainingSeconds / 60)
        if voice.engine == .seedVC {
            return voice.completedEpochs == 0
                ? "Ready · Seed-VC · from a reference clip"
                : "Ready · Seed-VC · \(minutes) · \(voice.completedEpochs) steps"
        }
        if voice.isTrained { return "Ready · \(minutes) · \(voice.completedEpochs) epochs" }
        if isTraining(voice) { return "Training…" }
        if voice.canResume { return "Paused at epoch \(voice.completedEpochs) of \(voice.targetEpochs) · \(minutes)" }
        return "Not trained"
    }

    private func isTraining(_ voice: VoiceProfile) -> Bool { workspace.trainingVoiceID == voice.id }

    @ViewBuilder
    private var trainingCard: some View {
        switch workspace.job {
        case .preparing(let voice, let progress):
            jobCard(title: progress < 1 ? "Preparing \"\(voice)\" — analyzing recordings" : "Saving \"\(voice)\"",
                    progress: progress, detail: nil, stoppable: progress < 1)
        case let .training(voice, epoch, epochs, step, steps, secondsLeft):
            jobCard(title: "Training \"\(voice)\" — epoch \(epoch) of \(epochs)",
                    progress: (Double(epoch - 1) + Double(step) / Double(steps)) / Double(epochs),
                    detail: secondsLeft.map { "About \(Self.format($0)) left · progress is saved every 5 epochs" },
                    stoppable: true)
        case .stopping:
            jobCard(title: "Stopping — saving progress…", progress: nil, detail: nil, stoppable: false)
        default:
            EmptyView()
        }
    }

    private func jobCard(title: String, progress: Double?, detail: String?, stoppable: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.callout)
            HStack {
                if let progress { ProgressCapsule(progress: progress) } else { ProgressView().controlSize(.small) }
                if stoppable {
                    Button("Stop") { workspace.stop() }
                        .help("Stops training; you can resume later from the last checkpoint.")
                }
            }
            if let detail { Text(detail).font(.caption).foregroundStyle(.secondary) }
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.background.opacity(0.6)))
    }

    static func format(_ seconds: Double) -> String {
        let f = DateComponentsFormatter()
        f.allowedUnits = seconds >= 3600 ? [.hour, .minute] : [.minute]
        f.unitsStyle = .short
        return f.string(from: max(60, seconds)) ?? ""
    }

    // MARK: Convert

    private var trainedVoices: [VoiceProfile] { workspace.voices.filter(\.isTrained) }

    private var convertSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Convert").font(.headline)
            if trainedVoices.isEmpty {
                Text("Once a voice has finished training, convert vocals to it here.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                Form {
                    Picker("Source", selection: $useMixStem) {
                        Text("Mixer stem").tag(true)
                        Text("Audio file").tag(false)
                    }
                    .pickerStyle(.segmented)
                    if useMixStem {
                        if let session = mix.session {
                            Picker("Stem", selection: $stemIndex) {
                                ForEach(session.stemNames.indices, id: \.self) { Text(session.stemNames[$0]).tag($0) }
                            }
                        } else {
                            Text("Nothing in the Mixer yet — separate a song first, or pick an audio file.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        HStack {
                            Text(fileURL?.lastPathComponent ?? "No file chosen")
                                .foregroundStyle(fileURL == nil ? .secondary : .primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button("Choose…") { showFileImporter = true }
                        }
                    }
                    Picker("Voice", selection: $voiceID) {
                        ForEach(trainedVoices) { Text($0.name).tag(Optional($0.id)) }
                    }
                    LabeledContent("Pitch shift") {
                        Stepper(value: $settings.pitchShift, in: -24 ... 24, step: 1) {
                            Text(settings.pitchShift == 0 ? "None" : String(format: "%+.0f semitones", settings.pitchShift))
                                .monospacedDigit()
                        }
                    }
                    if selectedVoice?.engine == .seedVC {
                        seedVCRows
                    } else {
                        LabeledContent("Voice likeness") {
                            Slider(value: $settings.indexRate, in: 0 ... 1)
                                .help("How strongly the sound is pulled toward the trained voice's recordings. Higher sounds more like the voice but can blur words.")
                        }
                        LabeledContent("Protect consonants") {
                            Slider(value: $settings.protect, in: 0 ... 0.5)
                                .help("Keeps breaths and consonants closer to the original. Lower protects more; far right turns it off.")
                        }
                    }
                    enhanceRow
                }
                .formStyle(.grouped)
                .scrollDisabled(true)
                .disabled(workspace.job != .idle)

                HStack {
                    Button("Convert") { startConversion() }
                        .buttonStyle(.borderedProminent)
                        .disabled(!canConvert)
                    if case .converting(let progress) = workspace.job {
                        ProgressCapsule(progress: progress)
                    } else if workspace.job == .loading {
                        ProgressView().controlSize(.small)
                    }
                    if workspace.job.isTraining {
                        Text("Available when training finishes or is stopped.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if let result = workspace.result { resultCard(result) }
            }
        }
    }

    private var selectedVoice: VoiceProfile? { trainedVoices.first { $0.id == voiceID } }

    /// Seed-VC's controls, and its shared models if they aren't downloaded yet.
    @ViewBuilder
    private var seedVCRows: some View {
        switch seedVCAssets.state {
        case .ready:
            LabeledContent("Quality") {
                Picker("Quality", selection: $settings.diffusionSteps) {
                    Text("Draft").tag(10)
                    Text("Good").tag(20)
                    Text("Best").tag(30)
                    Text("Max").tag(50)
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .help("Diffusion steps per 30 seconds of audio. More steps sound cleaner and take longer.")
            }
        case .downloading(let progress):
            LabeledContent("Seed-VC models") { ProgressView(value: progress).frame(width: 120) }
        case .failed(let message):
            LabeledContent("Seed-VC models") {
                Button("Retry") { seedVCAssets.download() }.help(message)
            }
        case .notDownloaded:
            LabeledContent("Seed-VC models") {
                Button("Download (~\(seedVCAssets.approximateSizeMB) MB)") { seedVCAssets.download() }
            }
        }
    }

    /// Optional clean-up pass with the Enhance section's Vocal Enhancer v2.
    @ViewBuilder
    private var enhanceRow: some View {
        let model = AIModel.apolloVocalEnhancerV2
        switch downloader.state(for: model) {
        case .ready:
            Toggle("Clean up with Vocal Enhancer", isOn: $enhance)
                .help("Runs the result through Vocal Enhancer v2 to reduce the raspy, compressed sound. Takes a little longer.")
        case .downloading(let progress):
            LabeledContent("Clean up with Vocal Enhancer") { ProgressView(value: progress).frame(width: 120) }
        default:
            LabeledContent("Clean up with Vocal Enhancer") {
                Button("Download (~\(model.approximateSizeMB) MB)") { downloader.download(model) }
            }
        }
    }

    private var canConvert: Bool {
        guard workspace.job == .idle, let voice = selectedVoice else { return false }
        if voice.engine == .seedVC, seedVCAssets.state != .ready { return false }
        return useMixStem ? mix.session.map { $0.stemNames.indices.contains(stemIndex) } ?? false : fileURL != nil
    }

    private func startConversion() {
        guard let voice = trainedVoices.first(where: { $0.id == voiceID }) else { return }
        player?.stop()
        workspace.clearResult()
        let enhanceReady = downloader.state(for: .apolloVocalEnhancerV2) == .ready
        workspace.convert(useMixStem ? .mixStem(index: stemIndex) : .file(fileURL!), voice: voice, settings: settings,
                          enhance: enhance && enhanceReady)
    }

    private func resultCard(_ result: VoiceWorkspace.Result) -> some View {
        HStack(spacing: 12) {
            Button {
                if player?.isPlaying == true {
                    player?.stop()
                    player = nil
                } else {
                    player = try? AVAudioPlayer(contentsOf: result.previewURL)
                    player?.play()
                }
            } label: {
                Image(systemName: player?.isPlaying == true ? "stop.fill" : "play.fill")
                    .foregroundStyle(.background)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(.primary))
            }
            .buttonStyle(.plain)
            Text(result.name).lineLimit(1).truncationMode(.middle)
            Spacer()
            if case .mixStem = result.source {
                Button("Replace in Mixer") { workspace.replaceStemInMix() }
                    .help("Swaps this stem in the Mixer/Spatialize session for the converted one.")
            }
            Button("Export…") { AudioExport.deliverMac([result.previewURL], quality: .current) }
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.background.opacity(0.6)))
    }

    private func pickDefaults() {
        if voiceID == nil || !trainedVoices.contains(where: { $0.id == voiceID }) {
            voiceID = trainedVoices.first?.id
        }
        if let names = mix.session?.stemNames, let vocals = names.firstIndex(where: { $0.localizedCaseInsensitiveContains("vocal") }) {
            stemIndex = vocals
        }
    }
}

/// "New Voice" sheet: name, recordings, how much usable audio they contain,
/// and the training length.
/// Also used as "Train…" for an existing voice whose recordings were never
/// prepared (`existing`): same form, minus the name.
private struct NewVoiceSheet: View {
    var existing: VoiceProfile?
    @Environment(\.dismiss) private var dismiss
    @State private var workspace = VoiceWorkspace.shared
    @State private var name = ""
    @State private var files: [URL] = []
    @State private var usableSeconds: Double?
    @State private var analyzing = false
    @State private var epochs = 200
    @State private var showImporter = false
    @State private var analyzeError: String?
    @State private var engine: VoiceModelKind = .rvc

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(existing.map { "Train \"\($0.name)\"" } ?? "New Voice").font(.title2.weight(.semibold))
            if existing == nil {
                Picker("Model", selection: $engine) {
                    Text("RVC").tag(VoiceModelKind.rvc)
                    Text("Seed-VC").tag(VoiceModelKind.seedVC)
                }
                .pickerStyle(.segmented)
            }
            if engine == .seedVC {
                seedVCInfo
            } else {
                rvcForm
            }
        }
        .padding(24)
        .frame(width: 520)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio], allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                files += urls.filter { !files.contains($0) }
                analyze()
            }
        }
    }

    /// Seed-VC voices are trained on Sunder Server, not on this Mac, and
    /// arrive here as .sundervoice files.
    private var seedVCInfo: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Seed-VC sounds far more natural than RVC, but training it takes a GPU, so it runs on Sunder Server: create a job there with Model set to Seed-VC, then import the .sundervoice it produces. A voice can also be made from one clean clip, without training, with `python -m sundervoice clip`.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                Button("Import Voice…") {
                    let panel = NSOpenPanel()
                    panel.title = "Import a voice"
                    panel.allowedContentTypes = [UTType(filenameExtension: VoiceWorkspace.fileExtension) ?? .zip]
                    panel.allowsMultipleSelection = false
                    guard panel.runModal() == .OK, let url = panel.url else { return }
                    workspace.importVoice(from: url)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private var rvcForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            Form {
                if existing == nil {
                    TextField("Name", text: $name)
                }
                LabeledContent("Recordings") {
                    VStack(alignment: .trailing, spacing: 4) {
                        ForEach(files, id: \.self) { Text($0.lastPathComponent).font(.caption).lineLimit(1).truncationMode(.middle) }
                        Button(files.isEmpty ? "Add Files…" : "Add More…") { showImporter = true }
                    }
                }
                LabeledContent("Usable audio") { usableLabel }
                LabeledContent("Training length") {
                    Stepper(value: $epochs, in: 10 ... 1000, step: 10) { Text("\(epochs) epochs").monospacedDigit() }
                }
            }
            .formStyle(.grouped)
            Text("Use clean, solo vocals — no music, reverb or backing vocals (run a song through Vocal Remover first). 10–30 minutes works best. Training runs on this Mac and can take a few hours; you can stop and resume it. Only use your own voice or voices you have permission to use.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                Button("Start Training") {
                    if let existing {
                        workspace.train(existing, files: files, epochs: epochs)
                    } else {
                        workspace.createVoice(name: name.trimmingCharacters(in: .whitespaces), files: files, epochs: epochs)
                    }
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .disabled((existing == nil && name.trimmingCharacters(in: .whitespaces).isEmpty) || files.isEmpty || analyzing || (usableSeconds ?? 0) < 60)
            }
        }
    }

    @ViewBuilder
    private var usableLabel: some View {
        if analyzing {
            ProgressView().controlSize(.small)
        } else if let analyzeError {
            Text(analyzeError).font(.caption).foregroundStyle(.red)
        } else if let seconds = usableSeconds {
            let minutes = seconds / 60
            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "%.1f minutes", minutes)).monospacedDigit()
                if seconds < 60 {
                    Text("Too short — add at least a minute.").font(.caption).foregroundStyle(.red)
                } else if minutes < 5 {
                    Text("Short — expect a rough likeness. 10–30 min is best.").font(.caption).foregroundStyle(.orange)
                }
            }
        } else {
            Text("—").foregroundStyle(.secondary)
        }
    }

    private func analyze() {
        analyzing = true
        analyzeError = nil
        let urls = files
        Task {
            do {
                let seconds = try await VoiceWorkspace.usableSeconds(of: urls)
                usableSeconds = seconds
                epochs = VoiceWorkspace.suggestedEpochs(forSeconds: seconds)
            } catch {
                analyzeError = error.localizedDescription
            }
            analyzing = false
        }
    }
}
#endif
