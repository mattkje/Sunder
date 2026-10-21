#if os(macOS)
import Foundation
import Observation
import VoiceKit

/// App-side state for the Voice Changer: the user's voices, the one running
/// voice job (training or converting -- they share the GPU and the feature
/// models, so only one at a time), and the latest conversion result.
/// Training can run for hours, so unlike separation it doesn't lock the
/// sidebar; progress is checkpointed and a stopped voice can be resumed.
@Observable
final class VoiceWorkspace {
    static let shared = VoiceWorkspace()

    enum Job: Equatable {
        case idle
        case loading // conversion starting up
        case preparing(voice: String, progress: Double)
        case training(voice: String, epoch: Int, epochs: Int, step: Int, steps: Int, secondsLeft: Double?)
        case stopping
        case converting(progress: Double)

        var isTraining: Bool {
            switch self {
            case .preparing, .training, .stopping: true
            default: false
            }
        }
    }

    enum Source: Hashable {
        case file(URL)
        case mixStem(index: Int)
    }

    struct Result {
        let name: String
        let channels: [[Float]] // 44.1 kHz; mono conversion duplicated, or the enhancer's stereo
        let previewURL: URL
        let source: Source
    }

    private(set) var voices: [VoiceProfile] = []
    private(set) var job: Job = .idle
    private(set) var trainingVoiceID: UUID?
    private(set) var result: Result?
    var errorMessage: String?

    let library: VoiceLibrary
    private var engine: VoiceEngine?
    private let stopFlag = StopFlag()
    /// Identifies the running job, so a progress update that lands after the
    /// job finished (they hop to the main actor asynchronously) is dropped.
    private var runID: UUID?
    /// Held while training so the Mac doesn't idle-sleep mid-run (the display
    /// may still sleep). Closing the lid still sleeps it.
    private var sleepActivity: NSObjectProtocol?

    private init() {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        library = VoiceLibrary(root: base.appendingPathComponent("Sunder/Voices", isDirectory: true))
        try? FileManager.default.createDirectory(at: library.root, withIntermediateDirectories: true)
        refresh()
    }

    func refresh() { voices = library.list() }

    // MARK: Training

    /// RVC-style budget: ~8k optimizer steps whatever the dataset size
    /// (batches of four ~3.4 s segments), clamped to 50...300 epochs.
    static func suggestedEpochs(forSeconds seconds: Double) -> Int {
        guard seconds > 0 else { return 200 }
        return min(300, max(50, Int((8000 * 4 * 3.4 / seconds).rounded())))
    }

    /// Seconds of usable vocal audio across `urls` (silence excluded).
    nonisolated static func usableSeconds(of urls: [URL]) async throws -> Double {
        try await Task.detached(priority: .userInitiated) {
            try urls.reduce(0) { $0 + (try VoiceEngine.usableSeconds(of: loadMono(url: $1))) }
        }.value
    }

    func createVoice(name: String, files: [URL], epochs: Int) {
        do {
            let profile = try library.create(name: name, epochs: epochs)
            refresh()
            runTraining(profile, files: files)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func resume(_ profile: VoiceProfile) { runTraining(profile, files: []) }

    /// Starts an existing, never-prepared voice over from new recordings.
    func train(_ profile: VoiceProfile, files: [URL], epochs: Int) {
        var profile = profile
        profile.targetEpochs = epochs
        profile.completedEpochs = 0
        do {
            try library.save(profile)
            refresh()
            runTraining(profile, files: files)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stop() {
        guard job.isTraining else { return }
        stopFlag.set(true)
        job = .stopping
    }

    private func runTraining(_ profile: VoiceProfile, files: [URL]) {
        guard job == .idle else { return }
        stopFlag.set(false)
        errorMessage = nil
        job = .preparing(voice: profile.name, progress: 0)
        trainingVoiceID = profile.id
        sleepActivity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled],
            reason: "Training voice \"\(profile.name)\"")
        let run = UUID()
        runID = run
        let library = self.library, stopFlag = self.stopFlag
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let engine = try await self.loadEngine()
                let sources = try files.map(Self.loadMono)
                var stepStart = Date()
                var secondsPerStep: Double?
                _ = try engine.train(profile, sources: sources, library: library, progress: { progress in
                    let job: Job
                    switch progress {
                    case .preparing(let fraction):
                        job = .preparing(voice: profile.name, progress: fraction)
                    case let .training(epoch, epochs, step, steps, _):
                        let elapsed = Date().timeIntervalSince(stepStart)
                        stepStart = Date()
                        // Smoothed; the first step of each epoch includes graph compilation.
                        if step > 1 { secondsPerStep = secondsPerStep.map { $0 * 0.95 + elapsed * 0.05 } ?? elapsed }
                        let remaining = Double((epochs - epoch) * steps + (steps - step))
                        job = .training(voice: profile.name, epoch: epoch, epochs: epochs, step: step, steps: steps,
                                        secondsLeft: secondsPerStep.map { $0 * remaining })
                    case .finishing:
                        job = .preparing(voice: profile.name, progress: 1)
                    }
                    Task { @MainActor in if self.runID == run, self.job != .stopping { self.job = job } }
                }, shouldStop: { stopFlag.get() })
                await MainActor.run { self.finishJob() }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.finishJob()
                }
            }
        }
    }

    func delete(_ profile: VoiceProfile) {
        try? library.delete(profile.id)
        refresh()
    }

    func rename(_ profile: VoiceProfile, to name: String) {
        var renamed = profile
        renamed.name = name
        try? library.save(renamed)
        refresh()
    }

    // MARK: Export / import

    /// A `.sundervoice` file is a plain zip of the voice's portable files
    /// (see `VoiceLibrary.stageExport`). File work only -- it never touches
    /// the engine, so it's fine while a training or conversion job runs.
    static let fileExtension = "sundervoice"

    func exportVoice(_ profile: VoiceProfile, to destination: URL) {
        let library = self.library
        runFileTask {
            let stage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
            defer { try? FileManager.default.removeItem(at: stage) }
            let voiceFolder = stage.appendingPathComponent("Voice", isDirectory: true)
            try library.stageExport(profile, into: voiceFolder)

            // `.forUploading` hands back a zip of the folder (Finder's own
            // "Compress"), which ZipReader reads back on import.
            var coordinatorError: NSError?
            var copyError: Error?
            NSFileCoordinator().coordinate(readingItemAt: voiceFolder, options: .forUploading, error: &coordinatorError) { zipURL in
                do {
                    try? FileManager.default.removeItem(at: destination)
                    try FileManager.default.copyItem(at: zipURL, to: destination)
                } catch {
                    copyError = error
                }
            }
            if let error = coordinatorError ?? copyError { throw error }
        }
    }

    func importVoice(from source: URL) {
        let library = self.library
        runFileTask {
            let access = source.startAccessingSecurityScopedResource()
            defer { if access { source.stopAccessingSecurityScopedResource() } }
            let stage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
            defer { try? FileManager.default.removeItem(at: stage) }
            do {
                try ZipReader.unzipItem(at: source, to: stage)
            } catch {
                throw VoiceError.invalidExport
            }
            // The archive nests the files one folder down; find them wherever
            // profile.json landed rather than assuming that folder's name.
            let profileURL = FileManager.default.enumerator(at: stage, includingPropertiesForKeys: nil)?
                .compactMap { $0 as? URL }
                .first { $0.lastPathComponent == "profile.json" }
            guard let folder = profileURL?.deletingLastPathComponent() else { throw VoiceError.invalidExport }
            _ = try library.importVoice(from: folder)
        }
    }

    private func runFileTask(_ work: @escaping @Sendable () throws -> Void) {
        errorMessage = nil
        Task.detached(priority: .userInitiated) { [self] in
            do {
                try work()
                await MainActor.run { refresh() }
            } catch {
                await MainActor.run { errorMessage = error.localizedDescription }
            }
        }
    }

    // MARK: Conversion

    /// `enhance` runs the result through the Vocal Enhancer v2 restoration
    /// model (the Enhance section's Apollo) -- it cleans up the raspy,
    /// band-limited sound RVC's vocoder leaves. Requires that model downloaded.
    /// Seed-VC voices also need the Seed-VC bundle (VoiceAssetDownloader.seedVC).
    func convert(_ source: Source, voice: VoiceProfile, settings: ConversionSettings, enhance: Bool) {
        guard job == .idle else { return }
        errorMessage = nil
        job = .loading
        let run = UUID()
        runID = run
        let library = self.library
        let (audio, name): ([Float], String)
        switch source {
        case .mixStem(let index):
            guard let session = MixWorkspace.shared.session, session.stems.indices.contains(index) else { job = .idle; return }
            audio = AudioIO.mixToMono(session.stems[index])
            name = session.stemNames[index]
        case .file(let url):
            audio = []
            name = url.deletingPathExtension().lastPathComponent
        }
        let title = "\(name) (\(voice.name))"
        let quality = OutputQuality.current
        let previewURL = AudioExport.tempFile(name: title, quality: quality)
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let engine = try await self.loadEngine()
                let seedVC = await MainActor.run { VoiceAssetDownloader.seedVC.seedVCAssets }
                let input: [Float]
                if case .file(let url) = source { input = try Self.loadMono(url: url) } else { input = audio }
                let share = enhance ? 0.6 : 1.0 // progress split between the two passes
                let report = { (fraction: Double) in
                    Task { @MainActor in if self.runID == run { self.job = .converting(progress: fraction) } }
                }
                let output = try engine.convert(input, with: voice, library: library, settings: settings, seedVC: seedVC) { report($0 * share) }
                var channels = [output, output]
                if enhance {
                    let enhancer = Demixer(model: try SeparatorModel(model: .apolloVocalEnhancerV2))
                    channels = try enhancer.separate(mix: channels) { report(share + $0 * (1 - share)) }[0]
                }
                try AudioIO.writeAudio(channels: channels, to: previewURL, quality: quality)
                await MainActor.run {
                    self.result = Result(name: title, channels: channels, previewURL: previewURL, source: source)
                    self.finishJob()
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.finishJob()
                }
            }
        }
    }

    /// Swaps the converted audio in for the stem it came from, in the shared
    /// Mixer/Spatialize session.
    func replaceStemInMix() {
        guard let result, case .mixStem(let index) = result.source,
              let session = MixWorkspace.shared.session, session.stems.indices.contains(index) else { return }
        let frames = session.stems[index].first?.count ?? result.channels[0].count
        var stems = session.stems
        let channels = stems[index].indices.map { result.channels[min($0, result.channels.count - 1)] }
        stems[index] = AudioIO.fit(channels: channels, toFrameCount: frames)
        var names = session.stemNames
        names[index] = result.name
        MixWorkspace.shared.updateSession(SeparationQueue.MixSession(inputURL: session.inputURL, stemNames: names, stems: stems))
    }

    func clearResult() { result = nil }

    // MARK: Helpers

    private func finishJob() {
        runID = nil
        trainingVoiceID = nil
        if let sleepActivity { ProcessInfo.processInfo.endActivity(sleepActivity) }
        sleepActivity = nil
        job = .idle
        refresh()
    }

    /// The engine loads both CoreML feature models, so it's built once, off
    /// the main actor, on first use.
    private func loadEngine() async throws -> VoiceEngine {
        // Checked first so a model bundle deleted in Settings isn't masked by
        // an engine that loaded it earlier.
        guard let assets = await MainActor.run(body: { VoiceAssetDownloader.shared.assets }) else {
            throw CocoaError(.fileNoSuchFile)
        }
        if let engine = await MainActor.run(body: { self.engine }) { return engine }
        let engine = try VoiceEngine(assets: assets)
        await MainActor.run { self.engine = engine }
        return engine
    }

    private nonisolated static func loadMono(url: URL) throws -> [Float] {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        return AudioIO.mixToMono(try AudioIO.loadAudio(url: url))
    }
}

/// Lock-protected flag read from the training thread.
private nonisolated final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func set(_ v: Bool) { lock.lock(); value = v; lock.unlock() }
    func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
}
#endif
