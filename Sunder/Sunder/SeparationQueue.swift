import Foundation
import Observation

/// The app's single separation pipeline, shared by every screen
/// (`SeparationScreen`) so at most one file is ever being processed at a
/// time. CoreML inference here is GPU/Neural-Engine- and memory-heavy
/// enough (see SeparatorModel.swift's compute-unit comments -- iOS already
/// sits near its jetsam ceiling for a single chunk of one run) that a
/// second concurrent run would only contend for the same hardware and risk
/// memory pressure, not go faster. Multiple files queue here and run one
/// after another; ContentView/SeparationScreen/SettingsView all disable
/// their interactive controls while `isRunning` so nothing can start a
/// second run, change models/settings, or navigate away mid-file.
@Observable
final class SeparationQueue {
    static let shared = SeparationQueue()

    struct Job {
        let inputURL: URL
        let model: AIModel
    }

    /// Where an unattended batch writes its stems. When a batch has one of
    /// these, finished files are written straight to disk (one subfolder per
    /// input file) and the queue moves on immediately, instead of stopping
    /// at the Mixer sheet after every file -- so a long multi-file batch can
    /// run without anyone at the keyboard.
    nonisolated struct AutoSave {
        let folder: URL
        let quality: OutputQuality
    }

    /// Everything a finished separation (or a direct file import -- see
    /// MixerScreen) produced, held in memory until the Mixer's Close/save
    /// actions decide what (if anything) gets written to disk -- see
    /// MixerView/MixerPlayer. `stemNames` is carried directly rather than as
    /// an `AIModel` reference so an imported session (no model involved) can
    /// build one the same way a separation does.
    struct MixSession {
        /// Distinct per construction -- lets MixerScreen force a fresh
        /// MixerView/MixerPlayer (`.id(session.id)`) whenever the stems
        /// themselves change (Add Stem, Align apply), since MixerPlayer
        /// builds its node graph once from an immutable `stems` array.
        let id = UUID()
        let inputURL: URL
        let stemNames: [String]
        let stems: [[[Float]]] // [stemIndex][channel][sample]
    }

    enum Status: Equatable {
        case idle
        case separating(inputURL: URL, indexInBatch: Int, totalInBatch: Int, progress: Double)
        case done(session: MixSession)
        case cancelled
        case failed(inputURL: URL, message: String)
        /// An auto-saved batch ran to the end. `failures` lists the input
        /// files (by name) that couldn't be separated or written -- an
        /// unattended batch skips those rather than stopping.
        case batchSaved(folder: URL, savedCount: Int, failures: [String])

        static func == (lhs: Status, rhs: Status) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.cancelled, .cancelled): return true
            case let (.separating(u1, i1, t1, p1), .separating(u2, i2, t2, p2)):
                return u1 == u2 && i1 == i2 && t1 == t2 && p1 == p2
            case let (.done(s1), .done(s2)): return s1.inputURL == s2.inputURL
            case let (.failed(u1, m1), .failed(u2, m2)): return u1 == u2 && m1 == m2
            case let (.batchSaved(f1, c1, x1), .batchSaved(f2, c2, x2)): return f1 == f2 && c1 == c2 && x1 == x2
            default: return false
            }
        }
    }

    private(set) var status: Status = .idle
    /// The most recent finished session, kept around after `status` moves on
    /// to `.idle`/the next queued file -- backs the persistent "Mixer"
    /// sidebar tab (MixerScreen), which needs something to show even once
    /// the post-separation sheet (driven by `status == .done`) is long gone.
    private(set) var lastSession: MixSession?
    private var pending: [Job] = []
    private var totalInBatch = 0
    private var autoSave: AutoSave?
    private var savedCount = 0
    private var failures: [String] = []
    private var currentTask: Task<Void, Never>?

    var isRunning: Bool {
        if case .separating = status { return true }
        return false
    }

    /// Starts a new batch immediately. No-op while already running --
    /// callers keep file selection disabled during a run (see
    /// SeparationScreen/ContentView's `.disabled(isRunning)`), so this
    /// should never actually be hit mid-batch. With `autoSave`, every file's
    /// stems are written to that folder and the batch never pauses for the
    /// Mixer; without it, each finished file opens the Mixer as before.
    func enqueue(_ jobs: [Job], autoSave: AutoSave? = nil) {
        guard !jobs.isEmpty, !isRunning else { return }
        pending = jobs
        totalInBatch = jobs.count
        self.autoSave = autoSave
        savedCount = 0
        failures = []
        runNext()
    }

    /// Cancels the in-flight job and drops the rest of the batch.
    func cancel() {
        currentTask?.cancel()
        pending.removeAll()
    }

    /// Clears a stale done/cancelled/failed status back to idle. No-op
    /// while running. Called when a screen (re)appears or the user starts
    /// picking new files, so an old result never lingers where it doesn't
    /// belong.
    func resetIfFinished() {
        guard !isRunning else { return }
        status = .idle
    }

    /// Called when the Mixer for a finished file is closed -- moves on to
    /// the next queued file (if any), or back to idle. Nothing is written to
    /// disk here; that already happened (or was skipped) inside the Mixer
    /// itself before it called this.
    func finishMixing() {
        guard case .done = status else { return }
        runNext()
    }

    private func runNext() {
        guard !pending.isEmpty else {
            if let autoSave {
                status = .batchSaved(folder: autoSave.folder, savedCount: savedCount, failures: failures)
                self.autoSave = nil
            } else {
                status = .idle
            }
            return
        }
        let job = pending.removeFirst()
        let index = totalInBatch - pending.count // 1-based position of `job` in the batch
        status = .separating(inputURL: job.inputURL, indexInBatch: index, totalInBatch: totalInBatch, progress: 0)

        let autoSave = autoSave
        currentTask = Task.detached(priority: .userInitiated) { [self] in
            do {
                let session = try Self.process(
                    inputURL: job.inputURL, model: job.model
                ) { progress in
                    Task { @MainActor [self] in
                        if case .separating(let url, let i, let t, _) = status, url == job.inputURL {
                            status = .separating(inputURL: url, indexInBatch: i, totalInBatch: t, progress: progress)
                        }
                    }
                }
                if let autoSave {
                    try Self.write(session, to: autoSave)
                    await MainActor.run {
                        lastSession = session
                        savedCount += 1
                        runNext()
                    }
                } else {
                    await MainActor.run {
                        status = .done(session: session)
                        lastSession = session
                    }
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.autoSave = nil
                    status = .cancelled
                }
            } catch {
                await MainActor.run {
                    if autoSave != nil {
                        // Unattended: note it and keep going with the rest.
                        failures.append(job.inputURL.lastPathComponent)
                        runNext()
                    } else {
                        pending.removeAll()
                        status = .failed(inputURL: job.inputURL, message: error.localizedDescription)
                    }
                }
            }
        }
    }

    // Explicitly nonisolated: this project defaults every declaration to
    // @MainActor (SWIFT_DEFAULT_ACTOR_ISOLATION), and without this the
    // compiler was letting this heavy, synchronous function run on the main
    // actor even when called from Task.detached -- freezing the UI for the
    // whole separation instead of only hopping back to MainActor for the
    // explicit `await MainActor.run` progress/completion updates above.
    private nonisolated static func process(
        inputURL: URL,
        model aiModel: AIModel,
        progress: @escaping (Double) -> Void
    ) throws -> MixSession {
        // NSOpenPanel-returned URLs already carry an active sandbox grant, but
        // drag-and-dropped URLs need this explicitly; harmless either way.
        let inputAccess = inputURL.startAccessingSecurityScopedResource()
        defer {
            if inputAccess { inputURL.stopAccessingSecurityScopedResource() }
        }

        let mix = try AudioIO.loadAudio(url: inputURL)
        guard let chain = aiModel.chainedSplit else {
            let stems = try separate(mix, with: aiModel, progress: progress)
            return MixSession(inputURL: inputURL, stemNames: aiModel.outputStemNames, stems: stems)
        }
        // Two passes, one model loaded at a time; each gets half the bar.
        var stems = try separate(mix, with: aiModel) { progress($0 / 2) }
        let split = try separate(stems[chain.stemIndex], with: chain.model) { progress(0.5 + $0 / 2) }
        stems.replaceSubrange(chain.stemIndex ... chain.stemIndex, with: split)
        return MixSession(inputURL: inputURL, stemNames: aiModel.outputStemNames, stems: stems)
    }

    private nonisolated static func separate(
        _ audio: [[Float]], with aiModel: AIModel, progress: (Double) -> Void
    ) throws -> [[[Float]]] {
        let demixer = Demixer(model: try SeparatorModel(model: aiModel))
        return try demixer.separate(mix: audio, isCancelled: { Task.isCancelled }, progress: progress)
    }

    /// Writes every stem of `session` into `<folder>/<input name>/`, named
    /// the same way MixerView's "Save All Separately" names them.
    private nonisolated static func write(_ session: MixSession, to autoSave: AutoSave) throws {
        let folderAccess = autoSave.folder.startAccessingSecurityScopedResource()
        defer {
            if folderAccess { autoSave.folder.stopAccessingSecurityScopedResource() }
        }

        let baseName = session.inputURL.deletingPathExtension().lastPathComponent
        let dir = autoSave.folder.appendingPathComponent(baseName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (i, name) in session.stemNames.enumerated() {
            let url = dir.appendingPathComponent("\(baseName)_\(name).\(autoSave.quality.fileExtension)")
            try? FileManager.default.removeItem(at: url)
            try AudioIO.writeAudio(channels: session.stems[i], to: url, quality: autoSave.quality)
        }
    }
}
