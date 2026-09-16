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

    /// Everything a finished separation produced, held in memory until the
    /// Mixer's Close/save actions decide what (if anything) gets written to
    /// disk -- see MixerView/MixerPlayer.
    struct MixSession {
        let inputURL: URL
        let model: AIModel
        let stems: [[[Float]]] // [stemIndex][channel][sample]
    }

    enum Status: Equatable {
        case idle
        case separating(inputURL: URL, indexInBatch: Int, totalInBatch: Int, progress: Double)
        case done(session: MixSession)
        case cancelled
        case failed(inputURL: URL, message: String)

        static func == (lhs: Status, rhs: Status) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.cancelled, .cancelled): return true
            case let (.separating(u1, i1, t1, p1), .separating(u2, i2, t2, p2)):
                return u1 == u2 && i1 == i2 && t1 == t2 && p1 == p2
            case let (.done(s1), .done(s2)): return s1.inputURL == s2.inputURL
            case let (.failed(u1, m1), .failed(u2, m2)): return u1 == u2 && m1 == m2
            default: return false
            }
        }
    }

    private(set) var status: Status = .idle
    private var pending: [Job] = []
    private var totalInBatch = 0
    private var currentTask: Task<Void, Never>?

    var isRunning: Bool {
        if case .separating = status { return true }
        return false
    }

    /// Starts a new batch immediately. No-op while already running --
    /// callers keep file selection disabled during a run (see
    /// SeparationScreen/ContentView's `.disabled(isRunning)`), so this
    /// should never actually be hit mid-batch.
    func enqueue(_ jobs: [Job]) {
        guard !jobs.isEmpty, !isRunning else { return }
        pending = jobs
        totalInBatch = jobs.count
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
            status = .idle
            return
        }
        let job = pending.removeFirst()
        let index = totalInBatch - pending.count // 1-based position of `job` in the batch
        status = .separating(inputURL: job.inputURL, indexInBatch: index, totalInBatch: totalInBatch, progress: 0)

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
                await MainActor.run {
                    status = .done(session: session)
                }
            } catch is CancellationError {
                await MainActor.run { status = .cancelled }
            } catch {
                await MainActor.run {
                    pending.removeAll()
                    status = .failed(inputURL: job.inputURL, message: error.localizedDescription)
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
        let model = try SeparatorModel(model: aiModel)
        let demixer = Demixer(model: model)
        let stems = try demixer.separate(mix: mix, isCancelled: { Task.isCancelled }, progress: progress)

        return MixSession(inputURL: inputURL, model: aiModel, stems: stems)
    }
}
