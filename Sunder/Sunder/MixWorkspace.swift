import Foundation
import Observation

/// The one shared "what's currently loaded for mixing/spatializing" session,
/// read and written by both `MixerScreen` and `SpatializeScreen` -- without
/// this, the two tabs could each show a different version of "the current
/// song" (e.g. after adding or aligning a stem in one but not the other),
/// which would be a real coherence bug, not just missing polish.
///
/// Distinct from `SeparationQueue.lastSession`: that's strictly the most
/// recent separation *result*; this is whatever the user is actively
/// editing across both tabs, which may be a separation result, a plain
/// import, or either with stems added/aligned on top. Falls back to
/// `SeparationQueue.lastSession` only until the user does something here
/// that supersedes it.
@Observable
final class MixWorkspace {
    static let shared = MixWorkspace()
    private init() {}

    private var explicitSession: SeparationQueue.MixSession?
    private(set) var isImporting = false
    var importError: String?

    var session: SeparationQueue.MixSession? {
        explicitSession ?? SeparationQueue.shared.lastSession
    }

    /// Each picked file becomes one stem, labeled with its own filename.
    /// Files commonly aren't exactly the same length (unlike stems from this
    /// app's own separation, which always match) -- shorter ones are
    /// zero-padded up to the longest via AudioIO.fit rather than truncating
    /// anything.
    func importFiles(_ urls: [URL]) {
        isImporting = true
        importError = nil
        Task.detached(priority: .userInitiated) {
            do {
                let loaded = try Self.loadFiles(urls)
                let maxFrameCount = loaded.map { $0.first?.count ?? 0 }.max() ?? 0
                let stems = loaded.map { AudioIO.fit(channels: $0, toFrameCount: maxFrameCount) }
                let stemNames = urls.map { $0.deletingPathExtension().lastPathComponent }
                let session = SeparationQueue.MixSession(
                    inputURL: URL(fileURLWithPath: "Imported Mix.wav"),
                    stemNames: stemNames,
                    stems: stems
                )
                await MainActor.run {
                    self.explicitSession = session
                    self.isImporting = false
                }
            } catch {
                await MainActor.run {
                    self.importError = error.localizedDescription
                    self.isImporting = false
                }
            }
        }
    }

    /// Appends picked file(s) as new stems onto whatever session is already
    /// showing. The overall frame count can grow (a longer added file), so
    /// every existing stem is re-fit to the new max alongside the new ones --
    /// AudioIO.fit is the same pad/truncate rule either direction.
    func addStems(_ urls: [URL]) {
        guard let base = session else { return importFiles(urls) }
        isImporting = true
        importError = nil
        Task.detached(priority: .userInitiated) {
            do {
                let loaded = try Self.loadFiles(urls)
                let baseFrameCount = base.stems.first?.first?.count ?? 0
                let maxFrameCount = max(baseFrameCount, loaded.map { $0.first?.count ?? 0 }.max() ?? 0)
                let existingFitted = base.stems.map { AudioIO.fit(channels: $0, toFrameCount: maxFrameCount) }
                let newFitted = loaded.map { AudioIO.fit(channels: $0, toFrameCount: maxFrameCount) }
                let newNames = urls.map { $0.deletingPathExtension().lastPathComponent }
                let session = SeparationQueue.MixSession(
                    inputURL: base.inputURL,
                    stemNames: base.stemNames + newNames,
                    stems: existingFitted + newFitted
                )
                await MainActor.run {
                    self.explicitSession = session
                    self.isImporting = false
                }
            } catch {
                await MainActor.run {
                    self.importError = error.localizedDescription
                    self.isImporting = false
                }
            }
        }
    }

    /// Called after MixerView's Align tool replaces one stem's audio --
    /// same "supersede the fallback" rule as import/addStems above.
    func updateSession(_ session: SeparationQueue.MixSession) {
        explicitSession = session
    }

    private nonisolated static func loadFiles(_ urls: [URL]) throws -> [[[Float]]] {
        try urls.map { url in
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            return try AudioIO.loadAudio(url: url)
        }
    }
}
