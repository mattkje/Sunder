import Foundation
import Observation

/// Downloads a model's zipped .mlmodelc from its GitHub Release asset into
/// Application Support, and reports per-model state so the UI can show
/// download progress. One instance, shared app-wide (@Observable so
/// ContentView's picker updates live as downloads progress).
@Observable
final class ModelDownloader {
    enum State: Equatable {
        case notDownloaded
        case downloading(progress: Double)
        case ready
        case failed(String)
    }

    static let shared = ModelDownloader()

    private var states: [AIModel: State] = [:]
    private var tasks: [AIModel: Task<Void, Never>] = [:]

    func state(for model: AIModel) -> State {
        if let cached = states[model] {
            return cached
        }
        let ready = FileManager.default.fileExists(atPath: Self.localModelURL(for: model).path)
        let initial: State = ready ? .ready : .notDownloaded
        states[model] = initial
        return initial
    }

    func download(_ model: AIModel) {
        if case .ready = state(for: model) { return }
        if case .downloading = state(for: model) { return }
        states[model] = .downloading(progress: 0)
        tasks[model] = Task.detached(priority: .userInitiated) { [self] in
            do {
                let zipURL = try await GitHubAssetDownloader.download(model.downloadURL) { progress in
                    Task { @MainActor [self] in
                        self.states[model] = .downloading(progress: progress)
                    }
                }
                try Self.install(zipURL: zipURL, model: model)
                await MainActor.run { self.states[model] = .ready }
            } catch is CancellationError {
                await MainActor.run { self.states[model] = .notDownloaded }
            } catch {
                await MainActor.run { self.states[model] = .failed(error.localizedDescription) }
            }
        }
    }

    func cancelDownload(_ model: AIModel) {
        tasks[model]?.cancel()
    }

    static func localModelURL(for model: AIModel) -> URL {
        modelsDirectory.appendingPathComponent("\(model.resourceName).mlmodelc")
    }

    private static var modelsDirectory: URL {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("Sunder/Models", isDirectory: true)
    }

    /// Unzips the downloaded archive and moves the resulting .mlmodelc into
    /// place, replacing any partial/previous copy.
    private nonisolated static func install(zipURL: URL, model: AIModel) throws {
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        let extractDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: extractDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: extractDir) }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-x", "-k", zipURL.path, extractDir.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ModelDownloaderError.unzipFailed
        }

        let extractedModelc = extractDir.appendingPathComponent("\(model.resourceName).mlmodelc")
        guard FileManager.default.fileExists(atPath: extractedModelc.path) else {
            throw ModelDownloaderError.unexpectedArchiveContents
        }

        let destination = localModelURL(for: model)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: extractedModelc, to: destination)
    }
}

enum ModelDownloaderError: LocalizedError {
    case unzipFailed
    case unexpectedArchiveContents

    var errorDescription: String? {
        switch self {
        case .unzipFailed: return "Could not unpack the downloaded model."
        case .unexpectedArchiveContents: return "Downloaded archive did not contain the expected model."
        }
    }
}

/// Thin URLSessionDownloadDelegate wrapper: downloads to a temp file while
/// reporting progress, retaining itself (and its session) for the duration
/// via GitHubAssetDownloader.active so nothing is deallocated mid-transfer.
private final class GitHubAssetDownloader: NSObject, URLSessionDownloadDelegate {
    private let progress: (Double) -> Void
    private let continuation: CheckedContinuation<URL, Error>
    private var session: URLSession!
    private var didResume = false
    private let lock = NSLock()

    private static var active: [ObjectIdentifier: GitHubAssetDownloader] = [:]
    private static let activeLock = NSLock()

    private init(progress: @escaping (Double) -> Void, continuation: CheckedContinuation<URL, Error>) {
        self.progress = progress
        self.continuation = continuation
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    static func download(_ url: URL, progress: @escaping (Double) -> Void) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            let downloader = GitHubAssetDownloader(progress: progress, continuation: continuation)
            let key = ObjectIdentifier(downloader)
            activeLock.lock(); active[key] = downloader; activeLock.unlock()
            downloader.onFinish = { activeLock.lock(); active[key] = nil; activeLock.unlock() }
            downloader.session.downloadTask(with: url).resume()
        }
    }

    private var onFinish: (() -> Void)?

    private func finish(_ result: Result<URL, Error>) {
        lock.lock()
        let alreadyResumed = didResume
        didResume = true
        lock.unlock()
        guard !alreadyResumed else { return }
        switch result {
        case .success(let url): continuation.resume(returning: url)
        case .failure(let error): continuation.resume(throwing: error)
        }
        onFinish?()
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        progress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let dest = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
        do {
            try FileManager.default.moveItem(at: location, to: dest)
            finish(.success(dest))
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            finish(.failure(error))
        }
    }
}
