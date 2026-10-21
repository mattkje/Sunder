#if os(macOS)
import CoreML
import Foundation
import Observation
import VoiceKit

/// Downloads one of the Voice Changer's model bundles -- each a zipped GitHub
/// Release asset of .mlpackages (compiled on-device like ModelDownloader's
/// models) and plain weight files -- and installs it into its own folder.
/// Kept apart from `AIModel`/`ModelDownloader` because these aren't
/// separation models: no spec, no picker, and a bundle's files are only ever
/// used together.
/// - `shared`: RVC -- ContentVec + RMVPE and the pretrained RVC v2
///   generator/discriminator every new voice is fine-tuned from. Every voice
///   needs it (Seed-VC uses its RMVPE).
/// - `seedVC`: the Whisper encoder and BigVGAN vocoder every Seed-VC voice
///   shares (tools/convert/build_coreml_seedvc.py).
@Observable
final class VoiceAssetDownloader {
    nonisolated struct ModelBundle: Sendable {
        let downloadURL: URL
        let approximateSizeMB: Int
        let folder: String
        let mlModels: [String]
        let files: [String]
    }

    static let shared = VoiceAssetDownloader(ModelBundle(
        downloadURL: URL(string: "https://github.com/mattkje/Sunder/releases/download/models-v1/VoiceChanger-v1.zip")!,
        approximateSizeMB: 710, folder: "VoiceChanger",
        mlModels: ["ContentVec", "RMVPE"], files: ["RVCBaseG.safetensors", "RVCBaseD.safetensors"]))

    static let seedVC = VoiceAssetDownloader(ModelBundle(
        downloadURL: URL(string: "https://github.com/mattkje/Sunder/releases/download/models-v1/SeedVC-v1.zip")!,
        approximateSizeMB: 620, folder: "SeedVC",
        mlModels: ["SeedVCWhisper", "SeedVCVocoder"], files: []))

    let bundle: ModelBundle
    private(set) var state: ModelDownloader.State
    private var task: Task<Void, Never>?

    private init(_ bundle: ModelBundle) {
        self.bundle = bundle
        state = Self.isInstalled(bundle) ? .ready : .notDownloaded
    }

    var approximateSizeMB: Int { bundle.approximateSizeMB }

    private nonisolated static func directory(_ bundle: ModelBundle) -> URL {
        ModelDownloader.modelsDirectory.appendingPathComponent(bundle.folder, isDirectory: true)
    }

    private nonisolated static func isInstalled(_ bundle: ModelBundle) -> Bool {
        let names = bundle.mlModels.map { "\($0).mlmodelc" } + bundle.files
        return names.allSatisfy { FileManager.default.fileExists(atPath: directory(bundle).appendingPathComponent($0).path) }
    }

    private func url(_ name: String) -> URL { Self.directory(bundle).appendingPathComponent(name) }

    /// RVC's files, non-nil once `shared` is downloaded.
    var assets: VoiceAssets? {
        guard state == .ready, self === Self.shared else { return nil }
        return VoiceAssets(
            contentVec: url("ContentVec.mlmodelc"), rmvpe: url("RMVPE.mlmodelc"),
            baseG: url("RVCBaseG.safetensors"), baseD: url("RVCBaseD.safetensors"))
    }

    /// Seed-VC's shared models, non-nil once `seedVC` is downloaded.
    var seedVCAssets: SeedVCAssets? {
        guard state == .ready, self === Self.seedVC else { return nil }
        return SeedVCAssets(whisper: url("SeedVCWhisper.mlmodelc"), vocoder: url("SeedVCVocoder.mlmodelc"))
    }

    func download() {
        switch state {
        case .ready, .downloading: return
        default: break
        }
        state = .downloading(progress: 0)
        let bundle = self.bundle
        task = Task.detached(priority: .userInitiated) { [self] in
            do {
                let zipURL = try await GitHubAssetDownloader.download(bundle.downloadURL) { progress in
                    Task { @MainActor [self] in self.state = .downloading(progress: progress) }
                }
                defer { try? FileManager.default.removeItem(at: zipURL) }
                try Self.install(bundle, zipURL: zipURL)
                await MainActor.run { self.state = .ready }
            } catch is CancellationError {
                await MainActor.run { self.state = .notDownloaded }
            } catch {
                await MainActor.run { self.state = .failed(error.localizedDescription) }
            }
        }
    }

    func cancelDownload() { task?.cancel() }

    func delete() {
        try? FileManager.default.removeItem(at: Self.directory(bundle))
        state = .notDownloaded
    }

    func installedSizeMB() -> Int? {
        guard let enumerator = FileManager.default.enumerator(at: Self.directory(bundle), includingPropertiesForKeys: [.fileSizeKey]) else { return nil }
        var total = 0
        for case let url as URL in enumerator {
            total += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        }
        return total > 0 ? total / (1024 * 1024) : nil
    }

    /// Unzips, compiles the .mlpackages for this Mac, and moves everything
    /// into place, replacing any previous/partial install.
    private nonisolated static func install(_ bundle: ModelBundle, zipURL: URL) throws {
        let fm = FileManager.default
        let extractDir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: extractDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: extractDir) }
        try ZipReader.unzipItem(at: zipURL, to: extractDir)

        let staging = extractDir.appendingPathComponent("installed", isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        for name in bundle.mlModels {
            let package = extractDir.appendingPathComponent("\(name).mlpackage")
            guard fm.fileExists(atPath: package.path) else { throw ModelDownloaderError.unexpectedArchiveContents }
            let compiled = try MLModel.compileModel(at: package)
            try fm.moveItem(at: compiled, to: staging.appendingPathComponent("\(name).mlmodelc"))
        }
        for name in bundle.files {
            let file = extractDir.appendingPathComponent(name)
            guard fm.fileExists(atPath: file.path) else { throw ModelDownloaderError.unexpectedArchiveContents }
            try fm.moveItem(at: file, to: staging.appendingPathComponent(name))
        }
        let directory = directory(bundle)
        try fm.createDirectory(at: ModelDownloader.modelsDirectory, withIntermediateDirectories: true)
        if fm.fileExists(atPath: directory.path) { try fm.removeItem(at: directory) }
        try fm.moveItem(at: staging, to: directory)
    }
}
#endif
