import CoreML
import Foundation
import MLX

/// Downloaded model files the engine needs (see Sunder's ModelDownloader).
public struct VoiceAssets: Sendable {
    public let contentVec: URL // compiled .mlmodelc
    public let rmvpe: URL      // compiled .mlmodelc
    public let baseG: URL      // RVCBaseG.safetensors
    public let baseD: URL      // RVCBaseD.safetensors

    public init(contentVec: URL, rmvpe: URL, baseG: URL, baseD: URL) {
        self.contentVec = contentVec; self.rmvpe = rmvpe; self.baseG = baseG; self.baseD = baseD
    }
}

/// The Seed-VC download (see Sunder's VoiceAssetDownloader): the parts every
/// Seed-VC voice shares. Seed-VC also needs `VoiceAssets.rmvpe`.
public struct SeedVCAssets: Sendable {
    public let whisper: URL // compiled .mlmodelc
    public let vocoder: URL // compiled .mlmodelc

    public init(whisper: URL, vocoder: URL) { self.whisper = whisper; self.vocoder = vocoder }
}

/// Which model a voice is.
public enum VoiceModelKind: String, Codable, Sendable {
    /// RVC v2: trained in the app (or on Sunder Server), converts in MLX.
    case rvc
    /// Seed-VC (singing): trained on Sunder Server, imported, converts in CoreML.
    case seedVC
}

/// A voice the user has created. Stored as `<library>/<id>/profile.json` next to
/// its model files.
public struct VoiceProfile: Codable, Identifiable, Sendable, Hashable {
    public let id: UUID
    public var name: String
    public var createdAt: Date
    /// Seconds of usable (non-silent) training audio.
    public var trainingSeconds: Double = 0
    /// RVC: epochs. Seed-VC: optimizer steps.
    public var targetEpochs: Int
    public var completedEpochs: Int = 0
    public var isTrained: Bool = false
    public var engine: VoiceModelKind = .rvc

    /// Training was started and its prepared dataset is on disk.
    public var canResume: Bool { !isTrained && trainingSeconds > 0 }

    init(id: UUID, name: String, createdAt: Date, targetEpochs: Int, engine: VoiceModelKind = .rvc) {
        self.id = id; self.name = name; self.createdAt = createdAt; self.targetEpochs = targetEpochs; self.engine = engine
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, createdAt, trainingSeconds, targetEpochs, completedEpochs, isTrained, engine
    }

    // Voices saved before Seed-VC have no `engine`: they're RVC.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        trainingSeconds = try c.decodeIfPresent(Double.self, forKey: .trainingSeconds) ?? 0
        targetEpochs = try c.decode(Int.self, forKey: .targetEpochs)
        completedEpochs = try c.decodeIfPresent(Int.self, forKey: .completedEpochs) ?? 0
        isTrained = try c.decodeIfPresent(Bool.self, forKey: .isTrained) ?? false
        engine = try c.decodeIfPresent(VoiceModelKind.self, forKey: .engine) ?? .rvc
    }
}

public enum TrainingProgress: Sendable {
    case preparing(fraction: Double)
    case training(epoch: Int, epochs: Int, step: Int, steps: Int, melLoss: Float)
    case finishing
}

/// On-disk layout of the user's voices.
public struct VoiceLibrary: Sendable {
    public let root: URL

    public init(root: URL) { self.root = root }

    public func directory(for id: UUID) -> URL { root.appending(path: id.uuidString, directoryHint: .isDirectory) }
    func modelURL(_ id: UUID) -> URL { directory(for: id).appending(path: "G.safetensors") }
    func indexURL(_ id: UUID) -> URL { directory(for: id).appending(path: "index.safetensors") }
    func datasetURL(_ id: UUID) -> URL { directory(for: id).appending(path: "dataset.safetensors") }
    func checkpointG(_ id: UUID) -> URL { directory(for: id).appending(path: "checkpoint_G.safetensors") }
    func checkpointD(_ id: UUID) -> URL { directory(for: id).appending(path: "checkpoint_D.safetensors") }
    func seedVCModel(_ id: UUID, _ name: String) -> URL { directory(for: id).appending(path: "\(name).mlmodelc") }
    func seedVCReference(_ id: UUID) -> URL { directory(for: id).appending(path: "reference.safetensors") }

    public func list() -> [VoiceProfile] {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return dirs.compactMap { dir in
            guard let data = try? Data(contentsOf: dir.appending(path: "profile.json")) else { return nil }
            return try? JSONDecoder().decode(VoiceProfile.self, from: data)
        }.sorted { $0.createdAt > $1.createdAt }
    }

    public func create(name: String, epochs: Int) throws -> VoiceProfile {
        let profile = VoiceProfile(id: UUID(), name: name, createdAt: Date(), targetEpochs: epochs)
        try FileManager.default.createDirectory(at: directory(for: profile.id), withIntermediateDirectories: true)
        try save(profile)
        return profile
    }

    public func save(_ profile: VoiceProfile) throws {
        try JSONEncoder().encode(profile).write(to: directory(for: profile.id).appending(path: "profile.json"), options: .atomic)
    }

    public func delete(_ id: UUID) throws {
        try FileManager.default.removeItem(at: directory(for: id))
    }

    /// The files a trained voice needs to convert, and all an export carries:
    /// training-only state (dataset, checkpoints) is never included.
    private static func portableFiles(_ engine: VoiceModelKind) -> [String] {
        switch engine {
        case .rvc: ["G.safetensors", "index.safetensors"]
        case .seedVC: SeedVCConverter.voiceModels.map { "\($0).mlmodelc" } + ["reference.safetensors"]
        }
    }

    /// Copies a trained voice's portable files into `folder` (created if
    /// needed), for the app to package however it likes.
    public func stageExport(_ profile: VoiceProfile, into folder: URL) throws {
        guard profile.isTrained else { throw VoiceError.notTrained }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for name in ["profile.json"] + Self.portableFiles(profile.engine) {
            try FileManager.default.copyItem(at: directory(for: profile.id).appending(path: name), to: folder.appending(path: name))
        }
    }

    /// Adds the voice staged in `folder` (as written by `stageExport`, or by
    /// Sunder Server) to the library under a fresh id, so importing the same
    /// export twice -- or back onto the Mac it came from -- never collides
    /// with an existing voice. Seed-VC models may arrive as .mlpackage
    /// (from the server) and are compiled for this Mac.
    public func importVoice(from folder: URL) throws -> VoiceProfile {
        guard let source = try? JSONDecoder().decode(VoiceProfile.self, from: Data(contentsOf: folder.appending(path: "profile.json"))),
              source.isTrained
        else { throw VoiceError.invalidExport }
        let fm = FileManager.default
        let files = Self.portableFiles(source.engine)
        func package(for file: String) -> URL? {
            guard file.hasSuffix(".mlmodelc") else { return nil }
            let url = folder.appending(path: file.replacingOccurrences(of: ".mlmodelc", with: ".mlpackage"))
            return fm.fileExists(atPath: url.path) ? url : nil
        }
        guard files.allSatisfy({ fm.fileExists(atPath: folder.appending(path: $0).path) || package(for: $0) != nil })
        else { throw VoiceError.invalidExport }

        var profile = VoiceProfile(id: UUID(), name: source.name, createdAt: Date(), targetEpochs: source.targetEpochs, engine: source.engine)
        profile.trainingSeconds = source.trainingSeconds
        profile.completedEpochs = source.completedEpochs
        profile.isTrained = true

        let dir = directory(for: profile.id)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        do {
            for name in files {
                if fm.fileExists(atPath: folder.appending(path: name).path) {
                    try fm.copyItem(at: folder.appending(path: name), to: dir.appending(path: name))
                } else if let package = package(for: name) {
                    let compiled = try MLModel.compileModel(at: package)
                    try fm.moveItem(at: compiled, to: dir.appending(path: name))
                }
            }
            // Written last: `list()` only shows directories with a profile,
            // so a half-copied import never appears as a voice.
            try save(profile)
        } catch {
            try? FileManager.default.removeItem(at: dir)
            throw error
        }
        return profile
    }
}

/// Entry point for training voices and converting audio with them. Heavy and
/// synchronous: call off the main actor, one operation at a time.
public final class VoiceEngine: @unchecked Sendable {
    private let assets: VoiceAssets
    private let features: VoiceFeatures

    /// Checkpoints are written this often (epochs) so a stopped run can resume.
    static let checkpointInterval = 5

    public init(assets: VoiceAssets) throws {
        self.assets = assets
        features = try VoiceFeatures(contentVecURL: assets.contentVec, rmvpeURL: assets.rmvpe)
    }

    /// Seconds of training material left after silence removal.
    public static func usableSeconds(of audio44k: [Float]) throws -> Double {
        try VoiceDataset.usableDuration(of: audio44k)
    }

    /// Trains (or resumes training) `profile`. `sources` (mono 44.1 kHz) are
    /// needed only on the first run; afterwards the prepared dataset is reused.
    /// Returns the updated profile; stopping early leaves a resumable checkpoint.
    public func train(
        _ profile: VoiceProfile, sources: [[Float]], library: VoiceLibrary,
        progress: @escaping (TrainingProgress) -> Void, shouldStop: @escaping () -> Bool
    ) throws -> VoiceProfile {
        var profile = profile
        let id = profile.id
        let dataset: VoiceDataset
        if FileManager.default.fileExists(atPath: library.datasetURL(id).path) {
            dataset = try VoiceDataset.load(from: library.datasetURL(id))
        } else {
            dataset = try VoiceDataset.build(sources: sources, features: features) { progress(.preparing(fraction: $0)) }
            guard !dataset.items.isEmpty else { throw VoiceError.noUsableAudio }
            try VoiceIndex.build(from: dataset.contentFrames).save(to: library.indexURL(id))
            try dataset.save(to: library.datasetURL(id))
            profile.trainingSeconds = Double(dataset.items.reduce(0) { $0 + $1.frames }) / 100
            try library.save(profile)
        }

        let trainer = try VoiceTrainer(dataset: dataset, baseG: assets.baseG, baseD: assets.baseD)
        if profile.completedEpochs > 0 {
            try trainer.resume(g: library.checkpointG(id), d: library.checkpointD(id))
        }

        let steps = trainer.stepsPerEpoch
        while profile.completedEpochs < profile.targetEpochs {
            let epoch = profile.completedEpochs + 1
            let finished = trainer.trainEpoch(epoch, shouldStop: shouldStop) { step, _, losses in
                progress(.training(epoch: epoch, epochs: profile.targetEpochs, step: step, steps: steps, melLoss: losses.mel))
            }
            guard finished else { break }
            guard trainer.weightsAreFinite else { throw VoiceError.trainingDiverged }
            profile.completedEpochs = epoch
            if epoch % Self.checkpointInterval == 0 || epoch == profile.targetEpochs {
                try trainer.saveCheckpoint(g: library.checkpointG(id), d: library.checkpointD(id))
                try library.save(profile)
            }
        }

        if profile.completedEpochs >= profile.targetEpochs {
            progress(.finishing)
            try trainer.saveVoiceModel(to: library.modelURL(id))
            profile.isTrained = true
            try library.save(profile)
            // Training state is large (dataset + D + full G) and no longer needed.
            for url in [library.checkpointG(id), library.checkpointD(id), library.datasetURL(id)] {
                try? FileManager.default.removeItem(at: url)
            }
        } else {
            // Stopped mid-run: checkpoint now so the saved epoch count and
            // weights agree on resume.
            try trainer.saveCheckpoint(g: library.checkpointG(id), d: library.checkpointD(id))
            try library.save(profile)
        }
        Memory.clearCache()
        return profile
    }

    /// Converts mono 44.1 kHz audio to `profile`'s voice (same length out).
    /// Seed-VC voices need `seedVC` (the shared Seed-VC models).
    public func convert(
        _ audio44k: [Float], with profile: VoiceProfile, library: VoiceLibrary,
        settings: ConversionSettings, seedVC: SeedVCAssets? = nil, progress: @escaping (Double) -> Void
    ) throws -> [Float] {
        guard profile.isTrained else { throw VoiceError.notTrained }
        switch profile.engine {
        case .rvc:
            let converter = try VoiceConverter(voiceModel: library.modelURL(profile.id), index: library.indexURL(profile.id), features: features)
            defer { Memory.clearCache() }
            return try converter.convert(audio44k, settings: settings, progress: progress)
        case .seedVC:
            guard let seedVC else { throw VoiceError.seedVCModelsMissing }
            let converter = try SeedVCConverter(
                shared: seedVC, dit: library.seedVCModel(profile.id, "SeedVCDiT"),
                regulator: library.seedVCModel(profile.id, "SeedVCRegulator"),
                reference: library.seedVCReference(profile.id), features: features)
            return try converter.convert(audio44k, settings: settings, progress: progress)
        }
    }
}

public enum VoiceError: LocalizedError {
    case noUsableAudio, notTrained, trainingDiverged, invalidExport, seedVCModelsMissing

    public var errorDescription: String? {
        switch self {
        case .noUsableAudio: "The recordings didn't contain enough non-silent vocal audio to train on."
        case .notTrained: "This voice hasn't finished training yet."
        case .trainingDiverged: "Training became numerically unstable. Resume to continue from the last saved checkpoint."
        case .invalidExport: "That file isn't an exported Sunder voice."
        case .seedVCModelsMissing: "This voice uses Seed-VC. Download the Seed-VC models in Settings first."
        }
    }
}

extension VoiceDataset {
    func save(to url: URL) throws {
        var arrays: [String: MLXArray] = ["content_frames": contentFrames]
        for (i, item) in items.enumerated() {
            arrays["\(i).wave"] = item.wave
            arrays["\(i).content"] = item.content
            arrays["\(i).pitch"] = item.pitch
            arrays["\(i).pitchf"] = item.pitchf
        }
        try MLX.save(arrays: arrays, url: url)
    }

    static func load(from url: URL) throws -> VoiceDataset {
        let arrays = try loadArrays(url: url)
        let count = arrays.keys.filter { $0.hasSuffix(".wave") }.count
        let items = (0 ..< count).map { i in
            TrainingItem(wave: arrays["\(i).wave"]!, content: arrays["\(i).content"]!,
                         pitch: arrays["\(i).pitch"]!, pitchf: arrays["\(i).pitchf"]!)
        }
        return VoiceDataset(items: items, contentFrames: arrays["content_frames"]!)
    }
}
