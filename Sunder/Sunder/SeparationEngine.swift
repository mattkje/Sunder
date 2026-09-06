import Foundation
import Observation

@Observable
final class SeparationEngine {
    enum State: Equatable {
        case idle
        case separating(progress: Double)
        case done(outputs: [URL])
        case cancelled
        case failed(String)
    }

    var state: State = .idle
    private var currentTask: Task<Void, Never>?

    func run(inputURL: URL, outputDir: URL, quality: OutputQuality, model: AIModel) {
        state = .separating(progress: 0)
        currentTask = Task.detached(priority: .userInitiated) { [self] in
            do {
                let outputs = try Self.process(inputURL: inputURL, outputDir: outputDir, quality: quality, model: model) { progress in
                    Task { @MainActor [self] in
                        self.state = .separating(progress: progress)
                    }
                }
                await MainActor.run { self.state = .done(outputs: outputs) }
            } catch is CancellationError {
                await MainActor.run { self.state = .cancelled }
            } catch {
                await MainActor.run { self.state = .failed(error.localizedDescription) }
            }
        }
    }

    func cancel() {
        currentTask?.cancel()
    }

    // Explicitly nonisolated: this project defaults every declaration to
    // @MainActor (SWIFT_DEFAULT_ACTOR_ISOLATION), and without this the
    // compiler was letting this heavy, synchronous function run on the main
    // actor even when called from Task.detached -- freezing the UI for the
    // whole separation instead of only hopping back to MainActor for the
    // explicit `await MainActor.run` progress/completion updates below.
    private nonisolated static func process(
        inputURL: URL,
        outputDir: URL,
        quality: OutputQuality,
        model aiModel: AIModel,
        progress: @escaping (Double) -> Void
    ) throws -> [URL] {
        // NSOpenPanel-returned URLs already carry an active sandbox grant, but
        // drag-and-dropped URLs need this explicitly; harmless either way.
        let inputAccess = inputURL.startAccessingSecurityScopedResource()
        let outputAccess = outputDir.startAccessingSecurityScopedResource()
        defer {
            if inputAccess { inputURL.stopAccessingSecurityScopedResource() }
            if outputAccess { outputDir.stopAccessingSecurityScopedResource() }
        }

        let mix = try AudioIO.loadAudio(url: inputURL)
        let model = try SeparatorModel(model: aiModel)
        let demixer = Demixer(model: model)
        let stems = try demixer.separate(mix: mix, isCancelled: { Task.isCancelled }, progress: progress)

        let baseName = inputURL.deletingPathExtension().lastPathComponent
        let ext = quality.fileExtension

        var outputs: [URL] = []
        for (i, name) in aiModel.spec.stemNames.enumerated() {
            let outURL = outputDir.appendingPathComponent("\(baseName)_\(name).\(ext)")
            try AudioIO.writeAudio(channels: stems[i], to: outURL, quality: quality)
            outputs.append(outURL)
        }
        return outputs
    }
}
