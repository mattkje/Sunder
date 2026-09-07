import CoreML
import Foundation

/// Thin wrapper around a CoreML model downloaded into Application Support by
/// ModelDownloader (see AIModel/ModelDownloader.swift -- nothing is bundled
/// in the app itself). Uses the generic MLModel/MLFeatureProvider API rather
/// than an Xcode-codegenned class so it doesn't depend on how that codegen
/// names things.
nonisolated final class SeparatorModel {
    private let model: MLModel
    let spec: ModelSpec

    init(model aiModel: AIModel = .melBandRoformerDeux) throws {
        let url = ModelDownloader.localModelURL(for: aiModel)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SeparatorError.modelNotFound
        }
        spec = aiModel.spec
        let config = MLModelConfiguration()
        #if os(iOS)
        // Both CPU+GPU and CPU-only still exceeded iOS's ~3.3GB jetsam
        // ceiling on a single chunk's forward pass -- the model's peak
        // memory on those backends is just too high on a phone. Try
        // .all (matches coremltools' compute_units=ALL default) so
        // CoreML's partitioner can put this graph's matmul-heavy layers on
        // the Neural Engine, which is far more memory-efficient than the
        // generic CPU/GPU backends for a transformer this size; the
        // gather/scatter-add ops it can't place there fall back to CPU/GPU
        // automatically. The known risk (see tools/convert/README.md) is
        // that *compiling* those ops for ANE was observed to take 20+
        // minutes in coremltools' own post-conversion validation load --
        // untested whether that reproduces in on-device ANE compilation,
        // so this is raced against a timeout rather than trusted blindly.
        config.computeUnits = .all
        model = try Self.loadWithTimeout(at: url, configuration: config, timeout: 90)
        #else
        // matches compute_units used at conversion time (tools/convert/build_coreml.py) --
        // including the ANE target made this graph's gather/scatter-add ops painfully
        // slow to compile there, so CPU+GPU only, consistently. macOS has no
        // memory ceiling anywhere near what iOS hits, so there's no reason
        // to risk the same ANE compile-time cost here.
        config.computeUnits = .cpuAndGPU
        model = try MLModel(contentsOf: url, configuration: config)
        #endif
    }

    /// stftRepr: real/imag STFT per channel, each flattened [freq][time]
    /// (index = f * spec.timeFrames + t), for both channels.
    /// Returns, per stem, per channel, the masked (real, imag) spectrum in
    /// the same flattened layout, ready for STFTProcessor.inverse.
    func separate(channelReal: [[Float]], channelImag: [[Float]]) throws -> [[(real: [Float], imag: [Float])]] {
        let F = spec.numFreqBins
        let T = spec.timeFrames
        let C = ModelConfig.channels

        let inputArray = try MLMultiArray(shape: [1, C, F, T, 2] as [NSNumber], dataType: .float32)
        inputArray.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, strides in
            for c in 0..<C {
                let real = channelReal[c]
                let imag = channelImag[c]
                for f in 0..<F {
                    for t in 0..<T {
                        let idx = ((c * F + f) * T + t) * 2
                        let src = f * T + t
                        buf[idx] = real[src]
                        buf[idx + 1] = imag[src]
                    }
                }
            }
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["stft_repr": MLFeatureValue(multiArray: inputArray)])
        let output = try model.prediction(from: input)
        guard let outArray = output.featureValue(for: "masked_spectra")?.multiArrayValue else {
            throw SeparatorError.missingOutput
        }

        var result: [[(real: [Float], imag: [Float])]] = []
        outArray.withUnsafeBufferPointer(ofType: Float.self) { buf in
            for n in 0..<spec.numStems {
                var perChannel: [(real: [Float], imag: [Float])] = []
                for c in 0..<C {
                    var real = [Float](repeating: 0, count: F * T)
                    var imag = [Float](repeating: 0, count: F * T)
                    for f in 0..<F {
                        for t in 0..<T {
                            let idx = (((n * C + c) * F + f) * T + t) * 2
                            let dst = f * T + t
                            real[dst] = buf[idx]
                            imag[dst] = buf[idx + 1]
                        }
                    }
                    perChannel.append((real: real, imag: imag))
                }
                result.append(perChannel)
            }
        }
        return result
    }

    #if os(iOS)
    /// MLModel(contentsOf:configuration:) has no timeout of its own, and a
    /// compute configuration that needs on-device Neural Engine compilation
    /// for unusual ops (see the .all comment above) has an unknown worst
    /// case. Race the async load against `timeout` on a background thread
    /// (this initializer already only ever runs off the main actor -- see
    /// SeparationEngine's nonisolated static process) so a slow compile
    /// reads as a clear error instead of an app that looks frozen.
    private static func loadWithTimeout(at url: URL, configuration: MLModelConfiguration, timeout: TimeInterval) throws -> MLModel {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Swift.Result<MLModel, Error>?
        MLModel.load(contentsOf: url, configuration: configuration) { loadResult in
            result = loadResult
            semaphore.signal()
        }
        guard semaphore.wait(timeout: .now() + timeout) == .success, let result else {
            throw SeparatorError.neuralEngineCompileTimedOut
        }
        return try result.get()
    }
    #endif
}

enum SeparatorError: LocalizedError {
    case modelNotFound
    case missingOutput
    case neuralEngineCompileTimedOut

    var errorDescription: String? {
        switch self {
        case .modelNotFound: return "Model has not been downloaded yet."
        case .missingOutput: return "CoreML model did not return the expected output."
        case .neuralEngineCompileTimedOut: return "Preparing this model took too long. Try again -- the device caches the compiled model after the first successful run."
        }
    }
}
