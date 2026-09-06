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
        // matches compute_units used at conversion time (tools/convert/build_coreml.py) --
        // including the ANE target made this graph's gather/scatter-add ops painfully
        // slow to compile there, so CPU+GPU only, consistently.
        config.computeUnits = .cpuAndGPU
        model = try MLModel(contentsOf: url, configuration: config)
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
}

enum SeparatorError: LocalizedError {
    case modelNotFound
    case missingOutput

    var errorDescription: String? {
        switch self {
        case .modelNotFound: return "Model has not been downloaded yet."
        case .missingOutput: return "CoreML model did not return the expected output."
        }
    }
}
