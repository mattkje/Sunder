import CoreML
import Foundation
import MLX
import MLXFFT

/// The frozen feature extractors, run in CoreML (converted by
/// tools/convert/build_coreml_rvc_features.py). Both take 16 kHz mono audio.
final class VoiceFeatures {
    private let contentVec: MLModel
    private let rmvpe: MLModel

    static let sampleRate: Double = 16000
    /// ContentVec: one 768-dim frame per 320 samples (20 ms).
    static let contentHop = 320
    /// RMVPE: one pitch frame per 160 samples (10 ms).
    static let pitchHop = 160

    init(contentVecURL: URL, rmvpeURL: URL) throws {
        let config = MLModelConfiguration()
        // CPU only: both models take flexible-length input, and on the GPU
        // CoreML compiles an MPSGraph per new length -- some lengths trip an
        // MPSGraph MLIR assertion ("MPSCommonRuntimeCanonicalization") that
        // aborts the process. Seen preparing ~30 min of recordings (hundreds
        // of distinct segment lengths). The CPU path doesn't use MPSGraph.
        config.computeUnits = .cpuOnly
        contentVec = try MLModel(contentsOf: contentVecURL, configuration: config)
        rmvpe = try MLModel(contentsOf: rmvpeURL, configuration: config)
    }

    // MARK: ContentVec

    /// 16 kHz audio -> [frames, 768] features at 50 fps, frames = (n - 400) / 320 + 1.
    /// Long audio runs in ~20 s windows with 1 s of context on each side.
    func content(_ audio: [Float]) throws -> [[Float]] {
        let total = max(0, (audio.count - 400) / Self.contentHop + 1)
        let coreFrames = 1000, contextFrames = 50 // 20 s, 1 s
        var out: [[Float]] = []
        out.reserveCapacity(total)
        var frame = 0
        while frame < total {
            let startFrame = max(0, frame - contextFrames)
            let endFrame = min(total, frame + coreFrames + contextFrames)
            let start = startFrame * Self.contentHop
            let end = min(audio.count, (endFrame - 1) * Self.contentHop + 400)
            let feats = try runContentVec(Array(audio[start ..< end]))
            let keepFrom = frame - startFrame
            let keepTo = min(feats.count, keepFrom + min(coreFrames, total - frame))
            out += feats[keepFrom ..< keepTo]
            frame += keepTo - keepFrom
            if keepTo <= keepFrom { break }
        }
        return out
    }

    private func runContentVec(_ chunk: [Float]) throws -> [[Float]] {
        var samples = chunk
        if samples.count < 16000 { samples += [Float](repeating: 0, count: 16000 - samples.count) }
        let input = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        samples.withUnsafeBufferPointer { src in
            input.dataPointer.assumingMemoryBound(to: Float.self).update(from: src.baseAddress!, count: samples.count)
        }
        let result = try contentVec.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio16k": input]))
        let features = result.featureValue(for: "features")!.multiArrayValue!
        let frames = features.shape[1].intValue, dim = features.shape[2].intValue
        let validFrames = min(frames, max(0, (chunk.count - 400) / Self.contentHop + 1))
        let pointer = features.dataPointer.assumingMemoryBound(to: Float.self)
        let strides = features.strides.map(\.intValue)
        return (0 ..< validFrames).map { t in (0 ..< dim).map { pointer[t * strides[1] + $0 * strides[2]] } }
    }

    // MARK: RMVPE

    /// 16 kHz audio -> f0 in Hz per 10 ms frame (0 = unvoiced), frames = n / 160 + 1.
    func pitch(_ audio: [Float], threshold: Float = 0.03) throws -> [Float] {
        let magnitude = Self.stftMagnitude(audio) // [T, 513]
        let total = magnitude.dim(0)
        let core = 2048, context = 256
        var f0: [Float] = []
        f0.reserveCapacity(total)
        var frame = 0
        while frame < total {
            let start = max(0, frame - context)
            let end = min(total, frame + core + context)
            let salience = try runRMVPE(magnitude[start ..< end])
            let keepFrom = frame - start
            let keepTo = keepFrom + min(core, total - frame)
            f0 += (keepFrom ..< keepTo).map { Self.decode(salience[$0], threshold: threshold) }
            frame += keepTo - keepFrom
        }
        return f0
    }

    private func runRMVPE(_ magnitude: MLXArray) throws -> [[Float]] {
        let frames = magnitude.dim(0)
        let padded = (frames + 31) / 32 * 32
        var values = magnitude.transposed().asArray(Float.self) // [513, frames] row-major
        if padded != frames {
            values = (0 ..< 513).flatMap { bin in
                Array(values[(bin * frames) ..< ((bin + 1) * frames)]) + [Float](repeating: 0, count: padded - frames)
            }
        }
        let input = try MLMultiArray(shape: [1, 513, NSNumber(value: padded)], dataType: .float32)
        values.withUnsafeBufferPointer { src in
            input.dataPointer.assumingMemoryBound(to: Float.self).update(from: src.baseAddress!, count: values.count)
        }
        let result = try rmvpe.prediction(from: MLDictionaryFeatureProvider(dictionary: ["magnitude": input]))
        let salience = result.featureValue(for: "salience")!.multiArrayValue!
        let pointer = salience.dataPointer.assumingMemoryBound(to: Float.self)
        let strides = salience.strides.map(\.intValue)
        return (0 ..< frames).map { t in (0 ..< 360).map { pointer[t * strides[1] + $0 * strides[2]] } }
    }

    /// RVC's `to_local_average_cents` + `decode`: salience-weighted mean of the
    /// 9 bins around the peak, 20 cents per bin, gated by peak salience.
    static func decode(_ salience: [Float], threshold: Float) -> Float {
        var peak: Float = 0, center = 0
        for (i, v) in salience.enumerated() where v > peak { peak = v; center = i }
        guard peak > threshold else { return 0 }
        var weighted: Float = 0, total: Float = 0
        for i in max(0, center - 4) ... min(359, center + 4) {
            weighted += salience[i] * (20 * Float(i) + 1997.3794084376191)
            total += salience[i]
        }
        let cents = weighted / total
        return 10 * pow(2, cents / 1200)
    }

    /// |STFT| with n_fft 1024, hop 160, periodic Hann, center=True (reflect),
    /// matching RMVPE's MelSpectrogram front end. -> [frames, 513].
    static func stftMagnitude(_ audio: [Float]) -> MLXArray {
        let n = 1024, hop = 160
        let x = reflectPadTime(MLXArray(audio).reshaped(1, audio.count, 1), left: n / 2, right: n / 2).reshaped(-1)
        let frames = (x.dim(0) - n) / hop + 1
        let index = (MLXArray(0 ..< Int32(frames)) * Int32(hop)).expandedDimensions(axis: 1) + MLXArray(0 ..< Int32(n)).expandedDimensions(axis: 0)
        let window = MLXArray((0 ..< n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) })
        let spec = rfft(x.take(index.flattened()).reshaped(frames, n) * window, axis: -1)
        return abs(spec)
    }
}

/// RVC's coarse pitch: mel-scaled f0 quantized to 1...255 (0 Hz -> 1).
enum PitchQuantizer {
    static let minMel = 1127 * log(1 + 50.0 / 700)
    static let maxMel = 1127 * log(1 + 1100.0 / 700)

    static func coarse(_ f0: Float) -> Int32 {
        var mel = 1127 * log(1 + Double(f0) / 700)
        if mel > 0 { mel = (mel - minMel) * 254 / (maxMel - minMel) + 1 }
        return Int32(min(255, max(1, mel.rounded(.toNearestOrEven))))
    }
}
