import Accelerate
import CoreML
import Foundation
import MLX
import MLXRandom

/// Seed-VC (singing, 44.1 kHz) conversion for one voice, in CoreML. Step for
/// step the same as tools/convert/verify_seedvc.py, which mirrors upstream
/// Seed-VC's inference; the models and their contracts are described in
/// tools/convert/models/seedvc/export.py.
///
/// Source -> Whisper content (50 fps) + RMVPE pitch (100 fps), both
/// nearest-resampled to the 86.13 fps mel rate -> the voice's regulator ->
/// 30 s chunks, each prefixed with the reference prompt, denoised by guided
/// Euler steps of the voice's DiT -> BigVGAN, chunks crossfaded.
final class SeedVCConverter {
    static let voiceModels = ["SeedVCDiT", "SeedVCRegulator"]

    static let sampleRate = 44100
    static let hop = 512
    /// Frames per DiT chunk, prompt included (30 s).
    static let maxContext = sampleRate / hop * 30
    static let overlapFrames = 16
    static let guidance: Float = 0.7
    /// Whisper's window: 30 s at 16 kHz; long sources run in windows overlapping by 5 s.
    static let whisperSamples = 480_000
    static let whisperOverlap = 80_000

    private let whisper: MLModel
    private let vocoder: MLModel
    private let dit: MLModel
    private let regulator: MLModel
    private let features: VoiceFeatures
    private let style: [Float]       // [192]
    private let promptMel: [Float]   // [128, P] row-major
    private let promptCond: [Float]  // [P, 768] row-major
    private let promptFrames: Int

    init(shared: SeedVCAssets, dit ditURL: URL, regulator regulatorURL: URL, reference: URL, features: VoiceFeatures) throws {
        // The DiT and vocoder are what make Seed-VC usable on a Mac, so they get
        // the GPU. Their input length only varies on a song's last chunk, which
        // keeps the per-length MPSGraph compiles (see VoiceFeatures) rare. The
        // regulator sees the whole song at once and is cheap: CPU.
        let gpu = MLModelConfiguration()
        gpu.computeUnits = .cpuAndGPU
        let cpu = MLModelConfiguration()
        cpu.computeUnits = .cpuOnly
        whisper = try MLModel(contentsOf: shared.whisper, configuration: gpu)
        vocoder = try MLModel(contentsOf: shared.vocoder, configuration: gpu)
        dit = try MLModel(contentsOf: ditURL, configuration: gpu)
        regulator = try MLModel(contentsOf: regulatorURL, configuration: cpu)
        self.features = features

        let arrays = try loadArrays(url: reference)
        guard let s = arrays["style"], let m = arrays["prompt_mel"], let c = arrays["prompt_cond"] else {
            throw VoiceError.invalidExport
        }
        style = s.asType(.float32).asArray(Float.self)
        promptMel = m.asType(.float32).asArray(Float.self)
        promptCond = c.asType(.float32).asArray(Float.self)
        promptFrames = m.dim(1)
    }

    /// Mono 44.1 kHz in, mono 44.1 kHz out (same length).
    func convert(_ audio44k: [Float], settings: ConversionSettings, progress: (Double) -> Void) throws -> [Float] {
        let frames = (audio44k.count - Self.hop) / Self.hop + 1 // Seed-VC's mel (center=False, (n_fft - hop) / 2 padding)
        guard frames > Self.overlapFrames else { return audio44k }
        let audio16k = try VoiceAudio.resample(audio44k, from: VoiceAudio.appSampleRate, to: VoiceFeatures.sampleRate)

        let content = try whisperContent(audio16k) // [n, 768]
        var f0 = try features.pitch(audio16k)
        if settings.pitchShift != 0 {
            let factor = pow(2, settings.pitchShift / 12)
            f0 = f0.map { $0 > 1 ? $0 * factor : $0 }
        }
        progress(0.05)

        let contentIndex = Self.nearest(count: content.count, to: frames)
        let pitchIndex = Self.nearest(count: f0.count, to: frames)
        let cond = try runRegulator(content: contentIndex.flatMap { content[$0] }, f0: pitchIndex.map { f0[$0] }, frames: frames)

        // Chunked generation, as upstream: each chunk is prompt + up to
        // `window` source frames; consecutive chunks overlap by 16 frames.
        let P = promptFrames
        let window = Self.maxContext - P
        let steps = max(1, settings.diffusionSteps)
        let chunkCount = max(1, Int(ceil(Double(max(0, frames - Self.overlapFrames)) / Double(window - Self.overlapFrames))))
        let overlapSamples = Self.overlapFrames * Self.hop
        var output: [Float] = []
        output.reserveCapacity(audio44k.count)
        var previous: [Float] = []
        var done = 0, chunk = 0
        MLXRandom.seed(0)
        while done < frames {
            let n = min(window, frames - done)
            let mel = try generate(cond: cond, from: done, count: n, steps: steps) { step in
                progress(0.05 + 0.9 * (Double(chunk) + Double(step) / Double(steps)) / Double(chunkCount))
            }
            let wave = try runVocoder(mel: mel, frames: n)
            let last = done + window >= frames
            let body = last ? wave : Array(wave.dropLast(overlapSamples))
            output += previous.isEmpty ? body : Self.crossfade(previous, body, overlap: overlapSamples)
            if last { break }
            previous = Array(wave.suffix(overlapSamples))
            done += n - Self.overlapFrames
            chunk += 1
        }
        if output.count < audio44k.count { output += [Float](repeating: 0, count: audio44k.count - output.count) }
        progress(1)
        // Seed-VC renders at the reference clip's level (training vocals are
        // peak-normalized), typically several times louder than a stem, and
        // leaves hiss where the source is silent: follow the source's loudness
        // envelope, as RVC's conversion does.
        return VoiceConverter.matchEnvelope(Array(output.prefix(audio44k.count)), to: audio44k, rate: 0.25)
    }

    // MARK: Stages

    /// Whisper encoder frames (50 fps, n / 320 + 1 of them) for 16 kHz audio.
    private func whisperContent(_ audio: [Float]) throws -> [[Float]] {
        func run(_ chunk: ArraySlice<Float>) throws -> [[Float]] {
            let input = try MLMultiArray(shape: [1, NSNumber(value: Self.whisperSamples)], dataType: .float32)
            let pointer = input.dataPointer.assumingMemoryBound(to: Float.self)
            pointer.initialize(repeating: 0, count: Self.whisperSamples)
            chunk.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: chunk.count) }
            let out = try whisper.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio16k": input]))
                .featureValue(for: "content")!.multiArrayValue!
            return Self.rows(out, count: chunk.count / 320 + 1)
        }
        if audio.count <= Self.whisperSamples { return try run(audio[...]) }
        // 30 s windows; after the first, each starts with the previous 5 s
        // and drops those frames (250) from its output.
        var out: [[Float]] = []
        var position = 0
        var buffer: ArraySlice<Float> = []
        while position < audio.count {
            let chunk: ArraySlice<Float>
            if buffer.isEmpty {
                chunk = audio[position ..< min(audio.count, position + Self.whisperSamples)]
            } else {
                let fresh = audio[position ..< min(audio.count, position + Self.whisperSamples - Self.whisperOverlap)]
                chunk = (buffer + fresh)[...]
            }
            let frames = try run(chunk)
            out += buffer.isEmpty ? frames : Array(frames.dropFirst(Self.whisperOverlap / 320))
            position += buffer.isEmpty ? chunk.count : chunk.count - Self.whisperOverlap
            buffer = chunk.suffix(Self.whisperOverlap)
        }
        return out
    }

    /// -> cond [frames, 768] row-major.
    private func runRegulator(content: [Float], f0: [Float], frames: Int) throws -> [Float] {
        let result = try regulator.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "content": try Self.array(content, [1, frames, 768]),
            "f0": try Self.array(f0, [1, frames]),
        ]))
        return Self.rows(result.featureValue(for: "cond")!.multiArrayValue!, count: frames).flatMap { $0 }
    }

    /// Denoises one chunk: prompt + `count` frames of `cond` from `start`.
    /// -> mel [128, count] row-major (the prompt part dropped).
    private func generate(cond: [Float], from start: Int, count: Int, steps: Int, step report: (Int) -> Void) throws -> [Float] {
        let P = promptFrames, L = P + count
        var condChunk = promptCond
        condChunk += cond[(start * 768) ..< ((start + count) * 768)]
        var promptX = [Float](repeating: 0, count: 128 * L)
        for c in 0 ..< 128 {
            for t in 0 ..< P { promptX[c * L + t] = promptMel[c * P + t] }
        }
        let fixed: [String: MLMultiArray] = [
            "prompt_x": try Self.array(promptX, [1, 128, L]),
            "cond": try Self.array(condChunk, [1, L, 768]),
            "style": try Self.array(style, [1, 192]),
            "cfg": try Self.array([Self.guidance], [1]),
        ]
        var x = MLXRandom.normal([128, L]).asArray(Float.self)
        Self.zeroPrompt(&x, length: L, prompt: P)
        let dt = 1 / Float(steps)
        for i in 0 ..< steps {
            var inputs = fixed
            inputs["x"] = try Self.array(x, [1, 128, L])
            inputs["t"] = try Self.array([Float(i) * dt], [1])
            let velocity = try dit.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
                .featureValue(for: "velocity")!.multiArrayValue!
            let v = Self.rows(velocity, count: 128).flatMap { $0 }
            x = vDSP.add(multiplication: (v, dt), x)
            Self.zeroPrompt(&x, length: L, prompt: P)
            report(i + 1)
        }
        return (0 ..< 128).flatMap { c in x[(c * L + P) ..< (c * L + L)] }
    }

    /// mel [128, frames] -> 44.1 kHz audio (frames * 512 samples).
    private func runVocoder(mel: [Float], frames: Int) throws -> [Float] {
        let out = try vocoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mel": try Self.array(mel, [1, 128, frames])]))
            .featureValue(for: "wave")!.multiArrayValue!
        return Self.rows(out, count: 1).flatMap { $0 }
    }

    // MARK: Helpers

    /// `F.interpolate(mode: "nearest")` source indices for `count` -> `target`.
    static func nearest(count: Int, to target: Int) -> [Int] {
        (0 ..< target).map { min(count - 1, Int(floor(Double($0) * Double(count) / Double(target)))) }
    }

    /// Upstream's equal-power crossfade of the previous chunk's tail into `next`.
    static func crossfade(_ tail: [Float], _ next: [Float], overlap: Int) -> [Float] {
        var out = next
        for i in 0 ..< min(overlap, out.count) {
            let phase = Double(i) / Double(max(1, overlap - 1)) * Double.pi / 2
            out[i] = out[i] * Float(pow(sin(phase), 2)) + tail[tail.count - overlap + i] * Float(pow(cos(phase), 2))
        }
        return out
    }

    private static func zeroPrompt(_ x: inout [Float], length: Int, prompt: Int) {
        for c in 0 ..< 128 { for t in 0 ..< prompt { x[c * length + t] = 0 } }
    }

    static func array(_ values: [Float], _ shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        values.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: values.count) }
        return array
    }

    /// The first `count` rows of a [1, rows, columns] (or [1, columns])
    /// output, honoring its strides and element type.
    static func rows(_ array: MLMultiArray, count: Int) -> [[Float]] {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        let rowStride = shape.count >= 3 ? strides[1] : 0
        let columnStride = strides[shape.count - 1]
        let columns = shape[shape.count - 1]
        let rows = shape.count >= 3 ? min(count, shape[1]) : 1
        // The models declare float32 outputs; anything else takes the slow,
        // type-agnostic path.
        guard array.dataType == .float32 else {
            return (0 ..< rows).map { r in (0 ..< columns).map { array[r * rowStride + $0 * columnStride].floatValue } }
        }
        let p = array.dataPointer.assumingMemoryBound(to: Float.self)
        return (0 ..< rows).map { r in (0 ..< columns).map { p[r * rowStride + $0 * columnStride] } }
    }
}
