import Accelerate
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Conversion settings: RVC's inference controls, plus Seed-VC's step count.
/// Each engine ignores the other's.
public struct ConversionSettings: Sendable, Equatable {
    /// Semitones to shift the source melody (e.g. +12 male -> female).
    public var pitchShift: Float = 0
    /// RVC: how strongly features are pulled toward the voice's training frames.
    public var indexRate: Float = 0.75
    /// RVC: keeps consonants and breaths from the source unconverted: lower
    /// values protect more, 0.5 disables protection.
    public var protect: Float = 0.33
    /// Seed-VC: diffusion steps per chunk -- more is cleaner and slower.
    public var diffusionSteps: Int = 30

    public init(pitchShift: Float = 0, indexRate: Float = 0.75, protect: Float = 0.33, diffusionSteps: Int = 30) {
        self.pitchShift = pitchShift
        self.indexRate = indexRate
        self.protect = protect
        self.diffusionSteps = diffusionSteps
    }
}

/// RVC's `Pipeline.vc` for one trained voice.
final class VoiceConverter {
    private let generator: RVCSynthesizer
    private let index: VoiceIndex
    private let features: VoiceFeatures

    init(voiceModel: URL, index indexURL: URL, features: VoiceFeatures) throws {
        generator = RVCSynthesizer()
        try RVCWeights.load(generator, from: voiceModel, verify: [.noUnusedKeys, .shapeMismatch])
        index = try VoiceIndex.load(from: indexURL)
        self.features = features
    }

    /// Mono 44.1 kHz in, mono 44.1 kHz out (same length).
    func convert(_ audio44k: [Float], settings: ConversionSettings, progress: (Double) -> Void) throws -> [Float] {
        let sr16 = VoiceFeatures.sampleRate, srOut = Double(RVCConfig.sampleRate)
        let source = try VoiceAudio.resample(audio44k, from: VoiceAudio.appSampleRate, to: sr16)
        guard source.count > 1600 else { return audio44k }

        // 1 s reflect padding on both sides, as RVC's pipeline does, so the
        // edges get the same context as the middle.
        let pad = Int(sr16)
        let padded = reflectPadTime(MLXArray(source).reshaped(1, -1, 1), left: min(pad, source.count - 1), right: min(pad, source.count - 1))
            .reshaped(-1).asArray(Float.self)
        let padUsed = min(pad, source.count - 1)

        let content = try features.content(padded)
        var f0 = try features.pitch(padded)
        progress(0.25)

        let raw = MLXArray(content.flatMap { $0 }, [content.count, 768])
        let blended = index.blend(raw, rate: settings.indexRate)
        let frames = min(padded.count / VoiceFeatures.pitchHop, content.count * 2, f0.count)
        var feats = repeated(blended, count: 2, axis: 0)[..<frames]
        let feats0 = repeated(raw, count: 2, axis: 0)[..<frames]

        let shift = pow(2, settings.pitchShift / 12)
        f0 = f0.prefix(frames).map { $0 * shift }
        if settings.protect < 0.5 {
            let keep = MLXArray(f0.map { $0 > 0 ? Float(1) : settings.protect }).expandedDimensions(axis: 1)
            feats = feats * keep + feats0 * (1 - keep)
        }
        let pitch = MLXArray(f0.map(PitchQuantizer.coarse))
        let pitchf = MLXArray(f0)

        let wave = synthesize(feats: feats, pitch: pitch, pitchf: pitchf) { progress(0.25 + 0.7 * $0) }
        let trim = padUsed * RVCConfig.sampleRate / Int(sr16)
        let body = Array(wave.dropFirst(trim).prefix(source.count * RVCConfig.sampleRate / Int(sr16)))
        var out = try VoiceAudio.resample(body, from: srOut, to: VoiceAudio.appSampleRate)
        if out.count < audio44k.count { out += [Float](repeating: 0, count: audio44k.count - out.count) }
        out = Array(out.prefix(audio44k.count))
        progress(1)
        return Self.matchEnvelope(out, to: audio44k, rate: 0.25)
    }

    /// RVC's `change_rms` (rms_mix_rate): pulls the output's loudness envelope
    /// (1 s windows, 0.5 s hop) toward the source's, so a converted stem keeps
    /// the original's dynamics in the mix. `rate` 1 keeps the output as is.
    static func matchEnvelope(_ output: [Float], to source: [Float], rate: Float) -> [Float] {
        let sr = Int(VoiceAudio.appSampleRate), window = sr, hop = sr / 2
        func envelope(_ x: [Float]) -> [Float] {
            let padded = [Float](repeating: 0, count: window / 2) + x + [Float](repeating: 0, count: window / 2)
            let frames = 1 + (padded.count - window) / hop
            let rms = (0 ..< frames).map { i -> Float in
                padded.withUnsafeBufferPointer { p in
                    var r: Float = 0
                    vDSP_rmsqv(p.baseAddress! + i * hop, 1, &r, vDSP_Length(window))
                    return r
                }
            }
            // Linear interpolation back to one value per sample.
            return (0 ..< x.count).map { n in
                let pos = Float(n) / Float(hop), i = min(Int(pos), rms.count - 1), j = min(i + 1, rms.count - 1)
                let t = pos - Float(i)
                return rms[i] * (1 - t) + rms[j] * t
            }
        }
        let a = envelope(source), b = envelope(output)
        return output.indices.map { n in output[n] * pow(a[n], 1 - rate) * pow(max(b[n], 1e-6), rate - 1) }
    }

    /// Runs the generator in 15 s windows with 1 s of context each side
    /// (attention cost is quadratic in length), crossfading 10 ms at seams.
    private func synthesize(feats: MLXArray, pitch: MLXArray, pitchf: MLXArray, progress: (Double) -> Void) -> [Float] {
        let total = feats.dim(0), core = 1500, context = 100, fade = 4 // frames
        let hop = RVCConfig.hop
        var output = [Float](repeating: 0, count: total * hop)
        var start = 0
        while start < total {
            let end = min(total, start + core)
            let lo = max(0, start - context), hi = min(total, end + context)
            let wave = generator.infer(
                phone: feats[lo ..< hi].expandedDimensions(axis: 0),
                pitch: pitch[lo ..< hi].expandedDimensions(axis: 0),
                pitchf: pitchf[lo ..< hi].expandedDimensions(axis: 0)).reshaped(-1)
            eval(wave)
            let samples = wave.asArray(Float.self)
            // Write [start - fade, end), fading in over the last `fade` frames
            // the previous window already wrote.
            let writeFrom = start == 0 ? 0 : start - fade
            for frame in writeFrom ..< end {
                for s in 0 ..< hop {
                    let global = frame * hop + s
                    let v = samples[global - lo * hop]
                    if frame < start {
                        let t = Float(global - writeFrom * hop) / Float(fade * hop)
                        output[global] = output[global] * (1 - t) + v * t
                    } else {
                        output[global] = v
                    }
                }
            }
            start = end
            progress(Double(end) / Double(total))
        }
        return output
    }
}
