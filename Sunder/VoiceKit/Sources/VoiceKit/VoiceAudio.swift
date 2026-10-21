import Accelerate
import AVFoundation

/// Mono Float32 helpers for the voice pipeline. VoiceKit's public API takes and
/// returns mono audio at `VoiceAudio.appSampleRate` (Sunder's 44.1 kHz);
/// internally it works at 16 kHz (features, pitch) and 40 kHz (the model).
public enum VoiceAudio {
    public static let appSampleRate: Double = 44100

    public enum Failure: Error { case converter, buffer }

    /// Band-limited resampling via AVAudioConverter (mastering-quality SRC).
    public static func resample(_ samples: [Float], from source: Double, to target: Double) throws -> [Float] {
        guard source != target, !samples.isEmpty else { return samples }
        guard let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: source, channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: target, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else { throw Failure.converter }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering

        guard let input = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(samples.count)) else { throw Failure.buffer }
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }

        let expected = Int((Double(samples.count) * target / source).rounded())
        guard let output = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: AVAudioFrameCount(expected + 8192)) else { throw Failure.buffer }
        var consumed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if consumed {
                status.pointee = .endOfStream
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return input
        }
        if let error { throw error }
        var result = Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
        // Pin the length so frame math downstream is exact.
        if result.count > expected { result.removeLast(result.count - expected) }
        if result.count < expected { result += [Float](repeating: 0, count: expected - result.count) }
        return result
    }

    static func peak(_ x: ArraySlice<Float>) -> Float {
        x.withUnsafeBufferPointer { p in
            var m: Float = 0
            vDSP_maxmgv(p.baseAddress!, 1, &m, vDSP_Length(p.count))
            return m
        }
    }

    /// 4th-order Butterworth high-pass (two biquads), removing rumble below
    /// `cutoff` before training -- RVC's preprocess uses a 48 Hz high-pass.
    static func highPass(_ x: [Float], cutoff: Double, sampleRate: Double) -> [Float] {
        let w0 = 2 * Double.pi * cutoff / sampleRate
        var coefficients: [Double] = []
        for q in [0.54119610, 1.30656296] { // Butterworth Q pair for 4th order
            let alpha = sin(w0) / (2 * q), cosw = cos(w0), a0 = 1 + alpha
            coefficients += [(1 + cosw) / 2 / a0, -(1 + cosw) / a0, (1 + cosw) / 2 / a0, -2 * cosw / a0, (1 - alpha) / a0]
        }
        guard let setup = vDSP.Biquad(coefficients: coefficients, channelCount: 1, sectionCount: 2, ofType: Float.self) else { return x }
        var biquad = setup
        return biquad.apply(input: x)
    }
}
