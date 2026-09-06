import Accelerate
import Foundation

/// STFT/ISTFT matching torch.stft/torch.istft exactly, for whatever
/// (nFFT, hopLength) a given model's ModelSpec specifies -- periodic Hann
/// window, center=True, normalized=False, winLength==nFFT (true of every
/// model added so far; asserted in init since a shorter zero-padded window
/// would need extra handling). Verified numerically against torch
/// (max abs diff ~3e-5 forward, ~1.5e-7 round-trip) during development --
/// see tools/convert for the reference generation used to check this.
///
/// vDSP_fft_zrip's real-FFT packing quirks this class accounts for:
///   - forward output is 2x the mathematical (unnormalized) DFT
///   - bin 0 (DC) and bin N/2 (Nyquist) share one packed slot: realp[0] is
///     DC's real part, imagp[0] is Nyquist's real part (both bins' imaginary
///     parts are mathematically zero for a real signal and are dropped --
///     torch's onesided irfft does the same)
///   - a forward-then-inverse round trip through zrip scales by 2*N, so an
///     inverse fed a "true" spectrum must be pre-multiplied by 2 and its
///     unpacked output divided by 2*N
nonisolated final class STFTProcessor {
    private let setup: FFTSetup
    private let window: [Float]
    private let windowSq: [Float]

    private let nFFT: Int
    private let hop: Int
    private let chunkSize: Int
    private let half: Int
    private let numFreq: Int
    private let pad: Int
    private let log2n: vDSP_Length

    init(nFFT: Int, hopLength: Int, winLength: Int, chunkSize: Int) {
        precondition(winLength == nFFT, "STFTProcessor assumes winLength == nFFT (true of every model added so far)")
        self.nFFT = nFFT
        self.hop = hopLength
        self.chunkSize = chunkSize
        half = nFFT / 2
        numFreq = half + 1
        pad = nFFT / 2
        log2n = vDSP_Length(log2(Double(nFFT)).rounded())
        guard let s = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Unable to create FFT setup")
        }
        setup = s

        var w = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            w[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(nFFT))
        }
        window = w
        windowSq = w.map { $0 * $0 }
    }

    deinit {
        vDSP_destroy_fftsetup(setup)
    }

    private func reflectPad(_ x: [Float], pad: Int) -> [Float] {
        let L = x.count
        var out = [Float](repeating: 0, count: L + 2 * pad)
        for i in 0..<pad { out[i] = x[pad - i] }
        for i in 0..<L { out[pad + i] = x[i] }
        for i in 0..<pad { out[pad + L + i] = x[L - 2 - i] }
        return out
    }

    /// Returns (real, imag), each flattened [freq][time] row-major
    /// (index = f * numFrames + t), for a signal of exactly `chunkSize`
    /// samples (as given at init).
    func forward(_ x: [Float]) -> (real: [Float], imag: [Float]) {
        precondition(x.count == chunkSize, "STFTProcessor.forward expects a full chunk")
        let padded = reflectPad(x, pad: pad)
        let T = (padded.count - nFFT) / hop + 1

        var real = [Float](repeating: 0, count: numFreq * T)
        var imag = [Float](repeating: 0, count: numFreq * T)

        var realp = [Float](repeating: 0, count: half)
        var imagp = [Float](repeating: 0, count: half)
        var frame = [Float](repeating: 0, count: nFFT)

        for t in 0..<T {
            let start = t * hop
            for n in 0..<nFFT {
                frame[n] = padded[start + n] * window[n]
            }
            realp.withUnsafeMutableBufferPointer { rp in
                imagp.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    frame.withUnsafeBufferPointer { fb in
                        fb.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { cplx in
                            vDSP_ctoz(cplx, 2, &split, 1, vDSP_Length(half))
                        }
                    }
                    vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
            real[0 * T + t] = realp[0] / 2
            real[half * T + t] = imagp[0] / 2
            for i in 1..<half {
                real[i * T + t] = realp[i] / 2
                imag[i * T + t] = imagp[i] / 2
            }
        }
        return (real, imag)
    }

    /// Inverse of `forward`: real/imag are flattened [freq][time]
    /// (index = f * T + t), returns exactly `chunkSize` samples.
    func inverse(real: [Float], imag: [Float], T: Int) -> [Float] {
        let paddedLen = (T - 1) * hop + nFFT
        var overlapBuf = [Float](repeating: 0, count: paddedLen)
        var winSumSq = [Float](repeating: 0, count: paddedLen)

        var realp = [Float](repeating: 0, count: half)
        var imagp = [Float](repeating: 0, count: half)
        var frameOut = [Float](repeating: 0, count: nFFT)

        for t in 0..<T {
            realp[0] = real[0 * T + t] * 2
            imagp[0] = real[half * T + t] * 2
            for i in 1..<half {
                realp[i] = real[i * T + t] * 2
                imagp[i] = imag[i * T + t] * 2
            }

            realp.withUnsafeMutableBufferPointer { rp in
                imagp.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Inverse))
                    frameOut.withUnsafeMutableBufferPointer { fo in
                        fo.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { cplx in
                            vDSP_ztoc(&split, 1, cplx, 2, vDSP_Length(half))
                        }
                    }
                }
            }

            let start = t * hop
            let normFactor = Float(1) / Float(2 * nFFT)
            for n in 0..<nFFT {
                let val = frameOut[n] * normFactor * window[n]
                overlapBuf[start + n] += val
                winSumSq[start + n] += windowSq[n]
            }
        }

        let eps: Float = 1e-11
        var out = [Float](repeating: 0, count: paddedLen)
        for i in 0..<paddedLen {
            out[i] = overlapBuf[i] / max(winSumSq[i], eps)
        }
        return Array(out[pad..<(pad + chunkSize)])
    }
}
