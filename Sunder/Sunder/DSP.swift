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
/// Every model through the BS/Mel-Roformer family uses nFFT=2048 (a power
/// of two), handled below via `vDSP_fft_zrip` -- Apple's classic radix-2
/// real FFT, fast but restricted to power-of-two lengths. Apollo
/// (restoration) uses nFFT=882 (20ms @ 44.1kHz = 2*3^2*7^2), which is
/// neither power-of-two nor expressible via `vDSP_DFT_zop`'s "arbitrary
/// length" API either -- confirmed empirically: that API silently returns
/// a nil setup for any length with a factor of 7 (or 11, 13), only
/// {2,3,5}-composite lengths work. So a non-power-of-two `nFFT` instead
/// goes through `BluesteinDFT` below (chirp-z transform on top of the same
/// proven `vDSP_fft_zip` power-of-two complex FFT), verified against a
/// NumPy `rfft`/`irfft` reference at n=882 specifically (max abs diff
/// ~7.6e-6 forward, ~2.5e-7 round-trip -- see tools/convert's Apollo
/// verification for the reference generation).
///
/// vDSP_fft_zrip's real-FFT packing quirks the power-of-two path accounts
/// for (irrelevant to the Bluestein path, which computes the true
/// unnormalized DFT directly with no packing trick):
///   - forward output is 2x the mathematical (unnormalized) DFT
///   - bin 0 (DC) and bin N/2 (Nyquist) share one packed slot: realp[0] is
///     DC's real part, imagp[0] is Nyquist's real part (both bins' imaginary
///     parts are mathematically zero for a real signal and are dropped --
///     torch's onesided irfft does the same)
///   - a forward-then-inverse round trip through zrip scales by 2*N, so an
///     inverse fed a "true" spectrum must be pre-multiplied by 2 and its
///     unpacked output divided by 2*N
///
/// Both paths present the same external (real, imag) packing to callers:
/// `real[0]`/`real[half]` hold the DC/Nyquist bins' (real-valued) content,
/// `imag[0]`/`imag[half]` are always zero, interior bins are plain
/// (real, imag) pairs.
nonisolated final class STFTProcessor {
    private let window: [Float]
    private let windowSq: [Float]

    private let nFFT: Int
    private let hop: Int
    private let chunkSize: Int
    private let half: Int
    private let numFreq: Int
    private let pad: Int

    /// Set when nFFT is a power of two; nil (and `bluestein` set instead)
    /// otherwise. Exactly one of the two is non-nil.
    private let radix2Setup: FFTSetup?
    private let log2n: vDSP_Length
    private let bluestein: BluesteinDFT?

    init(nFFT: Int, hopLength: Int, winLength: Int, chunkSize: Int) {
        precondition(winLength == nFFT, "STFTProcessor assumes winLength == nFFT (true of every model added so far)")
        self.nFFT = nFFT
        self.hop = hopLength
        self.chunkSize = chunkSize
        half = nFFT / 2
        numFreq = half + 1
        pad = nFFT / 2

        if nFFT & (nFFT - 1) == 0 {
            log2n = vDSP_Length(log2(Double(nFFT)).rounded())
            guard let s = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
                fatalError("Unable to create FFT setup")
            }
            radix2Setup = s
            bluestein = nil
        } else {
            log2n = 0
            radix2Setup = nil
            bluestein = BluesteinDFT(n: nFFT)
        }

        var w = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            w[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(nFFT))
        }
        window = w
        windowSq = w.map { $0 * $0 }
    }

    deinit {
        if let radix2Setup {
            vDSP_destroy_fftsetup(radix2Setup)
        }
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
        var frame = [Float](repeating: 0, count: nFFT)

        if let radix2Setup {
            var realp = [Float](repeating: 0, count: half)
            var imagp = [Float](repeating: 0, count: half)

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
                        vDSP_fft_zrip(radix2Setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                    }
                }
                real[0 * T + t] = realp[0] / 2
                real[half * T + t] = imagp[0] / 2
                for i in 1..<half {
                    real[i * T + t] = realp[i] / 2
                    imag[i * T + t] = imagp[i] / 2
                }
            }
        } else {
            let bluestein = bluestein!
            let zeroImag = [Float](repeating: 0, count: nFFT)
            for t in 0..<T {
                let start = t * hop
                for n in 0..<nFFT {
                    frame[n] = padded[start + n] * window[n]
                }
                let (r, i) = bluestein.forward(real: frame, imag: zeroImag)
                real[0 * T + t] = r[0]
                real[half * T + t] = r[half]
                for f in 1..<half {
                    real[f * T + t] = r[f]
                    imag[f * T + t] = i[f]
                }
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
        var frameOut = [Float](repeating: 0, count: nFFT)

        if let radix2Setup {
            var realp = [Float](repeating: 0, count: half)
            var imagp = [Float](repeating: 0, count: half)

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
                        vDSP_fft_zrip(radix2Setup, &split, 1, log2n, FFTDirection(kFFTDirection_Inverse))
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
        } else {
            let bluestein = bluestein!
            var fullReal = [Float](repeating: 0, count: nFFT)
            var fullImag = [Float](repeating: 0, count: nFFT)

            for t in 0..<T {
                fullReal[0] = real[0 * T + t]
                fullImag[0] = 0
                fullReal[half] = real[half * T + t]
                fullImag[half] = 0
                for f in 1..<half {
                    fullReal[f] = real[f * T + t]
                    fullImag[f] = imag[f * T + t]
                    fullReal[nFFT - f] = fullReal[f]
                    fullImag[nFFT - f] = -fullImag[f]
                }

                let (invReal, _) = bluestein.inverseUnnormalized(real: fullReal, imag: fullImag)

                let start = t * hop
                let normFactor = Float(1) / Float(nFFT)
                for n in 0..<nFFT {
                    let val = invReal[n] * normFactor * window[n]
                    overlapBuf[start + n] += val
                    winSumSq[start + n] += windowSq[n]
                }
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

/// Arbitrary-length DFT via Bluestein's algorithm (chirp-z transform),
/// built on top of `vDSP_fft_zip` (power-of-two complex FFT, unrestricted
/// and already relied on elsewhere in this file). Used for nFFT values
/// `vDSP_DFT_zop_CreateSetup` can't handle (any length with a factor of 7,
/// 11, or 13 -- confirmed empirically, not documented) -- specifically
/// Apollo's nFFT=882=2*3^2*7^2.
///
/// Precomputes the chirp sequence and the (fixed) filter's FFT once at
/// init since `n` never changes for a given STFTProcessor instance, so the
/// per-frame cost is two length-`m` FFTs plus O(n) elementwise work --
/// comparable to the cost already accepted for the power-of-two models'
/// own 2048-point transforms.
private final class BluesteinDFT {
    private let n: Int
    private let m: Int
    private let log2m: vDSP_Length
    private let setup: FFTSetup
    private let chirpReal: [Float]
    private let chirpImag: [Float]
    private var filterFFTReal: [Float]
    private var filterFFTImag: [Float]

    init(n: Int) {
        self.n = n
        var mm = 1
        while mm < 2 * n - 1 { mm *= 2 }
        m = mm
        log2m = vDSP_Length(log2(Double(mm)).rounded())
        guard let s = vDSP_create_fftsetup(log2m, FFTRadix(kFFTRadix2)) else {
            fatalError("Unable to create Bluestein FFT setup for n=\(n)")
        }
        setup = s

        // chirp[k] = exp(-i*pi*k^2/n); k^2 computed in Double to stay
        // accurate before the phase is wrapped and dropped to Float.
        var cr = [Float](repeating: 0, count: n)
        var ci = [Float](repeating: 0, count: n)
        for k in 0..<n {
            let angle = -Double.pi * Double(k) * Double(k) / Double(n)
            cr[k] = Float(cos(angle))
            ci[k] = Float(sin(angle))
        }
        chirpReal = cr
        chirpImag = ci

        // Convolution filter b[k] = conj(chirp[k]) for |k| < n, zero
        // elsewhere, wrapped circularly into length m (negative indices at
        // m-k); its FFT is fixed for the life of this instance.
        var br = [Float](repeating: 0, count: mm)
        var bi = [Float](repeating: 0, count: mm)
        for k in 0..<n {
            br[k] = cr[k]
            bi[k] = -ci[k]
        }
        for k in 1..<n {
            br[mm - k] = cr[k]
            bi[mm - k] = -ci[k]
        }
        filterFFTReal = br
        filterFFTImag = bi
        Self.fft(&filterFFTReal, &filterFFTImag, setup: setup, log2m: log2m, inverse: false)
    }

    deinit {
        vDSP_destroy_fftsetup(setup)
    }

    private static func fft(_ real: inout [Float], _ imag: inout [Float], setup: FFTSetup, log2m: vDSP_Length, inverse: Bool) {
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zip(setup, &split, 1, log2m, inverse ? FFTDirection(kFFTDirection_Inverse) : FFTDirection(kFFTDirection_Forward))
            }
        }
    }

    /// Unnormalized forward DFT: X[k] = sum_n x[n] * exp(-2*pi*i*k*n/n),
    /// matching numpy/torch's forward convention exactly (no 1/n scaling).
    func forward(real: [Float], imag: [Float]) -> (real: [Float], imag: [Float]) {
        var ar = [Float](repeating: 0, count: m)
        var ai = [Float](repeating: 0, count: m)
        for k in 0..<n {
            let xr = real[k], xi = imag[k]
            let cr = chirpReal[k], ci = chirpImag[k]
            ar[k] = xr * cr - xi * ci
            ai[k] = xr * ci + xi * cr
        }
        Self.fft(&ar, &ai, setup: setup, log2m: log2m, inverse: false)

        var pr = [Float](repeating: 0, count: m)
        var pi = [Float](repeating: 0, count: m)
        for i in 0..<m {
            let r0 = ar[i], i0 = ai[i], r1 = filterFFTReal[i], i1 = filterFFTImag[i]
            pr[i] = r0 * r1 - i0 * i1
            pi[i] = r0 * i1 + i0 * r1
        }
        Self.fft(&pr, &pi, setup: setup, log2m: log2m, inverse: true)
        let scale = Float(1.0 / Double(m))

        var outReal = [Float](repeating: 0, count: n)
        var outImag = [Float](repeating: 0, count: n)
        for k in 0..<n {
            let cr0 = pr[k] * scale
            let ci0 = pi[k] * scale
            let cr = chirpReal[k], ci = chirpImag[k]
            outReal[k] = cr0 * cr - ci0 * ci
            outImag[k] = cr0 * ci + ci0 * cr
        }
        return (outReal, outImag)
    }

    /// Unnormalized inverse-style DFT: x[n] = sum_k X[k] * exp(+2*pi*i*k*n/n)
    /// (no 1/n scaling -- callers divide by n themselves, matching the
    /// power-of-two path's own manual-normalization convention).
    func inverseUnnormalized(real: [Float], imag: [Float]) -> (real: [Float], imag: [Float]) {
        let conjIn = imag.map { -$0 }
        let (r, i) = forward(real: real, imag: conjIn)
        return (r, i.map { -$0 })
    }
}
