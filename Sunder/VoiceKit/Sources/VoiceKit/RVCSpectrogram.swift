import Foundation
import MLX
import MLXFFT

// RVC's `spectrogram_torch` / `spec_to_mel_torch` in MLX
// (tools/convert/models/rvc/train/mel_processing.py): reflect-pad by
// (n_fft - hop) / 2, no centering, periodic Hann, magnitude with a 2e-7 floor,
// slaney mel (basis shipped in RVCBaseG.safetensors), log with a 2e-6 clamp.
// Differentiable, so the generator's mel loss backpropagates through it.

struct RVCSpectrogram {
    let melBasis: MLXArray // [125, 1025]
    private let window: MLXArray
    private let frameOffsets: MLXArray

    init(melBasis: MLXArray) {
        self.melBasis = melBasis
        let n = RVCConfig.nFFT
        window = MLXArray((0 ..< n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) })
        frameOffsets = MLXArray(0 ..< Int32(n))
    }

    /// wave: [B, T, 1] -> magnitude [B, frames, 1025], frames = T / hop.
    func linear(_ wave: MLXArray) -> MLXArray {
        let n = RVCConfig.nFFT, hop = RVCConfig.hop
        let pad = (n - hop) / 2
        let x = reflectPadTime(wave, left: pad, right: pad).squeezed(axis: -1) // [B, L]
        let frames = (x.dim(1) - n) / hop + 1
        let starts = MLXArray(0 ..< Int32(frames)) * Int32(hop)
        let index = starts.expandedDimensions(axis: 1) + frameOffsets.expandedDimensions(axis: 0) // [F, n]
        let framed = x.take(index.flattened(), axis: 1).reshaped(x.dim(0), frames, n) * window
        let spec = rfft(framed, axis: -1)
        return sqrt(square(spec.realPart()) + square(spec.imaginaryPart()) + 2e-7)
    }

    /// magnitude [B, F, 1025] -> log-mel [B, F, 125].
    func mel(fromLinear spec: MLXArray) -> MLXArray {
        log(maximum(matmul(spec, melBasis.transposed()), 2e-6))
    }
}
