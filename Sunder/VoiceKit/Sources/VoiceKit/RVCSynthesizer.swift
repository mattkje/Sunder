import Foundation
import MLX
import MLXNN
import MLXRandom

// MLX port of RVC v2's `SynthesizerTrnMs768NSFsid` (40 kHz, f0) from
// tools/convert/models/rvc/models.py, channels-last. Shapes in comments are
// [batch, time, channels]; frame rate is 100 fps (hop 400 @ 40 kHz).

enum RVCConfig {
    static let sampleRate = 40000
    static let hop = 400
    static let nFFT = 2048
    static let specChannels = 1025
    static let melChannels = 125
    static let segmentFrames = 32 // 12800 samples
    static let hidden = 192
    static let inter = 192
    static let filter = 768
    static let heads = 2
    static let layers = 6
    static let gin = 256
    static let speakers = 109
    static let upsampleRates = [10, 10, 2, 2]
    static let upsampleKernels = [16, 16, 4, 4]
    static let upsampleInitial = 512
    static let resblockKernels = [3, 7, 11]
    static let resblockDilations = [1, 3, 5]
    static var upp: Int { upsampleRates.reduce(1, *) }
}

/// Self-attention with VITS's windowed relative position embeddings (window
/// 10, shared across heads).
final class RelativeMultiHeadAttention: Module {
    let conv_q: Conv1d, conv_k: Conv1d, conv_v: Conv1d, conv_o: Conv1d
    var emb_rel_k: MLXArray
    var emb_rel_v: MLXArray
    let heads: Int, dk: Int
    let window = 10

    init(channels: Int, heads: Int) {
        self.heads = heads
        dk = channels / heads
        conv_q = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        conv_k = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        conv_v = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        conv_o = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        emb_rel_k = MLXRandom.normal([1, 2 * window + 1, dk]) * pow(Float(dk), -0.5)
        emb_rel_v = MLXRandom.normal([1, 2 * window + 1, dk]) * pow(Float(dk), -0.5)
    }

    /// x: [B, T, C]; attnMask: [B, 1, T, T] (1 = attend).
    func callAsFunction(_ x: MLXArray, attnMask: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, t, heads, dk).transposed(0, 2, 1, 3) }
        let q = split(conv_q(x)) / sqrt(Float(dk))
        let k = split(conv_k(x))
        let v = split(conv_v(x))

        // Relative offset j - i for every (i, j), clipped into the embedding window.
        let pos = MLXArray(0 ..< Int32(t))
        let offset = pos.expandedDimensions(axis: 0) - pos.expandedDimensions(axis: 1) // [T, T]
        let inWindow = (abs(offset) .<= window).asType(.float32)
        let relIndex = clip(offset + window, min: 0, max: 2 * window).asType(.int32)

        var scores = matmul(q, k.transposed(0, 1, 3, 2))
        let relLogits = matmul(q, emb_rel_k[0].transposed()) // [B, H, T, 2W+1]
        let local = takeAlong(relLogits, broadcast(relIndex, to: [b, heads, t, t]), axis: 3)
        scores = scores + local * inWindow
        scores = which(attnMask .== 0, MLXArray(Float(-1e4)), scores)
        let p = softmax(scores, axis: -1)

        var out = matmul(p, v)
        // Relative values: weight for offset m - W at query i is p[i, i + m - W].
        let m = MLXArray(0 ..< Int32(2 * window + 1))
        let key = pos.expandedDimensions(axis: 1) + m.expandedDimensions(axis: 0) - Int32(window) // [T, 2W+1]
        let keyValid = ((key .>= 0) .&& (key .< t)).asType(.float32)
        let relWeights = takeAlong(p, broadcast(clip(key, min: 0, max: t - 1), to: [b, heads, t, 2 * window + 1]), axis: 3) * keyValid
        out = out + matmul(relWeights, emb_rel_v[0])
        return conv_o(out.transposed(0, 2, 1, 3).reshaped(b, t, heads * dk))
    }
}

final class FFN: Module {
    let conv_1: Conv1d, conv_2: Conv1d

    init(channels: Int, filter: Int, kernel: Int) {
        conv_1 = Conv1d(inputChannels: channels, outputChannels: filter, kernelSize: kernel, padding: kernel / 2)
        conv_2 = Conv1d(inputChannels: filter, outputChannels: channels, kernelSize: kernel, padding: kernel / 2)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        conv_2(relu(conv_1(x * mask)) * mask) * mask
    }
}

final class AttentionEncoder: Module {
    let attn_layers: [RelativeMultiHeadAttention]
    let norm_layers_1: [ChannelLayerNorm]
    let ffn_layers: [FFN]
    let norm_layers_2: [ChannelLayerNorm]

    override init() {
        let n = RVCConfig.layers, c = RVCConfig.hidden
        attn_layers = (0 ..< n).map { _ in RelativeMultiHeadAttention(channels: c, heads: RVCConfig.heads) }
        norm_layers_1 = (0 ..< n).map { _ in ChannelLayerNorm(c) }
        ffn_layers = (0 ..< n).map { _ in FFN(channels: c, filter: RVCConfig.filter, kernel: 3) }
        norm_layers_2 = (0 ..< n).map { _ in ChannelLayerNorm(c) }
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let attnMask = matmul(mask, mask.transposed(0, 2, 1)).expandedDimensions(axis: 1)
        var x = input * mask
        for i in 0 ..< attn_layers.count {
            x = norm_layers_1[i](x + attn_layers[i](x, attnMask: attnMask))
            x = norm_layers_2[i](x + ffn_layers[i](x, mask: mask))
        }
        return x * mask
    }
}

/// Prior encoder: ContentVec features + coarse pitch -> (m_p, logs_p).
final class TextEncoder: Module {
    let emb_phone: Linear
    let emb_pitch: Embedding
    let encoder: AttentionEncoder
    let proj: Conv1d

    override init() {
        emb_phone = Linear(768, RVCConfig.hidden)
        emb_pitch = Embedding(embeddingCount: 256, dimensions: RVCConfig.hidden)
        encoder = AttentionEncoder()
        proj = Conv1d(inputChannels: RVCConfig.hidden, outputChannels: RVCConfig.inter * 2, kernelSize: 1)
    }

    /// phone: [B, T, 768], pitch: [B, T] int32, mask: [B, T, 1].
    func callAsFunction(phone: MLXArray, pitch: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var x = (emb_phone(phone) + emb_pitch(pitch)) * sqrt(Float(RVCConfig.hidden))
        x = leakyRelu(x, negativeSlope: 0.1)
        x = encoder(x, mask: mask)
        let stats = proj(x) * mask
        return (stats[0..., 0..., ..<RVCConfig.inter], stats[0..., 0..., RVCConfig.inter...])
    }
}

/// Posterior encoder: linear spectrogram -> z (training only).
final class PosteriorEncoder: Module {
    let pre: Conv1d
    let enc: WN
    let proj: Conv1d

    override init() {
        pre = Conv1d(inputChannels: RVCConfig.specChannels, outputChannels: RVCConfig.hidden, kernelSize: 1)
        enc = WN(hidden: RVCConfig.hidden, kernel: 5, layers: 16, ginChannels: RVCConfig.gin)
        proj = Conv1d(inputChannels: RVCConfig.hidden, outputChannels: RVCConfig.inter * 2, kernelSize: 1)
    }

    func callAsFunction(_ spec: MLXArray, mask: MLXArray, g: MLXArray) -> (z: MLXArray, m: MLXArray, logs: MLXArray) {
        let h = enc(pre(spec) * mask, mask: mask, g: g)
        let stats = proj(h) * mask
        let m = stats[0..., 0..., ..<RVCConfig.inter]
        let logs = stats[0..., 0..., RVCConfig.inter...]
        let z = (m + MLXRandom.normal(m.shape) * exp(logs)) * mask
        return (z, m, logs)
    }
}

final class ResidualCouplingLayer: Module {
    let pre: Conv1d
    let enc: WN
    let post: Conv1d
    let half = RVCConfig.inter / 2

    override init() {
        pre = Conv1d(inputChannels: RVCConfig.inter / 2, outputChannels: RVCConfig.hidden, kernelSize: 1)
        enc = WN(hidden: RVCConfig.hidden, kernel: 5, layers: 3, ginChannels: RVCConfig.gin)
        post = Conv1d(inputChannels: RVCConfig.hidden, outputChannels: RVCConfig.inter / 2, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, g: MLXArray, reverse: Bool) -> MLXArray {
        let x0 = x[0..., 0..., ..<half]
        let x1 = x[0..., 0..., half...]
        let m = post(enc(pre(x0) * mask, mask: mask, g: g)) * mask
        let y1 = reverse ? (x1 - m) * mask : m + x1 * mask
        return concatenated([x0, y1], axis: -1)
    }
}

/// Four mean-only coupling layers, each followed by a channel flip.
final class ResidualCouplingBlock: Module {
    let flows: [ResidualCouplingLayer]

    override init() { flows = (0 ..< 4).map { _ in ResidualCouplingLayer() } }

    private func flip(_ x: MLXArray) -> MLXArray { x[0..., 0..., .stride(by: -1)] }

    func callAsFunction(_ input: MLXArray, mask: MLXArray, g: MLXArray, reverse: Bool = false) -> MLXArray {
        var x = input
        if reverse {
            for flow in flows.reversed() { x = flow(flip(x), mask: mask, g: g, reverse: true) }
        } else {
            for flow in flows { x = flip(flow(x, mask: mask, g: g, reverse: false)) }
        }
        return x
    }
}

/// Harmonic-plus-noise excitation from frame-rate f0 (`SourceModuleHnNSF`).
final class SourceModuleHnNSF: Module {
    let l_linear: Linear
    /// Test hook: zero the Gaussian noise terms for deterministic parity checks.
    var deterministic = false

    override init() { l_linear = Linear(1, 1) }

    /// f0: [B, T] Hz (0 = unvoiced) -> [B, T * upp, 1].
    func callAsFunction(_ f0: MLXArray, upp: Int) -> MLXArray {
        let (b, t) = (f0.dim(0), f0.dim(1))
        let f = f0.expandedDimensions(axis: -1) // [B, T, 1]
        var rad = f / Float(RVCConfig.sampleRate) * MLXArray(1 ... Int32(upp)).asType(.float32)
        let rad2 = remainder(rad[0..., 0..., (upp - 1)...] + 0.5, 1) - 0.5
        let acc = remainder(cumsum(rad2, axis: 1), 1)
        let shifted = concatenated([zeros([b, 1, 1]), acc[0..., ..<(t - 1), 0...]], axis: 1)
        rad = (rad + shifted).reshaped(b, t * upp, 1)
        var sine = sin(2 * Float.pi * rad) * 0.1
        let uv = repeated((f .> 0).asType(.float32), count: upp, axis: 1)
        if !deterministic {
            let noiseAmp = uv * 0.003 + (1 - uv) * (0.1 / 3)
            sine = sine * uv + noiseAmp * MLXRandom.normal(sine.shape)
        } else {
            sine = sine * uv
        }
        return tanh(l_linear(stopGradient(sine)))
    }
}

/// NSF-HiFiGAN decoder: latent frames + f0 -> 40 kHz waveform.
final class GeneratorNSF: Module {
    let m_source: SourceModuleHnNSF
    let conv_pre: Conv1d
    let cond: Conv1d
    let ups: [WNConvTranspose1d]
    let noise_convs: [Conv1d]
    let resblocks: [ResBlock1]
    let conv_post: Conv1d

    override init() {
        let rates = RVCConfig.upsampleRates, kernels = RVCConfig.upsampleKernels, c0 = RVCConfig.upsampleInitial
        m_source = SourceModuleHnNSF()
        conv_pre = Conv1d(inputChannels: RVCConfig.inter, outputChannels: c0, kernelSize: 7, padding: 3)
        cond = Conv1d(inputChannels: RVCConfig.gin, outputChannels: c0, kernelSize: 1)
        ups = rates.indices.map { i in
            WNConvTranspose1d(c0 >> i, c0 >> (i + 1), kernel: kernels[i], stride: rates[i], padding: (kernels[i] - rates[i]) / 2)
        }
        noise_convs = rates.indices.map { i in
            let c = c0 >> (i + 1)
            guard i + 1 < rates.count else { return Conv1d(inputChannels: 1, outputChannels: c, kernelSize: 1) }
            let s = rates[(i + 1)...].reduce(1, *)
            return Conv1d(inputChannels: 1, outputChannels: c, kernelSize: s * 2, stride: s, padding: s / 2)
        }
        resblocks = rates.indices.flatMap { i in
            RVCConfig.resblockKernels.map { ResBlock1(channels: c0 >> (i + 1), kernel: $0, dilations: RVCConfig.resblockDilations) }
        }
        conv_post = Conv1d(inputChannels: c0 >> rates.count, outputChannels: 1, kernelSize: 7, padding: 3, bias: false)
    }

    /// x: [B, T, 192], f0: [B, T], g: [B, 1, 256] -> [B, T * 400, 1].
    func callAsFunction(_ input: MLXArray, f0: MLXArray, g: MLXArray) -> MLXArray {
        let harmonic = m_source(f0, upp: RVCConfig.upp)
        var x = conv_pre(input) + cond(g)
        let nk = RVCConfig.resblockKernels.count
        for i in ups.indices {
            x = ups[i](leakyRelu(x, negativeSlope: lreluSlope)) + noise_convs[i](harmonic)
            var xs = resblocks[i * nk](x)
            for j in 1 ..< nk { xs = xs + resblocks[i * nk + j](x) }
            x = xs / Float(nk)
        }
        return tanh(conv_post(leakyRelu(x, negativeSlope: 0.01)))
    }
}

/// The full generator. Single-speaker fine-tuning always uses speaker id 0.
public final class RVCSynthesizer: Module {
    let enc_p: TextEncoder
    let dec: GeneratorNSF
    let enc_q: PosteriorEncoder
    let flow: ResidualCouplingBlock
    let emb_g: Embedding

    public override init() {
        enc_p = TextEncoder()
        dec = GeneratorNSF()
        enc_q = PosteriorEncoder()
        flow = ResidualCouplingBlock()
        emb_g = Embedding(embeddingCount: RVCConfig.speakers, dimensions: RVCConfig.gin)
    }

    func speaker(batch: Int) -> MLXArray {
        emb_g(zeros([batch, 1], type: Int32.self))
    }

    struct TrainOutput {
        let audio: MLXArray        // [B, 12800, 1]
        let zP: MLXArray, mP: MLXArray, logsP: MLXArray, logsQ: MLXArray
    }

    /// Training forward pass (`SynthesizerTrnMs256NSFsid.forward`). `starts`
    /// ([B] int32) picks each item's 32-frame decoder slice; the caller draws
    /// them so this stays free of host-side randomness (it runs compiled).
    func train(phone: MLXArray, pitch: MLXArray, pitchf: MLXArray, spec: MLXArray, mask: MLXArray, starts: MLXArray) -> TrainOutput {
        let g = speaker(batch: phone.dim(0))
        let (mP, logsP) = enc_p(phone: phone, pitch: pitch, mask: mask)
        let q = enc_q(spec, mask: mask, g: g)
        let zP = flow(q.z, mask: mask, g: g)
        let seg = RVCConfig.segmentFrames
        let audio = dec(sliceFrames(q.z, starts: starts, length: seg), f0: sliceFrames(pitchf, starts: starts, length: seg), g: g)
        return TrainOutput(audio: audio, zP: zP, mP: mP, logsP: logsP, logsQ: q.logs)
    }

    /// Inference (`infer`): features + pitch -> waveform [1, T * 400, 1].
    public func infer(phone: MLXArray, pitch: MLXArray, pitchf: MLXArray, noiseScale: Float = 0.66666) -> MLXArray {
        let mask = MLXArray.ones([1, phone.dim(1), 1])
        let g = speaker(batch: 1)
        let (mP, logsP) = enc_p(phone: phone, pitch: pitch, mask: mask)
        let zP = (mP + exp(logsP) * MLXRandom.normal(mP.shape) * noiseScale) * mask
        let z = flow(zP, mask: mask, g: g, reverse: true)
        return dec(z * mask, f0: pitchf, g: g)
    }
}
