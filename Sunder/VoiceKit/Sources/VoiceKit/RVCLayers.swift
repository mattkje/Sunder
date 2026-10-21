import MLX
import MLXNN

// Building blocks shared by the RVC synthesizer and discriminator. Everything
// is channels-last ([batch, time, channels]) as MLX convolutions expect;
// parameter names mirror the PyTorch checkpoint keys (snake_case on purpose)
// so `export_rvc_base.py`'s safetensors load with a plain `update(parameters:)`.

let lreluSlope: Float = 0.1

/// Conv1d with PyTorch-style weight normalization (`w = g * v / ||v||`, norm
/// per output channel). Kept as a reparameterization during training, as RVC does.
final class WNConv1d: Module, UnaryLayer {
    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray?
    let stride: Int, padding: Int, dilation: Int, groups: Int

    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int = 1, padding: Int = 0,
         dilation: Int = 1, groups: Int = 1, bias: Bool = true) {
        weight_v = MLXRandom.normal([outC, kernel, inC / groups]) * 0.01
        weight_g = sqrt(sum(weight_v * weight_v, axes: [1, 2], keepDims: true))
        self.bias = bias ? MLXArray.zeros([outC]) : nil
        self.stride = stride; self.padding = padding; self.dilation = dilation; self.groups = groups
    }

    var weight: MLXArray { weight_g * weight_v / sqrt(sum(weight_v * weight_v, axes: [1, 2], keepDims: true)) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        return bias.map { y + $0 } ?? y
    }
}

/// ConvTranspose1d with weight norm. PyTorch normalizes a transposed conv's
/// weight per *input* channel (its dim 0), hence the [1, 1, in] gain.
final class WNConvTranspose1d: Module, UnaryLayer {
    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray
    let stride: Int, padding: Int

    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int, padding: Int) {
        weight_v = MLXRandom.normal([outC, kernel, inC]) * 0.01
        weight_g = sqrt(sum(weight_v * weight_v, axes: [0, 1], keepDims: true))
        bias = MLXArray.zeros([outC])
        self.stride = stride; self.padding = padding
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weight_g * weight_v / sqrt(sum(weight_v * weight_v, axes: [0, 1], keepDims: true))
        return convTransposed1d(x, w, stride: stride, padding: padding) + bias
    }
}

/// Conv2d with weight norm, kernel/stride/padding along the height axis only
/// (the period discriminators convolve over [time / period, period]).
final class WNConv2d: Module, UnaryLayer {
    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray
    let stride: Int, padding: Int

    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int, padding: Int) {
        weight_v = MLXRandom.normal([outC, kernel, 1, inC]) * 0.01
        weight_g = sqrt(sum(weight_v * weight_v, axes: [1, 2, 3], keepDims: true))
        bias = MLXArray.zeros([outC])
        self.stride = stride; self.padding = padding
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weight_g * weight_v / sqrt(sum(weight_v * weight_v, axes: [1, 2, 3], keepDims: true))
        return conv2d(x, w, stride: .init((stride, 1)), padding: .init((padding, 0))) + bias
    }
}

/// VITS-style LayerNorm over the channel axis, with `gamma`/`beta` names.
final class ChannelLayerNorm: Module, UnaryLayer {
    var gamma: MLXArray
    var beta: MLXArray

    init(_ channels: Int) {
        gamma = MLXArray.ones([channels])
        beta = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = ((x - mean) * (x - mean)).mean(axis: -1, keepDims: true)
        return (x - mean) * (variance + 1e-5).rsqrt() * gamma + beta
    }
}

func paddingFor(kernel: Int, dilation: Int = 1) -> Int { (kernel * dilation - dilation) / 2 }

/// Reflect-pads the time axis (axis 1) of a [B, T, C] array, like
/// `F.pad(..., mode="reflect")`. MLX's `padded` has no reflect mode.
func reflectPadTime(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    let t = x.dim(1)
    var parts: [MLXArray] = []
    if left > 0 { parts.append(x.take(MLXArray((1 ... left).reversed().map { Int32($0) }), axis: 1)) }
    parts.append(x)
    if right > 0 { parts.append(x.take(MLXArray((0 ..< right).map { Int32(t - 2 - $0) }), axis: 1)) }
    return concatenated(parts, axis: 1)
}

/// x[b, starts[b] ..< starts[b] + length] for every b, along axis 1 (x is
/// [B, T] or [B, T, C]; starts is [B] int32). Gather-based so it compiles.
func sliceFrames(_ x: MLXArray, starts: MLXArray, length: Int) -> MLXArray {
    var index = starts.expandedDimensions(axis: 1) + MLXArray(0 ..< Int32(length)).expandedDimensions(axis: 0)
    if x.ndim == 3 { index = index.expandedDimensions(axis: -1) }
    return takeAlong(x, index, axis: 1)
}

/// [B, T] lengths -> [B, T, 1] float mask.
func sequenceMask(_ lengths: [Int], maxLength: Int) -> MLXArray {
    let rows = lengths.map { len in (0 ..< maxLength).map { Float($0 < len ? 1 : 0) } }
    return MLXArray(rows.flatMap { $0 }, [lengths.count, maxLength, 1])
}

/// WaveNet-style gated residual stack (VITS `modules.WN`), conditioned on the
/// speaker embedding `g`.
final class WN: Module {
    let in_layers: [WNConv1d]
    let res_skip_layers: [WNConv1d]
    let cond_layer: WNConv1d
    let hidden: Int

    init(hidden: Int, kernel: Int, layers: Int, ginChannels: Int) {
        self.hidden = hidden
        in_layers = (0 ..< layers).map { _ in
            WNConv1d(hidden, 2 * hidden, kernel: kernel, padding: paddingFor(kernel: kernel))
        }
        res_skip_layers = (0 ..< layers).map { i in
            WNConv1d(hidden, i < layers - 1 ? 2 * hidden : hidden, kernel: 1)
        }
        cond_layer = WNConv1d(ginChannels, 2 * hidden * layers, kernel: 1)
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray, g: MLXArray) -> MLXArray {
        var x = input
        var output = zeros(like: x)
        let cond = cond_layer(g)
        for (i, (inLayer, resSkip)) in zip(in_layers, res_skip_layers).enumerated() {
            let offset = i * 2 * hidden
            let a = inLayer(x) + cond[0..., 0..., offset ..< offset + 2 * hidden]
            let acts = tanh(a[0..., 0..., ..<hidden]) * sigmoid(a[0..., 0..., hidden...])
            let rs = resSkip(acts)
            if i < in_layers.count - 1 {
                x = (x + rs[0..., 0..., ..<hidden]) * mask
                output = output + rs[0..., 0..., hidden...]
            } else {
                output = output + rs
            }
        }
        return output * mask
    }
}

/// HiFi-GAN ResBlock1 (three dilated conv pairs).
final class ResBlock1: Module, UnaryLayer {
    let convs1: [WNConv1d]
    let convs2: [WNConv1d]

    init(channels: Int, kernel: Int, dilations: [Int]) {
        convs1 = dilations.map {
            WNConv1d(channels, channels, kernel: kernel, padding: paddingFor(kernel: kernel, dilation: $0), dilation: $0)
        }
        convs2 = dilations.map { _ in
            WNConv1d(channels, channels, kernel: kernel, padding: paddingFor(kernel: kernel))
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for (c1, c2) in zip(convs1, convs2) {
            let xt = c2(leakyRelu(c1(leakyRelu(x, negativeSlope: lreluSlope)), negativeSlope: lreluSlope))
            x = xt + x
        }
        return x
    }
}
