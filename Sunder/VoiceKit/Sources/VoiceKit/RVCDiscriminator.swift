import MLX
import MLXNN

// MLX port of RVC's `MultiPeriodDiscriminatorV2`: one scale discriminator
// plus period discriminators for periods 2, 3, 5, 7, 11, 17, 23, 37.

protocol WaveDiscriminator: Module {
    /// wave: [B, T, 1] -> (flattened logits [B, N], feature maps)
    func callAsFunction(_ wave: MLXArray) -> (MLXArray, [MLXArray])
}

final class DiscriminatorS: Module, WaveDiscriminator {
    let convs: [WNConv1d]
    let conv_post: WNConv1d

    override init() {
        convs = [
            WNConv1d(1, 16, kernel: 15, padding: 7),
            WNConv1d(16, 64, kernel: 41, stride: 4, padding: 20, groups: 4),
            WNConv1d(64, 256, kernel: 41, stride: 4, padding: 20, groups: 16),
            WNConv1d(256, 1024, kernel: 41, stride: 4, padding: 20, groups: 64),
            WNConv1d(1024, 1024, kernel: 41, stride: 4, padding: 20, groups: 256),
            WNConv1d(1024, 1024, kernel: 5, padding: 2),
        ]
        conv_post = WNConv1d(1024, 1, kernel: 3, padding: 1)
    }

    func callAsFunction(_ wave: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = wave
        var fmap: [MLXArray] = []
        for conv in convs {
            x = leakyRelu(conv(x), negativeSlope: lreluSlope)
            fmap.append(x)
        }
        x = conv_post(x)
        fmap.append(x)
        return (x.flattened(start: 1), fmap)
    }
}

final class DiscriminatorP: Module, WaveDiscriminator {
    let convs: [WNConv2d]
    let conv_post: WNConv2d
    let period: Int

    init(period: Int) {
        self.period = period
        let channels = [1, 32, 128, 512, 1024, 1024]
        convs = (0 ..< 5).map { i in
            WNConv2d(channels[i], channels[i + 1], kernel: 5, stride: i < 4 ? 3 : 1, padding: 2)
        }
        conv_post = WNConv2d(1024, 1, kernel: 3, stride: 1, padding: 1)
    }

    func callAsFunction(_ wave: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = wave
        let t = x.dim(1)
        if t % period != 0 { x = reflectPadTime(x, left: 0, right: period - t % period) }
        x = x.reshaped(x.dim(0), x.dim(1) / period, period, 1)
        var fmap: [MLXArray] = []
        for conv in convs {
            x = leakyRelu(conv(x), negativeSlope: lreluSlope)
            fmap.append(x)
        }
        x = conv_post(x)
        fmap.append(x)
        return (x.flattened(start: 1), fmap)
    }
}

public final class RVCDiscriminator: Module {
    let discriminators: [any WaveDiscriminator]

    public override init() {
        discriminators = [DiscriminatorS()] + [2, 3, 5, 7, 11, 17, 23, 37].map { DiscriminatorP(period: $0) }
    }

    func callAsFunction(_ wave: MLXArray) -> (logits: [MLXArray], fmaps: [[MLXArray]]) {
        var logits: [MLXArray] = [], fmaps: [[MLXArray]] = []
        for d in discriminators {
            let (l, f) = d(wave)
            logits.append(l)
            fmaps.append(f)
        }
        return (logits, fmaps)
    }
}

// MARK: - Losses (tools/convert/models/rvc/train/losses.py)

enum RVCLoss {
    static func discriminator(real: [MLXArray], fake: [MLXArray]) -> MLXArray {
        zip(real, fake).reduce(MLXArray(Float(0))) { acc, pair in
            acc + mean(square(1 - pair.0)) + mean(square(pair.1))
        }
    }

    static func generator(fake: [MLXArray]) -> MLXArray {
        fake.reduce(MLXArray(Float(0))) { $0 + mean(square(1 - $1)) }
    }

    static func featureMatching(real: [[MLXArray]], fake: [[MLXArray]]) -> MLXArray {
        var loss = MLXArray(Float(0))
        for (dr, dg) in zip(real, fake) {
            for (r, g) in zip(dr, dg) { loss = loss + mean(abs(stopGradient(r) - g)) }
        }
        return loss * 2
    }

    static func kl(zP: MLXArray, logsQ: MLXArray, mP: MLXArray, logsP: MLXArray, mask: MLXArray) -> MLXArray {
        let kl = logsP - logsQ - 0.5 + 0.5 * square(zP - mP) * exp(-2 * logsP)
        return sum(kl * mask) / sum(mask)
    }
}
