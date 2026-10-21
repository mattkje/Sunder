import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Fine-tunes the pretrained RVC v2 40 kHz generator/discriminator on one
/// voice, with RVC's objective (`train/train.py`): mel L1 x45 + KL + feature
/// matching + LSGAN, AdamW(1e-4, betas 0.8/0.99, eps 1e-9), lr x0.999875 per
/// epoch. Batches are length-bucketed like RVC's DistributedBucketSampler.
///
/// Each step is a generator update followed by a discriminator update on the
/// same generated audio, both as compiled MLX graphs. Compiling needs fixed
/// shapes, so every batch is padded to `batchSize` x `maxFrames` (masked).
final class VoiceTrainer {
    struct Losses: Sendable {
        let mel: Float, kl: Float, featureMatching: Float, generator: Float, discriminator: Float
    }

    let generator: RVCSynthesizer
    let discriminator: RVCDiscriminator
    private let spectrogram: RVCSpectrogram
    private let optimizerG: AdamW
    private let optimizerD: AdamW
    private let dataset: VoiceDataset
    private let batchSize: Int
    private let maxFrames: Int
    static let baseLearningRate: Float = 1e-4

    init(dataset: VoiceDataset, baseG: URL, baseD: URL, batchSize: Int = 4) throws {
        generator = RVCSynthesizer()
        let extras = try RVCWeights.load(generator, from: baseG, extras: ["mel_basis"])
        discriminator = RVCDiscriminator()
        try RVCWeights.load(discriminator, from: baseD)
        spectrogram = RVCSpectrogram(melBasis: extras["mel_basis"]!)
        optimizerG = AdamW(learningRate: Self.baseLearningRate, betas: (0.8, 0.99), eps: 1e-9, biasCorrection: true)
        optimizerD = AdamW(learningRate: Self.baseLearningRate, betas: (0.8, 0.99), eps: 1e-9, biasCorrection: true)
        self.dataset = dataset
        self.batchSize = batchSize
        maxFrames = dataset.items.map(\.frames).max() ?? RVCConfig.segmentFrames
    }

    /// Replaces the starting weights with a saved checkpoint (resume).
    func resume(g: URL, d: URL) throws {
        try RVCWeights.load(generator, from: g)
        try RVCWeights.load(discriminator, from: d)
    }

    var stepsPerEpoch: Int { (dataset.items.count + batchSize - 1) / batchSize }

    /// Consecutive items of similar length, in shuffled batch order; a short
    /// final batch is topped up with random items to keep the shape fixed.
    private func batches() -> [[Int]] {
        let order = dataset.items.indices.sorted { dataset.items[$0].frames < dataset.items[$1].frames }
        return stride(from: 0, to: order.count, by: batchSize).map { start in
            var batch = Array(order[start ..< min(order.count, start + batchSize)])
            while batch.count < batchSize { batch.append(order.randomElement()!) }
            return batch
        }.shuffled()
    }

    /// Batch tensors, all padded to `maxFrames`:
    /// content, pitch, pitchf, spec, mel, wave, mask, starts.
    private func makeBatch(_ indices: [Int]) -> [MLXArray] {
        let items = indices.map { dataset.items[$0] }
        func pad(_ x: MLXArray, frames: Int, scale: Int = 1) -> MLXArray {
            let extra = (maxFrames - frames) * scale
            guard extra > 0 else { return x }
            return padded(x, widths: [.init((0, extra))] + Array(repeating: .init((0, 0)), count: x.ndim - 1))
        }
        let spec = stacked(items.map { pad(spectrogram.linear($0.wave.reshaped(1, -1, 1))[0], frames: $0.frames) })
        let starts = items.map { Int32.random(in: 0 ... Int32($0.frames - RVCConfig.segmentFrames)) }
        let batch = [
            stacked(items.map { pad($0.content.asType(.float32), frames: $0.frames) }),
            stacked(items.map { pad($0.pitch, frames: $0.frames) }),
            stacked(items.map { pad($0.pitchf, frames: $0.frames) }),
            spec,
            spectrogram.mel(fromLinear: spec),
            stacked(items.map { pad($0.wave, frames: $0.frames, scale: RVCConfig.hop) }).expandedDimensions(axis: -1),
            sequenceMask(items.map(\.frames), maxLength: maxFrames),
            MLXArray(starts),
        ]
        eval(batch)
        return batch
    }

    /// Runs one epoch. `shouldStop` is polled between steps.
    func trainEpoch(_ epoch: Int, shouldStop: () -> Bool, onStep: (Int, Int, Losses) -> Void) -> Bool {
        // The learning rate is baked into the compiled graphs, so they're
        // rebuilt (retraced once) per epoch.
        let lr = Self.baseLearningRate * pow(0.999875, Float(epoch - 1))
        optimizerG.learningRate = lr
        optimizerD.learningRate = lr
        let generator = self.generator, discriminator = self.discriminator
        let optimizerG = self.optimizerG, optimizerD = self.optimizerD, spectrogram = self.spectrogram
        let seg = RVCConfig.segmentFrames, hop = RVCConfig.hop

        let lossG = valueAndGrad(model: generator) { (g: RVCSynthesizer, b: [MLXArray]) -> [MLXArray] in
            let out = g.train(phone: b[0], pitch: b[1], pitchf: b[2], spec: b[3], mask: b[6], starts: b[7])
            let real = sliceFrames(b[5], starts: b[7] * Int32(hop), length: seg * hop)
            let realMel = sliceFrames(b[4], starts: b[7], length: seg)
            let fakeMel = spectrogram.mel(fromLinear: spectrogram.linear(out.audio))
            let fake = discriminator(out.audio), realD = discriminator(real)
            let mel = mean(abs(realMel - fakeMel)) * 45
            let kl = RVCLoss.kl(zP: out.zP, logsQ: out.logsQ, mP: out.mP, logsP: out.logsP, mask: b[6])
            let fm = RVCLoss.featureMatching(real: realD.fmaps, fake: fake.fmaps)
            let adv = RVCLoss.generator(fake: fake.logits)
            return [mel + kl + fm + adv, stopGradient(out.audio), real, mel, kl, fm, adv]
        }
        let stepG = compile(
            inputs: [generator, optimizerG, discriminator, MLXRandom.globalState],
            outputs: [generator, optimizerG, MLXRandom.globalState]
        ) { (b: [MLXArray]) -> [MLXArray] in
            let (values, grads) = lossG(generator, b)
            let ok = Self.guardedUpdate(generator, optimizerG, loss: values[0], gradients: grads)
            return Array(values[1...]) + [ok]
        }

        let lossD = valueAndGrad(model: discriminator) { (d: RVCDiscriminator, pair: [MLXArray]) -> [MLXArray] in
            [RVCLoss.discriminator(real: d(pair[0]).logits, fake: d(pair[1]).logits)]
        }
        let stepD = compile(inputs: [discriminator, optimizerD], outputs: [discriminator, optimizerD]) { (pair: [MLXArray]) -> [MLXArray] in
            let (values, grads) = lossD(discriminator, pair)
            return values + [Self.guardedUpdate(discriminator, optimizerD, loss: values[0], gradients: grads)]
        }

        let all = batches()
        for (step, indices) in all.enumerated() {
            if shouldStop() { return false }
            let g = stepG(makeBatch(indices)) // fake, real, mel, kl, fm, adv, ok
            let d = stepD([g[1], g[0]])       // loss, ok
            eval(generator, discriminator, optimizerG, optimizerD, g, d)
            onStep(step + 1, all.count, Losses(
                mel: g[2].item(Float.self), kl: g[3].item(Float.self), featureMatching: g[4].item(Float.self),
                generator: g[5].item(Float.self), discriminator: d[0].item(Float.self)))
        }
        return true
    }

    /// Applies the optimizer step only if the loss and every gradient are
    /// finite, otherwise leaves the weights untouched -- what RVC gets for free
    /// from fp16 training's GradScaler, which skips overflowing steps. Without
    /// it one bad batch poisons every weight with NaN. Returns the ok flag.
    static func guardedUpdate(_ model: Module, _ optimizer: Optimizer, loss: MLXArray, gradients: ModuleParameters) -> MLXArray {
        let flat = gradients.flattened()
        let ok = flat.reduce(isFinite(loss).all()) { $0 .&& isFinite($1.1).all() }
        let before = model.parameters().flattened()
        optimizer.update(model: model, gradients: ModuleParameters.unflattened(flat.map { ($0.0, which(ok, $0.1, zeros(like: $0.1))) }))
        let after = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        model.update(parameters: ModuleParameters.unflattened(before.map { ($0.0, which(ok, after[$0.0]!, $0.1)) }))
        return ok
    }

    /// True if every weight is finite (checked before anything is saved).
    var weightsAreFinite: Bool {
        (generator.parameters().flattened() + discriminator.parameters().flattened())
            .allSatisfy { isFinite($0.1).all().item(Bool.self) }
    }

    func saveCheckpoint(g: URL, d: URL) throws {
        try RVCWeights.save(generator, to: g)
        try RVCWeights.save(discriminator, to: d)
    }

    /// Inference-only generator: no posterior encoder, float16 (~55 MB).
    func saveVoiceModel(to url: URL) throws {
        let keep = generator.parameters().flattened().filter { !$0.0.hasPrefix("enc_q.") }
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: keep.map { ($0.0, $0.1.asType(.float16)) }), url: url)
    }
}
