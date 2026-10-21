import Accelerate
import Foundation
import MLX

/// One training example, aligned at 100 fps (`frames` frames, 400 samples each).
struct TrainingItem {
    let wave: MLXArray      // [frames * 400] at 40 kHz
    let content: MLXArray   // [frames, 768] (ContentVec 50 fps, repeated x2)
    let pitch: MLXArray     // [frames] int32 coarse
    let pitchf: MLXArray    // [frames] Hz
    var frames: Int { pitch.dim(0) }
}

/// RVC's preprocessing (`train/preprocess.py`, `extract_f0`, `extract_hubert_feature`):
/// high-pass, silence slicing, 3.7 s segments with 0.3 s overlap, peak
/// normalization, then ContentVec features and RMVPE pitch per segment.
struct VoiceDataset {
    let items: [TrainingItem]
    /// Every ContentVec frame (50 fps) of the dataset, for the retrieval index.
    let contentFrames: MLXArray // [N, 768]

    static let segmentSeconds = 3.7
    static let overlapSeconds = 0.3

    /// Seconds of audio that survive silence slicing -- what the UI reports as
    /// "usable" training material.
    static func usableDuration(of audio44k: [Float]) throws -> Double {
        let audio = try VoiceAudio.resample(audio44k, from: VoiceAudio.appSampleRate, to: Double(RVCConfig.sampleRate))
        let sr = Double(RVCConfig.sampleRate)
        return Slicer(sampleRate: sr).slice(audio).reduce(0) { $0 + Double($1.count) } / sr
    }

    static func build(sources: [[Float]], features: VoiceFeatures, progress: (Double) -> Void) throws -> VoiceDataset {
        let sr = Double(RVCConfig.sampleRate)
        var segments: [[Float]] = []
        for source in sources {
            let audio = VoiceAudio.highPass(
                try VoiceAudio.resample(source, from: VoiceAudio.appSampleRate, to: sr), cutoff: 48, sampleRate: sr)
            for chunk in Slicer(sampleRate: sr).slice(audio) {
                segments += split(chunk, sampleRate: sr).compactMap(normalize)
            }
        }

        var items: [TrainingItem] = []
        var content: [MLXArray] = []
        for (i, segment) in segments.enumerated() {
            let wave16 = try VoiceAudio.resample(segment, from: sr, to: VoiceFeatures.sampleRate)
            let feats = try features.content(wave16)
            let f0 = try features.pitch(wave16)
            guard !feats.isEmpty else { continue }
            let feats50 = MLXArray(feats.flatMap { $0 }, [feats.count, 768])
            let frames = min(feats.count * 2, f0.count, segment.count / RVCConfig.hop, 900)
            guard frames >= RVCConfig.segmentFrames else { continue }
            let pitchf = Array(f0.prefix(frames))
            items.append(TrainingItem(
                wave: MLXArray(Array(segment.prefix(frames * RVCConfig.hop))),
                content: repeated(feats50, count: 2, axis: 0)[..<frames].asType(.float16),
                pitch: MLXArray(pitchf.map(PitchQuantizer.coarse)),
                pitchf: MLXArray(pitchf)))
            content.append(feats50.asType(.float16))
            eval(items[items.count - 1].content, content[content.count - 1])
            progress(Double(i + 1) / Double(segments.count))
        }
        return VoiceDataset(items: items, contentFrames: content.isEmpty ? MLXArray.zeros([0, 768]) : concatenated(content))
    }

    /// 3.7 s windows advancing by 3.4 s; the remainder becomes the last piece.
    private static func split(_ chunk: [Float], sampleRate sr: Double) -> [ArraySlice<Float>] {
        let per = Int(segmentSeconds * sr), step = Int((segmentSeconds - overlapSeconds) * sr)
        let tail = Int((segmentSeconds + overlapSeconds) * sr)
        var pieces: [ArraySlice<Float>] = []
        var start = 0
        while chunk.count - start > tail {
            pieces.append(chunk[start ..< start + per])
            start += step
        }
        pieces.append(chunk[start...])
        return pieces
    }

    /// RVC's `norm_write`: 75% peak-normalized to 0.9 + 25% original; clips
    /// peaking above 2.5 are rejected as broken.
    private static func normalize(_ x: ArraySlice<Float>) -> [Float]? {
        let peak = VoiceAudio.peak(x)
        guard peak > 0, peak <= 2.5 else { return nil }
        let alpha: Float = 0.75
        return x.map { $0 / peak * (0.9 * alpha) + (1 - alpha) * $0 }
    }
}

/// Port of RVC's `slicer2.Slicer` with the preprocess settings: -42 dB
/// threshold, 1.5 s min clip, 0.4 s min silence, 15 ms hop, 0.5 s silence kept.
struct Slicer {
    let threshold: Float, hop: Int, window: Int, minLength: Int, minInterval: Int, maxSilKept: Int

    init(sampleRate sr: Double, thresholdDB: Double = -42, minLengthMs: Double = 1500,
         minIntervalMs: Double = 400, hopMs: Double = 15, maxSilKeptMs: Double = 500) {
        let minIntervalSamples = sr * minIntervalMs / 1000
        threshold = Float(pow(10, thresholdDB / 20))
        hop = Int((sr * hopMs / 1000).rounded())
        window = min(Int(minIntervalSamples.rounded()), 4 * hop)
        minLength = Int((sr * minLengthMs / 1000 / Double(hop)).rounded())
        minInterval = Int((minIntervalSamples / Double(hop)).rounded())
        maxSilKept = Int((sr * maxSilKeptMs / 1000 / Double(hop)).rounded())
    }

    private func rms(_ x: [Float]) -> [Float] {
        let pad = window / 2
        let padded = [Float](repeating: 0, count: pad) + x + [Float](repeating: 0, count: pad)
        let frames = 1 + (padded.count - window) / hop
        return (0 ..< frames).map { i in
            padded.withUnsafeBufferPointer { p in
                var r: Float = 0
                vDSP_rmsqv(p.baseAddress! + i * hop, 1, &r, vDSP_Length(window))
                return r
            }
        }
    }

    private func argmin(_ a: [Float], _ range: Range<Int>) -> Int {
        let r = range.clamped(to: 0 ..< a.count)
        return r.min { a[$0] < a[$1] } ?? r.lowerBound
    }

    func slice(_ samples: [Float]) -> [[Float]] {
        guard samples.count > minLength else { return [samples] }
        let levels = rms(samples)
        var tags: [(Int, Int)] = []
        var silenceStart: Int?
        var clipStart = 0
        for (i, level) in levels.enumerated() {
            if level < threshold {
                if silenceStart == nil { silenceStart = i }
                continue
            }
            guard let s = silenceStart else { continue }
            let leading = s == 0 && i > maxSilKept
            let middle = i - s >= minInterval && i - clipStart >= minLength
            if !leading && !middle {
                silenceStart = nil
                continue
            }
            if i - s <= maxSilKept {
                let pos = argmin(levels, s ..< i + 1)
                tags.append(s == 0 ? (0, pos) : (pos, pos))
                clipStart = pos
            } else if i - s <= maxSilKept * 2 {
                let pos = argmin(levels, i - maxSilKept ..< s + maxSilKept + 1)
                let l = argmin(levels, s ..< s + maxSilKept + 1)
                let r = argmin(levels, i - maxSilKept ..< i + 1)
                if s == 0 {
                    tags.append((0, r)); clipStart = r
                } else {
                    tags.append((min(l, pos), max(r, pos))); clipStart = max(r, pos)
                }
            } else {
                let l = argmin(levels, s ..< s + maxSilKept + 1)
                let r = argmin(levels, i - maxSilKept ..< i + 1)
                tags.append(s == 0 ? (0, r) : (l, r))
                clipStart = r
            }
            silenceStart = nil
        }
        let total = levels.count
        if let s = silenceStart, total - s >= minInterval {
            let end = min(total, s + maxSilKept)
            tags.append((argmin(levels, s ..< end + 1), total + 1))
        }
        guard !tags.isEmpty else { return [samples] }
        func cut(_ a: Int, _ b: Int) -> [Float] {
            let lo = min(samples.count, a * hop), hi = min(samples.count, b * hop)
            return lo < hi ? Array(samples[lo ..< hi]) : []
        }
        var chunks: [[Float]] = []
        if tags[0].0 > 0 { chunks.append(cut(0, tags[0].0)) }
        for i in 0 ..< tags.count - 1 { chunks.append(cut(tags[i].1, tags[i + 1].0)) }
        if tags[tags.count - 1].1 < total { chunks.append(cut(tags[tags.count - 1].1, total)) }
        return chunks.filter { !$0.isEmpty }
    }
}
