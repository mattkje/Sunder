import Foundation

/// Post-processing for a finished mix bus (see MixerView's "Master" toggle):
/// brings the bounce to a fixed loudness target, then a lookahead peak
/// limiter keeps that gain-up from clipping. No EQ, no reference-track
/// matching -- deliberately the small, license-clean half of "mastering"
/// (see the plan's non-goals for the deferred, harder half).
nonisolated enum MasterEngine {
    private static let targetRMS: Float = 0.2 // ~-14 dBFS, a conventional streaming loudness target
    private static let maxGain: Float = 8 // guards against amplifying a near-silent bounce into noise
    private static let ceiling: Float = 0.98 // a hair of headroom below full scale
    private static let lookaheadFrames = 512
    private static let releaseFrames = 4410 // ~100ms at 44.1kHz

    static func process(_ channels: [[Float]]) -> [[Float]] {
        guard let frameCount = channels.first?.count, frameCount > 0 else { return channels }
        let rms = rmsLevel(channels)
        let gain = rms > 0.0001 ? min(targetRMS / rms, maxGain) : 1
        var boosted = channels.map { channel in channel.map { $0 * gain } }
        limit(&boosted)
        return boosted
    }

    private static func rmsLevel(_ channels: [[Float]]) -> Float {
        var sumSquares: Double = 0
        var count = 0
        for channel in channels {
            for sample in channel { sumSquares += Double(sample) * Double(sample) }
            count += channel.count
        }
        guard count > 0 else { return 0 }
        return Float((sumSquares / Double(count)).squareRoot())
    }

    /// Scans in lookahead-sized windows for the loudest sample across all
    /// channels; whenever a window's peak exceeds `ceiling`, ramps gain
    /// reduction in ahead of that peak (over the previous window) and
    /// releases it afterward, so the limiter reacts before the transient
    /// hits rather than clipping and correcting after the fact. Overlapping
    /// windows compose correctly because reductions are combined with
    /// `min`, always keeping the strictest one in force.
    private static func limit(_ channels: inout [[Float]]) {
        let frameCount = channels[0].count
        var gainReduction = [Float](repeating: 1, count: frameCount)

        var frame = 0
        while frame < frameCount {
            let windowEnd = min(frame + lookaheadFrames, frameCount)
            var peak: Float = 0
            for channel in channels {
                for i in frame..<windowEnd {
                    peak = max(peak, abs(channel[i]))
                }
            }
            if peak > ceiling {
                let reduction = ceiling / peak
                let rampStart = max(0, frame - lookaheadFrames)
                for i in rampStart..<windowEnd {
                    gainReduction[i] = min(gainReduction[i], reduction)
                }
                let releaseEnd = min(windowEnd + releaseFrames, frameCount)
                for i in windowEnd..<releaseEnd {
                    let t = Float(i - windowEnd) / Float(releaseFrames)
                    gainReduction[i] = min(gainReduction[i], reduction + (1 - reduction) * t)
                }
            }
            frame += lookaheadFrames
        }

        for c in 0..<channels.count {
            for i in 0..<frameCount {
                channels[c][i] *= gainReduction[i]
            }
        }
    }
}
