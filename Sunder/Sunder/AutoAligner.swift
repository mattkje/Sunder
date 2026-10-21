import Foundation

/// Automatic "quantize both tracks, then match grid to grid" alignment for
/// the Mixer's "Auto Align" button (MixerView) -- detects each stem's own
/// beat grid (BeatDetector), matches the two grids up, and time-stretches
/// the stem piecewise between matched beats via the existing, unmodified
/// TimeStretcher. Each segment gets its own locally-derived rate, so tempo
/// drift across the song is absorbed beat-by-beat instead of needing one
/// global ratio like the manual two-anchor tool.
///
/// This is deliberately not full audio alignment (chroma-feature DTW): a
/// cheap, bounded integer beat-offset search finds the best correspondence
/// between the two grids, and a poor match fails plainly rather than
/// guessing -- the manual tool remains the fallback for anything this can't
/// confidently handle (see AlignError).
nonisolated enum AutoAligner {
    enum AlignError: LocalizedError {
        case tooFewBeats
        case noConfidentMatch

        var errorDescription: String? {
            switch self {
            case .tooFewBeats:
                return "Couldn't find enough clear beats in one of these stems to auto-align -- try the manual Align tool instead."
            case .noConfidentMatch:
                return "Couldn't confidently match these two stems' beats -- try the manual Align tool instead."
            }
        }
    }

    private static let minMatchedBeats = 8
    private static let maxOffsetSearch = 8
    /// Average inter-beat-interval mismatch (seconds) above which a
    /// candidate offset is rejected as not a real match.
    private static let maxAverageIntervalError = 0.08

    static func align(stem: [[Float]], reference: [[Float]], toFrameCount frameCount: Int) throws -> [[Float]] {
        let stemBeats = BeatDetector.detectBeats(stem).times
        let referenceBeats = BeatDetector.detectBeats(reference).times
        guard stemBeats.count >= minMatchedBeats, referenceBeats.count >= minMatchedBeats else {
            throw AlignError.tooFewBeats
        }
        guard let pairs = matchGrids(stem: stemBeats, reference: referenceBeats) else {
            throw AlignError.noConfidentMatch
        }
        let stretched = try stitch(stem: stem, pairs: pairs)
        return AudioIO.fit(channels: stretched, toFrameCount: frameCount)
    }

    /// Scores each candidate integer beat-offset by how well consecutive
    /// inter-beat intervals match between the two grids at that offset, and
    /// keeps the best if it clears `maxAverageIntervalError`. Cheap (a
    /// handful of offsets × beat count) compared to full sequence alignment,
    /// and sufficient for "same song, maybe a bar of intro difference".
    private static func matchGrids(stem: [Double], reference: [Double]) -> [(stem: Double, reference: Double)]? {
        var bestOffset = 0
        var bestError = Double.infinity
        var bestCount = 0

        for offset in -maxOffsetSearch...maxOffsetSearch {
            var totalError = 0.0
            var count = 0
            for i in 0..<max(0, reference.count - 1) {
                let si = i + offset
                guard si >= 0, si + 1 < stem.count else { continue }
                let referenceInterval = reference[i + 1] - reference[i]
                let stemInterval = stem[si + 1] - stem[si]
                totalError += abs(referenceInterval - stemInterval)
                count += 1
            }
            guard count >= minMatchedBeats else { continue }
            let averageError = totalError / Double(count)
            if averageError < bestError {
                bestError = averageError
                bestOffset = offset
                bestCount = count
            }
        }

        guard bestCount >= minMatchedBeats, bestError <= maxAverageIntervalError else { return nil }

        var pairs: [(stem: Double, reference: Double)] = []
        for i in 0..<reference.count {
            let si = i + bestOffset
            guard si >= 0, si < stem.count else { continue }
            pairs.append((stem: stem[si], reference: reference[i]))
        }
        return pairs
    }

    /// Stretches the stem segment-by-segment between consecutive matched
    /// beats, each at its own locally-derived rate. Audio before the first
    /// matched beat and after the last is carried through unstretched --
    /// there's no matched interval to derive a rate from out there.
    private static func stitch(stem: [[Float]], pairs: [(stem: Double, reference: Double)]) throws -> [[Float]] {
        guard pairs.count >= 2, let frameCount = stem.first?.count, frameCount > 0 else { return stem }
        var out: [[Float]] = Array(repeating: [], count: stem.count)

        func frame(_ seconds: Double) -> Int {
            max(0, min(frameCount, Int((seconds * ModelConfig.sampleRate).rounded())))
        }

        let leadEnd = frame(pairs[0].stem)
        for c in 0..<stem.count { out[c].append(contentsOf: stem[c][0..<leadEnd]) }

        for i in 0..<(pairs.count - 1) {
            let sA = frame(pairs[i].stem), sB = frame(pairs[i + 1].stem)
            guard sB > sA else { continue }
            let referenceDuration = pairs[i + 1].reference - pairs[i].reference
            guard referenceDuration > 0.01 else { continue }

            let rate = Float((pairs[i + 1].stem - pairs[i].stem) / referenceDuration)
            let segment = stem.map { Array($0[sA..<sB]) }
            let stretched = TimeStretcher.validRateRange.contains(rate)
                ? try TimeStretcher.stretch(segment, rate: rate)
                : segment
            for c in 0..<out.count { out[c].append(contentsOf: stretched[c]) }
        }

        let tailStart = frame(pairs[pairs.count - 1].stem)
        if tailStart < frameCount {
            for c in 0..<stem.count { out[c].append(contentsOf: stem[c][tailStart...]) }
        }

        return out
    }
}
