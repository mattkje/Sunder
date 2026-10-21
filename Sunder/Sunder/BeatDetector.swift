import Foundation

/// Best-effort beat-grid detection for the Mixer's "Auto Align" tool (see
/// AutoAligner/MixerView) -- classic MIR building blocks (spectral-flux
/// onsets, windowed autocorrelation tempo, peak-snapped grid), no model.
/// Quality depends entirely on how detectable a track's attacks are: a full
/// instrumental or drum stem gives a clean grid, a legato solo vocal may
/// not -- callers (AutoAligner) treat a too-sparse result as a failure
/// rather than pretending it's reliable.
nonisolated enum BeatDetector {
    struct Beats {
        let times: [Double] // seconds, ascending
        let bpm: Double // representative tempo, for display/sanity only
    }

    private static let nFFT = 1024
    private static let hop = 512
    private static let tempoWindowSeconds = 8.0
    private static let tempoHopSeconds = 4.0
    private static let minBPM = 60.0
    private static let maxBPM = 200.0
    private static let peakMinSpacingSeconds = 0.1
    private static let snapToleranceSeconds = 0.06

    static func detectBeats(_ channels: [[Float]]) -> Beats {
        let mono = AudioIO.mixToMono(channels)
        guard mono.count > nFFT else { return Beats(times: [], bpm: 120) }

        let stft = STFTProcessor(nFFT: nFFT, hopLength: hop, winLength: nFFT, chunkSize: mono.count)
        let (real, imag) = stft.forward(mono)
        let numFreq = nFFT / 2 + 1
        let T = real.count / numFreq
        guard T > 2 else { return Beats(times: [], bpm: 120) }

        let envelope = spectralFlux(real: real, imag: imag, numFreq: numFreq, T: T)
        let frameRate = ModelConfig.sampleRate / Double(hop)
        let peaks = pickPeaks(envelope, frameRate: frameRate)
        let tempoCurve = estimateTempoCurve(envelope, frameRate: frameRate)
        let times = buildBeatGrid(peaks: peaks, frameRate: frameRate, duration: Double(T) / frameRate, tempoCurve: tempoCurve)
        let medianBPM = tempoCurve.map { $0.bpm }.sorted()[tempoCurve.count / 2]
        return Beats(times: times, bpm: medianBPM)
    }

    /// Per frame, the sum of positive magnitude increases over the previous
    /// frame across all bins -- the standard "spectral flux" onset strength
    /// function; works for percussive attacks and, less sharply, for vocal
    /// consonants/phrase starts.
    private static func spectralFlux(real: [Float], imag: [Float], numFreq: Int, T: Int) -> [Float] {
        var envelope = [Float](repeating: 0, count: T)
        var prevMag = [Float](repeating: 0, count: numFreq)
        for t in 0..<T {
            var flux: Float = 0
            for f in 0..<numFreq {
                let re = real[f * T + t]
                let im = imag[f * T + t]
                let mag = (re * re + im * im).squareRoot()
                let diff = mag - prevMag[f]
                if diff > 0 { flux += diff }
                prevMag[f] = mag
            }
            envelope[t] = flux
        }
        return envelope
    }

    private static func pickPeaks(_ envelope: [Float], frameRate: Double) -> [Int] {
        guard !envelope.isEmpty else { return [] }
        let mean = envelope.reduce(0, +) / Float(envelope.count)
        let variance = envelope.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(envelope.count)
        let threshold = mean + variance.squareRoot() * 0.5
        let minSpacing = max(1, Int(peakMinSpacingSeconds * frameRate))

        var peaks: [Int] = []
        var i = 1
        while i < envelope.count - 1 {
            if envelope[i] > threshold, envelope[i] >= envelope[i - 1], envelope[i] >= envelope[i + 1] {
                peaks.append(i)
                i += minSpacing
            } else {
                i += 1
            }
        }
        return peaks
    }

    /// One BPM estimate per overlapping window (autocorrelation of the
    /// onset envelope, octave ambiguity resolved by a log-Gaussian bias
    /// toward ~120 BPM), so the resulting curve can drift across the song
    /// instead of committing to one global tempo.
    private static func estimateTempoCurve(_ envelope: [Float], frameRate: Double) -> [(time: Double, bpm: Double)] {
        let windowFrames = Int(tempoWindowSeconds * frameRate)
        let hopFrames = max(1, Int(tempoHopSeconds * frameRate))
        guard envelope.count > windowFrames else {
            return [(Double(envelope.count) / 2 / frameRate, estimateBPM(envelope, frameRate: frameRate))]
        }

        var curve: [(Double, Double)] = []
        var start = 0
        while start + windowFrames <= envelope.count {
            let segment = Array(envelope[start..<(start + windowFrames)])
            let centerTime = (Double(start) + Double(windowFrames) / 2) / frameRate
            curve.append((centerTime, estimateBPM(segment, frameRate: frameRate)))
            start += hopFrames
        }
        return curve.isEmpty ? [(0, 120)] : curve
    }

    private static func estimateBPM(_ envelope: [Float], frameRate: Double) -> Double {
        let minLag = Int(frameRate * 60.0 / maxBPM)
        let maxLag = Int(frameRate * 60.0 / minBPM)
        guard minLag >= 1, maxLag < envelope.count else { return 120 }

        let mean = envelope.reduce(0, +) / Float(envelope.count)
        let centered = envelope.map { $0 - mean }

        var bestLag = minLag
        var bestScore = -Float.greatestFiniteMagnitude
        for lag in minLag...maxLag {
            var sum: Float = 0
            for i in 0..<(centered.count - lag) { sum += centered[i] * centered[i + lag] }
            let bpm = 60.0 * frameRate / Double(lag)
            let octaveBias = Float(exp(-0.5 * pow((log2(bpm) - log2(120.0)) / 0.6, 2)))
            let score = sum * octaveBias
            if score > bestScore { bestScore = score; bestLag = lag }
        }
        return 60.0 * frameRate / Double(bestLag)
    }

    /// Walks forward from the first strong onset, spacing each next beat by
    /// the tempo curve's *local* value (not one global BPM) and snapping to
    /// a nearby detected peak when there is one -- this is the
    /// "quantisation" step: a regular, tempo-informed grid that's still
    /// grounded in the track's actual attacks rather than a pure metronome.
    private static func buildBeatGrid(peaks: [Int], frameRate: Double, duration: Double, tempoCurve: [(time: Double, bpm: Double)]) -> [Double] {
        guard let firstPeak = peaks.first else { return [] }
        var beats: [Double] = []
        var t = Double(firstPeak) / frameRate
        while t < duration {
            beats.append(t)
            let bpm = localBPM(at: t, curve: tempoCurve)
            var next = t + 60.0 / bpm
            if let snapped = nearestPeak(to: next, peaks: peaks, frameRate: frameRate, tolerance: snapToleranceSeconds) {
                next = snapped
            }
            guard next > t + 0.05 else { break } // guards against a degenerate tempo estimate looping forever
            t = next
        }
        return beats
    }

    private static func localBPM(at time: Double, curve: [(time: Double, bpm: Double)]) -> Double {
        guard let first = curve.first, let last = curve.last else { return 120 }
        if time <= first.time { return first.bpm }
        if time >= last.time { return last.bpm }
        for i in 0..<(curve.count - 1) {
            let a = curve[i], b = curve[i + 1]
            if time >= a.time && time <= b.time {
                let f = (time - a.time) / (b.time - a.time)
                return a.bpm + (b.bpm - a.bpm) * f
            }
        }
        return last.bpm
    }

    private static func nearestPeak(to time: Double, peaks: [Int], frameRate: Double, tolerance: Double) -> Double? {
        let targetFrame = Int((time * frameRate).rounded())
        let toleranceFrames = Int(tolerance * frameRate)
        var best: Int?
        var bestDistance = Int.max
        for peak in peaks {
            let distance = abs(peak - targetFrame)
            if distance <= toleranceFrames, distance < bestDistance {
                bestDistance = distance
                best = peak
            }
        }
        return best.map { Double($0) / frameRate }
    }
}
