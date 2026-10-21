import Foundation

/// Best-effort musical key detection for the Mixer's pitch/tempo panel --
/// a chromagram (STFT energy folded into 12 pitch classes) correlated
/// against the Krumhansl-Schmuckler major/minor key profiles, no model.
/// Reliable on full mixes and harmonic stems; percussion-only audio has no
/// meaningful key and gives an arbitrary answer.
nonisolated enum KeyDetector {
    struct Key: Equatable {
        let tonic: Int // pitch class, 0 = C
        let isMinor: Bool

        static let names = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]

        var name: String { "\(Self.names[tonic]) \(isMinor ? "minor" : "major")" }

        /// This key moved by `semitones` (what a pitch shift turns it into).
        func transposed(by semitones: Int) -> Key {
            Key(tonic: ((tonic + semitones) % 12 + 12) % 12, isMinor: isMinor)
        }
    }

    private static let nFFT = 8192 // ~5.4 Hz bins at 44.1 kHz: enough to separate semitones above ~80 Hz
    private static let hop = 4096
    private static let minHz = 80.0
    private static let maxHz = 5000.0

    private static let majorProfile: [Double] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    private static let minorProfile: [Double] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    static func detect(_ channels: [[Float]]) -> Key? {
        let mono = AudioIO.mixToMono(channels)
        guard mono.count > nFFT * 4 else { return nil }

        let stft = STFTProcessor(nFFT: nFFT, hopLength: hop, winLength: nFFT, chunkSize: mono.count)
        let (real, imag) = stft.forward(mono)
        let numFreq = nFFT / 2 + 1
        let T = real.count / numFreq
        let binHz = ModelConfig.sampleRate / Double(nFFT)

        var chroma = [Double](repeating: 0, count: 12)
        for f in Int(minHz / binHz)...min(numFreq - 1, Int(maxHz / binHz)) {
            let hz = Double(f) * binHz
            let pitchClass = (Int((12 * log2(hz / 440)).rounded()) + 9 + 1200) % 12 // A = 9
            var energy = 0.0
            for t in 0..<T {
                let re = Double(real[f * T + t]), im = Double(imag[f * T + t])
                energy += (re * re + im * im).squareRoot()
            }
            chroma[pitchClass] += energy
        }
        guard chroma.contains(where: { $0 > 0 }) else { return nil }

        var best: (key: Key, score: Double)?
        for tonic in 0..<12 {
            for isMinor in [false, true] {
                let profile = isMinor ? minorProfile : majorProfile
                let rotated = (0..<12).map { profile[(($0 - tonic) % 12 + 12) % 12] }
                let score = correlation(chroma, rotated)
                if best == nil || score > best!.score { best = (Key(tonic: tonic, isMinor: isMinor), score) }
            }
        }
        return best?.key
    }

    private static func correlation(_ a: [Double], _ b: [Double]) -> Double {
        let ma = a.reduce(0, +) / Double(a.count), mb = b.reduce(0, +) / Double(b.count)
        var num = 0.0, da = 0.0, db = 0.0
        for i in a.indices {
            num += (a[i] - ma) * (b[i] - mb)
            da += (a[i] - ma) * (a[i] - ma)
            db += (b[i] - mb) * (b[i] - mb)
        }
        return da > 0 && db > 0 ? num / (da * db).squareRoot() : 0
    }
}
