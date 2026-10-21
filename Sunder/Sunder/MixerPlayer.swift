import AVFoundation
import Foundation
import Observation

/// Synchronized multi-stem playback for MixerView: one AVAudioPlayerNode per
/// stem, all feeding engine.mainMixerNode. Mute/solo toggle each node's own
/// `volume` between 0 and full; `automixEnabled` layers a per-stem gain (see
/// `automixGains`) on top of that instead of leaving every audible stem at a
/// flat 1.0 -- still no pan or per-stem EQ, just level balancing.
///
/// `isAudible` is the single source of truth for "is this stem currently
/// meant to be heard" -- both this class's live node volumes and
/// MixerView's "Save Mix" bounce read it, so the two never disagree.
@Observable
final class MixerPlayer {
    private let engine = AVAudioEngine()
    private let nodes: [AVAudioPlayerNode]
    private let format: AVAudioFormat
    private let stems: [[[Float]]]
    private let automixGains: [Float]
    let frameCount: Int
    let duration: Double

    private(set) var isPlaying = false
    private(set) var currentTime: Double = 0

    var muted: [Bool] {
        didSet { applyVolumes() }
    }
    var soloed: Set<Int> = [] {
        didSet { applyVolumes() }
    }
    var automixEnabled = false {
        didSet { applyVolumes() }
    }
    /// Live pitch (semitones) and tempo (playback-rate multiplier) for the
    /// whole mix, via one AVAudioUnitTimePitch on the summed stems. Saved
    /// files get the same change through `applyPitchAndTempo`.
    var pitchSemitones: Float = 0 {
        didSet { timePitch.pitch = pitchSemitones * 100 }
    }
    var tempo: Float = 1 {
        didSet { timePitch.rate = tempo }
    }
    private let timePitch = AVAudioUnitTimePitch()

    /// Absolute track position (seconds) that the nodes' current schedule
    /// starts at -- AVAudioPlayerNode's own playerTime resets to 0 on every
    /// (re)schedule, so the display timer adds this offset to get an
    /// absolute position. Updated by `scheduleAll`, i.e. on init and on
    /// every `seek`.
    private var scheduledStartTime: Double = 0
    private var displayTimer: Timer?

    init(stems: [[[Float]]], stemNames: [String]) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: ModelConfig.sampleRate,
            channels: AVAudioChannelCount(ModelConfig.channels),
            interleaved: false
        ) else {
            throw AudioIOError.formatCreationFailed
        }
        self.format = format
        self.stems = stems
        automixGains = MixerPlayer.computeAutomixGains(stems: stems, stemNames: stemNames)
        frameCount = stems.first?.first?.count ?? 0
        duration = Double(frameCount) / ModelConfig.sampleRate
        muted = Array(repeating: false, count: stems.count)

        // stems -> submix -> timePitch -> main mixer. The players' own
        // sample clocks keep counting source frames, so the position display
        // stays in track time whatever the tempo.
        let submix = AVAudioMixerNode()
        engine.attach(submix)
        engine.attach(timePitch)
        engine.connect(submix, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)
        var built: [AVAudioPlayerNode] = []
        for _ in stems {
            let node = AVAudioPlayerNode()
            engine.attach(node)
            engine.connect(node, to: submix, format: format)
            built.append(node)
        }
        nodes = built
        try engine.start()
        applyVolumes()
        scheduleAll(fromFrame: 0)
    }

    func isAudible(_ index: Int) -> Bool {
        soloed.isEmpty ? !muted[index] : soloed.contains(index)
    }

    /// The gain currently applied to a stem, independent of mute/solo:
    /// a flat 1.0 with Auto Mix off (today's behavior), or its precomputed
    /// `automixGains` entry with Auto Mix on.
    private func gain(for index: Int) -> Float {
        automixEnabled ? automixGains[index] : 1
    }

    /// Sums every currently-audible stem, sample by sample, each scaled by
    /// its current gain -- used by MixerView's "Save Mix" bounce. Reads live
    /// `isAudible`/`automixEnabled`, so it matches exactly what's playing at
    /// the moment the user hits save.
    func audibleMix() -> [[Float]] {
        let channels = stems.first?.count ?? 0
        var mix = Array(repeating: [Float](repeating: 0, count: frameCount), count: channels)
        for (i, stem) in stems.enumerated() where isAudible(i) {
            let g = gain(for: i)
            for c in 0..<channels {
                for s in 0..<frameCount {
                    mix[c][s] += stem[c][s] * g
                }
            }
        }
        return mix
    }

    /// The current pitch/tempo applied offline to `channels` (a bounce or a
    /// single stem) -- unchanged audio when both are at their defaults.
    func applyPitchAndTempo(_ channels: [[Float]]) throws -> [[Float]] {
        try TimeStretcher.stretch(channels, rate: tempo, pitchCents: pitchSemitones * 100)
    }

    func togglePlayPause() {
        isPlaying ? pause() : play()
    }

    func play() {
        guard !isPlaying else { return }
        nodes.forEach { $0.play() }
        isPlaying = true
        startDisplayTimer()
    }

    func pause() {
        guard isPlaying else { return }
        nodes.forEach { $0.pause() }
        isPlaying = false
        displayTimer?.invalidate()
    }

    /// AVAudioPlayerNode has no seek primitive -- rescheduling each node's
    /// buffer starting at the target frame is the standard workaround.
    func seek(to time: Double) {
        let wasPlaying = isPlaying
        nodes.forEach { $0.stop() }
        let clamped = max(0, min(time, duration))
        scheduleAll(fromFrame: Int(clamped * ModelConfig.sampleRate))
        currentTime = clamped
        if wasPlaying { nodes.forEach { $0.play() } }
    }

    private func scheduleAll(fromFrame startFrame: Int) {
        scheduledStartTime = Double(startFrame) / ModelConfig.sampleRate
        guard startFrame < frameCount else { return }
        for (i, node) in nodes.enumerated() {
            let sliced = stems[i].map { Array($0[startFrame...]) }
            guard let buffer = try? AudioIO.makeBuffer(channels: sliced, format: format) else { continue }
            node.scheduleBuffer(buffer, at: nil, options: [])
        }
    }

    private func applyVolumes() {
        for (i, node) in nodes.enumerated() {
            node.volume = isAudible(i) ? gain(for: i) : 0
        }
    }

    /// Fixed dB offset per stem role, applied on top of loudness
    /// normalization below -- brings vocals forward and pulls busier/lower
    /// stems back a little, a simple rule-based stand-in for a real
    /// automixing model (see the plan's non-goals for why: no pretrained
    /// checkpoint exists for non-drum stems). Keyed by the exact stemNames
    /// strings AIModel.spec produces, lowercased.
    private static let roleOffsetDB: [String: Float] = [
        "vocals": 3,
        "instrumental": 0,
        "drums": 0,
        "bass": -1,
        "other": -2,
        "guitar": -2,
        "piano": -2,
        "enhanced": 0,
    ]

    /// One static linear gain per stem: first normalizes each stem's RMS
    /// loudness to a common reference level (separation output loudness is
    /// otherwise uneven and would bias the balance below), then applies
    /// `roleOffsetDB`. `maxNormalizeGain` keeps a near-silent stem (e.g. an
    /// instrumental-only session's near-empty "vocals" bucket) from being
    /// amplified into audible noise.
    private static func computeAutomixGains(stems: [[[Float]]], stemNames: [String]) -> [Float] {
        let referenceRMS: Float = 0.1 // ~-20 dBFS
        let maxNormalizeGain: Float = 4
        return stems.enumerated().map { i, stem in
            let rms = rmsLevel(stem)
            let normalizeGain = rms > 0.0001 ? min(referenceRMS / rms, maxNormalizeGain) : 1
            let name = i < stemNames.count ? stemNames[i].lowercased() : ""
            let offsetDB = roleOffsetDB[name] ?? 0
            return normalizeGain * pow(10, offsetDB / 20)
        }
    }

    private static func rmsLevel(_ stem: [[Float]]) -> Float {
        var sumSquares: Double = 0
        var count = 0
        for channel in stem {
            for sample in channel { sumSquares += Double(sample) * Double(sample) }
            count += channel.count
        }
        guard count > 0 else { return 0 }
        return Float((sumSquares / Double(count)).squareRoot())
    }

    private func startDisplayTimer() {
        displayTimer?.invalidate()
        displayTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self, let node = self.nodes.first, let nodeTime = node.lastRenderTime,
                  let playerTime = node.playerTime(forNodeTime: nodeTime) else { return }
            let elapsed = Double(playerTime.sampleTime) / playerTime.sampleRate
            let t = self.scheduledStartTime + elapsed
            if t >= self.duration {
                self.pause()
                self.seek(to: 0)
            } else {
                self.currentTime = t
            }
        }
    }

    func stop() {
        displayTimer?.invalidate()
        nodes.forEach { $0.stop() }
        engine.stop()
    }

    deinit {
        displayTimer?.invalidate()
        engine.stop()
    }
}
