import AVFoundation
import Observation

/// Synchronized multi-stem playback for MixerView: one AVAudioPlayerNode per
/// stem, all feeding engine.mainMixerNode, muted/soloed by toggling each
/// node's own `volume` between 0 and 1 -- no per-stem gain, just on/off (see
/// MixerView's doc comment for why: mute/solo only was what was asked for).
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

    /// Absolute track position (seconds) that the nodes' current schedule
    /// starts at -- AVAudioPlayerNode's own playerTime resets to 0 on every
    /// (re)schedule, so the display timer adds this offset to get an
    /// absolute position. Updated by `scheduleAll`, i.e. on init and on
    /// every `seek`.
    private var scheduledStartTime: Double = 0
    private var displayTimer: Timer?

    init(stems: [[[Float]]]) throws {
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
        frameCount = stems.first?.first?.count ?? 0
        duration = Double(frameCount) / ModelConfig.sampleRate
        muted = Array(repeating: false, count: stems.count)

        var built: [AVAudioPlayerNode] = []
        for _ in stems {
            let node = AVAudioPlayerNode()
            engine.attach(node)
            engine.connect(node, to: engine.mainMixerNode, format: format)
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

    /// Sums every currently-audible stem, sample by sample -- used by
    /// MixerView's "Save Mix" bounce. Reads live `isAudible`, so it matches
    /// exactly what's playing at the moment the user hits save.
    func audibleMix() -> [[Float]] {
        let channels = stems.first?.count ?? 0
        var mix = Array(repeating: [Float](repeating: 0, count: frameCount), count: channels)
        for (i, stem) in stems.enumerated() where isAudible(i) {
            for c in 0..<channels {
                for s in 0..<frameCount {
                    mix[c][s] += stem[c][s]
                }
            }
        }
        return mix
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
            node.volume = isAudible(i) ? 1 : 0
        }
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
