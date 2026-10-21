import AVFoundation
import Foundation
import Observation

/// Live 3D placement for SpatializeView: one mono AVAudioPlayerNode per
/// stem, all routed through a single AVAudioEnvironmentNode (Apple's
/// built-in spatial audio renderer) instead of straight into
/// engine.mainMixerNode the way MixerPlayer's stereo nodes are. Point
/// sources for AVAudioEnvironmentNode must be mono -- a stereo signal has
/// no single position to place -- so each stem is downmixed once at init
/// via AudioIO.mixToMono; this is a separate signal path from MixerPlayer's,
/// not a mode on top of it.
///
/// Positions are normalized (x, z) in roughly -1...1 (left/right,
/// front/back; no height), scaled to `worldRadius` when applied to the
/// node's actual AVAudio3DPoint. The listener sits at the origin facing
/// -Z (forward), matching AVAudioEnvironmentNode's own default orientation.
///
/// Default rendering is `.equalPowerPanning`, not `.HRTF`: generic
/// (non-personalized) HRTF is notoriously weak at front/back and elevation
/// discrimination -- azimuth (left/right) is the one localization cue every
/// listener reliably picks up on without headphones or a personal HRTF
/// profile, and equal-power panning also doesn't lose the ~6-10dB a binaural
/// convolution typically does. `immersiveEnabled` lets someone try true
/// binaural HRTF on headphones if they want to compare.
@Observable
final class SpatialAudioPlayer {
    private let engine = AVAudioEngine()
    private let environment = AVAudioEnvironmentNode()
    private let nodes: [AVAudioPlayerNode]
    private let monoFormat: AVAudioFormat
    private let monoStems: [[Float]]
    let frameCount: Int
    let duration: Double

    private(set) var isPlaying = false
    private(set) var currentTime: Double = 0
    private(set) var positions: [SIMD2<Float>]

    var muted: [Bool] {
        didSet { applyVolumes() }
    }
    var soloed: Set<Int> = [] {
        didSet { applyVolumes() }
    }
    var immersiveEnabled = false {
        didSet { applyRenderingAlgorithm() }
    }

    /// How far a normalized ±1 coordinate sits from the listener, in the
    /// arbitrary distance units AVAudio3DPoint uses.
    private static let worldRadius: Float = 4

    private var scheduledStartTime: Double = 0
    private var displayTimer: Timer?

    init(stems: [[[Float]]], stemNames: [String]) throws {
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: ModelConfig.sampleRate, channels: 1, interleaved: false
        ), let stereoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: ModelConfig.sampleRate, channels: 2, interleaved: false
        ) else {
            throw AudioIOError.formatCreationFailed
        }
        self.monoFormat = monoFormat
        monoStems = stems.map { AudioIO.mixToMono($0) }
        frameCount = monoStems.first?.count ?? 0
        duration = Double(frameCount) / ModelConfig.sampleRate
        muted = Array(repeating: false, count: stems.count)
        positions = SpatialAudioPlayer.rolePresetPositions(stemNames: stemNames)

        engine.attach(environment)
        engine.connect(environment, to: engine.mainMixerNode, format: stereoFormat)
        Self.configureDistanceAndGain(environment)

        var built: [AVAudioPlayerNode] = []
        for _ in monoStems {
            let node = AVAudioPlayerNode()
            engine.attach(node)
            engine.connect(node, to: environment, format: monoFormat)
            built.append(node)
        }
        nodes = built
        try engine.start()
        for i in nodes.indices { applyPosition(i) }
        applyRenderingAlgorithm()
        applyVolumes()
        scheduleAll(fromFrame: 0)
    }

    /// Every position this app ever sets is within `worldRadius` (positions
    /// are clamped to the unit circle before scaling), so setting
    /// `referenceDistance` to that same radius means nothing ever falls
    /// outside the "full volume" zone -- distance-based attenuation, which
    /// was quietly cutting several dB off anything not dead-center, is
    /// effectively off. `outputVolume` adds a little makeup gain on top: a
    /// handful of already-mono-downmixed stems summed back together reads
    /// quieter than the original stereo mix even before any spatial
    /// processing, so this compensates rather than leaving it noticeably
    /// duller than a plain "Save Mix" from the Mixer tab.
    private static func configureDistanceAndGain(_ environment: AVAudioEnvironmentNode) {
        environment.distanceAttenuationParameters.referenceDistance = worldRadius
        environment.distanceAttenuationParameters.maximumDistance = worldRadius * 2
        environment.distanceAttenuationParameters.rolloffFactor = 0.3
        environment.outputVolume = 1.4
    }

    private func applyRenderingAlgorithm() {
        let algorithm: AVAudio3DMixingRenderingAlgorithm = immersiveEnabled ? .HRTF : .equalPowerPanning
        nodes.forEach { $0.renderingAlgorithm = algorithm }
    }

    func isAudible(_ index: Int) -> Bool {
        soloed.isEmpty ? !muted[index] : soloed.contains(index)
    }

    func setPosition(_ position: SIMD2<Float>, forStem index: Int) {
        guard positions.indices.contains(index) else { return }
        positions[index] = position
        applyPosition(index)
    }

    /// Snaps every stem back to its role-based preset -- see
    /// `rolePresetPositions`. Dragging remains free afterward; this is just
    /// a starting point, not a locked mode.
    func resetToAutoLayout(stemNames: [String]) {
        positions = SpatialAudioPlayer.rolePresetPositions(stemNames: stemNames)
        for i in nodes.indices { applyPosition(i) }
    }

    private func applyPosition(_ index: Int) {
        let p = positions[index]
        nodes[index].position = AVAudio3DPoint(
            x: p.x * Self.worldRadius, y: 0, z: p.y * Self.worldRadius
        )
    }

    /// Spreads every stem around the *full* circle at a shared radius,
    /// prioritizing distinct azimuth (left/right angle) over front/back
    /// placement -- azimuth is the cue every listener localizes reliably;
    /// front/back relies on weak, HRTF-quality-dependent spectral cues (see
    /// this class's own doc comment). An earlier version placed most stems
    /// on the exact front/back centerline (x=0), differing only in depth --
    /// with generic HRTF's poor front/back discrimination those all
    /// collapsed toward the same perceived spot. Vocals (or, for a 2-stem
    /// session, the instrumental) gets the front slot (angle 0) when
    /// present; everything else is spaced evenly around the remaining
    /// angles. Bass is the one deliberate exception -- low frequencies
    /// carry little directional information and real mixes keep bass
    /// centered -- so it sits close to the listener instead of out on the
    /// ring (a small offset, not exactly (0,0), so its dot doesn't sit
    /// directly on top of the listener glyph in the UI).
    private static func rolePresetPositions(stemNames: [String]) -> [SIMD2<Float>] {
        var positions = [SIMD2<Float>](repeating: .zero, count: stemNames.count)

        let bassIndices = stemNames.indices.filter { stemNames[$0].lowercased() == "bass" }
        for i in bassIndices { positions[i] = SIMD2(0, 0.12) }

        var spreadIndices = stemNames.indices.filter { !bassIndices.contains($0) }
        if let frontIndex = spreadIndices.first(where: {
            let name = stemNames[$0].lowercased()
            return name == "vocals" || name == "instrumental"
        }) {
            spreadIndices.removeAll { $0 == frontIndex }
            spreadIndices.insert(frontIndex, at: 0)
        }

        let radius: Float = 0.85
        let count = spreadIndices.count
        for (slot, i) in spreadIndices.enumerated() {
            let angle = count > 0 ? Float(slot) / Float(count) * 2 * .pi : 0
            positions[i] = SIMD2(sin(angle), -cos(angle)) * radius
        }
        return positions
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
            let sliced = [Array(monoStems[i][startFrame...])]
            guard let buffer = try? AudioIO.makeBuffer(channels: sliced, format: monoFormat) else { continue }
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

    // MARK: - Offline bounce

    /// Renders the currently-audible, currently-positioned stems through
    /// the same mono-source/environment-node graph offline, using the exact
    /// manual-rendering technique TimeStretcher.swift already established
    /// (a fresh engine, `enableManualRenderingMode(.offline, ...)`, loop
    /// `renderOffline` until the scheduled buffers drain) -- here applied to
    /// a spatial-render graph instead of a pitch-shift one.
    func spatializedMix() throws -> [[Float]] {
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: ModelConfig.sampleRate, channels: 1, interleaved: false
        ), let stereoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: ModelConfig.sampleRate, channels: 2, interleaved: false
        ) else {
            throw AudioIOError.formatCreationFailed
        }

        let renderEngine = AVAudioEngine()
        let renderEnvironment = AVAudioEnvironmentNode()
        renderEngine.attach(renderEnvironment)
        renderEngine.connect(renderEnvironment, to: renderEngine.mainMixerNode, format: stereoFormat)
        Self.configureDistanceAndGain(renderEnvironment)

        let algorithm: AVAudio3DMixingRenderingAlgorithm = immersiveEnabled ? .HRTF : .equalPowerPanning
        var renderNodes: [AVAudioPlayerNode] = []
        for i in monoStems.indices {
            let node = AVAudioPlayerNode()
            renderEngine.attach(node)
            renderEngine.connect(node, to: renderEnvironment, format: monoFormat)
            node.renderingAlgorithm = algorithm
            let p = positions[i]
            node.position = AVAudio3DPoint(x: p.x * Self.worldRadius, y: 0, z: p.y * Self.worldRadius)
            renderNodes.append(node)
        }

        let maxFrames: AVAudioFrameCount = 4096
        try renderEngine.enableManualRenderingMode(.offline, format: stereoFormat, maximumFrameCount: maxFrames)
        try renderEngine.start()

        for (i, node) in renderNodes.enumerated() where isAudible(i) {
            let buffer = try AudioIO.makeBuffer(channels: [monoStems[i]], format: monoFormat)
            node.scheduleBuffer(buffer, at: nil, options: [])
            node.play()
        }

        guard let renderBuffer = AVAudioPCMBuffer(pcmFormat: renderEngine.manualRenderingFormat, frameCapacity: renderEngine.manualRenderingMaximumFrameCount) else {
            throw AudioIOError.bufferCreationFailed
        }

        var out: [[Float]] = [[], []]
        renderLoop: while true {
            let status = try renderEngine.renderOffline(maxFrames, to: renderBuffer)
            switch status {
            case .success:
                guard let data = renderBuffer.floatChannelData else { break renderLoop }
                let length = Int(renderBuffer.frameLength)
                if length == 0 { break renderLoop }
                for c in 0..<2 { out[c].append(contentsOf: UnsafeBufferPointer(start: data[c], count: length)) }
            case .insufficientDataFromInputNode:
                break renderLoop
            case .cannotDoInCurrentContext:
                continue renderLoop
            case .error:
                throw AudioIOError.bufferCreationFailed
            @unknown default:
                break renderLoop
            }
        }

        renderEngine.stop()
        return AudioIO.fit(channels: out, toFrameCount: frameCount)
    }
}
