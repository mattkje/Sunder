import AVFoundation

/// Offline time-stretching and pitch-shifting for the Mixer -- the manual
/// "Align" tool (duration only) and the pitch/tempo controls applied to
/// saved files (see MixerView). `rate` changes duration without touching
/// pitch and `pitchCents` changes pitch without touching duration, using AVFoundation's own AVAudioUnitTimePitch (phase-vocoder
/// based) rendered through AVAudioEngine's manual/offline rendering mode
/// rather than real-time playback. No third-party dependency, no model.
nonisolated enum TimeStretcher {
    /// Below/above this, AVAudioUnitTimePitch's quality degrades sharply and
    /// a request this far off is almost certainly two mismatched points
    /// rather than a real tempo difference.
    static let validRateRange: ClosedRange<Float> = 0.25...4.0

    enum TimeStretchError: LocalizedError {
        case invalidRate
        case engineSetupFailed

        var errorDescription: String? {
            switch self {
            case .invalidRate: return "That rate is out of range -- check the marked points."
            case .engineSetupFailed: return "Could not set up offline audio rendering."
            }
        }
    }

    static func stretch(_ channels: [[Float]], rate: Float, pitchCents: Float = 0) throws -> [[Float]] {
        guard validRateRange.contains(rate) else { throw TimeStretchError.invalidRate }
        guard abs(rate - 1) > 0.001 || abs(pitchCents) > 0.5, let frameCount = channels.first?.count, frameCount > 0 else {
            return channels
        }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: ModelConfig.sampleRate,
            channels: AVAudioChannelCount(ModelConfig.channels),
            interleaved: false
        ) else {
            throw AudioIOError.formatCreationFailed
        }

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()
        timePitch.rate = rate
        timePitch.pitch = pitchCents
        engine.attach(player)
        engine.attach(timePitch)
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

        let maxFrames: AVAudioFrameCount = 4096
        do {
            try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: maxFrames)
        } catch {
            throw TimeStretchError.engineSetupFailed
        }
        try engine.start()

        let inputBuffer = try AudioIO.makeBuffer(channels: channels, format: format)
        player.scheduleBuffer(inputBuffer, at: nil)
        player.play()

        guard let renderBuffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: engine.manualRenderingMaximumFrameCount) else {
            throw TimeStretchError.engineSetupFailed
        }

        // A player node never reports running out of input -- once its buffer
        // is played it just renders silence -- so render exactly the expected
        // output length, plus the effect's processing latency (silence at the
        // start of the output, trimmed off below).
        let latency = Int((timePitch.auAudioUnit.latency * ModelConfig.sampleRate).rounded())
        let targetFrames = Int((Double(frameCount) / Double(rate)).rounded())
        let totalFrames = targetFrames + latency
        var outChannels: [[Float]] = Array(repeating: [], count: channels.count)
        var rendered = 0
        while rendered < totalFrames {
            let request = AVAudioFrameCount(min(Int(maxFrames), totalFrames - rendered))
            switch try engine.renderOffline(request, to: renderBuffer) {
            case .success:
                guard let data = renderBuffer.floatChannelData else { throw TimeStretchError.engineSetupFailed }
                let length = Int(renderBuffer.frameLength)
                for c in 0..<outChannels.count {
                    outChannels[c].append(contentsOf: UnsafeBufferPointer(start: data[c], count: length))
                }
                rendered += length
            case .cannotDoInCurrentContext:
                continue
            case .insufficientDataFromInputNode, .error:
                throw TimeStretchError.engineSetupFailed
            @unknown default:
                throw TimeStretchError.engineSetupFailed
            }
        }
        outChannels = outChannels.map { Array($0.dropFirst(latency).prefix(targetFrames)) }

        engine.stop()
        return outChannels
    }
}
