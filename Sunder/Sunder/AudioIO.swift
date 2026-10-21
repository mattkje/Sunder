import Accelerate
import AVFoundation
import Foundation

nonisolated enum AudioIO {
    /// Loads any audio file readable by AVFoundation and returns exactly
    /// ModelConfig.channels channels of Float32 PCM at ModelConfig.sampleRate.
    /// Mono input is duplicated to fill the required channel count.
    static func loadAudio(url: URL) throws -> [[Float]] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: ModelConfig.sampleRate,
            channels: AVAudioChannelCount(ModelConfig.channels),
            interleaved: false
        ) else {
            throw AudioIOError.formatCreationFailed
        }

        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw AudioIOError.bufferCreationFailed
        }
        try file.read(into: sourceBuffer)

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw AudioIOError.converterCreationFailed
        }

        let ratio = ModelConfig.sampleRate / sourceFormat.sampleRate
        let outFrameCapacity = AVAudioFrameCount(Double(sourceBuffer.frameLength) * ratio) + 4096
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrameCapacity) else {
            throw AudioIOError.bufferCreationFailed
        }

        var error: NSError?
        var consumed = false
        converter.convert(to: outBuffer, error: &error) { _, inputStatus in
            if consumed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            inputStatus.pointee = .haveData
            return sourceBuffer
        }
        if let error { throw error }

        let frameCount = Int(outBuffer.frameLength)
        guard let channelData = outBuffer.floatChannelData else {
            throw AudioIOError.noChannelData
        }

        var channelsOut: [[Float]] = []
        let availableChannels = Int(targetFormat.channelCount)
        for c in 0..<availableChannels {
            channelsOut.append(Array(UnsafeBufferPointer(start: channelData[c], count: frameCount)))
        }
        // targetFormat already requests ModelConfig.channels; AVAudioConverter
        // upmixes mono sources by duplicating into all requested channels.
        return channelsOut
    }

    /// Writes `channels` (each the same length) using the given quality
    /// preset. AVAudioFile.processingFormat is always deinterleaved Float32
    /// regardless of the on-disk format, so the same buffer-filling code
    /// writes lossless PCM and AAC alike -- AVAudioFile does the encoding.
    static func writeAudio(channels: [[Float]], to url: URL, quality: OutputQuality) throws {
        let settings = quality.fileSettings(channels: channels.count)
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let buffer = try makeBuffer(channels: channels, format: file.processingFormat)
        try file.write(from: buffer)
    }

    /// Zero-pads or truncates every channel to exactly `frameCount` --
    /// stems from this app's own separation always already match each other
    /// (same source audio), but a file imported into the Mixer, or a stem
    /// just time-stretched by the Align tool, generally won't. Shared by
    /// MixerScreen's import/add-stem paths and MixerView's align-apply so
    /// this "make every stem the same length" rule exists exactly once.
    static func fit(channels: [[Float]], toFrameCount frameCount: Int) -> [[Float]] {
        channels.map { channel in
            if channel.count == frameCount {
                return channel
            } else if channel.count < frameCount {
                return channel + [Float](repeating: 0, count: frameCount - channel.count)
            } else {
                return Array(channel.prefix(frameCount))
            }
        }
    }

    /// Averages every channel down to one -- BeatDetector needs mono for its
    /// onset analysis, and SpatialAudioPlayer needs mono per-stem sources
    /// since AVAudioEnvironmentNode point sources must be mono (a stereo
    /// signal has no single position to place). Shared here rather than
    /// duplicated so this rule exists exactly once.
    static func mixToMono(_ channels: [[Float]]) -> [Float] {
        guard let first = channels.first, !first.isEmpty else { return [] }
        guard channels.count > 1 else { return first }
        var mono = [Float](repeating: 0, count: first.count)
        for channel in channels {
            for i in 0..<min(channel.count, mono.count) { mono[i] += channel[i] }
        }
        var scale = Float(1) / Float(channels.count)
        vDSP_vsmul(mono, 1, &scale, &mono, 1, vDSP_Length(mono.count))
        return mono
    }

    /// Fills an AVAudioPCMBuffer from deinterleaved Float32 channel arrays --
    /// shared by `writeAudio` above (against the on-disk file's processing
    /// format) and MixerPlayer (against a plain playback engine format), so
    /// the buffer-filling loop exists exactly once.
    static func makeBuffer(channels: [[Float]], format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        let frameCount = channels[0].count
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            throw AudioIOError.bufferCreationFailed
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        guard let channelData = buffer.floatChannelData else {
            throw AudioIOError.noChannelData
        }
        for (c, data) in channels.enumerated() {
            data.withUnsafeBufferPointer { src in
                channelData[c].update(from: src.baseAddress!, count: frameCount)
            }
        }
        return buffer
    }
}

enum AudioIOError: LocalizedError {
    case formatCreationFailed
    case bufferCreationFailed
    case converterCreationFailed
    case noChannelData

    var errorDescription: String? {
        switch self {
        case .formatCreationFailed: return "Could not create audio format."
        case .bufferCreationFailed: return "Could not allocate audio buffer."
        case .converterCreationFailed: return "Could not create audio converter."
        case .noChannelData: return "Audio buffer has no channel data."
        }
    }
}
