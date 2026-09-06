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

        let frameCount = channels[0].count
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(frameCount)) else {
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
        try file.write(from: buffer)
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
