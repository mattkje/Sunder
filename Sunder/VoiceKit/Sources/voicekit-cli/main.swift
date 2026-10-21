import AVFoundation
import CoreML
import Foundation
import VoiceKit

// Headless VoiceKit runner.
//   voicekit-cli parity  <dir with RVCBaseG/D + rvc_reference.safetensors>
//   voicekit-cli train   <assets dir> <library dir> <name> <epochs> <audio files...>
//   voicekit-cli convert <assets dir> <library dir> <voice name> <in audio> <out.wav> [semitones] [seed-vc dir]
//   voicekit-cli import  <library dir> <unzipped .sundervoice's Voice folder>
// <assets dir> holds ContentVec/RMVPE (.mlpackage or .mlmodelc) and RVCBaseG/D.safetensors;
// <seed-vc dir> holds SeedVCWhisper/SeedVCVocoder (.mlpackage or .mlmodelc), needed for Seed-VC voices.

func loadMono44k(_ url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let format = file.processingFormat
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buffer)
    let n = Int(buffer.frameLength), channels = Int(format.channelCount)
    var mono = [Float](repeating: 0, count: n)
    for c in 0 ..< channels {
        let p = buffer.floatChannelData![c]
        for i in 0 ..< n { mono[i] += p[i] / Float(channels) }
    }
    return try VoiceAudio.resample(mono, from: format.sampleRate, to: VoiceAudio.appSampleRate)
}

func writeWav(_ samples: [Float], to url: URL) throws {
    let format = AVAudioFormat(standardFormatWithSampleRate: VoiceAudio.appSampleRate, channels: 1)!
    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
    try file.write(from: buffer)
}

func compiled(_ dir: URL, _ name: String) throws -> URL {
    let compiled = dir.appending(path: "\(name).mlmodelc")
    if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
    let temp = try MLModel.compileModel(at: dir.appending(path: "\(name).mlpackage"))
    try FileManager.default.moveItem(at: temp, to: compiled)
    return compiled
}

func engine(_ dir: URL) throws -> VoiceEngine {
    try VoiceEngine(assets: VoiceAssets(
        contentVec: compiled(dir, "ContentVec"), rmvpe: compiled(dir, "RMVPE"),
        baseG: dir.appending(path: "RVCBaseG.safetensors"), baseD: dir.appending(path: "RVCBaseD.safetensors")))
}

let args = CommandLine.arguments
let url = { (i: Int) in URL(fileURLWithPath: args[i]) }
setvbuf(stdout, nil, _IOLBF, 0)

switch args.count > 1 ? args[1] : "" {
case "parity" where args.count >= 3:
    let results = try RVCParity.run(baseDirectory: url(2))
    var ok = true
    for (name, err) in results {
        let pass = err < 1e-3
        ok = ok && pass
        print(String(format: "%@ %-18@ rel max err %.3e", pass ? "PASS" : "FAIL", name, err))
    }
    exit(ok ? 0 : 1)

case "train" where args.count >= 7:
    let library = VoiceLibrary(root: url(3))
    try FileManager.default.createDirectory(at: library.root, withIntermediateDirectories: true)
    let sources = try args[6...].map { try loadMono44k(URL(fileURLWithPath: $0)) }
    let usable = try sources.reduce(0) { $0 + (try VoiceEngine.usableSeconds(of: $1)) }
    print(String(format: "usable audio: %.1f s", usable))
    let profile = try library.list().first { $0.name == args[4] } ?? library.create(name: args[4], epochs: Int(args[5])!)
    let start = Date()
    var lastEpoch = 0
    let result = try engine(url(2)).train(profile, sources: sources, library: library, progress: { p in
        switch p {
        case .preparing(let f): if Int(f * 100) % 10 == 0 { print(String(format: "preparing %.0f%%", f * 100)) }
        case let .training(epoch, epochs, step, steps, mel):
            if step == steps || epoch != lastEpoch {
                lastEpoch = epoch
                print(String(format: "epoch %d/%d step %d/%d mel %.3f  (%.0f s)", epoch, epochs, step, steps, mel, Date().timeIntervalSince(start)))
            }
        case .finishing: print("saving voice model")
        }
    }, shouldStop: { false })
    print("trained=\(result.isTrained) epochs=\(result.completedEpochs) seconds=\(result.trainingSeconds) id=\(result.id)")

case "convert" where args.count >= 7:
    let library = VoiceLibrary(root: url(3))
    guard let profile = library.list().first(where: { $0.name == args[4] }) else { print("no voice \(args[4])"); exit(1) }
    let audio = try loadMono44k(url(5))
    let settings = ConversionSettings(pitchShift: args.count > 7 ? Float(args[7])! : 0)
    let seedVC = try args.count > 8 ? SeedVCAssets(whisper: compiled(url(8), "SeedVCWhisper"), vocoder: compiled(url(8), "SeedVCVocoder")) : nil
    let start = Date()
    var lastReport = -1
    let out = try engine(url(2)).convert(audio, with: profile, library: library, settings: settings, seedVC: seedVC) { p in
        if Int(p * 10) != lastReport { lastReport = Int(p * 10); print(String(format: "%.0f%%", p * 100)) }
    }
    try writeWav(out, to: url(6))
    print(String(format: "converted %.1f s of audio in %.1f s", Double(audio.count) / 44100, Date().timeIntervalSince(start)))

case "import" where args.count >= 4:
    let library = VoiceLibrary(root: url(2))
    try FileManager.default.createDirectory(at: library.root, withIntermediateDirectories: true)
    let profile = try library.importVoice(from: url(3))
    print("imported \(profile.name) (\(profile.engine.rawValue)) id=\(profile.id)")

default:
    print("usage: voicekit-cli parity|train|convert|import ... (see main.swift)")
    exit(2)
}
