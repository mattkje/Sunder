import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#endif

/// Opens automatically once a separation finishes (SeparationScreen presents
/// this as a sheet whenever `SeparationQueue.shared.status` is `.done`), and
/// is also embedded directly in the persistent "Mixer" sidebar tab
/// (MixerScreen), backed by `SeparationQueue.lastSession`. Nothing from the
/// separation is written to disk until one of the Save actions below runs --
/// the stems only exist in memory (`SeparationQueue.MixSession`) until then.
///
/// Mute/solo, plus an optional rule-based "Auto Mix" level balance and a
/// "Master" loudness/limiter pass on export -- see MixerPlayer and
/// MasterEngine. Both are off by default, so the original mute/solo/plain-sum
/// behavior is unchanged unless the user opts in.
struct MixerView: View {
    let session: SeparationQueue.MixSession
    /// nil when embedded as the persistent "Mixer" sidebar tab (MixerScreen)
    /// -- there's nothing to close there, so the toolbar's "Close" button and
    /// the sheet-sized fixed frame both only apply when this is non-nil, i.e.
    /// when SeparationScreen is presenting this as a sheet right after a
    /// separation finishes.
    let onClose: (() -> Void)?
    /// Called after the "Align" tool time-stretches a stem, with the
    /// session's stems otherwise unchanged except for that one -- nil when
    /// there's nowhere for the update to go (there always is one in
    /// practice, since only MixerScreen presents this, but it stays
    /// optional/nil-safe for the same reason `onClose` is).
    var onUpdateSession: ((SeparationQueue.MixSession) -> Void)? = nil

    @State private var player: MixerPlayer?
    @State private var setupError: String?
    @State private var masterEnabled = false
    @State private var alignEnabled = false
    @State private var alignStretchIndex = 0
    @State private var alignReferenceIndex = 1
    @State private var alignStretchA: Double?
    @State private var alignStretchB: Double?
    @State private var alignReferenceA: Double?
    @State private var alignReferenceB: Double?
    @State private var alignError: String?
    @State private var isAligning = false
    @State private var detectedBPM: Double?
    @State private var detectedKey: KeyDetector.Key?
    @State private var isAnalyzing = false
    @Environment(\.colorScheme) private var colorScheme
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue

    #if os(iOS)
    @State private var shareItems: [URL]?
    #endif

    private var quality: OutputQuality {
        OutputQuality(rawValue: qualityRaw) ?? .web
    }

    private var stemNames: [String] { session.stemNames }
    private var baseName: String { session.inputURL.deletingPathExtension().lastPathComponent }

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()

                if let player {
                    content(player)
                } else if let setupError {
                    Label(setupError, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .padding()
                } else {
                    ProgressView().onAppear { setUp() }
                }
            }
            .navigationTitle("Mixer")
            .toolbar {
                if let onClose {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Close") {
                            player?.stop()
                            onClose()
                        }
                    }
                }
            }
        }
        #if os(macOS)
        .frame(width: onClose != nil ? 480 : nil, height: onClose != nil ? 520 : nil)
        #endif
        #if os(iOS)
        .sheet(item: Binding(
            get: { shareItems.map { ShareItems(urls: $0) } },
            set: { shareItems = $0?.urls }
        )) { items in
            ShareSheet(items: items.urls)
        }
        #endif
    }

    @ViewBuilder
    private func content(_ player: MixerPlayer) -> some View {
        VStack(spacing: 0) {
            transport(player)
            Divider()
            mixControls(player)
            Divider()
            pitchTempoControls(player)
            if alignEnabled {
                Divider()
                alignPanel(player)
            }
            Divider()
            List {
                ForEach(Array(stemNames.enumerated()), id: \.offset) { i, name in
                    stemRow(player, index: i, name: name)
                }
            }
            #if os(macOS)
            .listStyle(.inset)
            #endif
            Divider()
            saveBar(player)
        }
    }

    private func transport(_ player: MixerPlayer) -> some View {
        VStack(spacing: 8) {
            Slider(
                value: Binding(
                    get: { player.currentTime },
                    set: { player.seek(to: $0) }
                ),
                in: 0...max(player.duration, 0.01)
            )
            HStack {
                Text(timeString(player.currentTime))
                Spacer()
                Button {
                    player.togglePlayPause()
                } label: {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 32))
                }
                .buttonStyle(.plain)
                Spacer()
                Text(timeString(player.duration))
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
        }
        .padding(16)
    }

    /// "Auto Mix" balances stem levels live and in the bounce (MixerPlayer);
    /// "Master" only affects the bounce written by "Save Mix" (see saveMix
    /// below) -- there's no live bus processing to preview it against.
    /// "Align" (see alignPanel) needs at least two stems to make sense.
    private func mixControls(_ player: MixerPlayer) -> some View {
        HStack(spacing: 12) {
            Toggle("Auto Mix", isOn: Binding(
                get: { player.automixEnabled },
                set: { player.automixEnabled = $0 }
            ))
            .toggleStyle(.button)
            Toggle("Master", isOn: $masterEnabled)
                .toggleStyle(.button)
            Toggle("Align", isOn: $alignEnabled)
                .toggleStyle(.button)
                .disabled(stemNames.count < 2)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    /// Whole-mix pitch (semitones) and tempo (percent), live and in every
    /// saved file, labelled with the detected key/BPM and what they become.
    private func pitchTempoControls(_ player: MixerPlayer) -> some View {
        let semitones = Int(player.pitchSemitones)
        let percent = Int((player.tempo * 100).rounded())
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 16) {
                Stepper(value: Binding(get: { semitones }, set: { player.pitchSemitones = Float($0) }), in: -12...12) {
                    Text("Pitch \(semitones == 0 ? "0" : String(format: "%+d", semitones))")
                        .monospacedDigit()
                }
                .fixedSize()
                Stepper(value: Binding(get: { percent }, set: { player.tempo = Float($0) / 100 }), in: 50...150) {
                    Text("Tempo \(percent)%").monospacedDigit()
                }
                .fixedSize()
                Spacer()
                if semitones != 0 || percent != 100 {
                    Button("Reset") {
                        player.pitchSemitones = 0
                        player.tempo = 1
                    }
                    .font(.caption)
                }
            }
            Text(analysisSummary(semitones: semitones, tempo: player.tempo))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private func analysisSummary(semitones: Int, tempo: Float) -> String {
        if isAnalyzing { return "Detecting key and tempo…" }
        var parts: [String] = []
        if let key = detectedKey {
            parts.append(semitones == 0 ? key.name : "\(key.name) → \(key.transposed(by: semitones).name)")
        }
        if let bpm = detectedBPM {
            let target = bpm * Double(tempo)
            parts.append(abs(tempo - 1) < 0.001 ? "\(Int(bpm.rounded())) BPM" : "\(Int(bpm.rounded())) → \(Int(target.rounded())) BPM")
        }
        return parts.isEmpty ? "Key and tempo couldn't be detected" : parts.joined(separator: " · ")
    }

    /// Detects key and tempo from all stems summed, off the main actor.
    private func analyze() {
        let stems = session.stems
        isAnalyzing = true
        Task.detached(priority: .utility) {
            guard let first = stems.first else { return }
            var mix = first
            for stem in stems.dropFirst() {
                for c in mix.indices { for s in mix[c].indices { mix[c][s] += stem[c][s] } }
            }
            let key = KeyDetector.detect(mix)
            let beats = BeatDetector.detectBeats(mix)
            await MainActor.run {
                detectedKey = key
                detectedBPM = beats.times.count >= 8 ? beats.bpm : nil
                isAnalyzing = false
            }
        }
    }

    /// Two-anchor manual tempo/offset match: the user solos the stem being
    /// stretched (with the existing per-row Solo toggle below) and the
    /// reference stem in turn, scrubbing to the same musical moment each
    /// time and tapping that stem's Mark button -- twice, at two distinct
    /// moments -- rather than this panel needing its own playback UI.
    /// `audioRate`/`shiftSeconds` are the exact linear fit through those two
    /// point-pairs; see TimeStretcher for how they're applied.
    private func alignPanel(_ player: MixerPlayer) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Stretch")
                Picker("Stretch", selection: $alignStretchIndex) {
                    ForEach(stemNames.indices, id: \.self) { i in Text(stemNames[i]).tag(i) }
                }
                .labelsHidden()
                Text("to match")
                Picker("Reference", selection: $alignReferenceIndex) {
                    ForEach(stemNames.indices, id: \.self) { i in Text(stemNames[i]).tag(i) }
                }
                .labelsHidden()
            }

            alignMarkRow(
                label: "Point A",
                stretchValue: alignStretchA, referenceValue: alignReferenceA,
                markStretch: { alignStretchA = player.currentTime },
                markReference: { alignReferenceA = player.currentTime }
            )
            alignMarkRow(
                label: "Point B",
                stretchValue: alignStretchB, referenceValue: alignReferenceB,
                markStretch: { alignStretchB = player.currentTime },
                markReference: { alignReferenceB = player.currentTime }
            )

            if let alignError {
                Label(alignError, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            } else if let rate = pendingAlignRate {
                Text("Stretch rate: \(rate, specifier: "%.3f")×")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack {
                if isAligning {
                    ProgressView().controlSize(.small)
                }
                Spacer()
                Button("Auto Align") { autoAlign() }
                    .disabled(isAligning || alignStretchIndex == alignReferenceIndex)
                Button("Apply") { applyAlignment() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isAligning || !canApplyAlignment)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private func alignMarkRow(
        label: String, stretchValue: Double?, referenceValue: Double?,
        markStretch: @escaping () -> Void, markReference: @escaping () -> Void
    ) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary).frame(width: 56, alignment: .leading)
            Button(stretchValue.map(timeString) ?? "Mark Stretch") { markStretch() }
                .buttonStyle(.bordered)
            Button(referenceValue.map(timeString) ?? "Mark Reference") { markReference() }
                .buttonStyle(.bordered)
        }
    }

    private var canApplyAlignment: Bool {
        alignStretchIndex != alignReferenceIndex
            && alignStretchA != nil && alignStretchB != nil
            && alignReferenceA != nil && alignReferenceB != nil
    }

    /// nil until all four points are marked; matches applyAlignment's own
    /// math so the preview and the applied result always agree.
    private var pendingAlignRate: Double? {
        guard let sa = alignStretchA, let sb = alignStretchB,
              let ra = alignReferenceA, let rb = alignReferenceB,
              abs(rb - ra) > 0.01 else { return nil }
        return (sb - sa) / (rb - ra)
    }

    /// Manual two-anchor path: one constant rate + start-offset shift,
    /// derived from the four marked points.
    private func applyAlignment() {
        alignError = nil
        guard let sa = alignStretchA, let sb = alignStretchB,
              let ra = alignReferenceA, let rb = alignReferenceB else { return }
        guard abs(sb - sa) > 0.25, abs(rb - ra) > 0.25 else {
            alignError = "Mark two points at least a quarter-second apart on each stem."
            return
        }
        let rate = Float((sb - sa) / (rb - ra))
        guard TimeStretcher.validRateRange.contains(rate) else {
            alignError = "That implies too extreme a tempo change -- check the marked points."
            return
        }

        let stretchIndex = alignStretchIndex
        let stemAudio = session.stems[stretchIndex]
        let frameCount = session.stems.first?.first?.count ?? 0
        let shiftSeconds = ra - sa / Double(rate)

        runAlignment(stretchIndex: stretchIndex) {
            let stretched = try TimeStretcher.stretch(stemAudio, rate: rate)
            let shiftFrames = Int((shiftSeconds * ModelConfig.sampleRate).rounded())
            let shifted = stretched.map { channel -> [Float] in
                if shiftFrames >= 0 {
                    return [Float](repeating: 0, count: shiftFrames) + channel
                } else {
                    return Array(channel.dropFirst(min(-shiftFrames, channel.count)))
                }
            }
            return AudioIO.fit(channels: shifted, toFrameCount: frameCount)
        }
    }

    /// Automatic beat-grid path: detects each stem's own beat grid, matches
    /// them, and stretches piecewise -- see AutoAligner. No marked points
    /// needed, just the stem/reference picker selection above.
    private func autoAlign() {
        alignError = nil
        guard alignStretchIndex != alignReferenceIndex else { return }
        let stretchIndex = alignStretchIndex
        let stemAudio = session.stems[stretchIndex]
        let referenceAudio = session.stems[alignReferenceIndex]
        let frameCount = session.stems.first?.first?.count ?? 0

        runAlignment(stretchIndex: stretchIndex) {
            try AutoAligner.align(stem: stemAudio, reference: referenceAudio, toFrameCount: frameCount)
        }
    }

    /// Shared tail for both alignment paths above: runs `compute` off the
    /// main thread, replaces `stretchIndex`'s stem with its result, and
    /// hands the rebuilt session up via `onUpdateSession` -- or surfaces the
    /// error inline (both `TimeStretcher` and `AutoAligner` throw plain,
    /// user-facing `LocalizedError`s).
    private func runAlignment(stretchIndex: Int, compute: @escaping () throws -> [[Float]]) {
        isAligning = true
        alignError = nil
        let stems = session.stems
        Task.detached(priority: .userInitiated) {
            do {
                var newStems = stems
                newStems[stretchIndex] = try compute()
                let newSession = SeparationQueue.MixSession(
                    inputURL: session.inputURL, stemNames: session.stemNames, stems: newStems
                )
                await MainActor.run {
                    isAligning = false
                    onUpdateSession?(newSession)
                }
            } catch {
                await MainActor.run {
                    isAligning = false
                    alignError = error.localizedDescription
                }
            }
        }
    }

    private func stemRow(_ player: MixerPlayer, index: Int, name: String) -> some View {
        HStack(spacing: 12) {
            Text(name).font(.body)
            Spacer()
            Toggle("Mute", isOn: Binding(
                get: { player.muted[index] },
                set: { player.muted[index] = $0 }
            ))
            .toggleStyle(.button)
            .tint(.red)

            Toggle("Solo", isOn: Binding(
                get: { player.soloed.contains(index) },
                set: { on in
                    if on { player.soloed.insert(index) } else { player.soloed.remove(index) }
                }
            ))
            .toggleStyle(.button)
            .tint(.yellow)

            Button {
                saveSingle(stemIndex: index, name: name)
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }

    private func saveBar(_ player: MixerPlayer) -> some View {
        HStack(spacing: 12) {
            Button("Save All Separately") { saveAllSeparately() }
            Spacer()
            Button("Save Mix") { saveMix(player) }
                .buttonStyle(.borderedProminent)
        }
        .padding(16)
    }

    private func setUp() {
        do {
            player = try MixerPlayer(stems: session.stems, stemNames: stemNames)
            player?.play()
            analyze()
        } catch {
            setupError = error.localizedDescription
        }
    }

    private func timeString(_ t: Double) -> String {
        let total = Int(t.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    // MARK: - Save actions

    private func saveSingle(stemIndex: Int, name: String) {
        let url = tempFile(name: "\(baseName)_\(name)")
        guard let player, let stem = try? player.applyPitchAndTempo(session.stems[stemIndex]),
              (try? AudioIO.writeAudio(channels: stem, to: url, quality: quality)) != nil else { return }
        deliver([url])
    }

    private func saveMix(_ player: MixerPlayer) {
        let url = tempFile(name: "\(baseName)_Mix")
        guard let shifted = try? player.applyPitchAndTempo(player.audibleMix()) else { return }
        let mix = masterEnabled ? MasterEngine.process(shifted) : shifted
        guard (try? AudioIO.writeAudio(channels: mix, to: url, quality: quality)) != nil else { return }
        deliver([url])
    }

    private func saveAllSeparately() {
        var urls: [URL] = []
        for (i, name) in stemNames.enumerated() {
            let url = tempFile(name: "\(baseName)_\(name)")
            guard let player, let stem = try? player.applyPitchAndTempo(session.stems[i]),
                  (try? AudioIO.writeAudio(channels: stem, to: url, quality: quality)) != nil else { continue }
            urls.append(url)
        }
        deliver(urls)
    }

    private func tempFile(name: String) -> URL {
        AudioExport.tempFile(name: name, quality: quality)
    }

    /// Writes land in a fresh temp dir (see `tempFile`) either way; this
    /// step is what actually gets them to a place the user chose.
    private func deliver(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        #if os(macOS)
        AudioExport.deliverMac(urls, quality: quality)
        #else
        shareItems = urls
        #endif
    }
}

#Preview {
    MixerView(
        session: .init(inputURL: URL(fileURLWithPath: "/tmp/test.mp3"), stemNames: ["Vocals", "Instrumental"], stems: []),
        onClose: {}
    )
}
