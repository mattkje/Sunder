import SwiftUI
#if os(iOS)
import UIKit
#endif

/// The "Spatialize" tab's player: a circular drag map for placing each stem
/// around the listener (via SpatialAudioPlayer/AVAudioEnvironmentNode) plus
/// transport and export. The listener is the filled dot at the center of
/// the ring; x is left/right, the map's vertical axis is front/back
/// (negative = in front, toward the "Front" label); there's no height axis.
///
/// "Auto" applies a role-based preset arrangement; dragging is always live
/// on top of that (no separate locked "manual mode" to fight with it) --
/// moving a dot updates the actual playing audio immediately, which is the
/// live-listen behavior itself, not a separate feature. "Immersive" swaps
/// in true binaural (HRTF) rendering for headphone listening; the default
/// (off) uses simple, reliable left/right panning instead -- see
/// SpatialAudioPlayer's own doc comment for why that's the default.
struct SpatializeView: View {
    let session: SeparationQueue.MixSession

    @State private var player: SpatialAudioPlayer?
    @State private var setupError: String?
    @Environment(\.colorScheme) private var colorScheme
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue

    #if os(iOS)
    @State private var shareItems: [URL]?
    #endif

    private var quality: OutputQuality { OutputQuality(rawValue: qualityRaw) ?? .web }
    private var stemNames: [String] { session.stemNames }
    private var baseName: String { session.inputURL.deletingPathExtension().lastPathComponent }

    var body: some View {
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
        .navigationTitle("Spatialize")
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
    private func content(_ player: SpatialAudioPlayer) -> some View {
        VStack(spacing: 0) {
            transport(player)
            Divider()
            map(player)
                .frame(minHeight: 260)
                .padding(16)
            Divider()
            stemList(player)
            Divider()
            saveBar(player)
        }
    }

    private func transport(_ player: SpatialAudioPlayer) -> some View {
        VStack(spacing: 8) {
            HStack {
                Text("Spatialize").font(.headline)
                Spacer()
                Toggle("Immersive", isOn: Binding(
                    get: { player.immersiveEnabled },
                    set: { player.immersiveEnabled = $0 }
                ))
                .toggleStyle(.button)
                .help("True binaural (HRTF) placement -- try it on headphones. Off uses simple left/right panning, which sounds correct and full-volume on any speakers or headphones.")
                Button("Auto") { player.resetToAutoLayout(stemNames: stemNames) }
            }
            Slider(
                value: Binding(get: { player.currentTime }, set: { player.seek(to: $0) }),
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

    /// Top-down drag map. Each stem is a labeled, colored dot; dragging
    /// updates `player`'s live position immediately (see SpatialAudioPlayer
    /// docs on why that alone is "live listen"). The small filled circle at
    /// the center is the listener; the ring around it is the field's
    /// boundary -- stems drag anywhere inside the ring, clamped radially
    /// (by distance from center, not per-axis), so the draggable area is
    /// actually circular rather than a square with unreachable corners.
    private func map(_ player: SpatialAudioPlayer) -> some View {
        GeometryReader { geo in
            let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
            let radius = min(geo.size.width, geo.size.height) / 2 - 24

            ZStack {
                Circle()
                    .fill(Color.secondary.opacity(0.06))
                    .frame(width: radius * 2, height: radius * 2)
                    .position(center)
                Circle()
                    .strokeBorder(Color.secondary.opacity(0.3), lineWidth: 1.5)
                    .frame(width: radius * 2, height: radius * 2)
                    .position(center)

                Text("Front")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .position(x: center.x, y: center.y - radius - 10)

                Circle() // the listener
                    .fill(Color.primary)
                    .frame(width: 12, height: 12)
                    .position(center)

                ForEach(stemNames.indices, id: \.self) { i in
                    stemDot(player, index: i, center: center, radius: radius)
                }
            }
        }
    }

    private func stemDot(_ player: SpatialAudioPlayer, index: Int, center: CGPoint, radius: CGFloat) -> some View {
        let position = player.positions[index]
        let point = CGPoint(
            x: center.x + CGFloat(position.x) * radius,
            y: center.y + CGFloat(position.y) * radius
        )

        return VStack(spacing: 2) {
            Circle()
                .fill(player.isAudible(index) ? Color.accentColor : Color.secondary)
                .frame(width: 28, height: 28)
            Text(stemNames[index]).font(.caption2).lineLimit(1)
        }
        .position(point)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    var x = Float((value.location.x - center.x) / radius)
                    var z = Float((value.location.y - center.y) / radius)
                    let magnitude = (x * x + z * z).squareRoot()
                    if magnitude > 1 {
                        x /= magnitude
                        z /= magnitude
                    }
                    player.setPosition(SIMD2(x, z), forStem: index)
                }
        )
    }

    private func stemList(_ player: SpatialAudioPlayer) -> some View {
        HStack(spacing: 12) {
            ForEach(stemNames.indices, id: \.self) { i in
                Toggle(stemNames[i], isOn: Binding(
                    get: { player.soloed.contains(i) },
                    set: { on in
                        if on { player.soloed.insert(i) } else { player.soloed.remove(i) }
                    }
                ))
                .toggleStyle(.button)
                .tint(.yellow)
                .font(.caption)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private func saveBar(_ player: SpatialAudioPlayer) -> some View {
        HStack {
            Spacer()
            Button("Save Mix") { saveMix(player) }
                .buttonStyle(.borderedProminent)
        }
        .padding(16)
    }

    private func setUp() {
        do {
            player = try SpatialAudioPlayer(stems: session.stems, stemNames: stemNames)
            player?.play()
        } catch {
            setupError = error.localizedDescription
        }
    }

    private func timeString(_ t: Double) -> String {
        let total = Int(t.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    private func saveMix(_ player: SpatialAudioPlayer) {
        guard let mix = try? player.spatializedMix() else { return }
        let url = AudioExport.tempFile(name: "\(baseName)_Spatial", quality: quality)
        guard (try? AudioIO.writeAudio(channels: mix, to: url, quality: quality)) != nil else { return }
        #if os(macOS)
        AudioExport.deliverMac([url], quality: quality)
        #else
        shareItems = [url]
        #endif
    }
}

#Preview {
    SpatializeView(
        session: .init(inputURL: URL(fileURLWithPath: "/tmp/test.mp3"), stemNames: ["Vocals", "Drums", "Bass", "Other"], stems: [])
    )
}
