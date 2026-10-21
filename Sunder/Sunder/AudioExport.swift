import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#endif

/// Shared "get a bounced file to where the user wants it" plumbing --
/// originally MixerView-only, now also used by SpatializeView, so it lives
/// here once instead of twice. macOS delivery (save panel / choose-folder)
/// needs no SwiftUI state and is fully self-contained; iOS delivery is a
/// share sheet, which needs a `@State` binding in the presenting view (see
/// `ShareSheet`/`ShareItems` below) -- each screen keeps its own tiny
/// `deliver(_:)` that calls into this for the macOS half.
enum AudioExport {
    /// Creates the (fresh, per-call) scratch directory up front -- AVAudioFile
    /// requires the parent directory to already exist before it will write.
    static func tempFile(name: String, quality: OutputQuality) -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("\(name).\(quality.fileExtension)")
    }

    #if os(macOS)
    static func deliverMac(_ urls: [URL], quality: OutputQuality) {
        guard !urls.isEmpty else { return }
        if urls.count == 1 {
            savePanel(suggesting: urls[0], quality: quality)
        } else {
            chooseFolderAndCopy(urls)
        }
    }

    private static func savePanel(suggesting sourceURL: URL, quality: OutputQuality) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = sourceURL.lastPathComponent
        panel.allowedContentTypes = quality.fileExtension == "wav" ? [.wav] : [.mpeg4Audio]
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        try? FileManager.default.removeItem(at: destination)
        try? FileManager.default.copyItem(at: sourceURL, to: destination)
    }

    private static func chooseFolderAndCopy(_ urls: [URL]) {
        let panel = NSOpenPanel()
        panel.title = "Choose where to save the exported tracks"
        panel.prompt = "Choose"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let outputDir = panel.url else { return }
        for url in urls {
            let destination = outputDir.appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.removeItem(at: destination)
            try? FileManager.default.copyItem(at: url, to: destination)
        }
        NSWorkspace.shared.activateFileViewerSelecting(urls.map { outputDir.appendingPathComponent($0.lastPathComponent) })
    }
    #endif
}

#if os(iOS)
struct ShareItems: Identifiable {
    let urls: [URL]
    var id: String { urls.map(\.absoluteString).joined() }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [URL]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif
