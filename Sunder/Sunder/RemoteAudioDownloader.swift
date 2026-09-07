import Foundation

/// Downloads a plain http(s) audio file the user pasted a link to, into a
/// local temp file AudioIO can read. Distinct from ModelDownloader's
/// GitHub-release-asset downloader (that one unzips and installs a model
/// into Application Support; this one just fetches one file for one-time
/// use as separation input).
nonisolated enum RemoteAudioDownloader {
    static func download(_ url: URL) async throws -> URL {
        let (tempURL, response) = try await URLSession.shared.download(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            try? FileManager.default.removeItem(at: tempURL)
            throw RemoteAudioDownloaderError.badResponse
        }

        // AVAudioFile picks its parser by file extension, so give the temp
        // file the source URL's extension (defaulting to mp3, the most
        // common case) rather than URLSession's extension-less temp name.
        let ext = url.pathExtension.isEmpty ? "mp3" : url.pathExtension
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(ext)
        try FileManager.default.moveItem(at: tempURL, to: destination)
        return destination
    }
}

enum RemoteAudioDownloaderError: LocalizedError {
    case badResponse

    var errorDescription: String? {
        switch self {
        case .badResponse: return "Could not download audio from that link."
        }
    }
}
