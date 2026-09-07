import Compression
import Foundation

/// Minimal ZIP reader for the archives ModelDownloader deals with (plain
/// `zip -r` output, single-disk, no ZIP64 -- every model asset is well
/// under the 4GB ZIP64 threshold). Replaces a ZIPFoundation SPM dependency
/// that repeatedly failed to resolve in Xcode's interactive package graph
/// on at least one real machine even after every standard fix (quit, reset
/// package caches, re-resolve, clean DerivedData) -- xcodebuild itself
/// always built and linked it fine there, so the failure was specific to
/// Xcode's own package resolution UI/state, not reproducible or diagnosable
/// from here. A dependency-free reader removes that failure mode entirely:
/// decompression is Apple's own Compression framework (raw DEFLATE, ZIP's
/// only compression method besides "stored"), so the only code here is
/// parsing the ZIP container itself.
nonisolated enum ZipReader {
    /// Extracts every entry in `zipURL` into `destination` (created if
    /// needed), preserving the archive's directory structure.
    static func unzipItem(at zipURL: URL, to destination: URL) throws {
        let data = try Data(contentsOf: zipURL, options: .alwaysMapped)
        let entries = try centralDirectoryEntries(in: data)

        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)

        for entry in entries {
            let outURL = destination.appendingPathComponent(entry.path)
            if entry.path.hasSuffix("/") {
                try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)
                continue
            }
            try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let contents = try extract(entry, from: data)
            try contents.write(to: outURL)
        }
    }

    // MARK: - Central directory

    private struct Entry {
        let path: String
        let compressionMethod: UInt16
        let compressedSize: Int
        let uncompressedSize: Int
        let localHeaderOffset: Int
    }

    /// Finds the End Of Central Directory record (searching backward from
    /// EOF, since an optional trailing comment can push it before the last
    /// 22 bytes) and reads every central directory entry it points to.
    private static func centralDirectoryEntries(in data: Data) throws -> [Entry] {
        let eocdSignature: [UInt8] = [0x50, 0x4B, 0x05, 0x06]
        let searchFloor = max(0, data.count - 22 - 65536) // max comment length is 16 bits
        guard let eocdStart = data.lastRange(of: eocdSignature, before: data.count, after: searchFloor) else {
            throw ZipReaderError.notAZip
        }

        let cdEntryCount = Int(data.readUInt16(at: eocdStart + 10))
        let cdOffset = Int(data.readUInt32(at: eocdStart + 16))

        var entries: [Entry] = []
        entries.reserveCapacity(cdEntryCount)

        var offset = cdOffset
        for _ in 0..<cdEntryCount {
            guard data.readUInt32(at: offset) == 0x02014b50 else {
                throw ZipReaderError.malformedCentralDirectory
            }
            let compressionMethod = data.readUInt16(at: offset + 10)
            let compressedSize = Int(data.readUInt32(at: offset + 20))
            let uncompressedSize = Int(data.readUInt32(at: offset + 24))
            let nameLength = Int(data.readUInt16(at: offset + 28))
            let extraLength = Int(data.readUInt16(at: offset + 30))
            let commentLength = Int(data.readUInt16(at: offset + 32))
            let localHeaderOffset = Int(data.readUInt32(at: offset + 42))

            let nameStart = offset + 46
            let path = String(decoding: data[nameStart..<(nameStart + nameLength)], as: UTF8.self)

            entries.append(Entry(
                path: path,
                compressionMethod: compressionMethod,
                compressedSize: compressedSize,
                uncompressedSize: uncompressedSize,
                localHeaderOffset: localHeaderOffset
            ))

            offset = nameStart + nameLength + extraLength + commentLength
        }
        return entries
    }

    // MARK: - Per-entry extraction

    /// Compressed data always immediately follows the *local* file header
    /// (whose variable-length name/extra fields can differ in length from
    /// the central directory's copy), so the local header has to be read
    /// too, purely to find where the actual bytes start.
    private static func extract(_ entry: Entry, from data: Data) throws -> Data {
        guard data.readUInt32(at: entry.localHeaderOffset) == 0x04034b50 else {
            throw ZipReaderError.malformedLocalHeader
        }
        let nameLength = Int(data.readUInt16(at: entry.localHeaderOffset + 26))
        let extraLength = Int(data.readUInt16(at: entry.localHeaderOffset + 28))
        let dataStart = entry.localHeaderOffset + 30 + nameLength + extraLength
        let compressed = data[dataStart..<(dataStart + entry.compressedSize)]

        switch entry.compressionMethod {
        case 0: // stored, no compression
            return Data(compressed)
        case 8: // deflate -- ZIP's raw deflate stream, no zlib/gzip wrapper
            return try inflate(compressed, uncompressedSize: entry.uncompressedSize)
        default:
            throw ZipReaderError.unsupportedCompressionMethod(entry.compressionMethod)
        }
    }

    private static func inflate(_ compressed: Data, uncompressedSize: Int) throws -> Data {
        guard uncompressedSize > 0 else { return Data() }
        var output = Data(count: uncompressedSize)
        let decodedCount = output.withUnsafeMutableBytes { outBuf -> Int in
            compressed.withUnsafeBytes { inBuf -> Int in
                compression_decode_buffer(
                    outBuf.bindMemory(to: UInt8.self).baseAddress!, uncompressedSize,
                    inBuf.bindMemory(to: UInt8.self).baseAddress!, compressed.count,
                    nil, COMPRESSION_ZLIB
                )
            }
        }
        guard decodedCount == uncompressedSize else {
            throw ZipReaderError.decompressionFailed
        }
        return output
    }
}

enum ZipReaderError: LocalizedError {
    case notAZip
    case malformedCentralDirectory
    case malformedLocalHeader
    case unsupportedCompressionMethod(UInt16)
    case decompressionFailed

    var errorDescription: String? {
        switch self {
        case .notAZip: return "Downloaded file is not a valid zip archive."
        case .malformedCentralDirectory, .malformedLocalHeader: return "Downloaded zip archive is malformed."
        case .unsupportedCompressionMethod(let method): return "Zip entry uses an unsupported compression method (\(method))."
        case .decompressionFailed: return "Could not decompress a zip archive entry."
        }
    }
}

private extension Data {
    func readUInt16(at offset: Int) -> UInt16 {
        UInt16(self[startIndex + offset]) | (UInt16(self[startIndex + offset + 1]) << 8)
    }

    func readUInt32(at offset: Int) -> UInt32 {
        UInt32(self[startIndex + offset])
            | (UInt32(self[startIndex + offset + 1]) << 8)
            | (UInt32(self[startIndex + offset + 2]) << 16)
            | (UInt32(self[startIndex + offset + 3]) << 24)
    }

    /// Searches backward for `pattern`, only within `[after, before)`, so
    /// the EOCD search can be bounded to the tail of a multi-hundred-MB
    /// file instead of scanning it all.
    func lastRange(of pattern: [UInt8], before: Int, after: Int) -> Int? {
        guard pattern.count > 0, before - after >= pattern.count else { return nil }
        var i = before - pattern.count
        while i >= after {
            var matched = true
            for j in 0..<pattern.count where self[startIndex + i + j] != pattern[j] {
                matched = false
                break
            }
            if matched { return i }
            i -= 1
        }
        return nil
    }
}
