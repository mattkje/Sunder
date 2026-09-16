import Foundation
import Observation
import SwiftUI

/// Checks the GitHub Releases list for a newer tagged app version than the
/// one currently running. The repo also hosts a `models-v1` release purely
/// to host model download assets (see AIModel.downloadURL) -- that tag
/// doesn't parse as a version, so it's filtered out here rather than
/// special-cased by name. One instance, shared app-wide (@Observable so
/// both the launch popup and Settings' "Check for Updates" row reflect the
/// same in-flight check).
@Observable
final class UpdateChecker {
    struct Release: Equatable {
        let version: String
        let htmlURL: URL
    }

    static let shared = UpdateChecker()

    private(set) var availableUpdate: Release?
    private(set) var isChecking = false
    private(set) var lastError: String?
    private(set) var hasChecked = false

    private static let releasesURL = URL(string: "https://api.github.com/repos/mattkje/Sunder/releases")!

    private var currentVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0"
    }

    private init() {}

    @MainActor
    func check() async {
        isChecking = true
        lastError = nil
        defer {
            isChecking = false
            hasChecked = true
        }
        do {
            let (data, _) = try await URLSession.shared.data(from: Self.releasesURL)
            let releases = try JSONDecoder().decode([GitHubReleaseResponse].self, from: data)
            let newest = releases
                .filter { !$0.draft && !$0.prerelease }
                .compactMap { release -> Release? in
                    guard let version = Self.semver(from: release.tag_name),
                          let url = URL(string: release.html_url) else { return nil }
                    return Release(version: version, htmlURL: url)
                }
                .max { Self.isVersion($0.version, lessThan: $1.version) }

            if let newest, Self.isVersion(currentVersion, lessThan: newest.version) {
                availableUpdate = newest
            } else {
                availableUpdate = nil
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    /// Clears the launch popup without re-checking; Settings' "Check for
    /// Updates" (or the next launch) can surface it again.
    func dismiss() {
        availableUpdate = nil
    }

    /// Strips a leading "v" and validates the rest is dotted integers, e.g.
    /// "v1.0.1" -> "1.0.1"; rejects non-version tags like "models-v1".
    private static func semver(from tag: String) -> String? {
        let stripped = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
        let parts = stripped.split(separator: ".")
        guard !parts.isEmpty, parts.allSatisfy({ Int($0) != nil }) else { return nil }
        return stripped
    }

    private static func isVersion(_ lhs: String, lessThan rhs: String) -> Bool {
        let l = lhs.split(separator: ".").compactMap { Int($0) }
        let r = rhs.split(separator: ".").compactMap { Int($0) }
        for i in 0..<max(l.count, r.count) {
            let a = i < l.count ? l[i] : 0
            let b = i < r.count ? r[i] : 0
            if a != b { return a < b }
        }
        return false
    }
}

private struct GitHubReleaseResponse: Decodable {
    let tag_name: String
    let html_url: String
    let draft: Bool
    let prerelease: Bool
}

/// Launch-time popup for `ContentView`, shown whenever
/// `UpdateChecker.shared.availableUpdate` is non-nil.
struct UpdateAvailableSheet: View {
    let update: UpdateChecker.Release

    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.openURL) private var openURL
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle().fill(.primary).frame(width: 56, height: 56)
                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(.background)
            }

            VStack(spacing: 6) {
                Text("Update Available")
                    .font(.title3.weight(.semibold))
                Text("Sunder \(update.version) is ready to download.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            HStack(spacing: 12) {
                Button("Later") { dismiss() }
                Button("View Release") {
                    openURL(update.htmlURL)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(28)
        .frame(minWidth: 320)
        .background(AppTheme.backgroundGradient(for: colorScheme))
    }
}
