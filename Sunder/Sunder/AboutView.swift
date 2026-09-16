import SwiftUI
#if os(macOS)
import AppKit
#else
import UIKit
#endif

#if os(macOS)
final class AboutWindowController: NSWindowController {
    static let shared = AboutWindowController()

    private init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 500),
            styleMask: [.titled, .closable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isMovableByWindowBackground = true
        window.center()
        window.setFrameAutosaveName("AboutWindow")
        window.contentView = NSHostingView(rootView: AboutView())
        super.init(window: window)
    }

    required init?(coder: NSCoder) { fatalError() }

    func showWindow() {
        window?.center()
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
#endif

/// The app icon, read from the asset catalog via each platform's own
/// bundle-icon API (there's no cross-platform `Image(appIcon:)`).
private var appIconImage: Image {
    #if os(macOS)
    return Image(nsImage: NSApp.applicationIconImage)
    #else
    let icons = Bundle.main.infoDictionary?["CFBundleIcons"] as? [String: Any]
    let primary = icons?["CFBundlePrimaryIcon"] as? [String: Any]
    let files = primary?["CFBundleIconFiles"] as? [String]
    if let name = files?.last, let uiImage = UIImage(named: name) {
        return Image(uiImage: uiImage)
    }
    return Image(systemName: "waveform")
    #endif
}

struct AboutView: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var showCredits = false

    private let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
    private let build   = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"

    var body: some View {
        #if os(iOS)
        iOSBody
        #else
        macBody
        #endif
    }

    #if os(iOS)
    /// Flat system background, no forced size -- this fills whatever sheet
    /// SettingsView presents it in, rather than the fixed 400x500 the
    /// macOS about *window* wants.
    private var iOSBody: some View {
        ZStack {
            Color(uiColor: .systemBackground).ignoresSafeArea()

            VStack(spacing: 0) {
                header
                Divider().padding(.horizontal, 32)
                descriptionAndAuthor
                Divider().padding(.horizontal, 32)
                links.padding(.vertical, 20)
                Spacer()
                modelAttribution
            }
        }
    }
    #else
    private var macBody: some View {
        ZStack {
            AppTheme.backgroundGradient(for: colorScheme)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                header
                Divider().padding(.horizontal, 32)
                descriptionAndAuthor
                Divider().padding(.horizontal, 32)
                links.padding(.vertical, 20)
                Spacer()
                modelAttribution
            }
        }
        .frame(width: 400, height: 500)
    }
    #endif

    private var header: some View {
        VStack(spacing: 12) {
            Group {
                appIconImage
                    .resizable()
                    .interpolation(.high)
            }
            .frame(width: 88, height: 88)
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .shadow(color: .black.opacity(0.18), radius: 12, x: 0, y: 4)

            Text("Sunder")
                .font(.system(size: 26, weight: .bold, design: .rounded))

            Text("Version \(version)")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(.quaternary, in: Capsule())
        }
        .padding(.top, 44)
        .padding(.bottom, 24)
    }

    private var descriptionAndAuthor: some View {
        VStack(spacing: 16) {
            Text("Separates a song into vocals and instrumental, entirely on-device, using a Mel-Band RoFormer model converted to Core ML.")
                .font(.system(size: 13))
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 8)

            HStack(spacing: 6) {
                Image(systemName: "person.fill")
                    .foregroundColor(.secondary)
                    .font(.caption)
                Text("Created by")
                    .foregroundColor(.secondary)
                Text("Matti Kjellstadli")
                    .fontWeight(.medium)
            }
            .font(.system(size: 13))
        }
        .padding(.horizontal, 40)
        .padding(.vertical, 20)
    }

    private var links: some View {
        HStack(spacing: 12) {
            LinkButton(
                title: "Website",
                systemImage: "globe",
                url: "https://mattikjellstadli.com/product/sunder"
            )
            LinkButton(
                title: "GitHub",
                systemImage: "chevron.left.forwardslash.chevron.right",
                url: "https://github.com/mattkje/Sunder"
            )
        }
    }

    /// Points at CreditsView for the full per-model attribution/license
    /// list and legal-use notice -- kept short here so About stays a quick
    /// glance, not a wall of small print.
    private var modelAttribution: some View {
        VStack(spacing: 8) {
            Button {
                showCredits = true
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 9))
                    Text("Model Credits & Licenses")
                    Image(systemName: "chevron.right")
                        .font(.system(size: 8))
                }
            }
            .buttonStyle(.plain)
            .foregroundColor(.secondary)

            // Not legal advice -- just makes clear where responsibility for
            // input audio sits, since this app can process any file a user
            // gives it, including copyrighted commercial recordings.
            Text("Only use Sunder on audio you own or have the rights to process. You're responsible for how you use its output.")
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
        .font(.system(size: 9.5))
        .foregroundColor(.secondary)
        .padding(.bottom, 14)
        .sheet(isPresented: $showCredits) { CreditsView() }
    }
}

private struct LinkButton: View {
    let title: String
    let systemImage: String
    let url: String

    @State private var isHovered = false

    var body: some View {
        Button {
            guard let u = URL(string: url) else { return }
            #if os(macOS)
            NSWorkspace.shared.open(u)
            #else
            UIApplication.shared.open(u)
            #endif
        } label: {
            Label(title, systemImage: systemImage)
                .font(.system(size: 13, weight: .medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(isHovered ? Color.accentColor.opacity(0.15) : Color.secondary.opacity(0.1))
                )
                .foregroundColor(isHovered ? .accentColor : .primary)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .animation(.easeInOut(duration: 0.15), value: isHovered)
    }
}
