import SwiftUI
import AppKit

final class AboutWindowController: NSWindowController {
    static let shared = AboutWindowController()

    private init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 460),
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

private struct AboutView: View {
    @Environment(\.colorScheme) private var colorScheme

    private let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
    private let build   = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"

    var body: some View {
        ZStack {

            LinearGradient(
                colors: colorScheme == .dark
                    ? [Color(nsColor: .windowBackgroundColor), Color(nsColor: .controlBackgroundColor)]
                    : [Color(nsColor: .controlBackgroundColor), Color(nsColor: .windowBackgroundColor)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                VStack(spacing: 12) {
                    Group {
                        Image(nsImage: NSApp.applicationIconImage)
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

                Divider().padding(.horizontal, 32)

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

                Divider().padding(.horizontal, 32)

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
                .padding(.vertical, 20)

                Spacer()

                // Model license footer
                HStack(spacing: 4) {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 9))
                    Text("Model: becruily/mel-band-roformer-deux")
                    Text("·")
                    Text("CC BY-NC 4.0")
                }
                .font(.system(size: 10))
                .foregroundColor(.secondary)
                .padding(.bottom, 16)
            }
        }
        .frame(width: 400, height: 460)
    }
}

private struct LinkButton: View {
    let title: String
    let systemImage: String
    let url: String

    @State private var isHovered = false

    var body: some View {
        Button {
            if let u = URL(string: url) { NSWorkspace.shared.open(u) }
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
