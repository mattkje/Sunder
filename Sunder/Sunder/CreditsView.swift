import SwiftUI
#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// Full model attribution + legal-use notice, split out of AboutView's
/// bottom section so it's an easy-to-find dedicated page (like uvronline's
/// own credits page) rather than small print at the bottom of About.
/// Presented as a sheet from AboutView on both platforms.
struct CreditsView: View {
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()

                Form {
                    Section {
                        ForEach(AIModel.allCases) { model in
                            modelRow(model)
                        }
                    } header: {
                        Text("Models")
                    } footer: {
                        Text("Some checkpoints are community-trained and mirrored -- not hosted by their original authors, whose original accounts are no longer available.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Section {
                        Text("Only use Sunder on audio you own or have the rights to process. Sunder isn't affiliated with the artists, labels, or rights holders of any audio you process with it -- you're responsible for how you use its output.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    } header: {
                        Text("Responsible Use")
                    }

                    Section {
                        Button {
                            openURL("https://github.com/mattkje/Sunder")
                        } label: {
                            Label("Sunder on GitHub", systemImage: "chevron.left.forwardslash.chevron.right")
                        }
                    }
                }
                .formStyle(.grouped)
                .scrollContentBackground(.hidden)
            }
            .navigationTitle("Credits")
            #if os(iOS)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            #endif
        }
        #if os(macOS)
        .frame(width: 460, height: 480)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            }
        }
        #endif
    }

    @ViewBuilder
    private func modelRow(_ model: AIModel) -> some View {
        let attribution = model.attribution
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName).font(.body)
                Text(attribution.author).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let license = attribution.license {
                Text(license)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Color.secondary.opacity(0.12)))
            }
        }
        .padding(.vertical, 2)
    }

    private func openURL(_ string: String) {
        guard let url = URL(string: string) else { return }
        #if os(macOS)
        NSWorkspace.shared.open(url)
        #else
        UIApplication.shared.open(url)
        #endif
    }
}

#Preview {
    CreditsView()
}
