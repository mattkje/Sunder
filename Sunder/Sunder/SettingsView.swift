import SwiftUI

struct SettingsView: View {
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue

    private var quality: OutputQuality {
        OutputQuality(rawValue: qualityRaw) ?? .web
    }

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 10) {
                    Slider(
                        value: Binding(
                            get: { Double(qualityRaw) },
                            set: { qualityRaw = Int($0.rounded()) }
                        ),
                        in: 0...Double(OutputQuality.allCases.count - 1),
                        step: 1
                    )
                    HStack {
                        Text(quality.label).font(.headline)
                        Spacer()
                        Text(quality.detail).font(.caption).foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 4)
            } header: {
                Text("Output Quality")
            } footer: {
                Text("Higher quality means larger files. Web is a good default for listening on a computer or phone.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 420, height: 220)
    }
}

#Preview {
    SettingsView()
}
