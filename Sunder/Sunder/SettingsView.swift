import SwiftUI

struct SettingsView: View {
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue
    @State private var downloader = ModelDownloader.shared
    #if os(iOS)
    @Environment(\.dismiss) private var dismiss
    @State private var showAbout = false
    #endif

    private var quality: OutputQuality {
        OutputQuality(rawValue: qualityRaw) ?? .web
    }

    var body: some View {
        #if os(iOS)
        NavigationStack {
            form
                .navigationTitle("Settings")
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
        #else
        form
        #endif
    }

    private var form: some View {
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

            Section {
                ForEach(AIModel.allCases) { model in
                    modelRow(model)
                }
            } header: {
                Text("Models")
            } footer: {
                Text("Models are downloaded on demand, not bundled with the app, so they don't add to its download size until you use them.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            #if os(iOS)
            Section {
                Button("About Sunder") { showAbout = true }
            }
            #endif
        }
        .formStyle(.grouped)
        #if os(macOS)
        .frame(width: 460, height: 420)
        #else
        .sheet(isPresented: $showAbout) { AboutView() }
        #endif
    }

    @ViewBuilder
    private func modelRow(_ model: AIModel) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName).font(.body)
                Text(model.detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            modelAction(model)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func modelAction(_ model: AIModel) -> some View {
        switch downloader.state(for: model) {
        case .ready:
            HStack(spacing: 8) {
                if let sizeMB = downloader.installedSizeMB(for: model) {
                    Text("\(sizeMB) MB")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Button("Delete", role: .destructive) {
                    downloader.delete(model)
                }
                .font(.caption)
            }
        case .notDownloaded:
            Button("Download (~\(model.approximateSizeMB) MB)") {
                downloader.download(model)
            }
            .font(.caption)
        case .downloading(let progress):
            HStack(spacing: 8) {
                ProgressView(value: progress).frame(width: 100)
                Button("Cancel") { downloader.cancelDownload(model) }
                    .font(.caption)
            }
        case .failed:
            Button("Retry") { downloader.download(model) }
                .font(.caption)
        }
    }
}

#Preview {
    SettingsView()
}
