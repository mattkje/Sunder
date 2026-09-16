import SwiftUI

struct SettingsView: View {
    @AppStorage(OutputQuality.storageKey) private var qualityRaw = OutputQuality.web.rawValue
    @State private var downloader = ModelDownloader.shared
    @State private var queue = SeparationQueue.shared
    @State private var updateChecker = UpdateChecker.shared
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.openURL) private var openURL
    #if os(iOS)
    @State private var showAbout = false
    #endif

    private var quality: OutputQuality {
        OutputQuality(rawValue: qualityRaw) ?? .web
    }

    private var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
    }

    var body: some View {
        ZStack {
            AppTheme.backgroundGradient(for: colorScheme).ignoresSafeArea()

            Form {
                Section {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text(quality.label).font(.headline)
                            Spacer()
                            Text(quality.detail).font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(Color.accentColor.opacity(0.1))
                        )

                        Slider(
                            value: Binding(
                                get: { Double(qualityRaw) },
                                set: { qualityRaw = Int($0.rounded()) }
                            ),
                            in: 0...Double(OutputQuality.allCases.count - 1),
                            step: 1
                        )
                    }
                    .padding(.vertical, 4)
                } header: {
                    Label("Output Quality", systemImage: "dial.high")
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
                    Label("Models", systemImage: "shippingbox")
                } footer: {
                    Text(queue.isRunning
                        ? "Locked while a separation is running."
                        : "Models are downloaded on demand, not bundled with the app, so they don't add to its download size until you use them.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Version \(appVersion)").font(.body)
                            updateStatusLabel
                        }
                        Spacer()
                        updateAction
                    }
                    .padding(.vertical, 2)
                } header: {
                    Label("Updates", systemImage: "arrow.down.circle")
                }

                #if os(iOS)
                Section {
                    Button("About Sunder") { showAbout = true }
                }
                #endif
            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
            .disabled(queue.isRunning)
        }
        .navigationTitle("Settings")
        #if os(iOS)
        .sheet(isPresented: $showAbout) { AboutView() }
        #endif
    }

    @ViewBuilder
    private func modelRow(_ model: AIModel) -> some View {
        HStack(spacing: 12) {
            Image(systemName: model.icon)
                .foregroundStyle(.background)
                .frame(width: 28, height: 28)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(.primary)
                )

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
                statusBadge("Ready" + (downloader.installedSizeMB(for: model).map { " · \($0) MB" } ?? ""), color: .green)
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

    @ViewBuilder
    private var updateStatusLabel: some View {
        if updateChecker.isChecking {
            Text("Checking…").font(.caption).foregroundStyle(.secondary)
        } else if let update = updateChecker.availableUpdate {
            Text("Update available: \(update.version)").font(.caption).foregroundStyle(.orange)
        } else if let error = updateChecker.lastError {
            Text(error).font(.caption).foregroundStyle(.red)
        } else if updateChecker.hasChecked {
            Text("Up to date").font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var updateAction: some View {
        if let update = updateChecker.availableUpdate {
            Button("View") { openURL(update.htmlURL) }
                .font(.caption)
        } else {
            Button("Check for Updates") {
                Task { await updateChecker.check() }
            }
            .font(.caption)
            .disabled(updateChecker.isChecking)
        }
    }

    private func statusBadge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Capsule().fill(color.opacity(0.15)))
    }
}

#Preview {
    SettingsView()
}
