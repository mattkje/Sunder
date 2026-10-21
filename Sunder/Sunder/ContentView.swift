import SwiftUI

/// The sidebar's full set of destinations: every `AppSection`, plus Settings
/// -- which used to be its own window on macOS (`Settings { SettingsView() }`
/// in SunderApp.swift) and a sheet on iOS. Both were an extra window/layer
/// for what's a single small screen, so it's a normal destination now like
/// any other, just visually separated at the bottom of the sidebar since
/// it's app configuration rather than a workflow.
private enum SidebarDestination: Hashable {
    case section(AppSection)
    case mixer
    case spatialize
    #if os(macOS)
    case voice
    #endif
    case settings
}

struct ContentView: View {
    @State private var selection: SidebarDestination? = .section(.vocalRemover)
    @State private var queue = SeparationQueue.shared
    @State private var updateChecker = UpdateChecker.shared

    var body: some View {
        NavigationSplitView {
            List(selection: $selection) {
                ForEach(AppSection.allCases) { section in
                    Label(section.displayName, systemImage: section.icon)
                        .tag(SidebarDestination.section(section))
                }
                Label("Mixer", systemImage: "slider.horizontal.3")
                    .tag(SidebarDestination.mixer)
                Label("Spatialize", systemImage: "move.3d")
                    .tag(SidebarDestination.spatialize)
                #if os(macOS)
                Label("Voice Changer", systemImage: "waveform.and.person.filled")
                    .tag(SidebarDestination.voice)
                #endif
            }
            .navigationTitle("Sunder")
            .disabled(queue.isRunning)
            .safeAreaInset(edge: .bottom) {
                VStack(spacing: 0) {
                    Divider()
                    settingsRow
                    Divider()
                    Text("By Matti Kjellstadli")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .padding(8)
                        .frame(maxWidth: .infinity)
                }
                .disabled(queue.isRunning)
            }
        } detail: {
            switch selection {
            case .settings:
                SettingsView()
            case .mixer:
                MixerScreen()
            case .spatialize:
                SpatializeScreen()
            #if os(macOS)
            case .voice:
                VoiceScreen()
            #endif
            default:
                SeparationScreen(section: currentSection)
                    .id(selection)
            }
        }
        #if os(macOS)
        .frame(minWidth: 620, minHeight: 420)
        #endif
        .task { await updateChecker.check() }
        .sheet(isPresented: Binding(
            get: { updateChecker.availableUpdate != nil },
            set: { if !$0 { updateChecker.dismiss() } }
        )) {
            if let update = updateChecker.availableUpdate {
                UpdateAvailableSheet(update: update)
            }
        }
    }

    private var currentSection: AppSection {
        if case .section(let section) = selection { return section }
        return .vocalRemover
    }

    private var settingsRow: some View {
        Button {
            selection = .settings
        } label: {
            Label("Settings", systemImage: "gearshape")
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(selection == .settings ? Color.accentColor : Color.clear)
                )
                .foregroundStyle(selection == .settings ? Color.white : Color.primary)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }
}

#Preview {
    ContentView()
}
