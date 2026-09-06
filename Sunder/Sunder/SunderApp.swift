import SwiftUI

@main
struct SunderApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Sunder") {
                    AboutWindowController.shared.showWindow()
                }
            }
        }

        Settings {
            SettingsView()
        }
    }
}
