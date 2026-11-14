import SwiftUI

@main struct LisaApp: App {
    @StateObject private var store = AppStore()
    var body: some Scene {
        #if os(macOS)
        Window("Lisa", id: "main") {
            ContentView().environmentObject(store)
                .frame(minWidth: 720, minHeight: 520)
        }
        .defaultSize(width: 1100, height: 760)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { Task { await store.newChat() } }
                    .keyboardShortcut("n").disabled(store.busy)
            }
        }
        #else
        WindowGroup { ContentView().environmentObject(store) }
        #endif
    }
}
