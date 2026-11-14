import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.scenePhase) private var scenePhase
    @State private var showModels = false
    @State private var showSettings = false
    @State private var showLogs = false
    @State private var importImages = false
    @State private var deletion: Conversation?
    @State private var visibility = NavigationSplitViewVisibility.all
    var body: some View {
        NavigationSplitView(columnVisibility: $visibility) {
            List {
                Button { Task { await store.newChat() } } label: { Label("New chat", systemImage: "square.and.pencil") }
                    .disabled(!store.ready || store.busy || store.switching)
                Section("Chats") {
                    ForEach(store.chats) { chat in
                        Button { Task { await store.selectChat(chat.id) } } label: {
                            HStack {
                                Text(chat.title).lineLimit(1)
                                Spacer()
                                if store.activeID == chat.id { Image(systemName: "checkmark").foregroundStyle(.tint) }
                            }.contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(store.busy || store.switching)
                        .contextMenu { Button("Delete chat", role: .destructive) { deletion = chat }.disabled(store.busy) }
                    }
                }
                Section("Models") {
                    ForEach(store.models.filter(\.local)) { model in
                        Button { Task { await store.loadModel(model.id) } } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(model.name).lineLimit(1)
                                    Text(model.sizeLabel).font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if store.loadingModel == model.id { ProgressView().controlSize(.small) }
                                else if store.loadedModel == model.id { Image(systemName: "checkmark.circle.fill").foregroundStyle(.tint) }
                            }
                        }.disabled(store.busy || store.switching || !model.fits)
                    }
                    Button { showModels = true } label: { Label("Model library", systemImage: "square.stack.3d.up") }
                }
            }
            .navigationTitle("Lisa")
            .navigationSplitViewColumnWidth(min: 210, ideal: 250, max: 340)
        } detail: {
            VStack(spacing: 0) {
                transcript
                Divider()
                composer.padding()
            }
            .navigationTitle(store.current?.title ?? "Lisa")
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    Button { showModels = true } label: { Label("Models", systemImage: "cube") }
                    Button { showSettings = true } label: { Label("Settings", systemImage: "slider.horizontal.3") }
                    Button { showLogs = true } label: { Label("Logs", systemImage: "text.alignleft") }
                }
            }
        }
        .task { await store.start() }
        .onChange(of: scenePhase) { _, phase in Task { await store.sceneChanged(phase) } }
        .onChange(of: store.draft) { _, _ in store.scheduleSave() }
        .sheet(isPresented: $showModels) { ModelLibrary().environmentObject(store) }
        .sheet(isPresented: $showSettings) { SettingsView().environmentObject(store) }
        .sheet(isPresented: $showLogs) { LogView().environmentObject(store) }
        .sheet(isPresented: $store.showBrowser, onDismiss: { store.browser.cancel() }) { BrowserView(controller: store.browser) }
        .fileImporter(isPresented: $importImages, allowedContentTypes: [.png, .jpeg, .webP], allowsMultipleSelection: true) { result in
            switch result {
            case .success(let urls): Task { await store.importImages(urls) }
            case .failure(let error): store.error = error.localizedDescription
            }
        }
        .alert("Lisa", isPresented: Binding(get: { store.error != nil }, set: { if !$0 { store.error = nil } })) {
            Button("OK") { store.error = nil }
        } message: { Text(store.error ?? "") }
        .confirmationDialog("Delete this chat?", isPresented: Binding(get: { deletion != nil }, set: { if !$0 { deletion = nil } }), titleVisibility: .visible) {
            Button("Delete chat", role: .destructive) { if let id = deletion?.id { Task { await store.deleteChat(id) } }; deletion = nil }
        } message: { Text("The saved conversation and its copied attachments will be deleted.") }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if store.current?.turns.isEmpty ?? true {
                    ContentUnavailableView {
                        Label("How can Lisa help?", systemImage: "sparkles")
                    } description: {
                        Text(store.loadedModel == nil ? "Choose a downloaded model, or visit the model library." : "Your model runs locally on this device.")
                    } actions: {
                        if store.loadedModel == nil { Button("Choose a model") { showModels = true } }
                    }.padding(.top, 80)
                }
                LazyVStack(alignment: .leading, spacing: 28) {
                    ForEach(Array((store.current?.turns ?? []).enumerated()), id: \.offset) { _, turn in
                        VStack(alignment: .leading, spacing: 16) {
                            HStack {
                                Spacer(minLength: 40)
                                VStack(alignment: .leading, spacing: 8) {
                                    ForEach(turn.images, id: \.self) { LocalImage(path: $0) }
                                    Text(turn.user).textSelection(.enabled)
                                }
                                .padding(14).background(.quaternary, in: RoundedRectangle(cornerRadius: 18))
                            }
                            HStack(alignment: .top, spacing: 12) {
                                Image(systemName: "sparkles").foregroundStyle(.tint).padding(.top, 3)
                                VStack(alignment: .leading, spacing: 10) {
                                    if turn.assistant.isEmpty && !turn.complete { ProgressView().controlSize(.small) }
                                    else { MarkdownMessage(text: turn.assistant) }
                                    if let image = turn.generated_image {
                                        LocalImage(path: image)
                                        ShareLink(item: URL(fileURLWithPath: image)) { Label("Share image", systemImage: "square.and.arrow.up") }
                                    }
                                }.frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }.frame(maxWidth: 800).padding(24).frame(maxWidth: .infinity)
            }
            .onChange(of: store.current?.turns.last?.assistant) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: store.current?.turns.count) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }
    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !store.attachments.isEmpty {
                ScrollView(.horizontal) {
                    HStack {
                        ForEach(store.attachments, id: \.self) { path in
                            HStack {
                                Text(URL(fileURLWithPath: path).lastPathComponent).lineLimit(1)
                                Button { store.attachments.removeAll { $0 == path } } label: { Image(systemName: "xmark.circle.fill") }
                            }.font(.caption).padding(8).background(.quaternary, in: Capsule())
                        }
                    }
                }
            }
            HStack(alignment: .bottom, spacing: 12) {
                Button { importImages = true } label: { Image(systemName: "paperclip") }
                    .help("Attach images").disabled(!store.supportsImages || store.busy || store.imageMode)
                TextField(store.imageMode ? "Describe an image" : "Message Lisa", text: $store.draft, axis: .vertical)
                    .textFieldStyle(.plain).lineLimit(1...6).onSubmit { Task { await store.send() } }
                    .disabled(!store.ready || store.busy)
                VoiceButton(controller: store.speech, disabled: !store.ready || store.busy) { store.draft = $0 } onError: { store.error = $0 }
                if store.busy {
                    Button { Task { await store.cancel() } } label: { Image(systemName: "stop.circle.fill").font(.title2) }
                        .help("Stop generation").disabled(store.loadingModel != nil)
                } else {
                    Button { Task { await store.send() } } label: { Image(systemName: "arrow.up.circle.fill").font(.title2) }
                        .help("Send message").disabled(!store.canSend)
                }
            }.padding(12).background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 16))
            Text(store.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
        }.frame(maxWidth: 800).frame(maxWidth: .infinity)
    }
}

struct LocalImage: View {
    let path: String
    var body: some View {
        #if os(macOS)
        if let image = NSImage(contentsOfFile: path) { Image(nsImage: image).resizable().scaledToFit().frame(maxWidth: 480, maxHeight: 360).clipShape(RoundedRectangle(cornerRadius: 12)) }
        #else
        if let image = UIImage(contentsOfFile: path) { Image(uiImage: image).resizable().scaledToFit().frame(maxWidth: 480, maxHeight: 360).clipShape(RoundedRectangle(cornerRadius: 12)) }
        #endif
    }
}

struct MarkdownMessage: View {
    let text: String
    var body: some View {
        let sections = text.components(separatedBy: "```")
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(sections.enumerated()), id: \.offset) { index, section in
                if index.isMultiple(of: 2) {
                    Text((try? AttributedString(markdown: section, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(section))
                        .textSelection(.enabled)
                } else {
                    let code = section.split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false).dropFirst().joined(separator: "\n")
                    ScrollView(.horizontal) { Text(code).font(.system(.body, design: .monospaced)).textSelection(.enabled).padding(12) }
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                }
            }
        }
    }
}

struct ModelLibrary: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            List(store.models) { model in
                VStack(alignment: .leading, spacing: 8) {
                    HStack { Text(model.name).font(.headline); Spacer(); Text(model.sizeLabel).font(.caption).foregroundStyle(.secondary) }
                    Text(model.detail).font(.subheadline).foregroundStyle(.secondary)
                    if !model.fits { Label("Exceeds this device’s memory budget", systemImage: "memorychip").font(.caption).foregroundStyle(.secondary) }
                    if store.downloading == model.id { ProgressView(value: store.downloadProgress) }
                    else if model.local {
                        Button(store.loadedModel == model.id ? "Loaded" : "Load model") { Task { await store.loadModel(model.id); dismiss() } }
                            .disabled(!model.fits || store.busy || store.loadedModel == model.id)
                    } else {
                        Button("Download") { Task { await store.download(model.id) } }
                            .disabled(!store.ready || store.downloading != nil || !model.fits)
                    }
                }.padding(.vertical, 8)
            }
            .navigationTitle("Model library")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }.frame(idealWidth: 600, idealHeight: 640)
    }
}

struct SettingsView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var address = "127.0.0.1:8080"
    var body: some View {
        NavigationStack {
            Form {
                Section("Conversation") {
                    Toggle("Allow web browsing", isOn: $store.webEnabled)
                    Toggle("Speak replies", isOn: $store.speakReplies)
                    Text("System instructions").font(.headline)
                    TextEditor(text: $store.system).frame(minHeight: 110).disabled(store.busy || !(store.current?.turns.isEmpty ?? true))
                    Text("System instructions apply from the first message of a new chat.").font(.caption).foregroundStyle(.secondary)
                }
                Section("Local API server") {
                    TextField("Listen address", text: $address).disabled(store.serverURL != nil)
                    if let url = store.serverURL { Text(url).textSelection(.enabled) }
                    Button(store.serverURL == nil ? "Start server" : "Stop server") { Task { await store.setServer(store.serverURL == nil, address: address) } }
                        .disabled(store.loadedModel == nil || store.busy)
                    #if os(iOS)
                    Text("The server stops when Lisa enters the background.").font(.caption).foregroundStyle(.secondary)
                    #endif
                }
            }.formStyle(.grouped)
            .navigationTitle("Settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { Task { await store.save() }; dismiss() } } }
        }.frame(idealWidth: 520, idealHeight: 540)
    }
}
struct LogView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            ScrollView { Text(store.logs.joined(separator: "\n")).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding() }
                .navigationTitle("Activity")
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }.frame(idealWidth: 640, idealHeight: 500)
    }
}
