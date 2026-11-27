import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.scenePhase) private var scenePhase
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    #endif
    @State private var showChats = false
    @State private var chatSearch = ""
    @State private var showChatSearch = false
    @State private var hoveredChat: String?
    @FocusState private var composerFocused: Bool
    @FocusState private var chatSearchFocused: Bool
    @State private var showModels = false
    @State private var showSettings = false
    @State private var showLogs = false
    @State private var importImages = false
    @State private var deletion: Conversation?
    @State private var visibility = NavigationSplitViewVisibility.all
    @State private var compactColumn = NavigationSplitViewColumn.detail
    var body: some View {
        navigation
        .task { await store.start() }
        .onChange(of: scenePhase) { _, phase in Task { await store.sceneChanged(phase) } }
        #if os(iOS)
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            Task { await store.memoryWarning() }
        }
        #endif
        .onChange(of: store.draft) { _, _ in store.scheduleSave() }
        .sheet(isPresented: $showModels) { ModelLibrary().environmentObject(store) }
        .sheet(isPresented: $showSettings) { SettingsView().environmentObject(store) }
        .sheet(isPresented: $showLogs) { LogView().environmentObject(store) }
        .sheet(isPresented: $store.showBrowser, onDismiss: { if store.browser.needsInput { store.browser.cancel() } }) { BrowserView(controller: store.browser) }
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

    @ViewBuilder private var navigation: some View {
        #if os(iOS)
        if horizontalSizeClass == .compact {
            GeometryReader { geometry in
                let drawerWidth = min(340, geometry.size.width * 0.78)
                ZStack(alignment: .leading) {
                    Color(uiColor: .systemBackground).ignoresSafeArea()
                    mobileSidebar
                        .frame(width: drawerWidth)
                        .accessibilityHidden(!showChats)
                    NavigationStack {
                        conversation
                            .toolbar(.hidden, for: .navigationBar)
                            .safeAreaInset(edge: .top, spacing: 0) { mobileChatHeader }
                    }
                    .background {
                        RoundedRectangle(cornerRadius: showChats ? 32 : 0)
                            .fill(Color(uiColor: .systemBackground)).ignoresSafeArea()
                    }
                    .overlay {
                        if showChats {
                            Color.black.opacity(0.08)
                                .ignoresSafeArea()
                                .contentShape(Rectangle())
                                .onTapGesture { showConversation() }
                                .accessibilityAddTraits(.isButton)
                                .accessibilityLabel("Close chats")
                        }
                    }
                    .mask {
                        RoundedRectangle(cornerRadius: showChats ? 32 : 0, style: .continuous)
                            .ignoresSafeArea()
                    }
                    .shadow(color: .black.opacity(showChats ? 0.12 : 0), radius: 12, x: -4, y: 0)
                    .offset(x: showChats ? drawerWidth : 0)
                    .simultaneousGesture(sidebarGesture)
                }
                .mask { Rectangle().ignoresSafeArea() }
            }
        } else {
            wideNavigation
        }
        #else
        wideNavigation
        #endif
    }

    #if os(iOS)
    private var mobileChatHeader: some View {
        HStack(spacing: 12) {
            Button {
                composerFocused = false
                withAnimation(.easeInOut(duration: 0.25)) { showChats.toggle() }
            } label: {
                Image(systemName: "sidebar.left").font(.title3)
                    .frame(width: 44, height: 44)
                    .lisaGlass(in: Circle(), interactive: true)
            }.accessibilityLabel("Open chats")
            Spacer(minLength: 0)
            Button { showModels = true } label: {
                HStack(spacing: 5) {
                    if store.loadingModel != nil { ProgressView().controlSize(.small) }
                    else if store.downloading != nil { ProgressView(value: store.downloadProgress).frame(width: 24) }
                    Text(store.models.first { $0.id == (store.loadedModel ?? store.current?.model) }?.name ?? "Lisa")
                        .font(.subheadline.weight(.semibold)).lineLimit(1)
                    Image(systemName: "chevron.down").font(.caption2)
                }
                .padding(.horizontal, 16).frame(height: 44)
                .lisaGlass(in: Capsule(), interactive: true)
            }.accessibilityLabel("Choose model")
            Spacer(minLength: 0)
            Button { Task { await store.newChat() } } label: {
                Image(systemName: "square.and.pencil").font(.title3)
                    .frame(width: 44, height: 44)
                    .lisaGlass(in: Circle(), interactive: true)
            }
            .accessibilityLabel("New chat")
            .disabled(!store.ready || store.busy || store.switching)
        }
        .buttonStyle(.plain).foregroundStyle(.primary)
        .padding(.horizontal, 16).padding(.top, 4).padding(.bottom, 8)
    }

    private var sidebarGesture: some Gesture {
        DragGesture(minimumDistance: 24).onEnded { value in
            let horizontal = value.translation.width
            guard abs(horizontal) > abs(value.translation.height) * 1.5 else { return }
            if !showChats && horizontal > 70 {
                composerFocused = false
                withAnimation(.easeInOut(duration: 0.25)) { showChats = true }
            } else if showChats && horizontal < -70 {
                showConversation()
            }
        }
    }

    private var mobileSidebar: some View {
        VStack(spacing: 0) {
            List {
                Group {
                    mobileMenuRow("New chat", icon: "square.and.pencil") {
                        Task { await store.newChat(); showConversation() }
                    }
                    .disabled(!store.ready || store.busy || store.switching)
                    mobileMenuRow("Models", icon: "square.stack.3d.up") { showConversation(); showModels = true }
                    mobileMenuRow("Settings", icon: "slider.horizontal.3") { showConversation(); showSettings = true }
                    mobileMenuRow("Activity", icon: "terminal") { showConversation(); showLogs = true }
                    mobileMenuRow("Search", icon: "magnifyingglass") { toggleChatSearch() }
                    if showChatSearch {
                        TextField("Search chats", text: $chatSearch)
                            .textFieldStyle(.plain).focused($chatSearchFocused).padding(12)
                            .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                    }
                    Color.clear.frame(height: 16).accessibilityHidden(true)
                }
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)
                .listRowInsets(EdgeInsets(top: 9, leading: 24, bottom: 9, trailing: 24))
                ForEach(store.visibleChats.filter { chatSearch.isEmpty || $0.title.localizedCaseInsensitiveContains(chatSearch) }) { chat in
                    Button { Task { await store.selectChat(chat.id); showConversation() } } label: {
                        Text(chat.title).font(.body).lineLimit(1)
                            .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .disabled(store.busy || store.switching)
                    .listRowInsets(EdgeInsets(top: 8, leading: 24, bottom: 8, trailing: 24))
                    .listRowSeparator(.hidden)
                    .listRowBackground(Color.clear)
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        Button(role: .destructive) { deletion = chat } label: { Label("Delete", systemImage: "trash") }
                            .disabled(store.busy || store.switching)
                    }
                    .contextMenu { Button("Delete chat", role: .destructive) { deletion = chat }.disabled(store.busy || store.switching) }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .scrollDismissesKeyboard(.interactively)
        }
        .padding(.top, 8)
        .foregroundStyle(.primary)
        .background(Color(uiColor: .systemBackground))
    }

    private func mobileMenuRow(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 16) {
                Image(systemName: icon).font(.title3).frame(width: 24)
                Text(title).font(.body)
            }.frame(maxWidth: .infinity, minHeight: 36, alignment: .leading)
        }.buttonStyle(.plain)
    }
    #endif

    private var wideNavigation: some View {
        NavigationSplitView(columnVisibility: $visibility, preferredCompactColumn: $compactColumn) {
            sidebar
        } detail: {
            conversation
        }
    }

    private func showConversation() {
        withAnimation(.easeInOut(duration: 0.2)) {
            showChats = false
            compactColumn = .detail
        }
    }

    private var sidebar: some View {
        VStack(spacing: 12) {
            VStack(spacing: 2) {
                sidebarMenuRow("New chat", icon: "square.and.pencil") {
                    Task { await store.newChat(); showConversation() }
                }
                .disabled(!store.ready || store.busy || store.switching)
                sidebarMenuRow("Models", icon: "square.stack.3d.up") { showModels = true }
                sidebarMenuRow("Settings", icon: "slider.horizontal.3") { showSettings = true }
                sidebarMenuRow("Activity", icon: "terminal") { showLogs = true }
                sidebarMenuRow("Search", icon: "magnifyingglass") { toggleChatSearch() }
                if showChatSearch {
                    TextField("Search chats", text: $chatSearch)
                        .textFieldStyle(.plain)
                        .focused($chatSearchFocused)
                        .font(.system(size: 13))
                        .padding(10)
                        .background(.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
                        #if os(macOS)
                        .onExitCommand {
                            showChatSearch = false; chatSearch = ""; chatSearchFocused = false
                        }
                        #endif
                }
            }.padding(.horizontal, 20).padding(.top, 12)
            #if os(macOS)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    Text("Chats")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                        .padding(.horizontal, 8).padding(.bottom, 6)
                    ForEach(store.visibleChats.filter { chatSearch.isEmpty || $0.title.localizedCaseInsensitiveContains(chatSearch) }) { chat in
                        Button { Task { await store.selectChat(chat.id); showConversation() } } label: {
                            Text(chat.title)
                                .font(.system(size: 13, weight: .regular))
                                .foregroundStyle(.primary)
                                .lineLimit(1).truncationMode(.tail)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 8).padding(.vertical, 7)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .background(
                            .primary.opacity(store.activeID == chat.id ? 0.08 : hoveredChat == chat.id ? 0.04 : 0),
                            in: RoundedRectangle(cornerRadius: 8)
                        )
                        .onHover { hoveredChat = $0 ? chat.id : nil }
                        .help(chat.title)
                        .disabled(store.busy || store.switching)
                        .contextMenu {
                            Button("Delete chat", role: .destructive) { deletion = chat }
                                .disabled(store.busy || store.switching)
                        }
                    }
                }
                .padding(.horizontal, 12).padding(.bottom, 8)
            }
            .padding(.top, 16)
            #else
            List {
                Section {
                    ForEach(store.visibleChats.filter { chatSearch.isEmpty || $0.title.localizedCaseInsensitiveContains(chatSearch) }) { chat in
                        Button { Task { await store.selectChat(chat.id); showConversation() } } label: {
                            Text(chat.title)
                                .font(.system(size: ChatLayout.sidebarFontSize))
                                .lineLimit(1)
                                .frame(maxWidth: .infinity, minHeight: ChatLayout.sidebarRowHeight, alignment: .leading)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(store.busy || store.switching)
                        .listRowSeparator(.hidden)
                        #if os(macOS)
                        .listRowInsets(EdgeInsets(top: 2, leading: 20, bottom: 2, trailing: 20))
                        #endif
                        .listRowBackground(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(.primary.opacity(store.activeID == chat.id ? 0.07 : 0))
                        )
                        .contextMenu {
                            Button("Delete chat", role: .destructive) { deletion = chat }
                                .disabled(store.busy || store.switching)
                        }
                        #if os(iOS)
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            Button(role: .destructive) { deletion = chat } label: { Label("Delete", systemImage: "trash") }
                                .disabled(store.busy || store.switching)
                        }
                        #endif
                    }
                }
            }
            .listStyle(.plain)
            #if os(macOS)
            .environment(\.defaultMinListRowHeight, 26)
            #endif
            .scrollContentBackground(.hidden)
            #endif
        }
        .navigationTitle("")
        .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 380)
    }

    private func toggleChatSearch() {
        withAnimation(.easeInOut(duration: 0.15)) { showChatSearch.toggle() }
        if !showChatSearch { chatSearch = "" }
        chatSearchFocused = showChatSearch
    }

    private func sidebarMenuRow(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon).frame(width: 20)
                Text(title)
            }
            .font(.system(size: 14))
            .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
            .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }

    private var modelSelector: some View {
        Button { showModels = true } label: {
            HStack(spacing: 8) {
                if store.loadingModel != nil { ProgressView().controlSize(.small) }
                Text(store.models.first { $0.id == (store.loadedModel ?? store.current?.model) }?.name ?? "Lisa")
                    .font(.system(size: 15, weight: .medium)).lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8).frame(minHeight: 36)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain).help("Choose model")
    }

    private var conversation: some View {
        VStack(spacing: 0) {
            transcript
            if store.loadingModel != nil, let phase = store.loadStatus {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(phase).font(.callout).foregroundStyle(.secondary)
                }.padding(.horizontal, 24).padding(.top, 8)
            }
            composer.padding(.horizontal, 24).padding(.top, 8)
                #if os(macOS)
                .padding(.bottom, 16)
                #else
                .padding(.bottom, 8)
                #endif
        }
        #if os(macOS)
        .navigationTitle("")
        .font(.system(size: 14))
        #else
        .navigationTitle(store.current.flatMap { $0.turns.isEmpty ? nil : $0.title } ?? "")
        #endif
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .background {
            Color(uiColor: showChats ? .secondarySystemBackground : .systemBackground)
                .ignoresSafeArea()
        }
        #endif
        .toolbar {
            #if os(macOS)
            if #available(macOS 26.0, *) {
                ToolbarItem(placement: .principal) { modelSelector }
                    .sharedBackgroundVisibility(.hidden)
            } else {
                ToolbarItem(placement: .principal) { modelSelector }
            }
            #endif
            ToolbarItemGroup(placement: .primaryAction) {
                #if os(iOS)
                if horizontalSizeClass == .compact {
                    Button { Task { await store.newChat() } } label: { Image(systemName: "square.and.pencil") }
                        .accessibilityLabel("New chat")
                        .disabled(!store.ready || store.busy || store.switching)
                } else {
                    conversationActions
                }
                #else
                Button { Task { await store.newChat() } } label: { Image(systemName: "square.and.pencil") }
                    .help("New chat (⌘N)")
                    .disabled(!store.ready || store.busy || store.switching)
                #endif
            }
        }
    }

    private var conversationActions: some View {
        Group {
            Button { showModels = true } label: { Label("Models", systemImage: "cube") }
            Button { showSettings = true } label: { Label("Settings", systemImage: "slider.horizontal.3") }
            Button { showLogs = true } label: { Label("Activity", systemImage: "terminal") }
        }
    }

    private var transcript: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: ChatLayout.turnSpacing) {
                let turns = store.current?.turns ?? []
                ForEach(Array(turns.enumerated()), id: \.offset) { index, turn in
                    // Only the last turn of a busy conversation can still be producing.
                    let live = store.busy && store.loadingModel == nil && index == turns.count - 1
                    VStack(alignment: .leading, spacing: ChatLayout.replySpacing) {
                        HStack {
                            Spacer(minLength: 40)
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(turn.images, id: \.self) { LocalImage(path: $0) }
                                Text(turn.user).textSelection(.enabled)
                            }
                            .padding(ChatLayout.bubblePadding).background(.quaternary, in: RoundedRectangle(cornerRadius: 18))
                            #if os(macOS)
                            .frame(maxWidth: 520, alignment: .trailing)
                            #endif
                        }
                        HStack(alignment: .top, spacing: 12) {
                            VStack(alignment: .leading, spacing: 10) {
                                if turn.assistant.isEmpty && !turn.complete {
                                    if live {
                                        HStack(spacing: 8) {
                                            ProgressView().controlSize(.small)
                                            if let status = store.replyStatus { Text(status).foregroundStyle(.secondary) }
                                        }
                                    } else {
                                        Label("Reply interrupted", systemImage: "exclamationmark.circle")
                                            .font(.callout).foregroundStyle(.secondary)
                                    }
                                }
                                else { MarkdownMessage(text: turn.assistant) }
                                if let image = turn.generated_image {
                                    GeneratedImageCard(path: image)
                                }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
            }.frame(maxWidth: ChatLayout.columnWidth).padding(24).frame(maxWidth: .infinity)
        }
        // Let layout resolve the scroll range before following content changes.
        // Short conversations stay at the top instead of overscrolling to a bottom sentinel.
        .defaultScrollAnchor(.bottom)
        .defaultScrollAnchor(.top, for: .alignment)
        .id(store.activeID)
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
            #if os(macOS)
            VStack(alignment: .leading, spacing: 12) {
                composerField.padding(.horizontal, 4)
                HStack(spacing: 8) {
                    attachmentButton
                    Spacer()
                    dictationButton
                    sendButton
                }
            }
            .padding(12)
            .buttonStyle(.plain)
            .lisaGlass(in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            #else
            HStack(alignment: .center, spacing: 8) {
                attachmentButton
                composerField
                dictationButton
                sendButton
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .lisaGlass(in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            #endif
        }.frame(maxWidth: ChatLayout.columnWidth).frame(maxWidth: .infinity)
    }

    private var composerField: some View {
        TextField(store.imageMode ? "Describe an image" : "Ask Lisa", text: $store.draft, axis: .vertical)
            .focused($composerFocused)
            .textFieldStyle(.plain).lineLimit(1...6).onSubmit { Task { await store.send() } }
            .frame(minHeight: 28)
            .disabled(!store.ready || store.busy)
    }

    private var attachmentButton: some View {
        Button { importImages = true } label: {
            Image(systemName: "plus").font(.title2).frame(width: 36, height: ChatLayout.composerControlHeight)
        }
        .help("Attach images").disabled(!store.supportsImages || store.busy || store.imageMode)
    }

    private var dictationButton: some View {
        VoiceButton(controller: store.speech, disabled: !store.ready || store.busy) { store.draft = $0 } onError: { store.error = $0 }
            .font(.title2).frame(width: 32, height: ChatLayout.composerControlHeight)
    }

    @ViewBuilder private var sendButton: some View {
        if store.busy {
            Button { Task { await store.cancel() } } label: {
                Image(systemName: "stop.circle.fill").font(.title).frame(width: 40, height: ChatLayout.composerControlHeight)
            }
            .help("Stop generation").disabled(store.loadingModel != nil)
        } else {
            Button { Task { await store.send() } } label: {
                Image(systemName: "arrow.up").font(.body.weight(.semibold))
                    .foregroundStyle(.white).frame(width: 32, height: 32)
                    .background(store.canSend ? Color.blue : Color.secondary.opacity(0.3), in: Circle())
                    .frame(width: 40, height: ChatLayout.composerControlHeight)
            }
            .help("Send message").disabled(!store.canSend)
        }
    }
}

private extension View {
    @ViewBuilder
    func lisaGlass<S: Shape>(in shape: S, tint: Color? = nil, interactive: Bool = false) -> some View {
        if #available(iOS 26.0, macOS 26.0, *) {
            glassEffect(.regular.tint(tint).interactive(interactive), in: shape)
        } else {
            background(.thinMaterial, in: shape)
                .background(tint ?? .clear, in: shape)
        }
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

/// A quiet tag beside a model's name: its family and quantization (filled)
/// or a feature (outlined).
struct ModelTag: View {
    let text: String
    var outlined = false
    var body: some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .lineLimit(1)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background {
                let shape = RoundedRectangle(cornerRadius: 4, style: .continuous)
                if outlined { shape.strokeBorder(.tertiary, lineWidth: 1) } else { shape.fill(.quaternary.opacity(0.6)) }
            }
    }
}

/// A model's tile: what it generates, text or images.
struct ModelKindMark: View {
    let generates: String
    var body: some View {
        RoundedRectangle(cornerRadius: 7, style: .continuous)
            .fill(.quaternary.opacity(0.6))
            .frame(width: 30, height: 30)
            .overlay(
                Image(systemName: generates == "image" ? "photo" : "text.bubble")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            )
            .accessibilityLabel(generates == "image" ? "Image generation" : "Text generation")
    }
}

/// The model library's sort orders.
enum ModelOrder: String, CaseIterable, Identifiable {
    case name, release, type
    var id: Self { self }
    var title: String {
        switch self {
        case .name: "Name"
        case .release: "Release"
        case .type: "Type"
        }
    }
    /// Name A to Z; newest release first; text models before image
    /// generators. Ties fall back to the name.
    func precedes(_ a: ModelChoice, _ b: ModelChoice) -> Bool {
        let byName = a.name.localizedStandardCompare(b.name) == .orderedAscending
        switch self {
        case .name: return byName
        case .release: return a.released != b.released ? a.released > b.released : byName
        case .type: return a.generates != b.generates ? a.generates == "text" : byName
        }
    }
}

struct ModelLibrary: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var generates = ""
    @State private var family = ""
    @State private var feature = ""
    @State private var order = ModelOrder.name
    /// One margin for the header, search, filters and rows.
    private let margin: CGFloat = 20
    private let mark: CGFloat = 30
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Models").font(.title2.weight(.semibold))
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, margin)
            .padding(.top, 18)
            .padding(.bottom, 12)
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search", text: $query).textFieldStyle(.plain)
                if !query.isEmpty {
                    Button { query = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(.quaternary.opacity(0.5)))
            .padding(.horizontal, margin)
            HStack(spacing: 10) {
                Picker("Generates", selection: $generates) {
                    Text("All").tag("")
                    Text("Text").tag("text")
                    Text("Image").tag("image")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                filterMenu("All families", selection: $family, options: families)
                filterMenu("All features", selection: $feature, options: features)
                Spacer()
                Picker("Sort", selection: $order) {
                    ForEach(ModelOrder.allCases) { Text("Sort: \($0.title)").tag($0) }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .help("Sort models")
            }
            .padding(.horizontal, margin)
            .padding(.vertical, 12)
            Divider()
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(shown) { model in
                        row(model)
                            .padding(.horizontal, margin)
                            .padding(.vertical, 9)
                        if model.id != shown.last?.id {
                            Divider().padding(.leading, margin + mark + 12)
                        }
                    }
                }
            }
            .overlay {
                if shown.isEmpty {
                    Text(fitting.isEmpty ? "No models fit the currently available memory." : "No models match your search.")
                        .font(.subheadline).foregroundStyle(.secondary).padding()
                }
            }
        }
        .frame(minWidth: 720, idealWidth: 820, minHeight: 420, idealHeight: 600)
        .task {
            do { try await store.refreshModels() }
            catch { store.error = error.localizedDescription }
        }
    }
    private var fitting: [ModelChoice] { store.models.filter(\.fits) }
    private var families: [String] {
        fitting.reduce(into: [String]()) { if !$0.contains($1.family) { $0.append($1.family) } }
    }
    private var features: [String] {
        fitting.flatMap(\.features).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
    }
    private func filterMenu(_ all: String, selection: Binding<String>, options: [String]) -> some View {
        Picker(all, selection: selection) {
            Text(all).tag("")
            Divider()
            ForEach(options, id: \.self) { Text($0).tag($0) }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .fixedSize()
    }
    /// Models passing the filters and matching every word of the search in
    /// their name, family, quantization, features or description.
    private var shown: [ModelChoice] {
        let words = query.lowercased().split(whereSeparator: \.isWhitespace)
        return fitting.filter { model in
            guard (generates.isEmpty || model.generates == generates)
                && (family.isEmpty || model.family == family)
                && (feature.isEmpty || model.features.contains(feature)) else { return false }
            let haystack = ([model.name, model.family, model.quantization, model.detail] + model.features)
                .joined(separator: " ").lowercased()
            return words.allSatisfy { haystack.contains($0) }
        }
        .sorted(by: order.precedes)
    }
    private func row(_ model: ModelChoice) -> some View {
        HStack(spacing: 12) {
            ModelKindMark(generates: model.generates)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    Text(model.name).font(.body.weight(.semibold)).lineLimit(1)
                    ModelTag(text: model.family)
                    ModelTag(text: model.quantization)
                    ForEach(model.features, id: \.self) { ModelTag(text: $0, outlined: true) }
                }
                Text(model.detail).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                    .help(model.detail)
            }
            Spacer(minLength: 12)
            Text(model.sizeLabel).font(.callout).monospacedDigit().foregroundStyle(.secondary)
            action(model).frame(width: 84, alignment: .trailing)
        }
        .contentShape(Rectangle())
    }
    @ViewBuilder private func action(_ model: ModelChoice) -> some View {
        if store.downloading == model.id {
            ProgressView(value: store.downloadProgress).controlSize(.small)
        } else if store.loadedModel == model.id {
            Label("In use", systemImage: "checkmark").font(.callout).foregroundStyle(.secondary)
        } else if model.local {
            Button("Load") { Task { await store.loadModel(model.id); dismiss() } }
                .controlSize(.small)
                .disabled(store.busy)
        } else {
            Button("Get") { Task { await store.download(model.id) } }
                .controlSize(.small)
                .disabled(!store.ready || store.downloading != nil)
        }
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
