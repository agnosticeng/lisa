import SwiftUI

/// Foundation parses block structure as well as inline formatting. Keep those
/// blocks separate so SwiftUI can lay out headings, list markers and tables.
struct MarkdownDocument {
    struct Block: Identifiable {
        let id: Int
        var text: AttributedString
        var kind: PresentationIntent.Kind = .paragraph
        var listDepth = 0
        var marker: String?
        var quoteDepth = 0
        var tableID: Int?
        var row = 0
        var column = 0
        var columnAlignment: PresentationIntent.TableColumn.Alignment = .left
    }
    struct Section: Identifiable {
        let id: Int
        var blocks: [Block]
        var isTable: Bool { blocks.first?.tableID != nil }
    }
    let sections: [Section]

    init(_ source: String) {
        let parsed = (try? AttributedString(markdown: source, options: .init(interpretedSyntax: .full))) ?? AttributedString(source)
        var sections: [Section] = []
        var seenItems: Set<Int> = []
        for (intent, range) in parsed.runs[\.presentationIntent] {
            let components = intent?.components ?? []
            var block = Block(id: components.first?.identity ?? sections.count, text: AttributedString(parsed[range]))
            block.text.presentationIntent = nil
            block.kind = components.first?.kind ?? .paragraph
            var item: (id: Int, ordinal: Int)?
            var ordered: Bool?
            var columns: [PresentationIntent.TableColumn] = []
            for component in components {
                switch component.kind {
                case .listItem(let ordinal):
                    if item == nil { item = (component.identity, ordinal) }
                case .orderedList:
                    block.listDepth += 1
                    if ordered == nil { ordered = true }
                case .unorderedList:
                    block.listDepth += 1
                    if ordered == nil { ordered = false }
                case .blockQuote: block.quoteDepth += 1
                case .table(let value): block.tableID = component.identity; columns = value
                case .tableHeaderRow: block.row = 0
                case .tableRow(let index): block.row = index
                case .tableCell(let index): block.column = index
                default: break
                }
            }
            if let item, seenItems.insert(item.id).inserted {
                block.marker = ordered == true ? "\(item.ordinal)." : "•"
            }
            if columns.indices.contains(block.column) { block.columnAlignment = columns[block.column].alignment }
            if let table = block.tableID, sections.last?.id == table {
                sections[sections.count - 1].blocks.append(block)
            } else {
                sections.append(Section(id: block.tableID ?? block.id, blocks: [block]))
            }
        }
        self.sections = sections
    }
}

struct MarkdownMessage: View {
    let text: String
    var body: some View {
        let document = MarkdownDocument(text)
        VStack(alignment: .leading, spacing: ChatLayout.paragraphSpacing) {
            ForEach(document.sections) { section in
                if section.isTable {
                    table(section.blocks)
                } else if let block = section.blocks.first {
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        if block.listDepth > 0 {
                            Text(block.marker ?? "")
                                .monospacedDigit().frame(minWidth: 22, alignment: .trailing)
                                .accessibilityHidden(block.marker == nil)
                        }
                        blockContent(block)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.leading, CGFloat(max(0, block.listDepth - 1) * 20 + block.quoteDepth * 14))
                    .overlay(alignment: .leading) {
                        if block.quoteDepth > 0 {
                            RoundedRectangle(cornerRadius: 2).fill(.secondary.opacity(0.3)).frame(width: 3)
                        }
                    }
                }
            }
        }
        .textSelection(.enabled)
    }

    @ViewBuilder private func blockContent(_ block: MarkdownDocument.Block) -> some View {
        switch block.kind {
        case .header(let level):
            Text(styled(block.text)).id(block.text).font(headingFont(level)).fontWeight(.semibold)
                .padding(.top, 6).accessibilityAddTraits(.isHeader)
        case .codeBlock(let language):
            VStack(alignment: .leading, spacing: 8) {
                if let language, !language.isEmpty { Text(language).font(.caption).foregroundStyle(.secondary) }
                ScrollView(.horizontal) {
                    Text(block.text).id(block.text).font(codeFont).fixedSize(horizontal: true, vertical: false)
                }
            }
            .padding(14).background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
        case .thematicBreak:
            Divider().padding(.vertical, 4)
        default:
            Text(styled(block.text)).id(block.text).lineSpacing(ChatLayout.lineSpacing)
        }
    }

    private var codeFont: Font {
        #if os(macOS)
        .system(size: 13, design: .monospaced)
        #else
        .system(.body, design: .monospaced)
        #endif
    }

    private func headingFont(_ level: Int) -> Font {
        #if os(macOS)
        return .system(size: level == 1 ? 22 : level == 2 ? 19 : level == 3 ? 16 : 14)
        #else
        switch level {
        case 1: .title
        case 2: .title2
        case 3: .title3
        default: .headline
        }
        #endif
    }

    private func styled(_ source: AttributedString) -> AttributedString {
        var result = source
        for run in source.runs {
            if run.inlinePresentationIntent?.contains(.code) == true {
                result[run.range].font = codeFont
                result[run.range].backgroundColor = .secondary.opacity(0.12)
            }
        }
        return result
    }

    private func table(_ blocks: [MarkdownDocument.Block]) -> some View {
        let rows = Array(Set(blocks.map(\.row))).sorted()
        return ScrollView(.horizontal) {
            Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 12) {
                ForEach(rows, id: \.self) { row in
                    GridRow(alignment: .top) {
                        ForEach(blocks.filter { $0.row == row }) { cell in
                            Text(styled(cell.text))
                                .fontWeight(row == 0 ? .semibold : .regular)
                                .frame(minWidth: 90, maxWidth: 260, alignment: alignment(cell.columnAlignment))
                                .gridColumnAlignment(horizontalAlignment(cell.columnAlignment))
                        }
                    }
                    if row == 0 { Divider().gridCellUnsizedAxes(.horizontal) }
                }
            }.padding(14)
        }.background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
    }

    private func alignment(_ value: PresentationIntent.TableColumn.Alignment) -> Alignment {
        switch value { case .center: .center; case .right: .trailing; default: .leading }
    }
    private func horizontalAlignment(_ value: PresentationIntent.TableColumn.Alignment) -> HorizontalAlignment {
        switch value { case .center: .center; case .right: .trailing; default: .leading }
    }
}
