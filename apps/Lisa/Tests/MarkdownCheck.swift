import Foundation

@main struct MarkdownCheck {
    static func main() {
        let source = """
        You are referring to **Special Relativity** (1905).

        ### Key Concepts:
        1. **Mass and Energy**: mass and energy are interchangeable.
        2. **Speed of Light**: constant in a vacuum.
           - Nested explanation

        > A quoted [reference](https://example.com) with `inline code`.

        ```swift
        let value = "``` inside code"

        print(value)
        ```

        | Model | Size |
        | :--- | ---: |
        | Qwen | 0.8B |
        """
        let document = MarkdownDocument(source)
        let blocks = document.sections.flatMap(\.blocks)
        precondition(blocks.contains { $0.kind == .header(level: 3) && String($0.text.characters) == "Key Concepts:" })
        precondition(blocks.filter { $0.marker != nil }.map(\.marker) == ["1.", "2.", "•"])
        precondition(blocks.first { $0.marker == "•" }?.listDepth == 2)
        let quote = blocks.first { $0.quoteDepth == 1 }!
        precondition(quote.text.runs.contains { $0.link == URL(string: "https://example.com") })
        precondition(quote.text.runs.contains { $0.inlinePresentationIntent?.contains(.code) == true })
        let code = blocks.first { if case .codeBlock = $0.kind { return true }; return false }!
        precondition(String(code.text.characters).contains("\"``` inside code\"\n\nprint(value)"))
        let table = document.sections.first { $0.isTable }!
        precondition(table.blocks.count == 4 && table.blocks.last?.row == 1)
        precondition(table.blocks.last?.columnAlignment == .right)
        precondition(blocks[0].text.runs.contains { $0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true })
        let streaming = MarkdownDocument("```swift\nlet x = 1").sections[0].blocks[0]
        precondition(String(streaming.text.characters).contains("let x = 1"))
        if case .codeBlock = streaming.kind {} else { preconditionFailure("Open fences must stay code while streaming") }
        let continuation = MarkdownDocument("1. First paragraph\n\n   Another paragraph\n\n2. Next item").sections.flatMap(\.blocks)
        precondition(continuation.map(\.marker) == ["1.", nil, "2."])
        precondition(MarkdownDocument("").sections.isEmpty)
        print("Markdown checks passed: headings, nested lists, inline formatting, links, quotes, code, tables, streaming and continuation paragraphs")
    }
}
