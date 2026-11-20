import SwiftUI

enum ChatLayout {
    #if os(macOS)
    static let columnWidth: CGFloat = 740
    static let composerControlHeight: CGFloat = 32
    static let sidebarFontSize: CGFloat = 13
    static let sidebarRowHeight: CGFloat = 22
    static let turnSpacing: CGFloat = 18
    static let replySpacing: CGFloat = 12
    static let bubblePadding: CGFloat = 10
    static let paragraphSpacing: CGFloat = 8
    static let lineSpacing: CGFloat = 2
    #else
    static let columnWidth: CGFloat = 800
    static let composerControlHeight: CGFloat = 44
    static let sidebarFontSize: CGFloat = 14
    static let sidebarRowHeight: CGFloat = 30
    static let turnSpacing: CGFloat = 28
    static let replySpacing: CGFloat = 16
    static let bubblePadding: CGFloat = 14
    static let paragraphSpacing: CGFloat = 14
    static let lineSpacing: CGFloat = 3
    #endif
}
