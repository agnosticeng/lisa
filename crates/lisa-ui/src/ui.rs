use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Sel};
use objc2::MainThreadMarker;
use objc2_app_kit::{
    NSBezelStyle, NSBorderType, NSButton, NSButtonType, NSColor, NSFocusRingType, NSFont,
    NSLayoutAttribute, NSScrollView, NSStackView, NSTextField, NSUserInterfaceLayoutOrientation,
    NSView, NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState,
    NSVisualEffectView,
};
use objc2_foundation::NSString;

type CGFloat = f64;

pub fn label(
    mtm: MainThreadMarker,
    text: &str,
    size: CGFloat,
    bold: bool,
    color: &NSColor,
) -> Retained<NSTextField> {
    let label = NSTextField::labelWithString(&NSString::from_str(text), mtm);
    let font = if bold {
        NSFont::boldSystemFontOfSize(size)
    } else {
        NSFont::systemFontOfSize(size)
    };
    label.setFont(Some(&font));
    label.setTextColor(Some(color));
    label
}

/// Configure a text field as a selectable static label that keeps its
/// attributed runs. `wrappingLabelWithString` is selectable by default, but its
/// cell is not rich text: clicking hands the string to AppKit's field editor,
/// which redraws it with the control's own font and drops the attributes (so
/// markdown text shrinks and headings lose bold). Marking the cell rich text
/// and allowing editing attributes preserves them through selection.
pub fn make_static(label: &NSTextField) {
    label.setEditable(false);
    label.setSelectable(true);
    label.setBezeled(false);
    label.setDrawsBackground(false);
    label.setAllowsEditingTextAttributes(true);
    label.setFocusRingType(NSFocusRingType::None);
}

#[allow(dead_code)]
pub fn wrapping_label(
    mtm: MainThreadMarker,
    text: &str,
    size: CGFloat,
    color: &NSColor,
) -> Retained<NSTextField> {
    let label = NSTextField::wrappingLabelWithString(&NSString::from_str(text), mtm);
    label.setFont(Some(&NSFont::systemFontOfSize(size)));
    label.setTextColor(Some(color));
    make_static(&label);
    label
}

pub fn button(
    mtm: MainThreadMarker,
    title: &str,
    target: Option<&AnyObject>,
    action: Option<Sel>,
    bordered: bool,
) -> Retained<NSButton> {
    let button = unsafe {
        NSButton::buttonWithTitle_target_action(&NSString::from_str(title), target, action, mtm)
    };
    button.setButtonType(NSButtonType::MomentaryPushIn);
    if bordered {
        button.setBezelStyle(NSBezelStyle::Push);
        button.setBordered(true);
    } else {
        button.setBordered(false);
    }
    button
}

#[allow(dead_code)]
pub fn stack(
    mtm: MainThreadMarker,
    orientation: NSUserInterfaceLayoutOrientation,
    spacing: CGFloat,
    alignment: NSLayoutAttribute,
) -> Retained<NSStackView> {
    let stack = NSStackView::new(mtm);
    stack.setOrientation(orientation);
    stack.setSpacing(spacing);
    stack.setAlignment(alignment);
    stack.setTranslatesAutoresizingMaskIntoConstraints(false);
    stack
}

pub fn scroll(mtm: MainThreadMarker, document: &NSView) -> Retained<NSScrollView> {
    let scroll = NSScrollView::new(mtm);
    scroll.setDocumentView(Some(document));
    scroll.setHasVerticalScroller(true);
    scroll.setDrawsBackground(false);
    scroll.setBorderType(NSBorderType::NoBorder);
    scroll.setTranslatesAutoresizingMaskIntoConstraints(false);
    scroll
}

pub fn rule(mtm: MainThreadMarker) -> Retained<NSView> {
    let rule = NSView::new(mtm);
    rule.setTranslatesAutoresizingMaskIntoConstraints(false);
    rule.heightAnchor().constraintEqualToConstant(1.0).setActive(true);
    rule
}

pub fn backdrop(mtm: MainThreadMarker) -> Retained<NSVisualEffectView> {
    let backdrop = NSVisualEffectView::new(mtm);
    backdrop.setMaterial(NSVisualEffectMaterial::UnderWindowBackground);
    backdrop.setBlendingMode(NSVisualEffectBlendingMode::BehindWindow);
    backdrop.setState(NSVisualEffectState::Active);
    backdrop.setTranslatesAutoresizingMaskIntoConstraints(false);
    backdrop
}

pub fn height(view: &NSView, value: CGFloat) {
    view.heightAnchor().constraintEqualToConstant(value).setActive(true);
}

#[allow(dead_code)]
pub fn size(view: &NSView, width: CGFloat, height: CGFloat) {
    view.setTranslatesAutoresizingMaskIntoConstraints(false);
    view.widthAnchor().constraintEqualToConstant(width).setActive(true);
    view.heightAnchor().constraintEqualToConstant(height).setActive(true);
}
