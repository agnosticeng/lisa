use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Sel};
use objc2::{AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSAccessibility, NSBezelStyle, NSBorderType, NSButton, NSButtonType, NSColor, NSFocusRingType,
    NSFont, NSFontWeightRegular, NSImage, NSImageSymbolConfiguration, NSImageSymbolScale,
    NSLayoutAttribute, NSLayoutConstraint, NSScrollView, NSStackView, NSTextAlignment, NSTextField,
    NSTrackingArea, NSTrackingAreaOptions, NSUserInterfaceLayoutOrientation, NSView,
    NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState, NSVisualEffectView,
};
use objc2_foundation::{NSObjectProtocol, NSPoint, NSRect, NSSize, NSString, NSUserDefaults};

type CGFloat = f64;

const MODEL_ROW_BACKGROUND_INSET: f64 = 6.0;
pub const MODEL_ROW_CONTENT_INSET: f64 = MODEL_ROW_BACKGROUND_INSET + 10.0;
pub const SIDEBAR_ICON_SIZE: f64 = 17.0;

pub fn sidebar_symbol_configuration() -> Retained<NSImageSymbolConfiguration> {
    NSImageSymbolConfiguration::configurationWithPointSize_weight_scale(
        13.0,
        unsafe { NSFontWeightRegular },
        NSImageSymbolScale::Medium,
    )
}

/// Hover is derived from the live pointer, not from the order in which AppKit
/// delivers tracking events while rows are replaced, clipped or moved.
fn pointer_inside(view: &NSView) -> bool {
    !view.isHiddenOrHasHiddenAncestor()
        && view.window().is_some_and(|window| {
            if !window.isKeyWindow() {
                return false;
            }
            let point =
                view.convertPoint_fromView(window.mouseLocationOutsideOfEventStream(), None);
            // With clipsToBounds off (AppKit's default), visibleRect can extend
            // into sibling rows. Require both our own bounds and ancestor clipping.
            view.mouse_inRect(point, view.bounds()) && view.mouse_inRect(point, view.visibleRect())
        })
}

#[derive(Debug)]
pub struct SidebarRowIvars {
    selected: bool,
    hovered: std::cell::Cell<bool>,
    inset_x: f64,
    inset_y: f64,
    action: Option<Retained<NSButton>>,
}

objc2::define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[ivars = SidebarRowIvars]
    #[derive(Debug)]
    pub struct SidebarRow;

    unsafe impl NSObjectProtocol for SidebarRow {}

    impl SidebarRow {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            // Clear our own backing layer even when no highlight is drawn.
            NSColor::clearColor().setFill();
            objc2_app_kit::NSRectFillUsingOperation(
                self.bounds(), objc2_app_kit::NSCompositingOperation::Copy,
            );
            if !self.ivars().selected && !self.ivars().hovered.get() { return; }
            let fill = NSColor::quaternaryLabelColor();
            if self.ivars().selected {
                fill.setFill();
            } else {
                fill.colorWithAlphaComponent(fill.alphaComponent() * 0.4).setFill();
            }
            let mut bounds = self.bounds();
            bounds.origin.x += self.ivars().inset_x;
            bounds.origin.y += self.ivars().inset_y;
            bounds.size.width = (bounds.size.width - 2.0 * self.ivars().inset_x).max(0.0);
            bounds.size.height = (bounds.size.height - 2.0 * self.ivars().inset_y).max(0.0);
            objc2_app_kit::NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(bounds, 8.0, 8.0).fill();
        }

        #[unsafe(method(mouseEntered:))]
        fn mouse_entered(&self, _event: &objc2_app_kit::NSEvent) {
            self.refresh_hover();
        }

        #[unsafe(method(mouseExited:))]
        fn mouse_exited(&self, _event: &objc2_app_kit::NSEvent) {
            self.refresh_hover();
        }

        #[unsafe(method(updateTrackingAreas))]
        fn update_tracking_areas(&self) {
            unsafe { let _: () = objc2::msg_send![super(self), updateTrackingAreas]; }
            self.refresh_hover();
        }
    }
);

impl SidebarRow {
    pub fn refresh_hover(&self) {
        let hovered = pointer_inside(self);
        if let Some(action) = &self.ivars().action {
            let hidden = !hovered && !self.ivars().selected;
            if action.isHidden() != hidden {
                action.setHidden(hidden);
            }
        }
        if self.ivars().hovered.replace(hovered) != hovered {
            self.setNeedsDisplay(true);
        }
    }

    fn new(
        mtm: MainThreadMarker,
        selected: bool,
        inset_x: f64,
        inset_y: f64,
        action: Option<Retained<NSButton>>,
    ) -> Retained<Self> {
        if let Some(button) = &action {
            button.setHidden(!selected);
        }
        let row = Self::alloc(mtm).set_ivars(SidebarRowIvars {
            selected,
            hovered: std::cell::Cell::new(false),
            inset_x,
            inset_y,
            action,
        });
        let row: Retained<Self> = unsafe { objc2::msg_send![super(row), init] };
        row.setWantsLayer(true);
        // InVisibleRect tracking must stay inside this row as well as its clip view.
        row.setClipsToBounds(true);
        unsafe {
            let tracking = NSTrackingArea::initWithRect_options_owner_userInfo(
                NSTrackingArea::alloc(),
                NSRect::ZERO,
                NSTrackingAreaOptions::MouseEnteredAndExited
                    | NSTrackingAreaOptions::ActiveInKeyWindow
                    | NSTrackingAreaOptions::EnabledDuringMouseDrag
                    | NSTrackingAreaOptions::InVisibleRect,
                Some(&row),
                None,
            );
            row.addTrackingArea(&tracking);
        }
        row
    }
}

/// Selection and hover use the same padded bounds, with distinct emphasis.
pub fn model_row(mtm: MainThreadMarker, selected: bool) -> Retained<NSView> {
    SidebarRow::new(mtm, selected, MODEL_ROW_BACKGROUND_INSET, 1.0, None).into_super()
}

pub fn chat_row(
    mtm: MainThreadMarker,
    selected: bool,
    delete: Retained<NSButton>,
) -> Retained<NSView> {
    let row = SidebarRow::new(mtm, selected, 0.0, 0.0, Some(delete));
    row.setTranslatesAutoresizingMaskIntoConstraints(false);
    row.into_super()
}

/// New chat is an action, so only hovering paints its background.
pub fn sidebar_action(mtm: MainThreadMarker) -> Retained<SidebarRow> {
    let row = SidebarRow::new(mtm, false, 0.0, 0.0, None);
    row.setTranslatesAutoresizingMaskIntoConstraints(false);
    row
}

pub fn refresh_sidebar_row(view: &NSView) {
    if let Some(row) = view.downcast_ref::<SidebarRow>() {
        row.refresh_hover();
    }
}

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

pub fn wrapping_label(
    mtm: MainThreadMarker,
    text: &str,
    size: CGFloat,
    color: &NSColor,
) -> Retained<NSTextField> {
    let label: Retained<NSTextField> = WrappingLabel::new(mtm).into_super();
    label.setStringValue(&NSString::from_str(text));
    label.setMaximumNumberOfLines(0);
    label.setContentCompressionResistancePriority_forOrientation(
        249.0,
        objc2_app_kit::NSLayoutConstraintOrientation::Horizontal,
    );
    label.setContentHuggingPriority_forOrientation(
        249.0,
        objc2_app_kit::NSLayoutConstraintOrientation::Horizontal,
    );
    if let Some(cell) = label.cell() {
        cell.setWraps(true);
        cell.setUsesSingleLineMode(false);
    }
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
    configure_scroll(NSScrollView::new(mtm), document)
}

struct ScrollPosition {
    origin: NSPoint,
    at_bottom: bool,
}

impl ScrollPosition {
    fn capture(scroll: &NSScrollView) -> Self {
        let visible = scroll.documentVisibleRect();
        Self {
            origin: visible.origin,
            at_bottom: scroll.documentView().is_some_and(|document| {
                let bounds = document.bounds();
                document.isFlipped()
                    && bounds.size.height > visible.size.height
                    && visible.origin.y + visible.size.height
                        >= bounds.origin.y + bounds.size.height - 0.5
            }),
        }
    }

    fn restore(&self, scroll: &NSScrollView) {
        let clip = scroll.contentView();
        let mut origin = self.origin;
        if self.at_bottom {
            if let Some(document) = scroll.documentView() {
                let bounds = document.bounds();
                origin.y = (bounds.origin.y + bounds.size.height - clip.bounds().size.height)
                    .max(bounds.origin.y);
            }
        }
        clip.scrollToPoint(origin);
        scroll.reflectScrolledClipView(&clip);
    }
}

/// Row reloads and selection restoration must not discard the user's position.
pub fn preserve_scroll_position(scroll: &NSScrollView, update: impl FnOnce()) {
    let position = ScrollPosition::capture(scroll);
    update();
    scroll.layoutSubtreeIfNeeded();
    position.restore(scroll);
}

// Keep the end of a sidebar list visible when its viewport shrinks. AppKit's
// default preserves the top offset, even if the user had scrolled to the end.
objc2::define_class!(
    #[unsafe(super(NSScrollView))]
    #[thread_kind = MainThreadOnly]
    #[derive(Debug)]
    struct SidebarScrollView;

    unsafe impl NSObjectProtocol for SidebarScrollView {}

    impl SidebarScrollView {
        #[unsafe(method(setFrameSize:))]
        fn set_frame_size(&self, size: NSSize) {
            let position = ScrollPosition::capture(self);
            unsafe { let _: () = objc2::msg_send![super(self), setFrameSize: size]; }
            if position.at_bottom {
                position.restore(self);
            }
        }
    }
);

pub fn sidebar_scroll(mtm: MainThreadMarker, document: &NSView) -> Retained<NSScrollView> {
    let scroll: Retained<SidebarScrollView> = unsafe {
        objc2::msg_send![SidebarScrollView::alloc(mtm), init]
    };
    configure_scroll(scroll.into_super(), document)
}

fn configure_scroll(scroll: Retained<NSScrollView>, document: &NSView) -> Retained<NSScrollView> {
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
    rule.heightAnchor()
        .constraintEqualToConstant(1.0)
        .setActive(true);
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
    view.heightAnchor()
        .constraintEqualToConstant(value)
        .setActive(true);
}

pub fn size(view: &NSView, width: CGFloat, height: CGFloat) {
    view.setTranslatesAutoresizingMaskIntoConstraints(false);
    view.widthAnchor()
        .constraintEqualToConstant(width)
        .setActive(true);
    view.heightAnchor()
        .constraintEqualToConstant(height)
        .setActive(true);
}

/// A symbol with a visible title, so compact controls remain discoverable.
pub fn symbol_button(
    mtm: MainThreadMarker,
    title: &str,
    symbol: &str,
    target: &AnyObject,
    action: Sel,
) -> Retained<NSButton> {
    let control = button(mtm, title, Some(target), Some(action), false);
    if let Some(image) = objc2_app_kit::NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str(symbol),
        Some(&NSString::from_str(title)),
    ) {
        control.setImage(Some(&image));
        control.setImagePosition(objc2_app_kit::NSCellImagePosition::ImageLeft);
    }
    control.setFont(Some(&NSFont::systemFontOfSize(12.0)));
    control.setToolTip(Some(&NSString::from_str(title)));
    control.setTranslatesAutoresizingMaskIntoConstraints(false);
    control
}

objc2::define_class!(
    #[unsafe(super(NSButton))]
    #[thread_kind = MainThreadOnly]
    #[derive(Debug)]
    struct SectionHeader;

    unsafe impl NSObjectProtocol for SectionHeader {}

    impl SectionHeader {
        #[unsafe(method(mouseEntered:))]
        fn mouse_entered(&self, _event: &objc2_app_kit::NSEvent) {
            self.refresh_hover();
        }

        #[unsafe(method(mouseExited:))]
        fn mouse_exited(&self, _event: &objc2_app_kit::NSEvent) {
            self.refresh_hover();
        }

        #[unsafe(method(updateTrackingAreas))]
        fn update_tracking_areas(&self) {
            unsafe { let _: () = objc2::msg_send![super(self), updateTrackingAreas]; }
            self.refresh_hover();
        }
    }
);

impl SectionHeader {
    fn refresh_hover(&self) {
        self.set_hovered(pointer_inside(self));
    }

    fn set_hovered(&self, hovered: bool) {
        let position = if hovered {
            objc2_app_kit::NSCellImagePosition::ImageRight
        } else {
            objc2_app_kit::NSCellImagePosition::NoImage
        };
        if self.imagePosition() != position {
            self.setImagePosition(position);
        }
    }
}

pub fn section_header(
    mtm: MainThreadMarker,
    title: &str,
    target: &AnyObject,
    action: Sel,
) -> Retained<NSButton> {
    let header: Retained<SectionHeader> =
        unsafe { objc2::msg_send![SectionHeader::alloc(mtm), initWithFrame: NSRect::ZERO] };
    header.setTitle(&NSString::from_str(title));
    header.setBordered(false);
    header.setButtonType(NSButtonType::MomentaryPushIn);
    header.setClipsToBounds(true);
    header.setTranslatesAutoresizingMaskIntoConstraints(false);
    unsafe {
        header.setTarget(Some(target));
        header.setAction(Some(action));
        // InVisibleRect keeps the tracking region in sync with the full heading.
        let tracking = NSTrackingArea::initWithRect_options_owner_userInfo(
            NSTrackingArea::alloc(),
            NSRect::ZERO,
            NSTrackingAreaOptions::MouseEnteredAndExited
                | NSTrackingAreaOptions::ActiveInKeyWindow
                | NSTrackingAreaOptions::EnabledDuringMouseDrag
                | NSTrackingAreaOptions::InVisibleRect,
            Some(&header),
            None,
        );
        header.addTrackingArea(&tracking);
    }
    header.into_super()
}

/// Keeps a sidebar section's content and layout collapsed together.
#[derive(Debug)]
pub struct SidebarSection {
    header: Retained<NSButton>,
    content: Retained<NSScrollView>,
    expanded_constraints: Vec<Retained<NSLayoutConstraint>>,
    collapsed_height: Retained<NSLayoutConstraint>,
    content_top: Retained<NSLayoutConstraint>,
    expanded_spacing: CGFloat,
    preference_key: Retained<NSString>,
}

impl SidebarSection {
    pub fn refresh_hover(&self) {
        if let Some(header) = self.header.downcast_ref::<SectionHeader>() {
            header.refresh_hover();
        }
    }

    pub fn new(
        header: Retained<NSButton>,
        content: Retained<NSScrollView>,
        expanded_constraints: Vec<Retained<NSLayoutConstraint>>,
        content_top: Retained<NSLayoutConstraint>,
        preference_key: &str,
    ) -> Self {
        header.setAlignment(NSTextAlignment::Left);
        header.setImagePosition(objc2_app_kit::NSCellImagePosition::NoImage);
        header.setImageHugsTitle(false);
        header.setFont(Some(&NSFont::boldSystemFontOfSize(11.0)));
        header.setContentTintColor(Some(&NSColor::secondaryLabelColor()));
        height(&header, 24.0);
        let section = Self {
            collapsed_height: content.heightAnchor().constraintEqualToConstant(0.0),
            expanded_spacing: content_top.constant(),
            header,
            content,
            expanded_constraints,
            content_top,
            preference_key: NSString::from_str(preference_key),
        };
        let collapsed = NSUserDefaults::standardUserDefaults().boolForKey(&section.preference_key);
        section.set_collapsed(collapsed);
        section
    }

    pub fn toggle(&self) {
        let collapsed = !self.content.isHidden();
        self.set_collapsed(collapsed);
        NSUserDefaults::standardUserDefaults().setBool_forKey(collapsed, &self.preference_key);
    }

    fn set_collapsed(&self, collapsed: bool) {
        // Remove the old height rules before enabling the new ones.
        self.collapsed_height.setActive(false);
        for constraint in &self.expanded_constraints {
            constraint.setActive(!collapsed);
        }
        self.content_top.setConstant(if collapsed {
            0.0
        } else {
            self.expanded_spacing
        });
        self.collapsed_height.setActive(collapsed);
        self.content.setHidden(collapsed);
        let symbol = if collapsed {
            "chevron.right"
        } else {
            "chevron.down"
        };
        // AppKit may change NoImage to ImageOnly when replacing the image.
        let image_position = self.header.imagePosition();
        self.header.setImage(
            NSImage::imageWithSystemSymbolName_accessibilityDescription(
                &NSString::from_str(symbol),
                None,
            )
            .as_deref(),
        );
        self.header.setImagePosition(image_position);
        let title = self.header.title();
        let action = if collapsed { "Expand" } else { "Collapse" };
        self.header
            .setToolTip(Some(&NSString::from_str(&format!("{action} {title}"))));
        self.header.setAccessibilityLabel(Some(&title));
        self.header.setAccessibilityExpanded(!collapsed);
    }
}

// AppKit otherwise keeps a wrapping label's intrinsic height at its old width
// when a split-view inspector opens or the window is resized.
objc2::define_class!(
    #[unsafe(super(NSTextField))]
    #[thread_kind = MainThreadOnly]
    #[derive(Debug)]
    pub struct WrappingLabel;

    unsafe impl NSObjectProtocol for WrappingLabel {}

    impl WrappingLabel {
        #[unsafe(method(layout))]
        fn layout(&self) {
            unsafe { let _: () = objc2::msg_send![super(self), layout]; }
            self.update_wrap_width();
        }

        #[unsafe(method(setFrameSize:))]
        fn set_frame_size(&self, size: objc2_foundation::NSSize) {
            unsafe { let _: () = objc2::msg_send![super(self), setFrameSize: size]; }
            self.update_wrap_width();
        }
    }
);

impl WrappingLabel {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        use objc2::MainThreadOnly;
        let this = Self::alloc(mtm);
        unsafe { objc2::msg_send![this, initWithFrame: objc2_foundation::NSRect::ZERO] }
    }

    fn update_wrap_width(&self) {
        let width = self.bounds().size.width;
        if width > 0.0 && (self.preferredMaxLayoutWidth() - width).abs() > 0.5 {
            self.setPreferredMaxLayoutWidth(width);
            self.invalidateIntrinsicContentSize();
        }
    }
}
