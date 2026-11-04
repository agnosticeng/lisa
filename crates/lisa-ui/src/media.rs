use std::path::Path;

use objc2::rc::Retained;
use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel,
};
use objc2_app_kit::*;
use objc2_foundation::{NSArray, NSObjectProtocol, NSRect, NSSize, NSString, NSURL};

use crate::ui;

#[derive(Debug)]
struct ImagePreviewIvars {
    image: Retained<NSImage>,
}

define_class!(
    #[unsafe(super(NSButton))]
    #[thread_kind = MainThreadOnly]
    #[ivars = ImagePreviewIvars]
    #[derive(Debug)]
    struct ImagePreview;

    unsafe impl NSObjectProtocol for ImagePreview {}

    impl ImagePreview {
        // The image's aspect ratio and card width determine its size, not the button cell.
        #[unsafe(method(intrinsicContentSize))]
        fn intrinsic_content_size(&self) -> NSSize {
            unsafe { NSSize::new(NSViewNoIntrinsicMetric, NSViewNoIntrinsicMetric) }
        }

        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            NSGraphicsContext::saveGraphicsState_class();
            let bounds = self.bounds();
            NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(bounds, 12.0, 12.0)
                .addClip();
            unsafe {
                self.ivars().image.drawInRect_fromRect_operation_fraction_respectFlipped_hints(
                    bounds, NSRect::ZERO, NSCompositingOperation::SourceOver, 1.0, true, None,
                );
            }
            if self.isHighlighted() {
                NSColor::colorWithWhite_alpha(0.0, 0.12).setFill();
                NSBezierPath::bezierPathWithRect(bounds).fill();
            }
            NSGraphicsContext::restoreGraphicsState_class();
        }

        #[unsafe(method(resetCursorRects))]
        fn reset_cursor_rects(&self) {
            self.addCursorRect_cursor(self.bounds(), &NSCursor::pointingHandCursor());
        }
    }
);

#[derive(Debug)]
pub struct GeneratedImageCardIvars {
    url: Retained<NSURL>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[ivars = GeneratedImageCardIvars]
    #[derive(Debug)]
    pub struct GeneratedImageCard;

    unsafe impl NSObjectProtocol for GeneratedImageCard {}

    impl GeneratedImageCard {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            NSColor::controlBackgroundColor().setFill();
            NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(self.bounds(), 18.0, 18.0)
                .fill();
        }

        #[unsafe(method(openImage:))]
        fn open_image(&self, _sender: &NSButton) {
            if !NSWorkspace::sharedWorkspace().openURL(&self.ivars().url) {
                let alert = NSAlert::new(self.mtm());
                alert.setMessageText(&NSString::from_str("Unable to open this image"));
                alert.setInformativeText(&NSString::from_str(
                    "The saved file may have been moved or deleted. Use Show in Finder to locate it.",
                ));
                alert.runModal();
            }
        }

        #[unsafe(method(revealImage:))]
        fn reveal_image(&self, _sender: &NSButton) {
            NSWorkspace::sharedWorkspace()
                .activateFileViewerSelectingURLs(&NSArray::from_slice(&[&*self.ivars().url]));
        }
    }
);

impl GeneratedImageCard {
    /// Retain the saved file with its card so actions always address the right image.
    pub fn new(mtm: MainThreadMarker, path: &Path) -> Option<Retained<Self>> {
        let file = NSString::from_str(&path.to_string_lossy());
        let image = NSImage::initWithContentsOfFile(NSImage::alloc(), &file)?;
        let size = image.size();
        if size.width <= 0.0 || size.height <= 0.0 {
            return None;
        }
        let this = Self::alloc(mtm).set_ivars(GeneratedImageCardIvars {
            url: NSURL::fileURLWithPath(&file),
        });
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        this.setTranslatesAutoresizingMaskIntoConstraints(false);

        let preview = ImagePreview::alloc(mtm).set_ivars(ImagePreviewIvars { image });
        let preview: Retained<ImagePreview> = unsafe { msg_send![super(preview), init] };
        preview.setTranslatesAutoresizingMaskIntoConstraints(false);
        preview.setTitle(&NSString::from_str(""));
        preview.setBordered(false);
        preview.setButtonType(NSButtonType::MomentaryPushIn);
        preview.setToolTip(Some(&NSString::from_str("Open full-size image")));
        preview.setAccessibilityLabel(Some(&NSString::from_str("Open generated image")));
        unsafe {
            preview.setTarget(Some(&this));
            preview.setAction(Some(sel!(openImage:)));
        }
        this.addSubview(&preview);

        let caption = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            3.0,
            NSLayoutAttribute::Leading,
        );
        caption.addArrangedSubview(&ui::label(
            mtm,
            "Generated image",
            12.0,
            true,
            &NSColor::labelColor(),
        ));
        caption.addArrangedSubview(&ui::label(
            mtm,
            &format!("{:.0} × {:.0} · Saved locally", size.width, size.height),
            11.0,
            false,
            &NSColor::secondaryLabelColor(),
        ));
        let open = ui::symbol_button(mtm, "Open", "arrow.up.right", &this, sel!(openImage:));
        open.setToolTip(Some(&NSString::from_str("Open full-size image")));
        let reveal = ui::symbol_button(mtm, "", "folder", &this, sel!(revealImage:));
        reveal.setToolTip(Some(&NSString::from_str("Show in Finder")));
        reveal.setAccessibilityLabel(Some(&NSString::from_str("Show in Finder")));
        ui::size(&reveal, 28.0, 28.0);
        let actions = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Horizontal,
            12.0,
            NSLayoutAttribute::CenterY,
        );
        actions.addArrangedSubview(&open);
        actions.addArrangedSubview(&reveal);
        this.addSubview(&caption);
        this.addSubview(&actions);

        NSLayoutConstraint::activateConstraints(&NSArray::from_slice(&[
            &*preview
                .topAnchor()
                .constraintEqualToAnchor_constant(&this.topAnchor(), 8.0),
            &*preview
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&this.leadingAnchor(), 8.0),
            &*preview
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&this.trailingAnchor(), -8.0),
            &*preview.heightAnchor().constraintEqualToAnchor_multiplier(
                &preview.widthAnchor(),
                size.height / size.width,
            ),
            &*caption
                .topAnchor()
                .constraintEqualToAnchor_constant(&preview.bottomAnchor(), 12.0),
            &*caption
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&this.leadingAnchor(), 16.0),
            &*caption
                .bottomAnchor()
                .constraintEqualToAnchor_constant(&this.bottomAnchor(), -14.0),
            &*actions
                .centerYAnchor()
                .constraintEqualToAnchor(&caption.centerYAnchor()),
            &*actions
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&this.trailingAnchor(), -12.0),
            &*caption
                .trailingAnchor()
                .constraintLessThanOrEqualToAnchor_constant(&actions.leadingAnchor(), -12.0),
            &*this
                .widthAnchor()
                .constraintLessThanOrEqualToConstant(480.0),
        ]));
        // Stay below the window's resizing priority so the card can shrink with it.
        let preferred = this.widthAnchor().constraintEqualToConstant(480.0);
        preferred.setPriority(NSLayoutPriorityDefaultLow);
        preferred.setActive(true);
        Some(this)
    }
}
