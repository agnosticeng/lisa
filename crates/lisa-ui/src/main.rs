mod app;
mod catalog;
mod chats;
mod media;
mod models;
mod ui;

use objc2::runtime::ProtocolObject;
use objc2::{AnyThread, MainThreadMarker};
use objc2_app_kit::{NSAlert, NSApplication, NSApplicationActivationPolicy, NSImage};
use objc2_foundation::{NSBundle, NSData, NSString};

use app::AppDelegate;

fn main() {
    let mtm = MainThreadMarker::new().expect("must run on the main thread");
    // Name the process before the app reads it, so the menu bar shows "lisa".
    objc2_foundation::NSProcessInfo::processInfo()
        .setProcessName(&objc2_foundation::NSString::from_str("lisa"));
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    if let Err(error) = lisa_engine::check_metal_support() {
        let alert = NSAlert::new(mtm);
        alert.setMessageText(&NSString::from_str("LISA needs a newer version of macOS"));
        alert.setInformativeText(&NSString::from_str(&error.to_string()));
        alert.runModal();
        return;
    }
    if let Err(error) = lisa_engine::initialize_kernels() {
        let alert = NSAlert::new(mtm);
        alert.setMessageText(&NSString::from_str("LISA could not load its kernels"));
        alert.setInformativeText(&NSString::from_str(&error.to_string()));
        alert.runModal();
        return;
    }
    // Bundles use their ICNS resource. Standalone launches share the tiny SVG.
    if NSBundle::mainBundle().bundleIdentifier().is_none() {
        let data = NSData::with_bytes(include_bytes!("../assets/Lisa.svg"));
        if let Some(icon) = NSImage::initWithData(NSImage::alloc(), &data) {
            unsafe { app.setApplicationIconImage(Some(&icon)) };
        }
    }

    let delegate = AppDelegate::new(mtm);
    let object = ProtocolObject::from_ref(&*delegate);
    app.setDelegate(Some(object));

    app.run();
}
