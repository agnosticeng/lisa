//! Generic native dispatch geometry, matching MLX c69303b device.cpp. GPU core
//! count is read once; zero preserves the upstream fallback when unavailable.
pub(crate) fn gpu_core_count() -> usize {
    static COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *COUNT.get_or_init(read_gpu_core_count)
}

#[cfg(not(target_os = "macos"))]
fn read_gpu_core_count() -> usize {
    0
}

#[cfg(target_os = "macos")]
fn read_gpu_core_count() -> usize {
    use std::ffi::{c_char, c_void};
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceGetMatchingServices(port: u32, matching: *mut c_void, iter: *mut u32) -> i32;
        fn IOIteratorNext(iter: u32) -> u32;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: *const c_void,
            allocator: *const c_void,
            options: u32,
        ) -> *const c_void;
        fn IOObjectRelease(object: u32) -> i32;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            allocator: *const c_void,
            value: *const c_char,
            encoding: u32,
        ) -> *const c_void;
        fn CFGetTypeID(value: *const c_void) -> usize;
        fn CFNumberGetTypeID() -> usize;
        fn CFNumberGetValue(number: *const c_void, kind: isize, value: *mut c_void) -> u8;
        fn CFRelease(value: *const c_void);
    }
    // SAFETY: C strings are static literals, returned handles are null-checked and released once,
    // and kind 9 (kCFNumberIntType) writes one i32 into `count`, as in the reference.
    unsafe {
        let key =
            CFStringCreateWithCString(std::ptr::null(), c"gpu-core-count".as_ptr(), 0x08000100);
        if key.is_null() {
            return 0;
        }
        let mut iter = 0;
        if IOServiceGetMatchingServices(0, IOServiceMatching(c"AGXAccelerator".as_ptr()), &mut iter)
            != 0
        {
            CFRelease(key);
            return 0;
        }
        let mut count = 0i32;
        loop {
            let entry = IOIteratorNext(iter);
            if entry == 0 {
                break;
            }
            let property = IORegistryEntryCreateCFProperty(entry, key, std::ptr::null(), 0);
            if !property.is_null() {
                if CFGetTypeID(property) == CFNumberGetTypeID() {
                    CFNumberGetValue(property, 9, (&mut count as *mut i32).cast());
                }
                CFRelease(property);
            }
            IOObjectRelease(entry);
            if count > 0 {
                break;
            }
        }
        IOObjectRelease(iter);
        CFRelease(key);
        count.max(0) as usize
    }
}
