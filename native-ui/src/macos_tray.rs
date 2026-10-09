#[cfg(target_os = "macos")]
mod platform {
    use objc::declare::ClassDecl;
    use objc::runtime::{BOOL, Class, Object, Sel, YES};
    use objc::{class, msg_send, sel, sel_impl};
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicPtr, Ordering};

    static TRAY_CLASS: OnceLock<&'static Class> = OnceLock::new();
    static TRAY_DELEGATE: AtomicPtr<Object> = AtomicPtr::new(ptr::null_mut());
    static TRAY_ITEM: AtomicPtr<Object> = AtomicPtr::new(ptr::null_mut());

    pub fn install() {
        unsafe {
            let class = *TRAY_CLASS.get_or_init(register_class);
            let delegate: *mut Object = msg_send![class, new];
            let status_bar: *mut Object = msg_send![class!(NSStatusBar), systemStatusBar];
            let status_item: *mut Object = msg_send![status_bar, statusItemWithLength: -1.0f64];
            let button: *mut Object = msg_send![status_item, button];
            if !button.is_null() {
                let title = ns_string("TW");
                let tooltip = ns_string("token-workshed");
                let _: () = msg_send![button, setTitle: title];
                let _: () = msg_send![button, setToolTip: tooltip];
            }

            let menu: *mut Object = msg_send![class!(NSMenu), new];
            let open_item = menu_item("Open token-workshed", sel!(onOpen:), delegate);
            let _: () = msg_send![menu, addItem: open_item];
            let separator: *mut Object = msg_send![class!(NSMenuItem), separatorItem];
            let _: () = msg_send![menu, addItem: separator];
            let quit_item = menu_item("Quit token-workshed", sel!(onQuit:), delegate);
            let _: () = msg_send![menu, addItem: quit_item];
            let _: () = msg_send![status_item, setMenu: menu];

            TRAY_DELEGATE.store(delegate, Ordering::SeqCst);
            TRAY_ITEM.store(status_item, Ordering::SeqCst);
        }
    }

    fn register_class() -> &'static Class {
        let superclass = class!(NSObject);
        let mut decl = ClassDecl::new("TokenWorkshedTrayController", superclass)
            .expect("TokenWorkshedTrayController class");
        unsafe {
            decl.add_method(
                sel!(onOpen:),
                on_open as extern "C" fn(&Object, Sel, *mut Object),
            );
            decl.add_method(
                sel!(onQuit:),
                on_quit as extern "C" fn(&Object, Sel, *mut Object),
            );
        }
        decl.register()
    }

    fn menu_item(title: &str, action: Sel, target: *mut Object) -> *mut Object {
        unsafe {
            let item: *mut Object = msg_send![class!(NSMenuItem), alloc];
            let item: *mut Object = msg_send![
                item,
                initWithTitle: ns_string(title)
                action: action
                keyEquivalent: ns_string("")
            ];
            let _: () = msg_send![item, setTarget: target];
            item
        }
    }

    fn ns_string(value: &str) -> *mut Object {
        unsafe {
            let string: *mut Object = msg_send![class!(NSString), alloc];
            let bytes = value.as_ptr() as *const c_void;
            let string: *mut Object = msg_send![
                string,
                initWithBytes: bytes
                length: value.len()
                encoding: 4usize
            ];
            string
        }
    }

    extern "C" fn on_open(_this: &Object, _cmd: Sel, _sender: *mut Object) {
        unsafe {
            let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
            let windows: *mut Object = msg_send![app, windows];
            let count: usize = msg_send![windows, count];
            for index in 0..count {
                let window: *mut Object = msg_send![windows, objectAtIndex: index];
                let _: () = msg_send![window, deminiaturize: ptr::null_mut::<Object>()];
                let _: () = msg_send![window, makeKeyAndOrderFront: ptr::null_mut::<Object>()];
            }
            let _: () = msg_send![app, activateIgnoringOtherApps: YES as BOOL];
        }
    }

    extern "C" fn on_quit(_this: &Object, _cmd: Sel, _sender: *mut Object) {
        unsafe {
            let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
            let _: () = msg_send![app, terminate: ptr::null_mut::<Object>()];
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    pub fn install() {}
}

pub fn install() {
    platform::install();
}
