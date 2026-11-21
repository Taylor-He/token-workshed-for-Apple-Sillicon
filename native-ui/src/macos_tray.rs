#[cfg(target_os = "macos")]
mod platform {
    use objc::declare::ClassDecl;
    use objc::runtime::{BOOL, Class, NO, Object, Sel, YES};
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
                let tooltip = ns_string("token-workshed NextCore");
                let _: () = msg_send![button, setTitle: ns_string("")];
                let _: () = msg_send![button, setImage: status_icon()];
                let _: () = msg_send![button, setToolTip: tooltip];
            }

            let menu: *mut Object = msg_send![class!(NSMenu), new];
            let open_item = menu_item("Open token-workshed NextCore", sel!(onOpen:), delegate);
            let _: () = msg_send![menu, addItem: open_item];
            let separator: *mut Object = msg_send![class!(NSMenuItem), separatorItem];
            let _: () = msg_send![menu, addItem: separator];
            let quit_item = menu_item("Quit token-workshed NextCore", sel!(onQuit:), delegate);
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

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NSPoint {
        x: f64,
        y: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NSSize {
        width: f64,
        height: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NSRect {
        origin: NSPoint,
        size: NSSize,
    }

    fn status_icon() -> *mut Object {
        unsafe {
            let image: *mut Object = msg_send![class!(NSImage), alloc];
            let image: *mut Object = msg_send![
                image,
                initWithSize: NSSize {
                    width: 18.0,
                    height: 18.0,
                }
            ];
            let _: () = msg_send![image, lockFocus];

            let white: *mut Object = msg_send![class!(NSColor), whiteColor];
            let _: () = msg_send![white, setFill];
            let _: () = msg_send![white, setStroke];
            let bench: *mut Object = msg_send![class!(NSBezierPath), bezierPath];
            let _: () = msg_send![
                bench,
                appendBezierPathWithRoundedRect: NSRect {
                    origin: NSPoint { x: 2.0, y: 12.5 },
                    size: NSSize {
                        width: 14.0,
                        height: 2.5,
                    },
                }
                xRadius: 1.0f64
                yRadius: 1.0f64
            ];
            let _: () = msg_send![bench, fill];

            let legs: *mut Object = msg_send![class!(NSBezierPath), bezierPath];
            let _: () = msg_send![legs, setLineWidth: 2.4f64];
            let _: () = msg_send![legs, setLineCapStyle: 1isize];
            let _: () = msg_send![legs, setLineJoinStyle: 1isize];
            let _: () = msg_send![legs, moveToPoint: NSPoint { x: 3.0, y: 11.5 }];
            let _: () = msg_send![legs, lineToPoint: NSPoint { x: 6.0, y: 4.0 }];
            let _: () = msg_send![legs, lineToPoint: NSPoint { x: 9.0, y: 10.0 }];
            let _: () = msg_send![legs, lineToPoint: NSPoint { x: 12.0, y: 4.0 }];
            let _: () = msg_send![legs, lineToPoint: NSPoint { x: 15.0, y: 11.5 }];
            let _: () = msg_send![legs, stroke];

            let _: () = msg_send![image, unlockFocus];
            let _: () = msg_send![image, setTemplate: NO as BOOL];
            image
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
