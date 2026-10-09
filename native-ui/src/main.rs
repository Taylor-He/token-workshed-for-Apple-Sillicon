mod api;
mod app;
mod appearance;
mod glass;
mod icons;
mod macos_tray;
mod persistence;
mod window_effects;
mod workshed;

use app::{AppFlags, TokenWorkshedApp};
use appearance::Page;
use cosmic::app::Settings;
use cosmic::iced::{Limits, Size};

const USE_TRANSPARENT_WINDOW: bool = false;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("ICED_BACKEND").is_none() {
        unsafe {
            std::env::set_var("ICED_BACKEND", "wgpu");
        }
    }
    #[cfg(target_os = "macos")]
    if std::env::var_os("WGPU_BACKEND").is_none() {
        unsafe {
            std::env::set_var("WGPU_BACKEND", "metal");
        }
    }
    if std::env::var_os("ICED_PRESENT_MODE").is_none() {
        unsafe {
            std::env::set_var("ICED_PRESENT_MODE", "vsync");
        }
    }
    if std::env::var_os("WGPU_POWER_PREF").is_none() {
        unsafe {
            std::env::set_var("WGPU_POWER_PREF", "low");
        }
    }

    let flags = AppFlags::from_env();
    let (window_size, minimum_size) = if flags.ide_companion {
        (Size::new(900.0, 760.0), Size::new(700.0, 600.0))
    } else {
        (Size::new(1240.0, 860.0), Size::new(980.0, 680.0))
    };
    let settings = Settings::default()
        .size(window_size)
        .size_limits(Limits::new(
            minimum_size,
            Size::new(f32::INFINITY, f32::INFINITY),
        ))
        .transparent(USE_TRANSPARENT_WINDOW)
        .resizable(Some(8.0))
        .client_decorations(false)
        .exit_on_close(false);

    cosmic::app::run::<TokenWorkshedApp>(settings, flags)?;
    Ok(())
}

impl AppFlags {
    fn from_env() -> Self {
        let mut api_base = "http://127.0.0.1:7862".to_string();
        let mut server_url = "http://127.0.0.1:8000".to_string();
        let mut manager_token = String::new();
        let mut initial_page = Page::Chat;
        let mut ide_companion = false;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--api-base" => {
                    if let Some(value) = args.next() {
                        api_base = value;
                    }
                }
                "--server-url" => {
                    if let Some(value) = args.next() {
                        server_url = value;
                    }
                }
                "--manager-token" => {
                    if let Some(value) = args.next() {
                        manager_token = value;
                    }
                }
                "--initial-page" => {
                    if let Some(value) = args.next() {
                        initial_page = Page::from_code(&value);
                    }
                }
                "--ide-companion" => ide_companion = true,
                _ => {}
            }
        }

        Self {
            api_base,
            server_url,
            manager_token,
            initial_page,
            ide_companion,
        }
    }
}
