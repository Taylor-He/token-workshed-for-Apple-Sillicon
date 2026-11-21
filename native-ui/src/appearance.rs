use crate::glass::{GlassState, GlassVariant, glass_container_style};
use cosmic::iced::{Background, Border, Color};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Accent {
    Blue,
    Black,
    Purple,
    Pink,
    Red,
    Orange,
    Yellow,
    Green,
    Gray,
    Rainbow,
}

impl Accent {
    pub const ALL: [Self; 10] = [
        Self::Blue,
        Self::Black,
        Self::Purple,
        Self::Pink,
        Self::Red,
        Self::Orange,
        Self::Yellow,
        Self::Green,
        Self::Gray,
        Self::Rainbow,
    ];

    pub const fn code(self) -> &'static str {
        match self {
            Self::Blue => "blue",
            Self::Black => "black",
            Self::Purple => "purple",
            Self::Pink => "pink",
            Self::Red => "red",
            Self::Orange => "orange",
            Self::Yellow => "yellow",
            Self::Green => "green",
            Self::Gray => "gray",
            Self::Rainbow => "rainbow",
        }
    }

    pub fn from_code(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "black" => Self::Black,
            "purple" => Self::Purple,
            "pink" => Self::Pink,
            "red" => Self::Red,
            "orange" => Self::Orange,
            "yellow" => Self::Yellow,
            "green" => Self::Green,
            "gray" => Self::Gray,
            "rainbow" => Self::Rainbow,
            _ => Self::Black,
        }
    }

    pub const fn display_color(self) -> Color {
        match self {
            Self::Blue => rgb8(0x2b, 0x6c, 0xff),
            Self::Black => rgb8(0x11, 0x11, 0x11),
            Self::Purple => rgb8(0x8b, 0x5c, 0xf6),
            Self::Pink => rgb8(0xff, 0x4f, 0xa3),
            Self::Red => rgb8(0xef, 0x44, 0x44),
            Self::Orange => rgb8(0xfb, 0x92, 0x3c),
            Self::Yellow => rgb8(0xfb, 0xbf, 0x24),
            Self::Green => rgb8(0x22, 0xc5, 0x5e),
            Self::Gray => rgb8(0x9c, 0xa3, 0xaf),
            Self::Rainbow => rgb8(0x2b, 0x6c, 0xff),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaterialMode {
    Frosted,
    Metal,
}

impl MaterialMode {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Frosted => "frosted",
            // Keep compatibility with old persisted value, but 3.0.0 only uses Frosted.
            Self::Metal => "frosted",
        }
    }

    pub fn from_code(value: &str) -> Self {
        let _ = value;
        Self::Frosted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Chat,
    Models,
    Workshed,
    Runtime,
    Logs,
    Community,
    Settings,
    About,
}

impl Page {
    pub const ALL: [Self; 8] = [
        Self::Chat,
        Self::Models,
        Self::Workshed,
        Self::Runtime,
        Self::Logs,
        Self::Community,
        Self::About,
        Self::Settings,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Chat => "Chat",
            Self::Models => "Model Settings",
            Self::Workshed => "Workshed",
            Self::Runtime => "Runtime & Agent",
            Self::Logs => "Terminal",
            Self::Community => "Community",
            Self::Settings => "App Settings",
            Self::About => "About",
        }
    }

    pub fn from_code(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "models" | "model" | "model-settings" => Self::Models,
            "workshed" | "workshop" | "quantize" | "quantization" => Self::Workshed,
            "server" | "server-settings" | "agent" | "runtime" | "runtime-agent" => Self::Runtime,
            "logs" | "log" => Self::Logs,
            // Keep old launch arguments working after the prompt page was
            // folded into Terminal.
            "prompt" | "system-prompt" => Self::Logs,
            "community" => Self::Community,
            "settings" | "app-settings" => Self::Settings,
            "about" => Self::About,
            _ => Self::Chat,
        }
    }
}

#[cfg(test)]
mod page_tests {
    use super::Page;

    #[test]
    fn legacy_server_and_agent_launch_codes_open_the_merged_page() {
        for code in ["server", "server-settings", "agent", "runtime-agent"] {
            assert_eq!(Page::from_code(code), Page::Runtime);
        }
        assert_eq!(
            Page::ALL
                .iter()
                .filter(|page| **page == Page::Runtime)
                .count(),
            1
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Palette {
    pub window_background: Color,
    pub inset_background: Color,
    pub panel_background: Color,
    pub panel_background_hover: Color,
    pub panel_border: Color,
    pub card_background: Color,
    pub card_background_strong: Color,
    pub card_border: Color,
    pub overlay_backdrop: Color,
    pub text_main: Color,
    pub text_soft: Color,
    pub icon_inactive: Color,
}

impl Palette {
    pub fn window_style(self) -> cosmic::iced::widget::container::Style {
        cosmic::iced::widget::container::Style::default()
            .background(Background::Color(self.window_background))
            .color(self.text_main)
            .border(Border {
                width: 0.0,
                radius: 0.0.into(),
                color: Color::TRANSPARENT,
            })
    }

    pub fn card_style(self, strong: bool) -> cosmic::iced::widget::container::Style {
        glass_container_style(
            self,
            if strong {
                GlassVariant::Panel
            } else {
                GlassVariant::Card
            },
            GlassState::Rest,
        )
    }

    pub fn pill_style(
        self,
        accent: Accent,
        active: bool,
    ) -> cosmic::iced::widget::container::Style {
        let _ = (accent, active);
        cosmic::iced::widget::container::Style::default()
            .background(Background::Color(rgb8(0xf8, 0xfa, 0xfc)))
            .color(self.text_main)
            .border(Border {
                width: 1.0,
                radius: HOVER_GLASS_RADIUS.into(),
                color: neutral_glass_border(HOVER_GLASS_BORDER_ALPHA),
            })
    }
}

pub const APP_PADDING: f32 = 10.0;
pub const PAGE_FRAME_INSET: f32 = 12.0;
pub const PAGE_SECTION_GAP: u16 = 12;
pub const CONTROL_HEIGHT: f32 = 34.0;
pub const ICON_CONTROL_SIZE: f32 = 32.0;
pub const CONTROL_TEXT_SIZE: u16 = 13;
pub const SIDEBAR_WIDTH: f32 = 34.0;
pub const SIDEBAR_BUTTON_SIZE: f32 = 34.0;
pub const SIDEBAR_ITEM_INSET: f32 = 8.0;
pub const SIDEBAR_ITEM_GAP: f32 = 8.0;
pub const SIDEBAR_ICON_SLOT_SIZE: f32 = 34.0;
pub const SIDEBAR_ICON_VIEWBOX: f32 = 20.0;
pub const SIDEBAR_ICON_HOVER_SCALE: f32 = 1.20;
pub const SIDEBAR_HOVER_PILL_WIDTH: f32 = 34.0;
pub const SIDEBAR_HOVER_PILL_HEIGHT: f32 = 32.0;
pub const CONTENT_RADIUS: f32 = 8.0;
pub const GLASS_BORDER_ALPHA: f32 = 0.05;
pub const GLASS_BORDER_HOVER_ALPHA: f32 = 0.075;
pub const GLASS_BORDER_PRESSED_ALPHA: f32 = 0.11;
pub const GLASS_BORDER_FOCUS_ALPHA: f32 = 0.22;
pub const GLASS_BORDER_DISABLED_ALPHA: f32 = 0.03;
pub const HOVER_GLASS_ALPHA: f32 = 0.26;
pub const HOVER_GLASS_BORDER_ALPHA: f32 = 0.025;
pub const HOVER_GLASS_RADIUS: f32 = 7.0;
pub const INPUT_FOCUS_GLASS_ALPHA: f32 = 0.60;
pub const SECTION_DIVIDER_INSET: f32 = 14.0;
pub const SIDEBAR_SEPARATOR_GUTTER_WIDTH: f32 = 6.0;
pub const SIDEBAR_ALUMINUM_LEFT_OFFSET: f32 = 1.0;
pub const ALUMINUM_FRAME_WIDTH: f32 = 3.0;
pub const SIDEBAR_ALUMINUM_WIDTH: f32 = ALUMINUM_FRAME_WIDTH;
pub const SIDEBAR_ALUMINUM_SHADE_ALPHA: f32 = 0.30;
pub const SIDEBAR_ALUMINUM_CORE_ALPHA: f32 = 0.44;
pub const SIDEBAR_ALUMINUM_HIGHLIGHT_ALPHA: f32 = 0.58;
pub const NUMBER_ALUMINUM_RULE_HEIGHT: f32 = ALUMINUM_FRAME_WIDTH;

pub fn neutral_glass_border(alpha: f32) -> Color {
    rgba(0x11, 0x11, 0x11, alpha)
}

pub fn palette(material: MaterialMode, blur_supported: bool) -> Palette {
    let _ = (material, blur_supported);
    Palette {
        // Keep the app canvas white and use neutral gray for content surfaces.
        // Opaque fills avoid clear-color bleed when child alpha surfaces are
        // not blended by the platform.
        window_background: rgb8(0xff, 0xff, 0xff),
        inset_background: rgb8(0xf5, 0xf6, 0xf7),
        panel_background: rgb8(0xff, 0xff, 0xff),
        panel_background_hover: rgb8(0xfa, 0xfa, 0xfa),
        panel_border: neutral_glass_border(GLASS_BORDER_ALPHA),
        card_background: rgb8(0xf3, 0xf4, 0xf5),
        card_background_strong: rgb8(0xf8, 0xf8, 0xf8),
        card_border: neutral_glass_border(GLASS_BORDER_ALPHA),
        overlay_backdrop: rgba(0x11, 0x11, 0x11, 0.10),
        text_main: rgb8(0x11, 0x11, 0x11),
        text_soft: rgb8(0x6b, 0x6b, 0x6b),
        icon_inactive: rgb8(0xc7, 0xcf, 0xdb),
    }
}

pub const fn rgb8(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

pub const fn rgba(r: u8, g: u8, b: u8, a: f32) -> Color {
    Color::from_rgba(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, a)
}
