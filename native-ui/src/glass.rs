use crate::appearance::{
    CONTENT_RADIUS, GLASS_BORDER_ALPHA, GLASS_BORDER_DISABLED_ALPHA, GLASS_BORDER_FOCUS_ALPHA,
    GLASS_BORDER_HOVER_ALPHA, GLASS_BORDER_PRESSED_ALPHA, Palette, rgba,
};
use cosmic::Element;
use cosmic::iced::widget::{button, container};
use cosmic::iced::{Background, Border, Color, Length, Shadow};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlassVariant {
    Window,
    Panel,
    Card,
    Control,
    Input,
    Floating,
    Toast,
    Subtle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlassState {
    Rest,
    Hover,
    Pressed,
    Focused,
    Selected,
    Disabled,
}

#[derive(Debug, Clone, Copy)]
struct FlatGlassSpec {
    fill: Color,
    border: Color,
    radius: f32,
}

/// Compatibility layer for callers that still compose a glass layer.
/// Caligo uses app-rendered surfaces, so the decorative canvas is intentionally empty.
pub fn glass_overlay<'a, Message: 'a>(
    _variant: GlassVariant,
    _state: GlassState,
) -> Element<'a, Message> {
    container(cosmic::iced::widget::text(""))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

pub fn glass_container_style(
    palette: Palette,
    variant: GlassVariant,
    state: GlassState,
) -> container::Style {
    let spec = flat_glass_spec(palette, variant, state);
    let border_width = match variant {
        GlassVariant::Panel | GlassVariant::Card | GlassVariant::Subtle => 0.0,
        _ => 1.0,
    };
    container::Style::default()
        .background(Background::Color(spec.fill))
        .color(palette.text_main)
        .border(Border {
            width: border_width,
            radius: spec.radius.into(),
            color: spec.border,
        })
        .shadow(Shadow::default())
}

pub fn glass_button_style(
    palette: Palette,
    variant: GlassVariant,
    state: GlassState,
    tint: Option<Color>,
) -> button::Style {
    let mut spec = flat_glass_spec(palette, variant, state);
    let mut text_color = palette.text_main;

    if let Some(tint) = tint {
        let alpha = match state {
            GlassState::Pressed => 0.78,
            GlassState::Hover | GlassState::Focused | GlassState::Selected => 0.90,
            GlassState::Disabled => 0.32,
            GlassState::Rest => 1.0,
        };
        spec.fill = Color::from_rgba(tint.r, tint.g, tint.b, alpha);
        spec.border = Color::TRANSPARENT;
        text_color = if matches!(state, GlassState::Disabled) {
            Color::from_rgba(1.0, 1.0, 1.0, 0.62)
        } else {
            Color::WHITE
        };
    }

    let mut style = button::Style::default().with_background(spec.fill);
    style.text_color = text_color;
    style.border = Border {
        width: if tint.is_some() { 0.0 } else { 1.0 },
        radius: spec.radius.into(),
        color: spec.border,
    };
    style.shadow = Shadow::default();
    style
}

pub fn state_from_status(status: button::Status, selected: bool) -> GlassState {
    if selected {
        GlassState::Selected
    } else {
        match status {
            button::Status::Hovered => GlassState::Hover,
            button::Status::Pressed => GlassState::Pressed,
            button::Status::Disabled => GlassState::Disabled,
            button::Status::Active => GlassState::Rest,
        }
    }
}

pub fn glass_radius(variant: GlassVariant) -> f32 {
    match variant {
        GlassVariant::Window => 0.0,
        GlassVariant::Panel
        | GlassVariant::Card
        | GlassVariant::Control
        | GlassVariant::Input
        | GlassVariant::Subtle => CONTENT_RADIUS,
        GlassVariant::Floating | GlassVariant::Toast => CONTENT_RADIUS,
    }
}

fn flat_glass_spec(palette: Palette, variant: GlassVariant, state: GlassState) -> FlatGlassSpec {
    let fill = match variant {
        GlassVariant::Window => palette.window_background,
        GlassVariant::Panel => palette.panel_background,
        GlassVariant::Card => palette.card_background,
        GlassVariant::Control => palette.inset_background,
        GlassVariant::Input => {
            if matches!(state, GlassState::Focused | GlassState::Selected) {
                // Opaque composite of the Caligo 0.60 focus surface.
                Color::from_rgb8(0xfb, 0xfc, 0xfd)
            } else {
                palette.inset_background
            }
        }
        GlassVariant::Floating | GlassVariant::Toast => palette.card_background_strong,
        GlassVariant::Subtle => palette.window_background,
    };

    let border_alpha = match state {
        GlassState::Focused | GlassState::Selected => GLASS_BORDER_FOCUS_ALPHA,
        GlassState::Hover => GLASS_BORDER_HOVER_ALPHA,
        GlassState::Pressed => GLASS_BORDER_PRESSED_ALPHA,
        GlassState::Disabled => GLASS_BORDER_DISABLED_ALPHA,
        GlassState::Rest => GLASS_BORDER_ALPHA,
    };
    let border = if matches!(variant, GlassVariant::Floating | GlassVariant::Toast) {
        rgba(0xff, 0xff, 0xff, border_alpha)
    } else {
        rgba(0x11, 0x11, 0x11, border_alpha)
    };

    FlatGlassSpec {
        fill,
        border,
        radius: glass_radius(variant),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::appearance::{Accent, MaterialMode, palette};

    #[test]
    fn caligo_surfaces_have_no_elevation() {
        let palette = palette(MaterialMode::Frosted, false);
        assert_eq!(
            glass_container_style(palette, GlassVariant::Card, GlassState::Rest)
                .shadow
                .blur_radius,
            0.0
        );
        assert_eq!(
            glass_button_style(palette, GlassVariant::Control, GlassState::Rest, None)
                .shadow
                .blur_radius,
            0.0
        );
    }

    #[test]
    fn panel_and_card_containers_have_no_outer_frame() {
        let palette = palette(MaterialMode::Frosted, false);
        for variant in [GlassVariant::Panel, GlassVariant::Card] {
            assert_eq!(
                glass_container_style(palette, variant, GlassState::Rest)
                    .border
                    .width,
                0.0
            );
        }
    }

    #[test]
    fn focused_input_uses_the_caligo_focus_alpha() {
        let palette = palette(MaterialMode::Frosted, false);
        let style = glass_container_style(palette, GlassVariant::Input, GlassState::Focused);
        assert_eq!(
            style.background,
            Some(Background::Color(Color::from_rgb8(0xfb, 0xfc, 0xfd)))
        );
        assert_eq!(style.border.radius, CONTENT_RADIUS.into());
        let _ = Accent::Black;
    }
}
