//! The quiet, Chat-aligned terminal surface. The command prompt is part of
//! the scrollable transcript, not a second composer below it.

use super::*;

const TERMINAL_TEXT: Color = Color::from_rgb8(0x11, 0x11, 0x11);
const TERMINAL_SURFACE: Color = Color::from_rgb8(0xf8, 0xfa, 0xfc);
const TERMINAL_PROMPT: &str = "vllm-mlx $";

pub(super) fn render(app: &TokenWorkshedApp) -> Element<'_, Message> {
    // This is the one real, local, stateful manager terminal. The API does
    // not offer shell switching, so do not draw a non-functional zsh menu.
    let context = widget::container(
        widget::row::with_capacity(2)
            .push(
                widget::icon::from_name("utilities-terminal-symbolic")
                    .size(16)
                    .icon()
                    .class(theme::Svg::custom(|_| cosmic::iced::widget::svg::Style {
                        color: Some(TERMINAL_TEXT),
                    })),
            )
            .push(widget::text("Local shell").size(13).class(TERMINAL_TEXT))
            .spacing(8)
            .align_y(Alignment::Center),
    )
    .padding([0, 10])
    .height(Length::Fixed(28.0))
    .style(|_| context_style());

    let toolbar = widget::container(context)
        .padding([0, 10])
        .height(Length::Fixed(38.0))
        .width(Length::Fill)
        .align_y(Alignment::Center)
        .style(|_| toolbar_style());

    let mut transcript = widget::column::with_capacity(app.developer_logs.len() * 2 + 1)
        .width(Length::Fill)
        .spacing(3);
    for (index, line) in app.developer_logs.iter().enumerate() {
        if needs_command_separator(index, line) {
            transcript = transcript.push(
                widget::container(app.caligo_horizontal_rule(false))
                    .padding([10, 0])
                    .width(Length::Fill),
            );
        }
        transcript = transcript.push(
            widget::text(line.as_str())
                .font(Font::MONOSPACE)
                .size(13)
                .class(TERMINAL_TEXT)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
        );
    }

    let input = widget::text_input("", &app.developer_terminal_input)
        .id(command_input_id())
        .padding([3, 0])
        .size(13)
        .font(Font::MONOSPACE)
        .style(inline_input_style())
        .on_input(Message::DeveloperTerminalChanged)
        .on_submit(|_| Message::DeveloperTerminalSubmit)
        .on_focus(Message::InputFocusChanged(
            InputField::DeveloperTerminal,
            true,
        ))
        .on_unfocus(Message::InputFocusChanged(
            InputField::DeveloperTerminal,
            false,
        ))
        .width(Length::Fill);
    transcript = transcript.push(
        widget::container(
            widget::row::with_capacity(2)
                .push(
                    widget::text(TERMINAL_PROMPT)
                        .font(Font::MONOSPACE)
                        .size(13)
                        .class(TERMINAL_TEXT),
                )
                .push(input)
                .spacing(8)
                .align_y(Alignment::Center)
                .width(Length::Fill),
        )
        .padding([8, 0]),
    );

    let output = widget::container(
        widget::scrollable(transcript)
            // Keep the current prompt visible as output is appended while
            // still allowing the user to scroll back through past commands.
            .anchor_bottom()
            .height(Length::Fill)
            .width(Length::Fill)
            .class(theme::iced::Scrollable::Transient),
    )
    .padding([12, 16])
    .height(Length::Fill)
    .width(Length::Fill)
    .style(|_| output_style());

    app.page_shell(
        "Terminal",
        "Run local commands and inspect output.",
        widget::column::with_capacity(2)
            .push(toolbar)
            .push(output)
            .spacing(8)
            .height(Length::Fill)
            .width(Length::Fill)
            .into(),
    )
}

pub(super) fn command_input_id() -> widget::Id {
    widget::Id::new("terminal-inline-command")
}

fn needs_command_separator(index: usize, line: &str) -> bool {
    index > 0 && line.starts_with("vllm-mlx $ ")
}

fn toolbar_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(TERMINAL_SURFACE))
        .color(TERMINAL_TEXT)
        .border(Border {
            width: 1.0,
            radius: CONTENT_RADIUS.into(),
            color: rgba(0x11, 0x11, 0x11, 0.05),
        })
}

fn context_style() -> container::Style {
    toolbar_style().background(Background::Color(Color::from_rgb8(0xf9, 0xfa, 0xfc)))
}

fn output_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(TERMINAL_SURFACE))
        .color(TERMINAL_TEXT)
        .border(Border {
            width: 0.0,
            radius: CONTENT_RADIUS.into(),
            color: Color::TRANSPARENT,
        })
}

fn inline_input_appearance() -> widget::text_input::Appearance {
    widget::text_input::Appearance {
        background: Background::Color(Color::TRANSPARENT),
        border_radius: 0.0.into(),
        border_offset: None,
        border_width: 0.0,
        border_color: Color::TRANSPARENT,
        label_color: TERMINAL_TEXT,
        placeholder_color: Color::from_rgb8(0x6b, 0x6b, 0x6b),
        selected_text_color: Color::WHITE,
        icon_color: Some(TERMINAL_TEXT),
        text_color: Some(TERMINAL_TEXT),
        selected_fill: TERMINAL_TEXT,
    }
}

fn inline_input_style() -> theme::TextInput {
    theme::TextInput::Custom {
        active: Box::new(|_| inline_input_appearance()),
        error: Box::new(|_| inline_input_appearance()),
        hovered: Box::new(|_| inline_input_appearance()),
        focused: Box::new(|_| inline_input_appearance()),
        disabled: Box::new(|_| inline_input_appearance()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_new_command_groups_get_a_separator() {
        assert!(!needs_command_separator(0, "vllm-mlx $ pwd"));
        assert!(needs_command_separator(3, "vllm-mlx $ pwd"));
        assert!(!needs_command_separator(4, "/project/token-workshed-0.2.0"));
        assert!(!needs_command_separator(5, "vllm-mlx: ready"));
    }

    #[test]
    fn inline_prompt_has_readable_black_text_without_a_composer_box() {
        let input = inline_input_appearance();
        assert_eq!(input.text_color, Some(TERMINAL_TEXT));
        assert_eq!(input.border_width, 0.0);
        assert_eq!(input.background, Background::Color(Color::TRANSPARENT));
        assert_eq!(input.selected_text_color, Color::WHITE);
        assert_eq!(input.selected_fill, TERMINAL_TEXT);
        assert_eq!(output_style().border.width, 0.0);
    }
}
