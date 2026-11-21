use crate::core::alignment;
use crate::core::layout;
use crate::core::mouse::{self, click};
use crate::core::renderer;
use crate::core::text::{Paragraph, Span};
use crate::core::widget::text::HasSelectableText;
use crate::core::widget::text::{
    self, Alignment, Catalog, Ellipsize, LineHeight, Shaping, Style, StyleFn,
    Wrapping,
};
use crate::core::widget::tree::{self, Tree};
use crate::core::{
    self, Clipboard, Color, Element, Event, Layout, Length, Pixels, Point,
    Rectangle, Shell, Size, Vector, Widget, keyboard, touch,
};
use unicode_segmentation::UnicodeSegmentation;

/// A bunch of [`Rich`] text.
pub struct Rich<
    'a,
    Link,
    Message,
    Theme = crate::Theme,
    Renderer = crate::Renderer,
> where
    Link: Clone + 'static,
    Theme: Catalog,
    Renderer: core::text::Renderer,
{
    spans: Box<dyn AsRef<[Span<'a, Link, Renderer::Font>]> + 'a>,
    size: Option<Pixels>,
    line_height: LineHeight,
    width: Length,
    height: Length,
    font: Option<Renderer::Font>,
    align_x: Alignment,
    align_y: alignment::Vertical,
    wrapping: Wrapping,
    ellipsize: Ellipsize,
    class: Theme::Class<'a>,
    hovered_link: Option<usize>,
    on_link_click: Option<Box<dyn Fn(Link) -> Message + 'a>>,
    selectable: bool,
}

impl<'a, Link, Message, Theme, Renderer>
    Rich<'a, Link, Message, Theme, Renderer>
where
    Link: Clone + 'static,
    Theme: Catalog,
    Renderer: core::text::Renderer,
    Renderer::Font: 'a,
{
    /// Creates a new empty [`Rich`] text.
    pub fn new() -> Self {
        Self {
            spans: Box::new([]),
            size: None,
            line_height: LineHeight::default(),
            width: Length::Shrink,
            height: Length::Shrink,
            font: None,
            align_x: Alignment::Default,
            align_y: alignment::Vertical::Top,
            wrapping: Wrapping::default(),
            ellipsize: Ellipsize::default(),
            class: Theme::default(),
            hovered_link: None,
            on_link_click: None,
            selectable: false,
        }
    }

    /// Creates a new [`Rich`] text with the given text spans.
    pub fn with_spans(
        spans: impl AsRef<[Span<'a, Link, Renderer::Font>]> + 'a,
    ) -> Self {
        Self {
            spans: Box::new(spans),
            ..Self::new()
        }
    }

    /// Sets the default size of the [`Rich`] text.
    pub fn size(mut self, size: impl Into<Pixels>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the default [`LineHeight`] of the [`Rich`] text.
    pub fn line_height(mut self, line_height: impl Into<LineHeight>) -> Self {
        self.line_height = line_height.into();
        self
    }

    /// Sets the default font of the [`Rich`] text.
    pub fn font(mut self, font: impl Into<Renderer::Font>) -> Self {
        self.font = Some(font.into());
        self
    }

    /// Sets the width of the [`Rich`] text boundaries.
    pub fn width(mut self, width: impl Into<Length>) -> Self {
        self.width = width.into();
        self
    }

    /// Sets the height of the [`Rich`] text boundaries.
    pub fn height(mut self, height: impl Into<Length>) -> Self {
        self.height = height.into();
        self
    }

    /// Centers the [`Rich`] text, both horizontally and vertically.
    pub fn center(self) -> Self {
        self.align_x(alignment::Horizontal::Center)
            .align_y(alignment::Vertical::Center)
    }

    /// Sets the [`alignment::Horizontal`] of the [`Rich`] text.
    pub fn align_x(mut self, alignment: impl Into<Alignment>) -> Self {
        self.align_x = alignment.into();
        self
    }

    /// Sets the [`alignment::Vertical`] of the [`Rich`] text.
    pub fn align_y(
        mut self,
        alignment: impl Into<alignment::Vertical>,
    ) -> Self {
        self.align_y = alignment.into();
        self
    }

    /// Sets the [`Wrapping`] strategy of the [`Rich`] text.
    pub fn wrapping(mut self, wrapping: Wrapping) -> Self {
        self.wrapping = wrapping;
        self
    }

    /// Sets the message that will be produced when a link of the [`Rich`] text
    /// is clicked.
    ///
    /// If the spans of the [`Rich`] text contain no links, you may need to call
    /// this method with `on_link_click(never)` in order for the compiler to infer
    /// the proper `Link` generic type.
    pub fn on_link_click(
        mut self,
        on_link_click: impl Fn(Link) -> Message + 'a,
    ) -> Self {
        self.on_link_click = Some(Box::new(on_link_click));
        self
    }

    /// Makes the [`Rich`] text selectable and copyable.
    pub fn selectable(mut self) -> Self {
        self.selectable = true;
        self
    }

    /// Sets the [`Ellipsize`] strategy of the [`Rich`] text.
    pub fn ellipsize(mut self, ellipsize: Ellipsize) -> Self {
        self.ellipsize = ellipsize;
        self
    }

    /// Sets the default style of the [`Rich`] text.
    #[must_use]
    pub fn style(mut self, style: impl Fn(&Theme) -> Style + 'a) -> Self
    where
        Theme::Class<'a>: From<StyleFn<'a, Theme>>,
    {
        self.class = (Box::new(style) as StyleFn<'a, Theme>).into();
        self
    }

    /// Sets the default [`Color`] of the [`Rich`] text.
    pub fn color(self, color: impl Into<Color>) -> Self
    where
        Theme::Class<'a>: From<StyleFn<'a, Theme>>,
    {
        self.color_maybe(Some(color))
    }

    /// Sets the default [`Color`] of the [`Rich`] text, if `Some`.
    pub fn color_maybe(self, color: Option<impl Into<Color>>) -> Self
    where
        Theme::Class<'a>: From<StyleFn<'a, Theme>>,
    {
        let color = color.map(Into::into);

        self.style(move |_theme| Style {
            color,
            ..Style::default()
        })
    }

    /// Sets the default style class of the [`Rich`] text.
    #[cfg(feature = "advanced")]
    #[must_use]
    pub fn class(mut self, class: impl Into<Theme::Class<'a>>) -> Self {
        self.class = class.into();
        self
    }
}

impl<'a, Link, Message, Theme, Renderer> Default
    for Rich<'a, Link, Message, Theme, Renderer>
where
    Link: Clone + 'a,
    Theme: Catalog,
    Renderer: core::text::Renderer,
    Renderer::Font: 'a,
{
    fn default() -> Self {
        Self::new()
    }
}

struct State<Link, P: Paragraph> {
    spans: Vec<Span<'static, Link, P::Font>>,
    span_pressed: Option<usize>,
    paragraph: P,
    content: String,
    selection: Option<SelectionState>,
    focused: bool,
    context_menu_position: Option<Point>,
}

#[derive(Debug, Clone, Default)]
struct SelectionState {
    anchor: usize,
    end: usize,
    dragging: bool,
    modifiers: keyboard::Modifiers,
    last_click: Option<click::Click>,
}

impl SelectionState {
    fn clear(&mut self) {
        self.anchor = 0;
        self.end = 0;
        self.dragging = false;
    }
}

impl<Link, P: Paragraph> crate::core::widget::operation::Focusable
    for State<Link, P>
{
    fn is_focused(&self) -> bool {
        self.focused
    }

    fn focus(&mut self) {
        self.focused = true;
    }

    fn unfocus(&mut self) {
        self.focused = false;
        self.context_menu_position = None;
        if let Some(selection) = &mut self.selection {
            selection.clear();
        }
    }
}

impl<Link, Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for Rich<'_, Link, Message, Theme, Renderer>
where
    Link: Clone + 'static,
    Theme: Catalog,
    Renderer: core::text::Renderer,
{
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State<Link, Renderer::Paragraph>>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::<Link, _> {
            spans: Vec::new(),
            span_pressed: None,
            paragraph: Renderer::Paragraph::default(),
            content: String::new(),
            selection: None,
            focused: false,
            context_menu_position: None,
        })
    }

    fn size(&self) -> Size<Length> {
        Size {
            width: self.width,
            height: self.height,
        }
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        layout(
            tree.state
                .downcast_mut::<State<Link, Renderer::Paragraph>>(),
            renderer,
            limits,
            self.width,
            self.height,
            self.spans.as_ref().as_ref(),
            self.line_height,
            self.size,
            self.font,
            self.align_x,
            self.align_y,
            self.wrapping,
            self.ellipsize,
        )
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        defaults: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        if !layout.bounds().intersects(viewport) {
            return;
        }

        let state = tree
            .state
            .downcast_ref::<State<Link, Renderer::Paragraph>>();

        let style = theme.style(&self.class);

        if let Some(selection) = &state.selection {
            let left = selection.anchor.min(selection.end);
            let right = selection.anchor.max(selection.end);
            if left != right {
                let lo = grapheme_to_byte(&state.content, left);
                let hi = grapheme_to_byte(&state.content, right);
                let anchor = layout.bounds().anchor(
                    state.paragraph.min_bounds(),
                    state.paragraph.align_x(),
                    state.paragraph.align_y(),
                );
                for rect in state.paragraph.highlight(
                    0,
                    (lo, core::text::Affinity::After),
                    (hi, core::text::Affinity::Before),
                ) {
                    renderer.fill_quad(
                        renderer::Quad {
                            bounds: Rectangle {
                                x: anchor.x + rect.x,
                                y: anchor.y + rect.y,
                                width: rect.width,
                                height: rect.height,
                            },
                            ..Default::default()
                        },
                        style.selected_fill,
                    );
                }
            }
        }

        for (index, span) in self.spans.as_ref().as_ref().iter().enumerate() {
            let is_hovered_link = self.on_link_click.is_some()
                && Some(index) == self.hovered_link;

            if span.highlight.is_some()
                || span.underline
                || span.strikethrough
                || is_hovered_link
            {
                let translation = layout.position() - Point::ORIGIN;
                let regions = state.paragraph.span_bounds(index);

                if let Some(highlight) = span.highlight {
                    for bounds in &regions {
                        let bounds = Rectangle::new(
                            bounds.position()
                                - Vector::new(
                                    span.padding.left,
                                    span.padding.top,
                                ),
                            bounds.size()
                                + Size::new(span.padding.x(), span.padding.y()),
                        );

                        renderer.fill_quad(
                            renderer::Quad {
                                bounds: bounds + translation,
                                border: highlight.border,
                                ..Default::default()
                            },
                            highlight.background,
                        );
                    }
                }

                if span.underline || span.strikethrough || is_hovered_link {
                    let size = span
                        .size
                        .or(self.size)
                        .unwrap_or(renderer.default_size());

                    let line_height = span
                        .line_height
                        .unwrap_or(self.line_height)
                        .to_absolute(size);

                    let color = span
                        .color
                        .or(style.color)
                        .unwrap_or(defaults.text_color);

                    let baseline = translation
                        + Vector::new(
                            0.0,
                            size.0 + (line_height.0 - size.0) / 2.0,
                        );

                    if span.underline || is_hovered_link {
                        for bounds in &regions {
                            renderer.fill_quad(
                                renderer::Quad {
                                    bounds: Rectangle::new(
                                        bounds.position() + baseline
                                            - Vector::new(0.0, size.0 * 0.08),
                                        Size::new(bounds.width, 1.0),
                                    ),
                                    ..Default::default()
                                },
                                color,
                            );
                        }
                    }

                    if span.strikethrough {
                        for bounds in &regions {
                            renderer.fill_quad(
                                renderer::Quad {
                                    bounds: Rectangle::new(
                                        bounds.position() + baseline
                                            - Vector::new(0.0, size.0 / 2.0),
                                        Size::new(bounds.width, 1.0),
                                    ),
                                    ..Default::default()
                                },
                                color,
                            );
                        }
                    }
                }
            }
        }

        text::draw(
            renderer,
            defaults,
            layout.bounds(),
            &state.paragraph,
            style,
            viewport,
        );
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        let state = tree
            .state
            .downcast_mut::<State<Link, Renderer::Paragraph>>();
        let bounds = layout.bounds();
        let content = state.content.as_str();
        let was_hovered = self.hovered_link.is_some();

        if let Some(position) = cursor.position_in(bounds) {
            self.hovered_link =
                state.paragraph.hit_span(position).and_then(|span| {
                    if self.spans.as_ref().as_ref().get(span)?.link.is_some() {
                        Some(span)
                    } else {
                        None
                    }
                });
        } else {
            self.hovered_link = None;
        }

        if was_hovered != self.hovered_link.is_some() {
            shell.request_redraw();
        }

        if matches!(
            event,
            Event::Mouse(mouse::Event::ButtonPressed(_))
                | Event::Touch(touch::Event::FingerPressed { .. })
        ) && cursor.position_over(bounds).is_none()
        {
            state.focused = false;
            state.context_menu_position = None;
            if let Some(selection) = &mut state.selection {
                selection.clear();
            }
        }

        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if state.context_menu_position.take().is_some() {
                    shell.capture_event();
                    return;
                }
                if let Some(position) = cursor.position_over(bounds) {
                    let clicked_link = self.hovered_link.is_some()
                        && self.on_link_click.is_some();
                    if clicked_link {
                        state.span_pressed = self.hovered_link;
                        shell.capture_event();
                    }
                    if self.selectable {
                        let paragraph = &state.paragraph;
                        let selection = state
                            .selection
                            .get_or_insert_with(SelectionState::default);
                        let anchor = bounds.anchor(
                            paragraph.min_bounds(),
                            paragraph.align_x(),
                            paragraph.align_y(),
                        );
                        let relative = Point::new(
                            position.x - anchor.x,
                            position.y - anchor.y,
                        );
                        let index =
                            hit_to_grapheme(paragraph, relative, content);
                        let new_click = click::Click::new(
                            position,
                            mouse::Button::Left,
                            selection.last_click.take(),
                        );
                        match new_click.kind() {
                            click::Kind::Single => {
                                if selection.modifiers.shift() {
                                    selection.end = index;
                                } else {
                                    selection.anchor = index;
                                    selection.end = index;
                                }
                                selection.dragging = true;
                            }
                            click::Kind::Double => {
                                selection.anchor =
                                    previous_start_of_word(content, index);
                                selection.end =
                                    next_end_of_word(content, index);
                                selection.dragging = true;
                            }
                            click::Kind::Triple => {
                                selection.anchor = 0;
                                selection.end = content.graphemes(true).count();
                                selection.dragging = true;
                            }
                        }
                        selection.last_click = Some(new_click);
                        state.focused = true;
                        shell.capture_event();
                    } else if clicked_link {
                        state.focused = true;
                    }
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                let selection_is_collapsed =
                    state.selection.as_ref().map_or(true, |selection| {
                        selection.anchor == selection.end
                    });
                match state.span_pressed {
                    Some(span)
                        if Some(span) == self.hovered_link
                            && selection_is_collapsed =>
                    {
                        if let (Some(on_link_clicked), Some(link)) = (
                            self.on_link_click.as_ref(),
                            self.spans
                                .as_ref()
                                .as_ref()
                                .get(span)
                                .and_then(|span| span.link.clone()),
                        ) {
                            shell.publish(on_link_clicked(link));
                        }
                    }
                    _ => {}
                }
                state.span_pressed = None;
                if let Some(selection) = &mut state.selection {
                    selection.dragging = false;
                }
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) => {
                if let Some(position) = cursor.position_over(bounds) {
                    state.context_menu_position = Some(position);
                    state.focused = true;
                    shell.capture_event();
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(
                mouse::Button::Right,
            )) => {
                if state.context_menu_position.is_some() {
                    shell.capture_event();
                }
            }
            Event::Mouse(mouse::Event::CursorMoved { position })
            | Event::Touch(touch::Event::FingerMoved { position, .. }) => {
                if self.selectable {
                    if let Some(selection) = &mut state.selection {
                        if selection.dragging {
                            let anchor = bounds.anchor(
                                state.paragraph.min_bounds(),
                                state.paragraph.align_x(),
                                state.paragraph.align_y(),
                            );
                            let relative = Point::new(
                                position.x - anchor.x,
                                position.y - anchor.y,
                            );
                            selection.end = hit_to_grapheme(
                                &state.paragraph,
                                relative,
                                content,
                            );
                            shell.capture_event();
                        }
                    }
                }
            }
            Event::Touch(touch::Event::FingerPressed { .. }) => {
                if let Some(position) = cursor.position_over(bounds) {
                    if self.selectable && self.hovered_link.is_none() {
                        let paragraph = &state.paragraph;
                        let anchor = bounds.anchor(
                            paragraph.min_bounds(),
                            paragraph.align_x(),
                            paragraph.align_y(),
                        );
                        let relative = Point::new(
                            position.x - anchor.x,
                            position.y - anchor.y,
                        );
                        let index =
                            hit_to_grapheme(paragraph, relative, content);
                        let selection = state
                            .selection
                            .get_or_insert_with(SelectionState::default);
                        selection.anchor = index;
                        selection.end = index;
                        selection.dragging = true;
                        state.focused = true;
                        shell.capture_event();
                    }
                }
            }
            Event::Touch(touch::Event::FingerLifted { .. }) => {
                if let Some(selection) = &mut state.selection {
                    selection.dragging = false;
                }
            }
            Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                modifiers,
                physical_key,
                ..
            }) if self.selectable && state.focused => {
                let selection =
                    state.selection.get_or_insert_with(SelectionState::default);
                let count = content.graphemes(true).count();
                let mut handled = false;
                if modifiers.command() {
                    match key.to_latin(*physical_key) {
                        Some('c') => {
                            let left = selection.anchor.min(selection.end);
                            let right = selection.anchor.max(selection.end);
                            if left != right {
                                let selected: String = content
                                    .graphemes(true)
                                    .skip(left)
                                    .take(right - left)
                                    .collect();
                                clipboard.write(
                                    crate::core::clipboard::Kind::Standard,
                                    selected,
                                );
                            }
                            shell.capture_event();
                            handled = true;
                        }
                        Some('a') => {
                            selection.anchor = 0;
                            selection.end = count;
                            shell.capture_event();
                            handled = true;
                        }
                        _ => {}
                    }
                }
                if handled {
                    return;
                }

                match key {
                    keyboard::Key::Named(keyboard::key::Named::ArrowLeft) => {
                        if modifiers.shift() {
                            selection.end = selection.end.saturating_sub(1);
                        } else {
                            let position = selection
                                .anchor
                                .min(selection.end)
                                .saturating_sub(1);
                            selection.anchor = position;
                            selection.end = position;
                        }
                        shell.capture_event();
                    }
                    keyboard::Key::Named(keyboard::key::Named::ArrowRight) => {
                        if modifiers.shift() {
                            selection.end = (selection.end + 1).min(count);
                        } else {
                            let position =
                                (selection.anchor.max(selection.end) + 1)
                                    .min(count);
                            selection.anchor = position;
                            selection.end = position;
                        }
                        shell.capture_event();
                    }
                    keyboard::Key::Named(keyboard::key::Named::Home) => {
                        if modifiers.shift() {
                            selection.end = 0;
                        } else {
                            selection.anchor = 0;
                            selection.end = 0;
                        }
                        shell.capture_event();
                    }
                    keyboard::Key::Named(keyboard::key::Named::End) => {
                        if modifiers.shift() {
                            selection.end = count;
                        } else {
                            selection.anchor = count;
                            selection.end = count;
                        }
                        shell.capture_event();
                    }
                    _ => {}
                }
            }
            Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) => {
                if let Some(selection) = &mut state.selection {
                    selection.modifiers = *modifiers;
                }
            }
            _ => {}
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        if self.hovered_link.is_some() {
            mouse::Interaction::Pointer
        } else if self.selectable && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Text
        } else {
            mouse::Interaction::None
        }
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn crate::core::widget::Operation,
    ) {
        let state = tree
            .state
            .downcast_mut::<State<Link, Renderer::Paragraph>>();
        operation.text(None, layout.bounds(), &state.content);
        if self.selectable {
            operation.focusable(None, layout.bounds(), state);
        }
    }
}

impl<Link, Message, Theme, Renderer> HasSelectableText
    for Rich<'_, Link, Message, Theme, Renderer>
where
    Link: Clone + 'static,
    Theme: Catalog,
    Renderer: core::text::Renderer,
{
    fn selected_text(&self, tree: &Tree) -> Option<String> {
        let state = tree
            .state
            .downcast_ref::<State<Link, Renderer::Paragraph>>();
        let selection = state.selection.as_ref()?;
        let left = selection.anchor.min(selection.end);
        let right = selection.anchor.max(selection.end);
        if left == right {
            return None;
        }
        Some(
            state
                .content
                .graphemes(true)
                .skip(left)
                .take(right - left)
                .collect(),
        )
    }

    fn select_all(&self, tree: &mut Tree) {
        let state = tree
            .state
            .downcast_mut::<State<Link, Renderer::Paragraph>>();
        let count = state.content.graphemes(true).count();
        let selection =
            state.selection.get_or_insert_with(SelectionState::default);
        selection.anchor = 0;
        selection.end = count;
    }

    fn is_focused(&self, tree: &Tree) -> bool {
        tree.state
            .downcast_ref::<State<Link, Renderer::Paragraph>>()
            .focused
    }

    fn context_menu_position(&self, tree: &Tree) -> Option<Point> {
        tree.state
            .downcast_ref::<State<Link, Renderer::Paragraph>>()
            .context_menu_position
    }

    fn set_context_menu_position(&self, tree: &mut Tree, pos: Option<Point>) {
        tree.state
            .downcast_mut::<State<Link, Renderer::Paragraph>>()
            .context_menu_position = pos;
    }
}

fn grapheme_to_byte(content: &str, grapheme_index: usize) -> usize {
    content
        .graphemes(true)
        .take(grapheme_index)
        .map(str::len)
        .sum()
}

fn hit_to_grapheme<P: Paragraph>(
    paragraph: &P,
    point: Point,
    content: &str,
) -> usize {
    match paragraph.hit_test(point) {
        Some(hit) => content[..hit.cursor().min(content.len())]
            .graphemes(true)
            .count(),
        None => content.graphemes(true).count(),
    }
}

fn previous_start_of_word(content: &str, grapheme_index: usize) -> usize {
    let graphemes: Vec<&str> = content.graphemes(true).collect();
    let clamped = grapheme_index.min(graphemes.len());
    let before: String = graphemes[..clamped].concat();
    UnicodeSegmentation::split_word_bound_indices(before.as_str())
        .filter(|(_, word)| !word.trim_start().is_empty())
        .next_back()
        .map_or(0, |(index, word)| {
            clamped
                - word.graphemes(true).count()
                - before[index + word.len()..].graphemes(true).count()
        })
}

fn next_end_of_word(content: &str, grapheme_index: usize) -> usize {
    let graphemes: Vec<&str> = content.graphemes(true).collect();
    let clamped = grapheme_index.min(graphemes.len());
    let after: String = graphemes[clamped..].concat();
    UnicodeSegmentation::split_word_bound_indices(after.as_str())
        .find(|(_, word)| !word.trim_start().is_empty())
        .map_or(graphemes.len(), |(index, word)| {
            clamped
                + word.graphemes(true).count()
                + after[..index].graphemes(true).count()
        })
}

fn layout<Link, Renderer>(
    state: &mut State<Link, Renderer::Paragraph>,
    renderer: &Renderer,
    limits: &layout::Limits,
    width: Length,
    height: Length,
    spans: &[Span<'_, Link, Renderer::Font>],
    line_height: LineHeight,
    size: Option<Pixels>,
    font: Option<Renderer::Font>,
    align_x: Alignment,
    align_y: alignment::Vertical,
    wrapping: Wrapping,
    ellipsize: Ellipsize,
) -> layout::Node
where
    Link: Clone,
    Renderer: core::text::Renderer,
{
    layout::sized(limits, width, height, |limits| {
        let bounds = limits.max();

        let size = size.unwrap_or_else(|| renderer.default_size());
        let font = font.unwrap_or_else(|| renderer.default_font());

        let text_with_spans = || core::Text {
            content: spans,
            bounds,
            size,
            line_height,
            font,
            align_x,
            align_y,
            shaping: Shaping::Advanced,
            wrapping,
            ellipsize,
        };

        if state.spans != spans {
            state.paragraph =
                Renderer::Paragraph::with_spans(text_with_spans());
            state.spans = spans.iter().cloned().map(Span::to_static).collect();
            state.content =
                state.spans.iter().map(|span| span.text.as_ref()).collect();
        } else {
            match state.paragraph.compare(core::Text {
                content: (),
                bounds,
                size,
                line_height,
                font,
                align_x,
                align_y,
                shaping: Shaping::Advanced,
                wrapping,
                ellipsize,
            }) {
                core::text::Difference::None => {}
                core::text::Difference::Bounds => {
                    state.paragraph.resize(bounds);
                }
                core::text::Difference::Shape => {
                    state.paragraph =
                        Renderer::Paragraph::with_spans(text_with_spans());
                }
            }
        }

        state.paragraph.min_bounds()
    })
}

impl<'a, Link, Message, Theme, Renderer>
    FromIterator<Span<'a, Link, Renderer::Font>>
    for Rich<'a, Link, Message, Theme, Renderer>
where
    Link: Clone + 'a,
    Theme: Catalog,
    Renderer: core::text::Renderer,
    Renderer::Font: 'a,
{
    fn from_iter<T: IntoIterator<Item = Span<'a, Link, Renderer::Font>>>(
        spans: T,
    ) -> Self {
        Self::with_spans(spans.into_iter().collect::<Vec<_>>())
    }
}

impl<'a, Link, Message, Theme, Renderer>
    From<Rich<'a, Link, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Link: Clone + 'a,
    Theme: Catalog + 'a,
    Renderer: core::text::Renderer + 'a,
{
    fn from(
        text: Rich<'a, Link, Message, Theme, Renderer>,
    ) -> Element<'a, Message, Theme, Renderer> {
        Element::new(text)
    }
}
