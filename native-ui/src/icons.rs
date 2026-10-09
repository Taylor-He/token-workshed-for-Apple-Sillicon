use crate::appearance::{Page, SIDEBAR_ICON_VIEWBOX};
use cosmic::iced::widget::Canvas;
use cosmic::iced::widget::canvas::{self, Geometry, Path, Stroke, stroke};
use cosmic::iced::{Color, Length, Point, Rectangle, Size, border, mouse};
use cosmic::{Element, Renderer, Theme};

#[derive(Debug, Clone, Copy)]
struct SidebarIcon {
    page: Page,
    color: Color,
}

#[derive(Debug, Clone, Copy)]
enum UtilityIconKind {
    Send,
    Plus,
}

#[derive(Debug, Clone, Copy)]
struct UtilityIcon {
    kind: UtilityIconKind,
    color: Color,
}

#[derive(Debug, Clone, Copy)]
struct BlockNotch {
    color: Color,
    fill: Color,
}

pub fn sidebar_icon<Message: 'static>(
    page: Page,
    color: Color,
    scale: f32,
) -> Element<'static, Message> {
    let size = (SIDEBAR_ICON_VIEWBOX * scale).round();
    Canvas::<SidebarIcon, Message, Theme, Renderer>::new(SidebarIcon { page, color })
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .into()
}

pub fn send_icon<Message: 'static>(color: Color, size: f32) -> Element<'static, Message> {
    utility_icon(UtilityIconKind::Send, color, size)
}

pub fn plus_icon<Message: 'static>(color: Color, size: f32) -> Element<'static, Message> {
    utility_icon(UtilityIconKind::Plus, color, size)
}

/// Draws the small interlocking tab used by the immutable Workshed root block.
///
/// The canvas is intentionally limited to the outline. All editable content
/// remains native widgets, so focus, text input, and accessibility semantics
/// are unaffected by the decorative block silhouette.
pub fn block_notch<Message: 'static>(
    color: Color,
    fill: Color,
    width: f32,
    height: f32,
) -> Element<'static, Message> {
    Canvas::<BlockNotch, Message, Theme, Renderer>::new(BlockNotch { color, fill })
        .width(Length::Fixed(width))
        .height(Length::Fixed(height))
        .into()
}

fn utility_icon<Message: 'static>(
    kind: UtilityIconKind,
    color: Color,
    size: f32,
) -> Element<'static, Message> {
    Canvas::<UtilityIcon, Message, Theme, Renderer>::new(UtilityIcon { kind, color })
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .into()
}

impl<Message> canvas::Program<Message, Theme, Renderer> for SidebarIcon {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let edge = bounds.width.min(bounds.height);
        let scale = edge / 24.0;
        let origin_x = (bounds.width - edge) / 2.0;
        let origin_y = (bounds.height - edge) / 2.0;
        let stroke = Stroke {
            style: stroke::Style::Solid(self.color),
            width: 1.75 * scale,
            line_cap: stroke::LineCap::Round,
            line_join: stroke::LineJoin::Round,
            ..Stroke::default()
        };

        let p = |x: f32, y: f32| Point::new(origin_x + x * scale, origin_y + y * scale);
        let sz = |w: f32, h: f32| Size::new(w * scale, h * scale);
        let rr = |v: f32| border::Radius::from(v * scale);

        match self.page {
            Page::Chat => {
                frame.stroke(
                    &Path::rounded_rectangle(p(4.0, 5.0), sz(16.0, 12.0), rr(5.0)),
                    stroke,
                );
                frame.stroke(
                    &Path::new(|b| {
                        b.move_to(p(8.0, 17.0));
                        b.line_to(p(8.0, 20.0));
                        b.line_to(p(11.1, 17.4));
                        b.line_to(p(15.0, 17.4));
                    }),
                    stroke,
                );
            }
            Page::Models => {
                for y in [4.5, 10.0, 15.5] {
                    frame.stroke(
                        &Path::rounded_rectangle(p(3.5, y), sz(17.0, 4.2), rr(1.2)),
                        stroke,
                    );
                }
                for y in [6.6, 12.1, 17.6] {
                    frame.stroke(&Path::circle(p(8.0, y), 0.85 * scale), stroke);
                }
            }
            Page::Workshed => {
                // Interlocking blocks read as a compact model-workbench mark
                // at the 20 px sidebar scale.
                frame.stroke(
                    &Path::rounded_rectangle(p(4.0, 14.6), sz(4.0, 5.0), rr(1.0)),
                    stroke,
                );
                frame.stroke(
                    &Path::rounded_rectangle(p(10.0, 10.2), sz(4.0, 9.4), rr(1.0)),
                    stroke,
                );
                frame.stroke(
                    &Path::rounded_rectangle(p(16.0, 5.8), sz(4.0, 13.8), rr(1.0)),
                    stroke,
                );
                frame.stroke(&Path::line(p(5.0, 9.0), p(9.0, 5.0)), stroke);
                frame.stroke(&Path::line(p(7.2, 5.0), p(9.0, 5.0)), stroke);
                frame.stroke(&Path::line(p(9.0, 5.0), p(9.0, 6.8)), stroke);
            }
            Page::Server => {
                frame.stroke(
                    &Path::rounded_rectangle(p(3.5, 4.5), sz(17.0, 4.4), rr(1.2)),
                    stroke,
                );
                frame.stroke(
                    &Path::rounded_rectangle(p(3.5, 10.0), sz(17.0, 4.4), rr(1.2)),
                    stroke,
                );
                frame.stroke(&Path::circle(p(17.6, 17.8), 2.8 * scale), stroke);
                frame.stroke(&Path::line(p(17.6, 15.5), p(17.6, 17.8)), stroke);
                frame.stroke(&Path::line(p(17.6, 17.8), p(18.9, 17.8)), stroke);
                frame.stroke(&Path::circle(p(8.0, 6.7), 0.85 * scale), stroke);
                frame.stroke(&Path::circle(p(8.0, 12.2), 0.85 * scale), stroke);
            }
            Page::Logs => {
                frame.stroke(
                    &Path::rounded_rectangle(p(3.5, 4.5), sz(17.0, 15.0), rr(2.2)),
                    stroke,
                );
                frame.stroke(
                    &Path::new(|b| {
                        b.move_to(p(7.2, 9.4));
                        b.line_to(p(10.1, 12.0));
                        b.line_to(p(7.2, 14.6));
                    }),
                    stroke,
                );
                frame.stroke(&Path::line(p(12.4, 14.6), p(16.6, 14.6)), stroke);
            }
            Page::Community => {
                frame.stroke(&Path::circle(p(12.0, 12.0), 6.6 * scale), stroke);
                frame.stroke(&Path::circle(p(9.2, 10.7), 0.7 * scale), stroke);
                frame.stroke(&Path::circle(p(14.8, 10.7), 0.7 * scale), stroke);
                frame.stroke(
                    &Path::new(|b| {
                        b.move_to(p(9.0, 13.4));
                        b.quadratic_curve_to(p(12.0, 16.0), p(15.0, 13.4));
                    }),
                    stroke,
                );
                frame.stroke(
                    &Path::new(|b| {
                        b.move_to(p(6.2, 12.1));
                        b.quadratic_curve_to(p(7.7, 11.8), p(8.8, 13.2));
                    }),
                    stroke,
                );
                frame.stroke(
                    &Path::new(|b| {
                        b.move_to(p(17.8, 12.1));
                        b.quadratic_curve_to(p(16.3, 11.8), p(15.2, 13.2));
                    }),
                    stroke,
                );
                frame.stroke(&Path::line(p(7.4, 8.2), p(5.8, 6.9)), stroke);
                frame.stroke(&Path::line(p(16.6, 8.2), p(18.2, 6.9)), stroke);
            }
            Page::Settings => {
                frame.stroke(&Path::circle(p(12.0, 12.0), 2.5 * scale), stroke);
                let gear = Path::new(|b| {
                    b.move_to(p(19.0, 12.0));
                    b.line_to(p(20.8, 10.4));
                    b.line_to(p(19.0, 6.0));
                    b.line_to(p(16.5, 7.0));
                    b.line_to(p(14.9, 6.1));
                    b.line_to(p(14.5, 3.5));
                    b.line_to(p(9.5, 3.5));
                    b.line_to(p(9.1, 6.1));
                    b.line_to(p(7.5, 7.0));
                    b.line_to(p(5.0, 6.0));
                    b.line_to(p(3.0, 9.5));
                    b.line_to(p(5.1, 11.1));
                    b.line_to(p(5.0, 12.9));
                    b.line_to(p(3.0, 14.5));
                    b.line_to(p(5.0, 18.0));
                    b.line_to(p(7.5, 17.0));
                    b.line_to(p(9.1, 17.9));
                    b.line_to(p(9.5, 20.5));
                    b.line_to(p(14.5, 20.5));
                    b.line_to(p(14.9, 17.9));
                    b.line_to(p(16.5, 17.0));
                    b.line_to(p(19.0, 18.0));
                    b.line_to(p(21.0, 14.5));
                    b.line_to(p(18.9, 12.9));
                    b.close();
                });
                frame.stroke(&gear, stroke);
            }
            Page::About => {
                frame.stroke(&Path::circle(p(12.0, 12.0), 8.0 * scale), stroke);
                frame.stroke(&Path::line(p(12.0, 10.8), p(12.0, 15.8)), stroke);
                frame.stroke(&Path::line(p(10.9, 15.8), p(13.1, 15.8)), stroke);
                frame.stroke(&Path::circle(p(12.0, 8.2), 0.9 * scale), stroke);
            }
        }

        vec![frame.into_geometry()]
    }
}

impl<Message> canvas::Program<Message, Theme, Renderer> for UtilityIcon {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let edge = bounds.width.min(bounds.height);
        let scale = edge / 24.0;
        let origin_x = (bounds.width - edge) / 2.0;
        let origin_y = (bounds.height - edge) / 2.0;
        let p = |x: f32, y: f32| Point::new(origin_x + x * scale, origin_y + y * scale);
        let stroke = Stroke {
            style: stroke::Style::Solid(self.color),
            width: 1.75 * scale,
            line_cap: stroke::LineCap::Round,
            line_join: stroke::LineJoin::Round,
            ..Stroke::default()
        };

        match self.kind {
            UtilityIconKind::Send => {
                frame.stroke(&Path::line(p(12.0, 17.0), p(12.0, 7.0)), stroke);
                frame.stroke(
                    &Path::new(|b| {
                        b.move_to(p(7.5, 11.5));
                        b.line_to(p(12.0, 7.0));
                        b.line_to(p(16.5, 11.5));
                    }),
                    stroke,
                );
            }
            UtilityIconKind::Plus => {
                frame.stroke(&Path::line(p(12.0, 6.0), p(12.0, 18.0)), stroke);
                frame.stroke(&Path::line(p(6.0, 12.0), p(18.0, 12.0)), stroke);
            }
        }

        vec![frame.into_geometry()]
    }
}

impl<Message> canvas::Program<Message, Theme, Renderer> for BlockNotch {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let center = bounds.width / 2.0;
        let inset = 1.0;
        let tip_y = (bounds.height - 1.0).max(inset);
        let half_width = ((bounds.width - 2.0) / 2.0).max(1.0);
        let left = Point::new(center - half_width, inset);
        let tip = Point::new(center, tip_y);
        let right = Point::new(center + half_width, inset);
        let fill = Path::new(|builder| {
            builder.move_to(left);
            builder.line_to(tip);
            builder.line_to(right);
            builder.close();
        });
        frame.fill(&fill, self.fill);
        frame.stroke(
            &Path::new(|builder| {
                builder.move_to(left);
                builder.line_to(tip);
                builder.line_to(right);
            }),
            Stroke {
                style: stroke::Style::Solid(self.color),
                width: 1.5,
                line_cap: stroke::LineCap::Round,
                line_join: stroke::LineJoin::Round,
                ..Stroke::default()
            },
        );

        vec![frame.into_geometry()]
    }
}
