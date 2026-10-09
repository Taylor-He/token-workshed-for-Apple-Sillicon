use super::*;

const WORKSHED_COMPACT_WIDTH: f32 = 640.0;
const WORKSHED_TEXT: Color = Color::from_rgb8(0x11, 0x11, 0x11);
const WORKSHED_MUTED_TEXT: Color = Color::from_rgb8(0x6b, 0x6b, 0x6b);
const WORKSHED_PANEL_WIDTH: f32 = 240.0;
const WORKSHED_LABEL_WIDTH: f32 = 180.0;
const WORKSHED_BLOCK_CATEGORIES: [(&str, &str); 9] = [
    ("models", "Models"),
    ("data", "Data"),
    ("distill", "Distill"),
    ("train", "Train"),
    ("quantize", "Quantize"),
    ("test_compare", "Test & Compare"),
    ("control", "Control"),
    ("logic", "Logic"),
    ("deliver", "Deliver"),
];

pub(super) fn render(app: &TokenWorkshedApp) -> Element<'_, Message> {
    let workspace = widget::responsive(move |size| {
        let available_width = size.width.max(0.0);
        let panel_open = app.workshed_inspector_open || app.workshed_library_open;
        let center_width = workshed_center_width(available_width, panel_open);
        let title = widget::column::with_capacity(2)
            .push(widget::text("Workshed").size(16).class(WORKSHED_TEXT))
            .push(
                widget::text(
                    "Create and optimize local models with a simple, configurable workflow.",
                )
                .size(12)
                .class(WORKSHED_MUTED_TEXT)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
            )
            .spacing(3)
            .width(Length::Fill);
        let actions = widget::row::with_capacity(2)
            .push(app.workshed_compact_button("New work order", Message::WorkshedNewPressed))
            .push(app.workshed_compact_button(
                if app.workshed_library_open {
                    "Close add panel"
                } else {
                    "+ Add block"
                },
                Message::WorkshedPanelToggled("library".into()),
            ))
            .spacing(8)
            .align_y(Alignment::Center);
        let header: Element<'_, Message> = if center_width < WORKSHED_COMPACT_WIDTH {
            widget::column::with_capacity(2)
                .push(title)
                .push(actions)
                .spacing(8)
                .into()
        } else {
            widget::row::with_capacity(2)
                .push(title)
                .push(actions)
                .spacing(12)
                .align_y(Alignment::Start)
                .into()
        };
        let center = widget::column::with_capacity(2)
            .push(header)
            .push(app.workshed_stack(center_width))
            .spacing(PAGE_SECTION_GAP)
            .width(Length::Fill)
            .height(Length::Fill);
        let mut workspace = widget::row::with_capacity(3).push(center);
        if panel_open {
            let divider = widget::container(widget::space())
                .width(Length::Fixed(1.0))
                .height(Length::Fill)
                .style(|_| {
                    container::Style::default().background(Background::Color(WORKSHED_RULE))
                });
            workspace = workspace.spacing(PAGE_SECTION_GAP).push(divider);
        }
        if app.workshed_library_open {
            workspace = workspace.push(app.workshed_add_panel());
        } else if app.workshed_inspector_open {
            workspace = workspace.push(app.workshed_inspector_panel());
        }
        workspace.height(Length::Fill).into()
    })
    .width(Length::Fill)
    .height(Length::Fill);

    let mut body = widget::column::with_capacity(3)
        .push(workspace.height(Length::Fill))
        .spacing(PAGE_SECTION_GAP)
        .width(Length::Fill)
        .height(Length::Fill);
    if app.workshed_preflight_open && !app.workshed_preflight.is_null() {
        body = body.push(app.workshed_plan_review());
    }
    body = body.push(app.workshed_footer());

    let page = widget::container(body)
        .padding(0)
        .width(Length::Fill)
        .height(Length::Fill);
    app.page_frame(iced::widget::Themer::new(Some(workshed_theme()), page).into())
}

impl TokenWorkshedApp {
    // A transient chooser keeps the canvas focused without removing editing
    // entry points. It is deliberately not the former persistent Block Library.
    fn workshed_add_panel(&self) -> Element<'_, Message> {
        let heading = widget::row::with_capacity(3)
            .push(widget::text("Add block").size(14).class(WORKSHED_TEXT))
            .push(widget::space().width(Length::Fill))
            .push(
                self.workshed_link_button("Close", Message::WorkshedPanelToggled("library".into())),
            )
            .align_y(Alignment::Center);
        let orders = workshed_order_choices(&self.workshed_work_orders, &self.workshed_order);
        let current_id = string_at(&self.workshed_order, "id");
        let selected_order = orders
            .iter()
            .find(|choice| choice.id == current_id)
            .cloned();
        let order_picker =
            iced::widget::pick_list(orders, selected_order, |choice: WorkshedOrderChoice| {
                Message::WorkshedWorkOrderSelectPressed(choice.id)
            })
            .placeholder("Current work order")
            .padding([7, 9])
            .text_size(12)
            .text_wrap(text::Wrapping::WordOrGlyph)
            .width(Length::Fill);
        let category_labels = WORKSHED_BLOCK_CATEGORIES
            .iter()
            .map(|(_, label)| label.to_string())
            .collect::<Vec<_>>();
        let selected_category = WORKSHED_BLOCK_CATEGORIES
            .iter()
            .find(|(id, _)| *id == self.workshed_category)
            .map(|(_, label)| label.to_string());
        let category_picker =
            iced::widget::pick_list(category_labels, selected_category, |label: String| {
                let id = WORKSHED_BLOCK_CATEGORIES
                    .iter()
                    .find(|(_, candidate)| *candidate == label)
                    .map(|(id, _)| *id)
                    .unwrap_or("models");
                Message::WorkshedCategoryPressed(id.to_string())
            })
            .padding([7, 9])
            .text_size(13)
            .width(Length::Fill);
        let query = self.workshed_search.trim().to_ascii_lowercase();
        let items = workshed_catalog_items(&self.workshed_capabilities, &self.workshed_category)
            .into_iter()
            .filter(|(_, label, _, _)| {
                query.is_empty() || label.to_ascii_lowercase().contains(&query)
            })
            .collect::<Vec<_>>();
        let mut choices = widget::column::with_capacity(items.len().max(1))
            .spacing(8)
            .width(Length::Fill);
        if items.is_empty() {
            choices = choices.push(self.workshed_help_text("No blocks match this search."));
        }
        for (type_id, label, available, reason) in items {
            let title = widget::text(label.clone())
                .size(13)
                .class(WORKSHED_TEXT)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph);
            let mut choice = widget::column::with_capacity(2).push(title).spacing(5);
            if !available {
                choice = choice.push(self.workshed_help_text(if reason.is_empty() {
                    "Unavailable on this Mac".to_string()
                } else {
                    reason
                }));
            }
            let button = iced::widget::button(choice)
                .padding([10, 10])
                .width(Length::Fill)
                .height(Length::Shrink)
                .class(iced_button_class(rounded_selector_button_style()))
                .name(format!("Add {label}"));
            choices = choices.push(if available {
                button.on_press(Message::WorkshedBlockAddPressed(type_id))
            } else {
                button
            });
        }
        let body = widget::column::with_capacity(8)
            .push(heading)
            .push(self.workshed_help_text("Work order"))
            .push(widget::container(order_picker).style(|_| workshed_control_style()))
            .push(self.caligo_horizontal_rule(false))
            .push(widget::container(category_picker).style(|_| workshed_control_style()))
            .push(self.workshed_library_search())
            .push(
                widget::scrollable(
                    widget::container(choices)
                        .padding([0, 10, 0, 0])
                        .width(Length::Fill),
                )
                .width(Length::Fill)
                .height(Length::Fill),
            )
            .spacing(10)
            .width(Length::Fill)
            .height(Length::Fill);
        widget::container(body)
            .padding([0, 0, 0, 0])
            .width(Length::Fixed(WORKSHED_PANEL_WIDTH))
            .height(Length::Fill)
            .clip(true)
            .style(|_| container::Style::default().color(WORKSHED_TEXT))
            .into()
    }

    fn workshed_stack(&self, available_width: f32) -> Element<'_, Message> {
        let compact = available_width < WORKSHED_COMPACT_WIDTH;
        let mut blocks = self
            .workshed_order
            .get("blocks")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter(|block| string_at(block, "type_id") != "model_development")
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        blocks.sort_by_key(|block| {
            let lane = match string_at(block, "lane").as_str() {
                "source" => 0,
                "improve" => 1,
                "verify" => 2,
                "deliver" => 3,
                _ => 4,
            };
            let order = block.get("order").and_then(Value::as_i64).unwrap_or(0);
            (lane, order)
        });

        let mut stack = widget::column::with_capacity(blocks.len() * 2 + 2).spacing(0);
        stack = stack.push(self.workshed_root_block(compact));
        for (index, block) in blocks.iter().enumerate() {
            stack = stack
                .push(self.workshed_connector(index == 0))
                .push(self.workshed_step_block(block, compact, available_width));
        }
        if blocks.is_empty() {
            stack = stack
                .push(self.workshed_connector(true))
                .push(self.workshed_empty_slot());
        }

        widget::scrollable(
            widget::container(stack)
                .padding([0, 8, 0, 0])
                .width(Length::Fill),
        )
        .height(Length::Fill)
        .class(theme::iced::Scrollable::Transient)
        .into()
    }

    fn workshed_root_block(&self, compact: bool) -> Element<'_, Message> {
        let name = self.workshed_model_name_control();
        if compact {
            let name_row = widget::row::with_capacity(3)
                .push(widget::text("(").size(15))
                .push(widget::container(name).width(Length::Fill))
                .push(widget::text(")").size(15))
                .spacing(4)
                .align_y(Alignment::Center)
                .width(Length::Fill);
            let content = widget::column::with_capacity(2)
                .push(name_row)
                .push(
                    widget::text("model development")
                        .size(14)
                        .class(WORKSHED_TEXT)
                        .width(Length::Fill)
                        .center(),
                )
                .spacing(5)
                .width(Length::Fill);
            self.workshed_surface(content, "root", None)
        } else {
            let row = widget::row::with_capacity(3)
                .push(widget::text("(").size(15))
                .push(widget::container(name).width(Length::Fixed(270.0)))
                .push(widget::text(") model development").size(15))
                .spacing(8)
                .align_y(Alignment::Center);
            let centered = widget::container(row)
                .width(Length::Fill)
                .align_x(Alignment::Center);
            self.workshed_surface(centered, "root", Some(64.0))
        }
    }

    fn workshed_step_block(
        &self,
        block: &Value,
        compact: bool,
        available_width: f32,
    ) -> Element<'_, Message> {
        match string_at(block, "type_id").as_str() {
            "load_model" => self.workshed_load_model_block(block, compact, available_width),
            "quantize_mlx" => self.workshed_quantize_block(block, compact),
            "register_model" => self.workshed_register_block(block, compact),
            _ => self.workshed_generic_block(block, compact),
        }
    }

    fn workshed_surface<'a>(
        &self,
        content: impl Into<Element<'a, Message>>,
        kind: &'static str,
        height: Option<f32>,
    ) -> Element<'a, Message> {
        widget::container(content.into())
            .padding([12, 16])
            .width(Length::Fill)
            .height(height.map_or(Length::Shrink, Length::Fixed))
            .style(move |_| workshed_surface_style(kind))
            .into()
    }

    fn workshed_connector(&self, first: bool) -> Element<'_, Message> {
        let _ = first;
        let line = widget::container(widget::space())
            .width(Length::Fixed(1.0))
            .height(Length::Fixed(12.0))
            .style(|_| {
                container::Style::default()
                    .background(Background::Color(Color::from_rgb8(0xc7, 0xcf, 0xdb)))
            });
        let port = widget::container(widget::space())
            .width(Length::Fixed(5.0))
            .height(Length::Fixed(5.0))
            .style(|_| {
                container::Style::default()
                    .background(Background::Color(WORKSHED_TEXT))
                    .border(Border {
                        radius: 3.0.into(),
                        ..Border::default()
                    })
            });
        widget::container(
            widget::column::with_capacity(2)
                .push(line)
                .push(port)
                .align_x(Alignment::Center),
        )
        .width(Length::Fill)
        .height(Length::Fixed(18.0))
        .align_x(Alignment::Center)
        .into()
    }

    fn workshed_empty_slot(&self) -> Element<'_, Message> {
        let content = workshed_centered(
            widget::text("Your work order is ready for development steps.")
                .size(13)
                .class(WORKSHED_MUTED_TEXT),
        );
        self.workshed_surface(content, "empty", Some(64.0))
    }

    fn workshed_icon(&self, category: &str) -> Element<'_, Message> {
        workshed_icon(workshed_category_icon_name(category), 18)
    }

    fn workshed_load_model_block(
        &self,
        block: &Value,
        compact: bool,
        available_width: f32,
    ) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let definition = workshed::definition_for(&self.workshed_capabilities, "load_model");
        let params = workshed::effective_params(block, &definition);
        let source_kind = params
            .get("source_kind")
            .and_then(Value::as_str)
            .unwrap_or("local");
        let reference = params.get("ref").and_then(Value::as_str).unwrap_or("");
        let subtitle = if reference.trim().is_empty() {
            "Select a downloaded model, local folder, or Hugging Face ID"
        } else {
            match source_kind {
                "local" => "Local weights · This Mac",
                "managed" => "Managed model · This Mac",
                _ => "Hugging Face · downloaded when needed",
            }
        };
        let picker_open = self.workshed_model_picker.as_deref() == Some(block_id.as_str());
        let selector_width = (available_width
            - if compact {
                120.0
            } else {
                WORKSHED_LABEL_WIDTH + 120.0
            })
        .max(100.0);
        let controls = widget::row::with_capacity(2)
            .push(self.workshed_dropdown_button(
                &workshed_model_reference_label(reference, source_kind),
                Message::WorkshedModelPickerToggled(block_id.clone()),
                selector_width,
            ))
            .push(self.workshed_compact_button(
                if picker_open { "Done" } else { "Change" },
                Message::WorkshedModelPickerToggled(block_id.clone()),
            ))
            .spacing(8)
            .align_y(Alignment::Center)
            .width(Length::Fill);
        let detail = widget::column::with_capacity(2)
            .push(controls)
            .push(
                widget::text(subtitle)
                    .size(12)
                    .class(WORKSHED_MUTED_TEXT)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph),
            )
            .spacing(6)
            .width(Length::Fill);
        let title = self.workshed_block_title("Load base model", "models", block_id.clone());
        let mut body = widget::column::with_capacity(5)
            .spacing(10)
            .width(Length::Fill);
        if compact {
            body = body.push(title).push(detail);
        } else {
            body = body.push(
                widget::row::with_capacity(2)
                    .push(widget::container(title).width(Length::Fixed(WORKSHED_LABEL_WIDTH)))
                    .push(detail)
                    .spacing(8)
                    .align_y(Alignment::Start)
                    .width(Length::Fill),
            );
        }
        if picker_open {
            body = body.push(self.workshed_model_source_selector(block, &definition));
            if let Some(field) = workshed::field(&definition, "ref") {
                body = body.push(self.workshed_param_control(block, &definition, &field, &[]));
            }
        }
        self.workshed_surface(body, "block", None)
    }

    fn workshed_block_title(
        &self,
        label: &str,
        category: &str,
        block_id: String,
    ) -> Element<'_, Message> {
        iced::widget::button(
            widget::row::with_capacity(2)
                .push(self.workshed_icon(category))
                .push(
                    widget::text(label.to_string())
                        .size(13)
                        .class(WORKSHED_TEXT),
                )
                .spacing(10)
                .align_y(Alignment::Center),
        )
        .padding([8, 0])
        .height(Length::Fixed(34.0))
        .class(iced_button_class(transparent_text_button_style()))
        .name(format!("Configure {label}"))
        .on_press(Message::WorkshedBlockSelectPressed(block_id))
        .into()
    }

    fn workshed_model_source_selector(
        &self,
        block: &Value,
        definition: &BlockDefinition,
    ) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let params = workshed::effective_params(block, definition);
        let current = params
            .get("source_kind")
            .and_then(Value::as_str)
            .unwrap_or("local");
        let Some(field) = workshed::field(definition, "source_kind") else {
            return widget::space().into();
        };
        let mut choices = widget::row::with_capacity(3).spacing(5);
        if let ParamKind::Enum(values) = &field.kind {
            for choice in values {
                let value = choice.value.as_str().unwrap_or("").to_string();
                if value.is_empty() {
                    continue;
                }
                let label = match value.as_str() {
                    "local" => "Local",
                    "huggingface" => "Hugging Face",
                    "managed" => "Managed",
                    _ => choice.label.as_str(),
                };
                choices = choices.push(self.small_toggle(
                    label,
                    value == current,
                    Message::WorkshedParamSet {
                        block_id: block_id.clone(),
                        field: field.key.clone(),
                        value: Value::String(value),
                    },
                ));
            }
        }
        widget::column::with_capacity(2)
            .push(widget::text("Model source").size(11))
            .push(choices)
            .spacing(4)
            .into()
    }

    fn workshed_quantize_block(&self, block: &Value, compact: bool) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let definition = workshed::definition_for(&self.workshed_capabilities, "quantize_mlx");
        let params = workshed::effective_params(block, &definition);
        let preset_id = params
            .get("preset_id")
            .and_then(Value::as_str)
            .unwrap_or("mlx-balanced");
        let bits = self.workshed_effective_quant_bits(block, &definition, preset_id);
        let bits_picker: Element<'_, Message> = if let Some(field) =
            workshed::field(&definition, "bits")
        {
            if let ParamKind::Enum(choices) = &field.kind {
                let entries = choices
                    .iter()
                    .filter_map(|choice| {
                        choice
                            .value
                            .as_i64()
                            .map(|n| (format!("{n}-bit"), choice.value.clone()))
                    })
                    .collect::<Vec<_>>();
                let options = entries
                    .iter()
                    .map(|(label, _)| label.clone())
                    .collect::<Vec<_>>();
                let selected = entries
                    .iter()
                    .find(|(_, value)| value.as_i64() == Some(bits))
                    .map(|(label, _)| label.clone());
                let id = block_id.clone();
                let picker = iced::widget::pick_list(options, selected, move |label: String| {
                    let value = entries
                        .iter()
                        .find(|(candidate, _)| candidate == &label)
                        .map(|(_, value)| value.clone())
                        .unwrap_or(Value::Null);
                    Message::WorkshedParamSet {
                        block_id: id.clone(),
                        field: "bits".into(),
                        value,
                    }
                })
                .padding([7, 9])
                .text_size(13)
                .width(Length::Fill);
                widget::container(picker)
                    .width(Length::FillPortion(4))
                    .style(|_| workshed_control_style())
                    .into()
            } else {
                self.workshed_dropdown_button(
                    &format!("{bits}-bit"),
                    Message::WorkshedBlockSelectPressed(block_id.clone()),
                    130.0,
                )
            }
        } else {
            self.workshed_dropdown_button(
                &format!("{bits}-bit"),
                Message::WorkshedBlockSelectPressed(block_id.clone()),
                130.0,
            )
        };
        let controls = widget::row::with_capacity(3)
            .push(bits_picker)
            .push(self.workshed_preset_picker(block, &definition, preset_id))
            .push(self.workshed_settings_button(block_id.clone()))
            .spacing(10)
            .align_y(Alignment::Center)
            .width(Length::Fill);
        let detail = widget::column::with_capacity(2)
            .push(controls)
            .push(
                widget::text("MLX · This Mac")
                    .size(12)
                    .class(WORKSHED_MUTED_TEXT),
            )
            .spacing(6)
            .width(Length::Fill);
        let title = self.workshed_block_title("Quantize", "quantize", block_id);
        let content: Element<'_, Message> = if compact {
            widget::column::with_capacity(2)
                .push(title)
                .push(detail)
                .spacing(8)
                .width(Length::Fill)
                .into()
        } else {
            widget::row::with_capacity(2)
                .push(widget::container(title).width(Length::Fixed(WORKSHED_LABEL_WIDTH)))
                .push(detail)
                .spacing(8)
                .align_y(Alignment::Start)
                .width(Length::Fill)
                .into()
        };
        self.workshed_surface(content, "block", None)
    }

    fn workshed_settings_button(&self, block_id: String) -> Element<'_, Message> {
        iced::widget::button(
            widget::container(workshed_icon("preferences-system-symbolic", 15))
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(Alignment::Center)
                .align_y(Alignment::Center),
        )
        .padding(0)
        .width(Length::Fixed(32.0))
        .height(Length::Fixed(32.0))
        .class(iced_button_class(rounded_selector_button_style()))
        .name("Open block settings")
        .on_press(Message::WorkshedBlockSelectPressed(block_id))
        .into()
    }

    fn workshed_effective_quant_bits(
        &self,
        block: &Value,
        definition: &BlockDefinition,
        preset_id: &str,
    ) -> i64 {
        let params = workshed::effective_params(block, definition);
        if let Some(bits) = params.get("bits").and_then(Value::as_i64) {
            return bits;
        }
        self.workshed_raw_presets(&definition.type_id)
            .and_then(|presets| {
                presets
                    .into_iter()
                    .find(|preset| string_at(preset, "id") == preset_id)
            })
            .and_then(|preset| {
                preset
                    .get("params")
                    .and_then(|params| params.get("bits"))
                    .and_then(Value::as_i64)
            })
            .unwrap_or(4)
    }

    fn workshed_preset_picker(
        &self,
        block: &Value,
        definition: &BlockDefinition,
        selected: &str,
    ) -> Element<'_, Message> {
        let presets = self
            .workshed_raw_presets(&definition.type_id)
            .unwrap_or_default();
        let entries = presets
            .iter()
            .map(|preset| (string_at(preset, "label"), string_at(preset, "id")))
            .filter(|(label, id)| !label.is_empty() && !id.is_empty())
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return widget::text("No compatible presets").size(12).into();
        }
        let options = entries
            .iter()
            .map(|(label, _)| label.clone())
            .collect::<Vec<_>>();
        let selected_label = entries
            .iter()
            .find(|(_, id)| id == selected)
            .map(|(label, _)| label.clone());
        let selection = entries;
        let block_id = string_at(block, "id");
        let picker = iced::widget::pick_list(options, selected_label, move |label: String| {
            let value = selection
                .iter()
                .find(|(candidate, _)| candidate == &label)
                .map(|(_, id)| id.clone())
                .unwrap_or_default();
            Message::WorkshedParamSet {
                block_id: block_id.clone(),
                field: "preset_id".into(),
                value: Value::String(value),
            }
        })
        .placeholder("Preset")
        .padding([7, 9])
        .text_size(13)
        .text_wrap(text::Wrapping::WordOrGlyph)
        .width(Length::Fill);
        widget::container(picker)
            .width(Length::FillPortion(5))
            .style(|_| workshed_control_style())
            .into()
    }

    fn workshed_register_block(&self, block: &Value, compact: bool) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let definition = workshed::definition_for(&self.workshed_capabilities, "register_model");
        let params = workshed::effective_params(block, &definition);
        let test_enabled = params
            .get("test_when_finished")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let output = if let Some(mut field) = workshed::field(&definition, "model_id") {
            if field.placeholder.is_empty() {
                field.placeholder = "Same as work order name".into();
            }
            self.workshed_param_text_input(&block_id, &field, params.get("model_id"))
        } else {
            widget::text("Loading model settings…").size(12).into()
        };
        let destination = widget::container(
            widget::row::with_capacity(2)
                .push(workshed_icon("folder-symbolic", 13))
                .push(widget::text("Managed Models").size(13).class(WORKSHED_TEXT))
                .spacing(8)
                .align_y(Alignment::Center),
        )
        .padding([7, 9])
        .height(Length::Fixed(38.0))
        .width(Length::Fill)
        .align_y(Alignment::Center)
        .style(|_| workshed_control_style());
        let controls: Element<'_, Message> = if compact {
            widget::column::with_capacity(2)
                .push(output)
                .push(
                    widget::row::with_capacity(2)
                        .push(widget::text("to").size(13))
                        .push(destination)
                        .spacing(8)
                        .align_y(Alignment::Center),
                )
                .spacing(8)
                .width(Length::Fill)
                .into()
        } else {
            widget::row::with_capacity(3)
                .push(widget::container(output).width(Length::Fill))
                .push(widget::text("to").size(13))
                .push(destination)
                .spacing(10)
                .align_y(Alignment::Center)
                .width(Length::Fill)
                .into()
        };
        let test = widget::checkbox(test_enabled)
            .label("Test when finished")
            .size(16)
            .text_size(13)
            .on_toggle(move |enabled| Message::WorkshedParamSet {
                block_id: block_id.clone(),
                field: "test_when_finished".into(),
                value: Value::Bool(enabled),
            });
        let detail = widget::column::with_capacity(2)
            .push(controls)
            .push(test)
            .spacing(8)
            .width(Length::Fill);
        let title = self.workshed_block_title("Save as", "deliver", string_at(block, "id"));
        let content: Element<'_, Message> = if compact {
            widget::column::with_capacity(2)
                .push(title)
                .push(detail)
                .spacing(8)
                .width(Length::Fill)
                .into()
        } else {
            widget::row::with_capacity(2)
                .push(widget::container(title).width(Length::Fixed(WORKSHED_LABEL_WIDTH)))
                .push(detail)
                .spacing(8)
                .align_y(Alignment::Start)
                .width(Length::Fill)
                .into()
        };
        self.workshed_surface(content, "block", None)
    }

    fn workshed_generic_block(&self, block: &Value, compact: bool) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let type_id = string_at(block, "type_id");
        let definition = workshed::definition_for(&self.workshed_capabilities, &type_id);
        let title = definition.label.clone();
        let summary = workshed::summary_parts(block, &definition, 3).join(" · ");
        let edit =
            self.workshed_link_button("Configure", Message::WorkshedBlockSelectPressed(block_id));
        let title = widget::row::with_capacity(2)
            .push(self.workshed_icon(&definition.category))
            .push(
                widget::text(title)
                    .size(13)
                    .class(WORKSHED_TEXT)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph),
            )
            .spacing(10)
            .align_y(Alignment::Center);
        let heading: Element<'_, Message> = if compact {
            widget::column::with_capacity(2)
                .push(title)
                .push(edit)
                .spacing(5)
                .width(Length::Fill)
                .into()
        } else {
            widget::row::with_capacity(3)
                .push(title)
                .push(widget::space().width(Length::Fill))
                .push(edit)
                .align_y(Alignment::Center)
                .width(Length::Fill)
                .into()
        };
        let mut body = widget::column::with_capacity(4).push(heading).spacing(6);
        if !summary.is_empty() {
            body = body.push(
                widget::text(summary)
                    .size(12)
                    .class(WORKSHED_MUTED_TEXT)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph)
                    .center(),
            );
        }
        if !definition.available {
            body = body.push(
                widget::text(definition.unavailable_reason)
                    .size(11)
                    .class(WORKSHED_MUTED_TEXT)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph)
                    .center(),
            );
        }
        self.workshed_surface(body, "block", None)
    }

    fn workshed_inline_advanced(&self, block_id: &str) -> Element<'_, Message> {
        let open = self.workshed_advanced_blocks.contains(block_id);
        self.workshed_link_button(
            if open { "Hide advanced" } else { "Advanced" },
            Message::WorkshedAdvancedToggled(block_id.to_string()),
        )
    }

    fn workshed_dropdown_button(
        &self,
        label: &str,
        message: Message,
        width: f32,
    ) -> Element<'_, Message> {
        let label = workshed_fitted_selector_label(label, width);
        let content = widget::row::with_capacity(2)
            .push(
                widget::text(label)
                    .size(13)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::None),
            )
            .push(workshed_icon("pan-down-symbolic", 12))
            .spacing(4)
            .align_y(Alignment::Center);
        let clipped_content = widget::container(content)
            .width(Length::Fixed((width - 20.0).max(0.0)))
            .height(Length::Fixed(20.0))
            .clip(true)
            .align_y(Alignment::Center);
        iced::widget::button(clipped_content)
            .padding([7, 10])
            .width(Length::Fixed(width))
            .height(Length::Fixed(34.0))
            .class(iced_button_class(rounded_selector_button_style()))
            .on_press(message)
            .into()
    }

    fn workshed_link_button(&self, label: &str, message: Message) -> Element<'_, Message> {
        iced::widget::button(widget::text(label.to_string()).size(12))
            .padding([4, 6])
            .class(iced_button_class(secondary_button_style(
                self.active_palette(),
                Accent::Black,
                false,
            )))
            .on_press(message)
            .into()
    }

    fn workshed_raw_presets(&self, type_id: &str) -> Option<Vec<Value>> {
        self.workshed_capabilities
            .get("blocks")
            .and_then(Value::as_array)
            .and_then(|blocks| {
                blocks
                    .iter()
                    .find(|item| string_at(item, "type_id") == type_id)
            })
            .and_then(|item| item.get("presets"))
            .and_then(Value::as_array)
            .cloned()
    }

    fn workshed_footer(&self) -> Element<'_, Message> {
        if let Some(run) = &self.workshed_run {
            let status = string_at(run, "status");
            let active = !matches!(
                status.as_str(),
                "completed" | "failed" | "cancelled" | "interrupted"
            );
            if !active {
                let summary = match status.as_str() {
                    "completed" => "Last run completed · 100%".to_string(),
                    "failed" => format!("Failed · {}", string_at(run, "error")),
                    "cancelled" => "Cancelled · validated artifacts were kept".to_string(),
                    _ => "Interrupted · retry from the last completed step".to_string(),
                };
                let mut row = widget::row::with_capacity(4)
                    .push(
                        widget::text(summary)
                            .size(13)
                            .width(Length::Fill)
                            .wrapping(text::Wrapping::WordOrGlyph),
                    )
                    .push(
                        self.workshed_compact_button("Open logs", Message::PagePressed(Page::Logs)),
                    );
                if status == "completed" {
                    row = row.push(self.workshed_compact_button(
                        "View result",
                        Message::PagePressed(Page::Models),
                    ));
                }
                row = row
                    .push(self.workshed_compact_button("Run again", Message::WorkshedRunPressed));
                return self.workshed_footer_shell(row.spacing(10).align_y(Alignment::Center));
            }
            if active {
                let stage = string_at(run, "stage");
                let message = string_at(run, "message");
                let progress = run.get("progress").and_then(Value::as_f64).unwrap_or(0.0) * 100.0;
                let eta = run
                    .get("eta_seconds")
                    .and_then(Value::as_u64)
                    .map(|seconds| {
                        if seconds >= 60 {
                            format!("~{}m {}s", seconds / 60, seconds % 60)
                        } else {
                            format!("~{seconds}s")
                        }
                    })
                    .unwrap_or_else(|| "calculating".into());
                let allowed = run
                    .get("allowed_actions")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default();
                let detail = widget::column::with_capacity(3)
                    .push(
                        widget::text(format!("{stage} · {progress:.0}% · ETA {eta}"))
                            .size(13)
                            .width(Length::Fill)
                            .wrapping(text::Wrapping::WordOrGlyph),
                    )
                    .push(
                        iced::widget::progress_bar(0.0..=100.0, progress as f32)
                            .girth(6.0)
                            .class(theme::ProgressBar::custom(|_| {
                                iced::widget::progress_bar::Style {
                                    background: Background::Color(WORKSHED_RULE),
                                    bar: Background::Color(WORKSHED_TEXT),
                                    border: Border {
                                        radius: 3.0.into(),
                                        ..Border::default()
                                    },
                                }
                            })),
                    )
                    .push(
                        widget::text(message)
                            .size(11)
                            .class(WORKSHED_MUTED_TEXT)
                            .width(Length::Fill)
                            .wrapping(text::Wrapping::WordOrGlyph),
                    )
                    .spacing(6)
                    .width(Length::Fill);
                let mut row = widget::row::with_capacity(3).push(detail).push(
                    self.workshed_compact_button("Open logs", Message::PagePressed(Page::Logs)),
                );
                if allowed.contains("cancel") {
                    row = row.push(
                        iced::widget::button(centered_text("Cancel".into(), 12))
                            .height(Length::Fixed(34.0))
                            .padding([0, 16])
                            .class(iced_button_class(primary_button_style(Accent::Black)))
                            .on_press(Message::WorkshedActionPressed("cancel".into())),
                    );
                }
                return self.workshed_footer_shell(row.spacing(10).align_y(Alignment::Center));
            }
        }

        let ready_text = if !self.workshed_preflight.is_null() {
            if self.workshed_preflight_ok() {
                "Ready · preflight passed"
            } else if self.workshed_preflight_open {
                "Review the plan before running"
            } else {
                "Ready"
            }
        } else {
            "Ready"
        };
        let run_label = self.workshed_run_label();
        let mut action = iced::widget::button(centered_text(run_label, 12))
            .padding([0, 16])
            .height(Length::Fixed(34.0))
            .width(Length::Fixed(160.0))
            .class(iced_button_class(primary_button_style(Accent::Black)));
        if self.workshed_field_errors.is_empty() {
            action = action.on_press(Message::WorkshedRunPressed);
        }
        let row = widget::row::with_capacity(4)
            .push(
                widget::text(ready_text)
                    .size(13)
                    .width(Length::FillPortion(1))
                    .wrapping(text::Wrapping::WordOrGlyph),
            )
            .push(widget::space().width(Length::Fill))
            .push(action)
            .spacing(12)
            .align_y(Alignment::Center);
        self.workshed_footer_shell(row)
    }

    fn workshed_run_label(&self) -> String {
        let blocks = self
            .workshed_order
            .get("blocks")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let has_quantize = blocks
            .iter()
            .any(|block| string_at(block, "type_id") == "quantize_mlx");
        let has_other_work = blocks.iter().any(|block| {
            !matches!(
                string_at(block, "type_id").as_str(),
                "model_development" | "load_model" | "quantize_mlx" | "register_model"
            )
        });
        if has_quantize && !has_other_work {
            "Run quantization".into()
        } else {
            "Run work order".into()
        }
    }

    fn workshed_compact_button(&self, label: &str, message: Message) -> Element<'_, Message> {
        iced::widget::button(widget::text(label.to_string()).size(12))
            .padding([6, 10])
            .height(Length::Fixed(34.0))
            .class(iced_button_class(secondary_button_style(
                self.active_palette(),
                Accent::Black,
                false,
            )))
            .on_press(message)
            .into()
    }

    fn workshed_footer_shell<'a>(
        &self,
        content: impl Into<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        let divider = widget::container(widget::space())
            .width(Length::Fill)
            .height(Length::Fixed(1.0))
            .style(|_| container::Style::default().background(Background::Color(WORKSHED_RULE)));
        widget::column::with_capacity(2)
            .push(divider)
            .push(
                widget::container(content.into())
                    .padding([14, 4])
                    .width(Length::Fill),
            )
            .spacing(0)
            .width(Length::Fill)
            .into()
    }

    fn workshed_plan_review(&self) -> Element<'_, Message> {
        let errors = workshed_messages(&self.workshed_preflight, "errors");
        let warnings = workshed_messages(&self.workshed_preflight, "warnings");
        let consents = self
            .workshed_preflight
            .get("required_consents")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let estimate = self
            .workshed_preflight
            .get("estimates")
            .unwrap_or(&Value::Null);
        let estimate_text = format!(
            "{} steps · {} MB output · {} GB memory",
            estimate
                .get("step_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            estimate
                .get("estimated_output_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                / 1024
                / 1024,
            estimate
                .get("estimated_memory_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                / 1024
                / 1024
                / 1024,
        );
        let mut content = widget::column::with_capacity(6)
            .push(widget::text("Preflight plan").size(14))
            .push(
                widget::text(estimate_text)
                    .size(12)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph),
            )
            .spacing(6);
        for error in errors.iter().take(3) {
            content = content.push(
                widget::text(format!("Fix · {error}"))
                    .size(11)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph),
            );
        }
        for warning in warnings.iter().take(2) {
            content = content.push(
                widget::text(format!("Note · {warning}"))
                    .size(11)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph),
            );
        }
        if consents
            .iter()
            .any(|item| item.as_str() == Some("toolchain_install"))
        {
            content = content.push(
                widget::text(
                    "Preparation will download the listed, versioned tools into Workshed's managed tool folder.",
                )
                .size(11)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
            );
        }
        if consents
            .iter()
            .any(|item| item.as_str() == Some("trust_remote_code"))
        {
            content = content.push(
                widget::text(
                    "This plan requires permission to run code from the selected model repository.",
                )
                .size(11)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
            );
        }
        let mut actions = widget::row::with_capacity(2).spacing(8);
        if self.workshed_preflight_ok() && !consents.is_empty() {
            actions = actions.push(
                self.workshed_compact_button("Approve & run", Message::WorkshedConfirmRunPressed),
            );
        }
        actions =
            actions.push(self.workshed_compact_button(
                "Close",
                Message::WorkshedPanelToggled("preflight".into()),
            ));
        content = content.push(actions);
        widget::container(content)
            .padding([10, 14])
            .width(Length::Fill)
            .style(|_| workshed_surface_style("review"))
            .into()
    }
}

const WORKSHED_RULE: Color = Color::from_rgb8(0xdf, 0xe3, 0xe9);

fn workshed_theme() -> Theme {
    let mut theme = Theme::light().cosmic().clone();
    let black = theme::CosmicColor::new(0.0667, 0.0667, 0.0667, 1.0);
    theme.accent.base = black;
    theme.accent.hover = black;
    theme.accent.pressed = black;
    theme.accent.selected = black;
    theme.accent.on = theme::CosmicColor::new(1.0, 1.0, 1.0, 1.0);
    Theme::custom(std::sync::Arc::new(theme))
}

fn workshed_icon(name: &'static str, size: u16) -> Element<'static, Message> {
    widget::icon::from_name(name)
        .size(size)
        .icon()
        .class(theme::Svg::custom(|_| cosmic::iced::widget::svg::Style {
            color: Some(WORKSHED_TEXT),
        }))
        .into()
}

fn workshed_centered<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    widget::container(content.into())
        .width(Length::Fill)
        .align_x(Alignment::Center)
        .into()
}

fn workshed_center_width(available_width: f32, inspector_open: bool) -> f32 {
    let available_width = available_width.max(0.0);
    if inspector_open {
        (available_width - WORKSHED_PANEL_WIDTH - 2.0 * PAGE_SECTION_GAP as f32 - 1.0).max(0.0)
    } else {
        available_width
    }
}

fn workshed_model_reference_label(reference: &str, source_kind: &str) -> String {
    let reference = reference.trim();
    if reference.is_empty() {
        return "Choose base model".to_string();
    }
    if source_kind != "local" {
        return reference.to_string();
    }
    // HF snapshots end in an opaque commit. Recover the repo from its cache
    // directory for display only; the editable reference remains unchanged.
    for component in std::path::Path::new(reference).components() {
        if let Some(repo) = component
            .as_os_str()
            .to_str()
            .and_then(|part| part.strip_prefix("models--"))
        {
            if !repo.is_empty() {
                return repo.replace("--", "/");
            }
        }
    }
    std::path::Path::new(reference)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(reference)
        .to_string()
}

fn workshed_fitted_selector_label(label: &str, width: f32) -> String {
    // Reserve padding, arrow, and spacing, and budget conservatively for the
    // 13px UI font. Clipping is also enabled as a final glyph-width guard.
    let max_chars = (((width - 38.0).max(0.0) / 8.0).floor() as usize).max(1);
    let chars = label.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        label.to_string()
    } else if max_chars == 1 {
        "…".to_string()
    } else {
        format!("{}…", chars[..max_chars - 1].iter().collect::<String>())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkshedOrderChoice {
    id: String,
    label: String,
}

impl std::fmt::Display for WorkshedOrderChoice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label)
    }
}

fn workshed_order_choices(orders: &[Value], current: &Value) -> Vec<WorkshedOrderChoice> {
    let current_id = string_at(current, "id");
    let mut orders = orders.to_vec();
    if !current_id.is_empty() {
        orders.retain(|order| string_at(order, "id") != current_id);
        orders.insert(0, current.clone());
    }
    orders
        .into_iter()
        .filter(|order| !string_at(order, "id").is_empty())
        .map(|order| {
            let id = string_at(&order, "id");
            let name = string_at(&order, "name");
            WorkshedOrderChoice {
                label: workshed_fitted_selector_label(
                    &format!(
                        "{} · {}",
                        if name.trim().is_empty() {
                            "Untitled"
                        } else {
                            name.as_str()
                        },
                        &id.chars().take(6).collect::<String>()
                    ),
                    280.0,
                ),
                id,
            }
        })
        .collect()
}

fn workshed_surface_style(kind: &str) -> container::Style {
    let background = match kind {
        "review" => Color::from_rgb8(0xf8, 0xfa, 0xfc),
        _ => Color::from_rgb8(0xf9, 0xfa, 0xfc),
    };
    container::Style::default()
        .background(Background::Color(background))
        .color(WORKSHED_TEXT)
        .border(Border {
            width: 1.0,
            radius: CONTENT_RADIUS.into(),
            color: Color::from_rgb8(0xdc, 0xe1, 0xe9),
        })
        .shadow(Shadow::default())
}

fn workshed_select_surface<'a>(
    content: impl Into<Element<'a, Message>>,
    width: f32,
) -> Element<'a, Message> {
    widget::container(content.into())
        .width(Length::Fixed(width))
        .style(|_| workshed_control_style())
        .into()
}

fn workshed_control_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::from_rgb8(0xff, 0xff, 0xff)))
        .color(WORKSHED_TEXT)
        .border(Border {
            width: 1.0,
            radius: CONTENT_RADIUS.into(),
            color: Color::from_rgb8(0xdc, 0xe1, 0xe9),
        })
        .shadow(Shadow::default())
}

#[cfg(test)]
mod layout_tests {
    use super::{
        WORKSHED_COMPACT_WIDTH, workshed_center_width, workshed_control_style,
        workshed_fitted_selector_label, workshed_model_reference_label, workshed_order_choices,
    };
    use crate::appearance::CONTENT_RADIUS;
    use serde_json::json;

    #[test]
    fn workshed_selectors_use_the_shared_content_radius() {
        assert_eq!(
            workshed_control_style().border.radius,
            CONTENT_RADIUS.into()
        );
    }

    #[test]
    fn workshed_native_controls_use_monochrome_accents() {
        let theme = super::workshed_theme();
        let accent: cosmic::iced::Color = theme.cosmic().accent.base.into();
        assert!((accent.r - accent.g).abs() < f32::EPSILON);
        assert!((accent.g - accent.b).abs() < f32::EPSILON);
        assert!(accent.r < 0.1);
    }

    #[test]
    fn center_width_never_exceeds_the_available_canvas() {
        assert_eq!(workshed_center_width(620.0, false), 620.0);
        assert_eq!(workshed_center_width(980.0, false), 980.0);
        assert_eq!(workshed_center_width(640.0, true), 375.0);
    }

    #[test]
    fn side_panel_widths_trigger_compact_block_layouts() {
        assert!(workshed_center_width(640.0, true) < WORKSHED_COMPACT_WIDTH);
        assert!(workshed_center_width(760.0, false) >= WORKSHED_COMPACT_WIDTH);
    }

    #[test]
    fn local_snapshot_labels_show_the_repository_not_the_commit() {
        assert_eq!(
            workshed_model_reference_label(
                "/Users/test/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-350m/snapshots/3b17b717b8f2f5d305b0a92c1491e239aeda19c8",
                "local",
            ),
            "ibm-granite/granite-4.0-h-350m"
        );
        assert_eq!(
            workshed_model_reference_label("/Volumes/Models/My Model", "local"),
            "My Model"
        );
        assert_eq!(
            workshed_model_reference_label("org/base-model", "huggingface"),
            "org/base-model"
        );
        assert_eq!(
            workshed_model_reference_label("", "local"),
            "Choose base model"
        );
    }

    #[test]
    fn selector_labels_remain_bounded_at_compact_widths() {
        let name = "模型很长This-is-a-very-long-model-name-with-no-space";
        let narrow = workshed_fitted_selector_label(name, 132.0);
        let wide = workshed_fitted_selector_label(name, 245.0);
        assert!(narrow.chars().count() <= 11);
        assert!(wide.chars().count() <= 25);
        assert!(narrow.ends_with('…'));
        assert_eq!(workshed_fitted_selector_label("short", 245.0), "short");
        assert_eq!(workshed_fitted_selector_label(name, 0.0), "…");
    }

    #[test]
    fn order_chooser_keeps_the_original_and_current_order() {
        let original = json!({"id": "local-draft", "name": "Original work order"});
        let current = json!({"id": "work-order-test", "name": "Temporary quantization"});
        let choices = workshed_order_choices(&[original, current.clone()], &current);
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].id, "work-order-test");
        assert_eq!(choices[1].id, "local-draft");
    }
}
