use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    String,
    Integer,
    Number,
    Boolean,
    Enum(Vec<EnumChoice>),
    IntegerOrString(Vec<EnumChoice>),
    StringArray,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumChoice {
    pub value: Value,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParamField {
    pub key: String,
    pub label: String,
    pub description: String,
    pub kind: ParamKind,
    pub default: Option<Value>,
    pub required: bool,
    pub nullable: bool,
    pub advanced: bool,
    pub summary: bool,
    pub quick: bool,
    pub bindable: bool,
    pub group: String,
    pub order: i64,
    pub unit: String,
    pub placeholder: String,
    pub widget: String,
    pub options_provider: Value,
    pub options_context: Map<String, Value>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub exclusive_minimum: bool,
    pub exclusive_maximum: bool,
    pub minimum_length: Option<usize>,
    pub maximum_length: Option<usize>,
    pub minimum_items: Option<usize>,
    pub maximum_items: Option<usize>,
    pub step: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortDefinition {
    pub name: String,
    pub label: String,
    pub type_name: String,
    pub required: bool,
    pub binding_modes: Vec<String>,
    pub literal_field: Option<ParamField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockDefinition {
    pub schema_version: u64,
    pub definition_version: u64,
    pub type_id: String,
    pub label: String,
    pub category: String,
    pub shape: String,
    pub runners: Vec<String>,
    pub allowed_lanes: Vec<String>,
    pub available: bool,
    pub unavailable_reason: String,
    pub fields: Vec<ParamField>,
    pub inputs: Vec<PortDefinition>,
    pub outputs: Vec<PortDefinition>,
    pub default_params: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkshedIssue {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub block_id: String,
    pub field: String,
    pub port: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputBinding {
    Literal(Value),
    Auto,
    Inherited(String),
    Edge {
        block_id: String,
        port: String,
        type_name: String,
    },
    Missing,
}

pub fn definition_for(capabilities: &Value, type_id: &str) -> BlockDefinition {
    let raw = capabilities
        .get("blocks")
        .and_then(Value::as_array)
        .and_then(|items| items.iter().find(|item| string(item, "type_id") == type_id))
        .or_else(|| {
            capabilities
                .get("root_block")
                .filter(|item| string(item, "type_id") == type_id)
        });
    parse_definition(raw, type_id)
}

fn parse_definition(raw: Option<&Value>, type_id: &str) -> BlockDefinition {
    let has_catalog_definition = raw.is_some();
    let raw = raw.unwrap_or(&Value::Null);
    let label = nonempty(string(raw, "label")).unwrap_or_else(|| fallback_label(type_id));
    let category = nonempty(string(raw, "category")).unwrap_or_else(|| fallback_category(type_id));
    let default_params = raw
        .get("default_params")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let schema = raw.get("param_schema").unwrap_or(&Value::Null);
    let ui = raw.get("ui_schema").unwrap_or(&Value::Null);
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let advanced = string_set(ui.get("advanced"));
    let summaries = string_set(ui.get("summary_fields"));
    let quick = string_set(ui.get("quick_fields"));
    let mut group_by_field = BTreeMap::<String, String>::new();
    if let Some(groups) = ui.get("groups").and_then(Value::as_array) {
        for group in groups {
            let group_id = nonempty(string(group, "label"))
                .or_else(|| nonempty(string(group, "title")))
                .or_else(|| nonempty(string(group, "id")))
                .unwrap_or_else(|| "Parameters".into());
            for field in group
                .get("fields")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                group_by_field.insert(field.to_string(), group_id.clone());
            }
        }
    }
    let ordered = ordered_fields(ui, schema);
    let widgets = ui
        .get("widgets")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut fields = Vec::new();
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (fallback_order, key) in ordered.iter().enumerate() {
            let Some(property) = properties.get(key) else {
                continue;
            };
            let x_ui = property
                .get("x-ui")
                .or_else(|| property.get("x_ui"))
                .unwrap_or(&Value::Null);
            let widget_value = widgets.get(key).unwrap_or(&Value::Null);
            let widget = widget_value
                .as_str()
                .map(str::to_string)
                .or_else(|| nonempty(string(widget_value, "control")))
                .or_else(|| nonempty(string(widget_value, "widget")))
                .or_else(|| nonempty(string(x_ui, "control")))
                .unwrap_or_default();
            let default = property
                .get("default")
                .cloned()
                .or_else(|| default_params.get(key).cloned());
            let kind = parse_kind(property, &widget);
            let nullable = declared_types(property).contains("null")
                || property
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| values.iter().any(Value::is_null));
            let options_provider = x_ui
                .get("options_provider_by")
                .or_else(|| x_ui.get("options_provider"))
                .or_else(|| x_ui.get("provider"))
                .or_else(|| widget_value.get("options_provider"))
                .or_else(|| widget_value.get("provider"))
                .or_else(|| ui.get("options_providers").and_then(|items| items.get(key)))
                .cloned()
                .unwrap_or(Value::Null);
            let options_context = x_ui
                .get("options_context")
                .or_else(|| widget_value.get("options_context"))
                .or_else(|| ui.get("options_context").and_then(|items| items.get(key)))
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            fields.push(ParamField {
                key: key.clone(),
                label: nonempty(string(property, "title"))
                    .unwrap_or_else(|| semantic_field_label(type_id, key)),
                description: string(property, "description"),
                kind,
                default,
                required: required.contains(key),
                nullable,
                advanced: bool_at(x_ui, "advanced")
                    || advanced.contains(key)
                    || string(x_ui, "section").eq_ignore_ascii_case("advanced"),
                summary: bool_at(x_ui, "summary") || summaries.contains(key),
                quick: bool_at(x_ui, "quick") || quick.contains(key),
                bindable: bool_at(x_ui, "bindable"),
                group: nonempty(string(x_ui, "section"))
                    .or_else(|| group_by_field.get(key).cloned())
                    .unwrap_or_else(|| "Parameters".into()),
                order: x_ui
                    .get("order")
                    .and_then(Value::as_i64)
                    .unwrap_or(fallback_order as i64),
                unit: string(x_ui, "unit"),
                placeholder: nonempty(string(x_ui, "placeholder"))
                    .or_else(|| first_example(property))
                    .unwrap_or_default(),
                widget,
                options_provider,
                options_context,
                minimum: property
                    .get("minimum")
                    .or_else(|| property.get("exclusiveMinimum"))
                    .and_then(Value::as_f64),
                maximum: property
                    .get("maximum")
                    .or_else(|| property.get("exclusiveMaximum"))
                    .and_then(Value::as_f64),
                exclusive_minimum: property.get("exclusiveMinimum").is_some(),
                exclusive_maximum: property.get("exclusiveMaximum").is_some(),
                minimum_length: property
                    .get("minLength")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize),
                maximum_length: property
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize),
                minimum_items: property
                    .get("minItems")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize),
                maximum_items: property
                    .get("maxItems")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize),
                step: property
                    .get("multipleOf")
                    .and_then(Value::as_f64)
                    .or_else(|| x_ui.get("step").and_then(Value::as_f64)),
            });
        }
    }
    apply_preset_labels(raw, &mut fields);
    if fields.is_empty() {
        fields = fallback_fields(type_id);
    }
    fields.sort_by(|left, right| left.order.cmp(&right.order).then(left.key.cmp(&right.key)));
    let (inputs, outputs) = if has_catalog_definition {
        (
            parse_ports(raw.get("inputs"), true),
            parse_ports(raw.get("outputs"), false),
        )
    } else {
        fallback_ports(type_id)
    };
    BlockDefinition {
        schema_version: raw
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(2),
        definition_version: raw
            .get("definition_version")
            .and_then(Value::as_u64)
            .unwrap_or(1),
        type_id: type_id.into(),
        label,
        category,
        shape: nonempty(string(raw, "shape")).unwrap_or_else(|| "action".into()),
        runners: string_list(raw.get("runners")),
        allowed_lanes: string_list(raw.get("allowed_lanes")),
        available: raw
            .get("available")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        unavailable_reason: string(raw, "unavailable_reason"),
        fields,
        inputs,
        outputs,
        default_params,
    }
}

fn ordered_fields(ui: &Value, schema: &Value) -> Vec<String> {
    let mut ordered = Vec::<String>::new();
    if let Some(items) = ui.get("order").and_then(Value::as_array) {
        for key in items.iter().filter_map(Value::as_str) {
            if !ordered.iter().any(|item| item == key) {
                ordered.push(key.into());
            }
        }
    }
    if let Some(groups) = ui.get("groups").and_then(Value::as_array) {
        for group in groups {
            for key in group
                .get("fields")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                if !ordered.iter().any(|item| item == key) {
                    ordered.push(key.into());
                }
            }
        }
    }
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for key in properties.keys() {
            if !ordered.iter().any(|item| item == key) {
                ordered.push(key.clone());
            }
        }
    }
    ordered
}

fn parse_kind(property: &Value, widget: &str) -> ParamKind {
    let mut choices = Vec::new();
    if let Some(values) = property.get("enum").and_then(Value::as_array) {
        for value in values {
            choices.push(EnumChoice {
                value: value.clone(),
                label: choice_label(value),
            });
        }
    } else if let Some(values) = property.get("oneOf").and_then(Value::as_array) {
        for item in values {
            if let Some(value) = item.get("const") {
                choices.push(EnumChoice {
                    value: value.clone(),
                    label: nonempty(string(item, "title")).unwrap_or_else(|| scalar_text(value)),
                });
            }
        }
    }
    let types = declared_types(property);
    let one_of_types = property
        .get("oneOf")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .flat_map(declared_types)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    if one_of_types.contains("integer") && !choices.is_empty() {
        return ParamKind::IntegerOrString(choices);
    }
    if !choices.is_empty() || matches!(widget, "select" | "segmented" | "radio" | "preset-cards") {
        return ParamKind::Enum(choices);
    }
    if types.contains("integer") {
        ParamKind::Integer
    } else if types.contains("number") {
        ParamKind::Number
    } else if types.contains("boolean") {
        ParamKind::Boolean
    } else if types.contains("array")
        && property
            .get("items")
            .and_then(|items| items.get("type"))
            .and_then(Value::as_str)
            == Some("string")
    {
        ParamKind::StringArray
    } else {
        ParamKind::String
    }
}

fn declared_types(property: &Value) -> BTreeSet<&str> {
    match property.get("type") {
        Some(Value::String(value)) => std::iter::once(value.as_str()).collect(),
        Some(Value::Array(values)) => values.iter().filter_map(Value::as_str).collect(),
        _ => BTreeSet::new(),
    }
}

fn choice_label(value: &Value) -> String {
    match value {
        Value::Null => "Default".into(),
        Value::String(value) if value == "huggingface" => "Hugging Face".into(),
        Value::String(value) => humanize(value),
        _ => scalar_text(value),
    }
}

fn apply_preset_labels(raw: &Value, fields: &mut [ParamField]) {
    let Some(presets) = raw.get("presets").and_then(Value::as_array) else {
        return;
    };
    let labels = presets
        .iter()
        .filter_map(|preset| {
            let id = nonempty(string(preset, "id"))?;
            let label = nonempty(string(preset, "label")).unwrap_or_else(|| humanize(&id));
            Some((id, label))
        })
        .collect::<BTreeMap<_, _>>();
    for field in fields {
        if field.key != "preset_id" {
            continue;
        }
        if let ParamKind::Enum(choices) = &mut field.kind {
            for choice in choices {
                if let Some(id) = choice.value.as_str() {
                    if let Some(label) = labels.get(id) {
                        choice.label.clone_from(label);
                    }
                }
            }
        }
    }
}

fn parse_ports(value: Option<&Value>, input: bool) -> Vec<PortDefinition> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let name = string(item, "name");
                    if name.is_empty() {
                        return None;
                    }
                    let literal_field = item.get("literal_schema").and_then(|schema| {
                        if !schema.is_object() {
                            return None;
                        }
                        let x_ui = schema
                            .get("x-ui")
                            .or_else(|| schema.get("x_ui"))
                            .unwrap_or(&Value::Null);
                        Some(ParamField {
                            key: name.clone(),
                            label: nonempty(string(schema, "title"))
                                .or_else(|| nonempty(string(item, "label")))
                                .unwrap_or_else(|| humanize(&name)),
                            description: string(schema, "description"),
                            kind: parse_kind(schema, &string(x_ui, "control")),
                            default: schema.get("default").cloned(),
                            required: item
                                .get("required")
                                .and_then(Value::as_bool)
                                .unwrap_or(true),
                            nullable: declared_types(schema).contains("null"),
                            advanced: false,
                            summary: false,
                            quick: false,
                            bindable: false,
                            group: "Inputs".into(),
                            order: 0,
                            unit: string(x_ui, "unit"),
                            placeholder: string(x_ui, "placeholder"),
                            widget: string(x_ui, "control"),
                            options_provider: x_ui
                                .get("options_provider_by")
                                .or_else(|| x_ui.get("options_provider"))
                                .or_else(|| x_ui.get("provider"))
                                .cloned()
                                .unwrap_or(Value::Null),
                            options_context: x_ui
                                .get("options_context")
                                .and_then(Value::as_object)
                                .cloned()
                                .unwrap_or_default(),
                            minimum: schema
                                .get("minimum")
                                .or_else(|| schema.get("exclusiveMinimum"))
                                .and_then(Value::as_f64),
                            maximum: schema
                                .get("maximum")
                                .or_else(|| schema.get("exclusiveMaximum"))
                                .and_then(Value::as_f64),
                            exclusive_minimum: schema.get("exclusiveMinimum").is_some(),
                            exclusive_maximum: schema.get("exclusiveMaximum").is_some(),
                            minimum_length: schema
                                .get("minLength")
                                .and_then(Value::as_u64)
                                .map(|value| value as usize),
                            maximum_length: schema
                                .get("maxLength")
                                .and_then(Value::as_u64)
                                .map(|value| value as usize),
                            minimum_items: schema
                                .get("minItems")
                                .and_then(Value::as_u64)
                                .map(|value| value as usize),
                            maximum_items: schema
                                .get("maxItems")
                                .and_then(Value::as_u64)
                                .map(|value| value as usize),
                            step: schema.get("multipleOf").and_then(Value::as_f64),
                        })
                    });
                    Some(PortDefinition {
                        name: name.clone(),
                        label: nonempty(string(item, "label")).unwrap_or_else(|| humanize(&name)),
                        type_name: nonempty(string(item, "type")).unwrap_or_else(|| "Value".into()),
                        required: item
                            .get("required")
                            .and_then(Value::as_bool)
                            .unwrap_or(input),
                        binding_modes: item
                            .get("binding_modes")
                            .and_then(Value::as_array)
                            .map(|modes| {
                                modes
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_string)
                                    .collect()
                            })
                            .unwrap_or_else(|| {
                                if input {
                                    vec!["edge".into(), "literal".into()]
                                } else {
                                    Vec::new()
                                }
                            }),
                        literal_field,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn effective_params(block: &Value, definition: &BlockDefinition) -> Map<String, Value> {
    let mut result = definition.default_params.clone();
    for field in &definition.fields {
        if !result.contains_key(&field.key) {
            if let Some(default) = &field.default {
                result.insert(field.key.clone(), default.clone());
            }
        }
    }
    if let Some(params) = block.get("params").and_then(Value::as_object) {
        result.extend(params.clone());
    }
    result
}

pub fn field_text(value: Option<&Value>, field: &ParamField) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", "),
        Some(value) => match &field.kind {
            ParamKind::Enum(choices) | ParamKind::IntegerOrString(choices) => choices
                .iter()
                .find(|choice| choice.value == *value)
                .map(|choice| choice.label.clone())
                .unwrap_or_else(|| scalar_text(value)),
            _ => scalar_text(value),
        },
    }
}

pub fn parse_field_text(field: &ParamField, text: &str) -> Result<Value, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() && (!field.required || field.nullable) {
        return Ok(Value::Null);
    }
    let value = match &field.kind {
        ParamKind::String => Value::String(text.to_string()),
        ParamKind::Integer => trimmed
            .parse::<i64>()
            .map(Value::from)
            .map_err(|_| format!("{} must be a whole number.", field.label))?,
        ParamKind::Number => trimmed
            .parse::<f64>()
            .map(Value::from)
            .map_err(|_| format!("{} must be a number.", field.label))?,
        ParamKind::Boolean => match trimmed.to_ascii_lowercase().as_str() {
            "true" | "on" | "yes" | "1" => Value::Bool(true),
            "false" | "off" | "no" | "0" => Value::Bool(false),
            _ => return Err(format!("{} must be on or off.", field.label)),
        },
        ParamKind::Enum(choices) => choices
            .iter()
            .find(|choice| {
                scalar_text(&choice.value) == trimmed || choice.label.eq_ignore_ascii_case(trimmed)
            })
            .map(|choice| choice.value.clone())
            .ok_or_else(|| format!("Choose a supported value for {}.", field.label))?,
        ParamKind::IntegerOrString(choices) => {
            if let Some(choice) = choices.iter().find(|choice| {
                scalar_text(&choice.value) == trimmed || choice.label.eq_ignore_ascii_case(trimmed)
            }) {
                choice.value.clone()
            } else {
                trimmed.parse::<i64>().map(Value::from).map_err(|_| {
                    format!("{} must be a supported value or whole number.", field.label)
                })?
            }
        }
        ParamKind::StringArray => Value::Array(
            text.lines()
                .flat_map(|line| line.split(','))
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(|item| Value::String(item.to_string()))
                .collect(),
        ),
    };
    validate_numeric(field, &value)?;
    validate_collection(field, &value)?;
    Ok(value)
}

fn validate_numeric(field: &ParamField, value: &Value) -> Result<(), String> {
    let Some(number) = value.as_f64() else {
        return Ok(());
    };
    if field
        .minimum
        .is_some_and(|minimum| number < minimum || (field.exclusive_minimum && number <= minimum))
    {
        return Err(format!(
            "{} must be {} {}.",
            field.label,
            if field.exclusive_minimum {
                "greater than"
            } else {
                "at least"
            },
            scalar_number(field.minimum.unwrap())
        ));
    }
    if field
        .maximum
        .is_some_and(|maximum| number > maximum || (field.exclusive_maximum && number >= maximum))
    {
        return Err(format!(
            "{} must be {} {}.",
            field.label,
            if field.exclusive_maximum {
                "less than"
            } else {
                "at most"
            },
            scalar_number(field.maximum.unwrap())
        ));
    }
    Ok(())
}

fn validate_collection(field: &ParamField, value: &Value) -> Result<(), String> {
    if let Some(text) = value.as_str() {
        let length = text.chars().count();
        if field.minimum_length.is_some_and(|minimum| length < minimum) {
            return Err(format!(
                "{} must contain at least {} characters.",
                field.label,
                field.minimum_length.unwrap()
            ));
        }
        if field.maximum_length.is_some_and(|maximum| length > maximum) {
            return Err(format!(
                "{} must contain at most {} characters.",
                field.label,
                field.maximum_length.unwrap()
            ));
        }
    }
    if let Some(items) = value.as_array() {
        if field
            .minimum_items
            .is_some_and(|minimum| items.len() < minimum)
        {
            return Err(format!(
                "{} needs at least {} items.",
                field.label,
                field.minimum_items.unwrap()
            ));
        }
        if field
            .maximum_items
            .is_some_and(|maximum| items.len() > maximum)
        {
            return Err(format!(
                "{} allows at most {} items.",
                field.label,
                field.maximum_items.unwrap()
            ));
        }
    }
    Ok(())
}

pub fn set_param(order: &mut Value, block_id: &str, key: &str, value: Option<Value>) -> bool {
    let Some(block) = find_block_mut(order, block_id) else {
        return false;
    };
    if !block.get("params").is_some_and(Value::is_object) {
        block["params"] = json!({});
    }
    let Some(params) = block.get_mut("params").and_then(Value::as_object_mut) else {
        return false;
    };
    match value {
        Some(Value::Null) | None => {
            params.remove(key);
            if let Some(origins) = block
                .get_mut("param_origins")
                .and_then(Value::as_object_mut)
            {
                origins.remove(key);
            }
        }
        Some(value) => {
            params.insert(key.to_string(), value);
            if !block.get("param_origins").is_some_and(Value::is_object) {
                block["param_origins"] = json!({});
            }
            if let Some(origins) = block
                .get_mut("param_origins")
                .and_then(Value::as_object_mut)
            {
                origins.insert(key.to_string(), json!({"kind": "user"}));
            }
        }
    }
    true
}

pub fn summary_parts(block: &Value, definition: &BlockDefinition, limit: usize) -> Vec<String> {
    let params = effective_params(block, definition);
    let mut fields = definition
        .fields
        .iter()
        .filter(|field| field.summary)
        .collect::<Vec<_>>();
    if fields.is_empty() {
        fields = definition
            .fields
            .iter()
            .filter(|field| !field.advanced && params.contains_key(&field.key))
            .take(limit)
            .collect();
    }
    fields
        .into_iter()
        .filter_map(|field| {
            let value = params.get(&field.key)?;
            let mut rendered = field_text(Some(value), field);
            if rendered.trim().is_empty() {
                return None;
            }
            if rendered.chars().count() > 24 {
                rendered = format!("{}…", rendered.chars().take(23).collect::<String>());
            }
            let unit = if field.unit.is_empty() {
                String::new()
            } else {
                format!(" {}", field.unit)
            };
            Some(format!("{} {}{}", field.label, rendered, unit))
        })
        .take(limit)
        .collect()
}

pub fn issues(value: &Value) -> Vec<WorkshedIssue> {
    let mut result = Vec::new();
    for (key, severity) in [("errors", "error"), ("warnings", "warning")] {
        for item in value
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(message) = item.as_str() {
                result.push(WorkshedIssue {
                    severity: severity.into(),
                    code: String::new(),
                    message: message.into(),
                    block_id: String::new(),
                    field: String::new(),
                    port: String::new(),
                });
                continue;
            }
            let location = item.get("location").unwrap_or(&Value::Null);
            let mut field = nonempty(string(item, "field"))
                .or_else(|| nonempty(string(item, "param")))
                .or_else(|| nonempty(string(location, "field")))
                .unwrap_or_default();
            if let Some(stripped) = field.strip_prefix("params.") {
                field = stripped.into();
            }
            let code = string(item, "code");
            let message = string(item, "message");
            let mut port = nonempty(string(item, "port"))
                .or_else(|| nonempty(string(location, "port")))
                .unwrap_or_default();
            if field.is_empty() {
                field = inferred_issue_field(&code, &message);
            }
            if port.is_empty() {
                port = inferred_issue_port(&code, &message);
            }
            result.push(WorkshedIssue {
                severity: severity.into(),
                code,
                message,
                block_id: nonempty(string(item, "block_id"))
                    .or_else(|| nonempty(string(location, "block_id")))
                    .unwrap_or_default(),
                field,
                port,
            });
        }
    }
    result
}

fn inferred_issue_field(code: &str, message: &str) -> String {
    if let Some(rest) = message.strip_prefix("Parameter '") {
        if let Some((field, _)) = rest.split_once('\'') {
            return field.trim_start_matches("params.").to_string();
        }
    }
    match code {
        "model_source" | "dataset_source" => "ref".into(),
        "model_name" => "model_name".into(),
        "quantize_params" => "preset_id".into(),
        "output_name" => "output_name".into(),
        _ => String::new(),
    }
}

fn inferred_issue_port(code: &str, message: &str) -> String {
    if matches!(
        code,
        "missing_input" | "binding" | "binding_kind" | "binding_literal" | "binding_inherited"
    ) {
        if let Some((_, rest)) = message.split_once('\'') {
            if let Some((port, _)) = rest.split_once('\'') {
                return port.to_string();
            }
        }
    }
    String::new()
}

pub fn input_binding(order: &Value, block: &Value, port: &str) -> InputBinding {
    let block_id = string(block, "id");
    if let Some(edge) = order
        .get("edges")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|edge| {
            edge.get("to").is_some_and(|target| {
                string(target, "block_id") == block_id && string(target, "port") == port
            })
        })
    {
        let source = edge.get("from").unwrap_or(&Value::Null);
        return InputBinding::Edge {
            block_id: string(source, "block_id"),
            port: string(source, "port"),
            type_name: string(source, "type"),
        };
    }
    let binding = block
        .get("input_bindings")
        .and_then(|bindings| bindings.get(port));
    match binding
        .and_then(|binding| binding.get("kind"))
        .and_then(Value::as_str)
    {
        Some("literal") => InputBinding::Literal(
            binding
                .and_then(|item| item.get("value"))
                .cloned()
                .unwrap_or(Value::Null),
        ),
        Some("auto") => InputBinding::Auto,
        Some("inherited") => {
            InputBinding::Inherited(binding.map(|item| string(item, "name")).unwrap_or_default())
        }
        _ => block
            .get("params")
            .and_then(|params| params.get(port))
            .cloned()
            .map(InputBinding::Literal)
            .unwrap_or(InputBinding::Missing),
    }
}

pub fn set_input_binding(
    order: &mut Value,
    block_id: &str,
    port: &str,
    binding: InputBinding,
) -> bool {
    remove_incoming_edge(order, block_id, port);
    let Some(block) = find_block_mut(order, block_id) else {
        return false;
    };
    if !block.get("input_bindings").is_some_and(Value::is_object) {
        block["input_bindings"] = json!({});
    }
    let bindings = block
        .get_mut("input_bindings")
        .and_then(Value::as_object_mut)
        .expect("input bindings object");
    match binding {
        InputBinding::Literal(value) => {
            bindings.insert(
                port.into(),
                json!({"kind": "literal", "value": value, "origin": "user"}),
            );
        }
        InputBinding::Auto => {
            bindings.insert(port.into(), json!({"kind": "auto", "origin": "user"}));
        }
        InputBinding::Inherited(name) => {
            bindings.insert(
                port.into(),
                json!({"kind": "inherited", "name": name, "origin": "user"}),
            );
        }
        InputBinding::Edge { .. } | InputBinding::Missing => {
            bindings.remove(port);
        }
    }
    true
}

pub fn auto_connect(
    order: &mut Value,
    capabilities: &Value,
    block_id: &str,
    target_port: &str,
    target_type: &str,
) -> bool {
    let blocks = order
        .get("blocks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let Some(target_index) = blocks
        .iter()
        .position(|block| string(block, "id") == block_id)
    else {
        return false;
    };
    let candidate = blocks[..target_index].iter().rev().find_map(|candidate| {
        let definition = definition_for(capabilities, &string(candidate, "type_id"));
        definition.outputs.iter().find_map(|output| {
            if compatible_types(&output.type_name, target_type) {
                Some((
                    string(candidate, "id"),
                    output.name.clone(),
                    output.type_name.clone(),
                ))
            } else {
                None
            }
        })
    });
    let Some((source_id, source_port, source_type)) = candidate else {
        return false;
    };
    remove_incoming_edge(order, block_id, target_port);
    if let Some(block) = find_block_mut(order, block_id) {
        if let Some(bindings) = block
            .get_mut("input_bindings")
            .and_then(Value::as_object_mut)
        {
            bindings.remove(target_port);
        }
    }
    if !order.get("edges").is_some_and(Value::is_array) {
        order["edges"] = json!([]);
    }
    order
        .get_mut("edges")
        .and_then(Value::as_array_mut)
        .expect("edges array")
        .push(json!({
            "from": {"block_id": source_id, "port": source_port, "type": source_type},
            "to": {"block_id": block_id, "port": target_port, "type": target_type}
        }));
    true
}

pub fn compatible_types(left: &str, right: &str) -> bool {
    left == right || matches!((left, right), ("Number", "Integer") | ("Integer", "Number"))
}

fn remove_incoming_edge(order: &mut Value, block_id: &str, target_port: &str) {
    if let Some(edges) = order.get_mut("edges").and_then(Value::as_array_mut) {
        edges.retain(|edge| {
            !edge.get("to").is_some_and(|target| {
                string(target, "block_id") == block_id && string(target, "port") == target_port
            })
        });
    }
}

pub fn block<'a>(order: &'a Value, block_id: &str) -> Option<&'a Value> {
    order
        .get("blocks")
        .and_then(Value::as_array)
        .and_then(|blocks| blocks.iter().find(|block| string(block, "id") == block_id))
}

fn find_block_mut<'a>(order: &'a mut Value, block_id: &str) -> Option<&'a mut Value> {
    order
        .get_mut("blocks")
        .and_then(Value::as_array_mut)
        .and_then(|blocks| {
            blocks
                .iter_mut()
                .find(|block| string(block, "id") == block_id)
        })
}

pub fn field(definition: &BlockDefinition, key: &str) -> Option<ParamField> {
    definition
        .fields
        .iter()
        .find(|field| field.key == key)
        .cloned()
}

pub fn value_type_for_field(field: &ParamField) -> &'static str {
    match field.kind {
        ParamKind::Integer => "Integer",
        ParamKind::Number => "Number",
        ParamKind::Boolean => "Boolean",
        ParamKind::StringArray => "List",
        ParamKind::String | ParamKind::Enum(_) | ParamKind::IntegerOrString(_) => "String",
    }
}

pub fn literal_field(port: &PortDefinition) -> ParamField {
    port.literal_field.clone().unwrap_or_else(|| ParamField {
        key: port.name.clone(),
        label: port.label.clone(),
        description: String::new(),
        kind: match port.type_name.as_str() {
            "Integer" => ParamKind::Integer,
            "Number" => ParamKind::Number,
            "Boolean" => ParamKind::Boolean,
            "List" | "CandidateSet" => ParamKind::StringArray,
            _ => ParamKind::String,
        },
        default: None,
        required: port.required,
        nullable: !port.required,
        advanced: false,
        summary: false,
        quick: false,
        bindable: false,
        group: "Inputs".into(),
        order: 0,
        unit: String::new(),
        placeholder: match port.type_name.as_str() {
            "Model" => "Hub id, local path, or managed model".into(),
            "Dataset" => "Hub dataset id or local JSONL path".into(),
            _ => String::new(),
        },
        widget: String::new(),
        options_provider: Value::Null,
        options_context: Map::new(),
        minimum: None,
        maximum: None,
        exclusive_minimum: false,
        exclusive_maximum: false,
        minimum_length: None,
        maximum_length: None,
        minimum_items: None,
        maximum_items: None,
        step: None,
    })
}

/// Resolve a provider solely from capability metadata and current parameter
/// values. The manager owns the allowlist; the native client never invents a
/// provider id. Both the simple V2 string form and conditional metadata forms
/// are accepted so a field such as `ref` can follow its `source_kind` value.
pub fn options_provider(field: &ParamField, params: &Map<String, Value>) -> Option<String> {
    resolve_provider_value(&field.options_provider, params, 0)
}

fn resolve_provider_value(
    value: &Value,
    params: &Map<String, Value>,
    depth: usize,
) -> Option<String> {
    if depth > 6 {
        return None;
    }
    if let Some(provider) = value.as_str() {
        return nonempty(provider.to_string());
    }
    if let Some(values) = value.as_array() {
        return values
            .iter()
            .find_map(|value| resolve_provider_value(value, params, depth + 1));
    }
    let object = value.as_object()?;
    for key in ["provider", "id"] {
        if let Some(provider) = object.get(key).and_then(Value::as_str) {
            if !provider.trim().is_empty() {
                return Some(provider.to_string());
            }
        }
    }

    let discriminator = ["field", "depends_on", "source_field", "discriminator"]
        .iter()
        .find_map(|key| object.get(*key).and_then(Value::as_str));
    let mappings = ["providers", "cases", "by_value", "values", "map"]
        .iter()
        .find_map(|key| object.get(*key).and_then(Value::as_object));
    if let Some(field) = discriminator {
        if let Some(selected) = params.get(field).map(scalar_text) {
            let provider = mappings
                .and_then(|mappings| mappings.get(&selected))
                .or_else(|| object.get(&selected));
            return provider
                .and_then(|provider| resolve_provider_value(provider, params, depth + 1));
        }
    }

    // Also accept the compact `{ "source_kind": { "local": "…" } }`
    // form. Only keys present in current params participate in resolution.
    for (field, mapping) in object {
        let Some(selected) = params.get(field).map(scalar_text) else {
            continue;
        };
        let Some(provider) = mapping
            .as_object()
            .and_then(|items| items.get(&selected))
            .and_then(|value| resolve_provider_value(value, params, depth + 1))
        else {
            continue;
        };
        return Some(provider);
    }

    object
        .get("default")
        .and_then(|value| resolve_provider_value(value, params, depth + 1))
}

fn fallback_fields(type_id: &str) -> Vec<ParamField> {
    let mut fields = Vec::new();
    let mut push = |key: &str,
                    label: &str,
                    kind: ParamKind,
                    default: Option<Value>,
                    advanced: bool,
                    summary: bool,
                    quick: bool| {
        let order = fields.len() as i64;
        fields.push(ParamField {
            key: key.into(),
            label: label.into(),
            description: String::new(),
            kind,
            default,
            required: false,
            nullable: true,
            advanced,
            summary,
            quick,
            bindable: false,
            group: if advanced {
                "Advanced".into()
            } else {
                "Parameters".into()
            },
            order,
            unit: String::new(),
            placeholder: String::new(),
            widget: String::new(),
            options_provider: Value::Null,
            options_context: Map::new(),
            minimum: None,
            maximum: None,
            exclusive_minimum: false,
            exclusive_maximum: false,
            minimum_length: None,
            maximum_length: None,
            minimum_items: None,
            maximum_items: None,
            step: None,
        });
    };
    match type_id {
        "model_development" => push(
            "model_name",
            "Model name",
            ParamKind::String,
            None,
            false,
            true,
            false,
        ),
        "load_model" => {
            push(
                "source_kind",
                "Source",
                ParamKind::Enum(vec![
                    choice("huggingface", "Hugging Face"),
                    choice("local", "Local folder"),
                    choice("managed", "Managed model"),
                ]),
                Some(Value::String("huggingface".into())),
                false,
                true,
                true,
            );
            push(
                "ref",
                "Model reference",
                ParamKind::String,
                None,
                false,
                true,
                false,
            );
            push(
                "revision",
                "Revision",
                ParamKind::String,
                None,
                true,
                false,
                false,
            );
        }
        "load_dataset" => {
            push(
                "source_kind",
                "Source",
                ParamKind::Enum(vec![
                    choice("huggingface", "Hugging Face"),
                    choice("local", "Local folder"),
                    choice("managed", "Managed dataset"),
                ]),
                Some(Value::String("huggingface".into())),
                false,
                true,
                true,
            );
            push(
                "ref",
                "Dataset reference",
                ParamKind::String,
                None,
                false,
                true,
                false,
            );
            push(
                "revision",
                "Revision",
                ParamKind::String,
                None,
                true,
                false,
                false,
            );
        }
        "train_lora" | "train_qlora" | "train_dora" => {
            push(
                "model",
                "Model",
                ParamKind::String,
                None,
                false,
                false,
                false,
            );
            push(
                "dataset",
                "Dataset",
                ParamKind::String,
                None,
                false,
                false,
                false,
            );
            push(
                "rank",
                "Rank",
                ParamKind::Integer,
                Some(json!(8)),
                false,
                true,
                true,
            );
            push(
                "epochs",
                "Epochs",
                ParamKind::Integer,
                Some(json!(3)),
                false,
                true,
                false,
            );
            push(
                "learning_rate",
                "Learning rate",
                ParamKind::Number,
                Some(json!(0.0002)),
                true,
                false,
                false,
            );
            push(
                "output_name",
                "Output name",
                ParamKind::String,
                None,
                true,
                false,
                false,
            );
        }
        "train_sft" | "train_full" | "train_dpo" => {
            push(
                "model",
                "Model",
                ParamKind::String,
                None,
                false,
                false,
                false,
            );
            push(
                "dataset",
                "Dataset",
                ParamKind::String,
                None,
                false,
                false,
                false,
            );
            push(
                "epochs",
                "Epochs",
                ParamKind::Integer,
                Some(json!(3)),
                false,
                true,
                true,
            );
            push(
                "learning_rate",
                "Learning rate",
                ParamKind::Number,
                Some(json!(0.0002)),
                true,
                false,
                false,
            );
            push(
                "output_name",
                "Output name",
                ParamKind::String,
                None,
                true,
                false,
                false,
            );
        }
        "quantize_mlx" => {
            push(
                "model",
                "Model",
                ParamKind::String,
                None,
                false,
                false,
                false,
            );
            push(
                "preset",
                "Preset",
                ParamKind::Enum(vec![
                    choice("mlx-compressed", "Compressed"),
                    choice("mlx-balanced", "Balanced"),
                    choice("mlx-quality", "Quality"),
                ]),
                Some(Value::String("mlx-balanced".into())),
                false,
                true,
                true,
            );
            push(
                "bits",
                "Bits",
                ParamKind::Integer,
                Some(json!(4)),
                false,
                true,
                false,
            );
            push(
                "group_size",
                "Group size",
                ParamKind::Integer,
                Some(json!(128)),
                true,
                true,
                false,
            );
            push(
                "mode",
                "Mode",
                ParamKind::Enum(vec![choice("affine", "Affine"), choice("mxfp4", "MXFP4")]),
                Some(Value::String("affine".into())),
                true,
                false,
                false,
            );
            push(
                "output_name",
                "Output name",
                ParamKind::String,
                None,
                true,
                false,
                false,
            );
        }
        "quality_gate" => {
            push(
                "metric",
                "Metric",
                ParamKind::String,
                Some(Value::String("loss".into())),
                false,
                true,
                false,
            );
            push(
                "threshold",
                "Threshold",
                ParamKind::Number,
                Some(json!(1.0)),
                false,
                true,
                true,
            );
        }
        "repeat" | "retry" => push(
            "count",
            "Count",
            ParamKind::Integer,
            Some(json!(2)),
            false,
            true,
            true,
        ),
        "register_model" => push(
            "output_name",
            "Output name",
            ParamKind::String,
            None,
            false,
            true,
            false,
        ),
        _ => push(
            "output_name",
            "Output name",
            ParamKind::String,
            None,
            true,
            false,
            false,
        ),
    }
    fields
}

fn choice(value: &str, label: &str) -> EnumChoice {
    EnumChoice {
        value: Value::String(value.into()),
        label: label.into(),
    }
}

fn fallback_label(type_id: &str) -> String {
    match type_id {
        "model_development" => "model development".into(),
        "load_model" => "load base model".into(),
        "load_dataset" => "prepare dataset".into(),
        "generate_teacher_responses" => "generate teacher responses".into(),
        "response_distill" => "distill responses".into(),
        "logits_distill" => "distill compatible logits".into(),
        "train_sft" => "train with SFT".into(),
        "train_lora" => "train with LoRA".into(),
        "train_qlora" => "train with QLoRA".into(),
        "train_dora" => "train with DoRA".into(),
        "train_full" => "full fine-tune".into(),
        "train_dpo" => "train with DPO".into(),
        "fuse_adapter" => "fuse adapter".into(),
        "quantize_mlx" => "quantize with MLX".into(),
        "evaluate_loss" => "evaluate loss".into(),
        "evaluate_lm" => "run reproducible lm-eval".into(),
        "benchmark_model" => "benchmark speed and memory".into(),
        "compare_models" => "compare candidates".into(),
        "quality_gate" => "quality gate".into(),
        "register_model" => "register new model".into(),
        "reveal_artifact" => "show in Finder".into(),
        "export_report" => "export report".into(),
        "repeat" => "repeat N times".into(),
        "for_each" => "for each item".into(),
        "if_else" => "if / else".into(),
        "retry" => "retry N times".into(),
        "stop" => "stop work order".into(),
        _ => type_id.replace('_', " "),
    }
}

fn fallback_port(name: &str, type_name: &str, input: bool) -> PortDefinition {
    PortDefinition {
        name: name.into(),
        label: humanize(name),
        type_name: type_name.into(),
        required: input,
        binding_modes: if input {
            vec!["edge".into(), "literal".into(), "auto".into()]
        } else {
            Vec::new()
        },
        literal_field: None,
    }
}

fn fallback_ports(type_id: &str) -> (Vec<PortDefinition>, Vec<PortDefinition>) {
    let ports = |values: &[(&str, &str)], input: bool| {
        values
            .iter()
            .map(|(name, type_name)| fallback_port(name, type_name, input))
            .collect::<Vec<_>>()
    };
    let (inputs, outputs): (&[(&str, &str)], &[(&str, &str)]) = match type_id {
        "load_model" => (&[], &[("model", "Model")]),
        "load_dataset" => (&[], &[("dataset", "Dataset")]),
        "generate_teacher_responses" => (
            &[("teacher", "Model"), ("dataset", "Dataset")],
            &[("dataset", "Dataset")],
        ),
        "response_distill" => (
            &[("student", "Model"), ("dataset", "Dataset")],
            &[("model", "Model")],
        ),
        "logits_distill" => (
            &[
                ("teacher", "Model"),
                ("student", "Model"),
                ("dataset", "Dataset"),
            ],
            &[("model", "Model")],
        ),
        "train_sft" | "train_full" | "train_dpo" => (
            &[("model", "Model"), ("dataset", "Dataset")],
            &[("model", "Model")],
        ),
        "train_lora" | "train_qlora" | "train_dora" => (
            &[("model", "Model"), ("dataset", "Dataset")],
            &[("adapter", "Adapter")],
        ),
        "fuse_adapter" => (
            &[("model", "Model"), ("adapter", "Adapter")],
            &[("model", "Model")],
        ),
        "quantize_mlx" => (&[("model", "Model")], &[("model", "Model")]),
        "evaluate_loss" => (
            &[("model", "Model"), ("dataset", "Dataset")],
            &[("metric", "Metric")],
        ),
        "evaluate_lm" => (&[("model", "Model")], &[("report", "Report")]),
        "benchmark_model" => (&[("model", "Model")], &[("metric", "Metric")]),
        "compare_models" => (&[("candidates", "CandidateSet")], &[("report", "Report")]),
        "quality_gate" => (&[("metric", "Metric")], &[("passed", "Boolean")]),
        "register_model" => (&[("model", "Model")], &[("artifact", "Artifact")]),
        "reveal_artifact" => (&[("artifact", "Artifact")], &[("artifact", "Artifact")]),
        "export_report" => (&[("report", "Report")], &[("artifact", "Artifact")]),
        "for_each" => (&[("items", "List")], &[("candidates", "CandidateSet")]),
        "if_else" => (&[("condition", "Boolean")], &[]),
        _ => (&[], &[]),
    };
    (ports(inputs, true), ports(outputs, false))
}

fn semantic_field_label(type_id: &str, key: &str) -> String {
    match (type_id, key) {
        ("load_model", "ref") => "Model reference".into(),
        ("load_dataset", "ref") => "Dataset reference".into(),
        (_, "source_kind") => "Source".into(),
        (_, "preset_id") => "Preset".into(),
        _ => humanize(key),
    }
}

fn first_example(property: &Value) -> Option<String> {
    match property.get("examples") {
        Some(Value::String(value)) => nonempty(value.clone()),
        Some(Value::Array(values)) => values
            .iter()
            .find_map(|value| value.as_str().and_then(|value| nonempty(value.to_string()))),
        _ => None,
    }
}

fn fallback_category(type_id: &str) -> String {
    if type_id.starts_with("load_") || type_id == "model_development" {
        "models".into()
    } else if type_id.starts_with("train_") || type_id == "fuse_adapter" {
        "train".into()
    } else if type_id.contains("quantize") {
        "quantize".into()
    } else if matches!(type_id, "repeat" | "retry" | "for_each") {
        "control".into()
    } else if matches!(type_id, "if_else" | "stop") {
        "logic".into()
    } else if matches!(
        type_id,
        "register_model" | "reveal_artifact" | "export_report"
    ) {
        "deliver".into()
    } else {
        "test_compare".into()
    }
}

fn string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn bool_at(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn string_set(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn nonempty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

fn humanize(value: &str) -> String {
    let text = value.replace(['_', '-'], " ");
    let mut chars = text.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
        .unwrap_or_default()
}

fn scalar_text(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(value) => {
            if *value {
                "On".into()
            } else {
                "Off".into()
            }
        }
        Value::Number(value) => value.to_string(),
        Value::Null => String::new(),
        value => value.to_string(),
    }
}

fn scalar_number(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities() -> Value {
        json!({
            "schema_version": 2,
            "blocks": [{
                "type_id": "train_lora",
                "label": "train with LoRA",
                "category": "train",
                "param_schema": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer", "title": "Rank", "default": 8, "minimum": 1, "maximum": 64, "x-ui": {"summary": true, "quick": true, "bindable": true}},
                        "optimizer": {"type": "string", "enum": ["adamw", "adafactor"]},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "revision": {"type": "string"}
                    },
                    "required": ["rank"]
                },
                "ui_schema": {
                    "order": ["optimizer", "rank", "tags", "revision"],
                    "advanced": ["revision"],
                    "groups": [{"id": "training", "label": "Training", "fields": ["optimizer", "rank"]}],
                    "widgets": {"optimizer": "segmented"}
                },
                "default_params": {"optimizer": "adamw"},
                "inputs": [{"name": "model", "type": "Model", "required": true, "binding_modes": ["edge", "literal"], "literal_schema": {"type": "string", "title": "Base model"}}],
                "outputs": [{"name": "adapter", "type": "Adapter"}]
            }]
        })
    }

    #[test]
    fn parses_v2_schema_and_ui_hints() {
        let definition = definition_for(&capabilities(), "train_lora");
        assert_eq!(definition.fields[0].key, "optimizer");
        assert_eq!(definition.fields[1].key, "rank");
        assert!(definition.fields[1].summary);
        assert!(definition.fields[1].quick);
        assert!(definition.fields[1].bindable);
        assert!(definition.fields[3].advanced);
        assert_eq!(
            definition.inputs[0].literal_field.as_ref().unwrap().label,
            "Base model"
        );
    }

    #[test]
    fn offline_fallback_keeps_catalog_labels_and_typed_ports() {
        let source = definition_for(&Value::Null, "load_model");
        assert_eq!(source.label, "load base model");
        assert!(source.inputs.is_empty());
        assert_eq!(source.outputs[0].name, "model");
        assert_eq!(source.outputs[0].type_name, "Model");

        let train = definition_for(&Value::Null, "train_lora");
        assert_eq!(train.label, "train with LoRA");
        assert_eq!(
            train
                .inputs
                .iter()
                .map(|port| port.type_name.as_str())
                .collect::<Vec<_>>(),
            vec!["Model", "Dataset"]
        );
        assert_eq!(train.outputs[0].type_name, "Adapter");
    }

    #[test]
    fn parses_and_bounds_numeric_drafts() {
        let definition = definition_for(&capabilities(), "train_lora");
        let rank = field(&definition, "rank").unwrap();
        assert_eq!(parse_field_text(&rank, "16"), Ok(json!(16)));
        assert!(parse_field_text(&rank, "-").is_err());
        assert!(parse_field_text(&rank, "65").is_err());
    }

    #[test]
    fn effective_values_and_summaries_include_defaults() {
        let definition = definition_for(&capabilities(), "train_lora");
        let block = json!({"id": "train", "params": {"rank": 16}});
        let effective = effective_params(&block, &definition);
        assert_eq!(effective.get("optimizer"), Some(&json!("adamw")));
        assert_eq!(summary_parts(&block, &definition, 3), vec!["Rank 16"]);
    }

    #[test]
    fn bindings_are_mutually_exclusive() {
        let mut order = json!({
            "blocks": [
                {"id": "source", "type_id": "load_model", "params": {}},
                {"id": "train", "type_id": "train_lora", "params": {}, "input_bindings": {"model": {"kind": "literal", "value": "old"}}}
            ],
            "edges": [{"from": {"block_id": "source", "port": "model", "type": "Model"}, "to": {"block_id": "train", "port": "model", "type": "Model"}}]
        });
        assert!(set_input_binding(
            &mut order,
            "train",
            "model",
            InputBinding::Literal(json!("managed/model"))
        ));
        assert!(order["edges"].as_array().unwrap().is_empty());
        assert_eq!(
            order["blocks"][1]["input_bindings"]["model"]["kind"],
            "literal"
        );
    }

    #[test]
    fn issues_keep_block_and_field_targets() {
        let parsed = issues(
            &json!({"errors": [{"code": "range", "message": "Too high", "block_id": "train", "field": "params.rank"}]}),
        );
        assert_eq!(parsed[0].block_id, "train");
        assert_eq!(parsed[0].field, "rank");
    }

    #[test]
    fn v1_definition_gets_safe_fallback_fields() {
        let definition = definition_for(
            &json!({"blocks": [{"type_id": "quantize_mlx", "param_schema": {"type": "object", "additionalProperties": true}}]}),
            "quantize_mlx",
        );
        assert!(definition.fields.iter().any(|field| field.key == "bits"));
        assert!(
            definition
                .fields
                .iter()
                .any(|field| field.key == "preset" && field.quick)
        );
    }

    #[test]
    fn conditional_option_provider_matches_backend_v2_shape() {
        let capabilities = json!({
            "blocks": [{
                "schema_version": 2,
                "definition_version": 2,
                "type_id": "load_model",
                "category": "models",
                "label": "load base model",
                "param_schema": {
                    "type": "object",
                    "properties": {
                        "source_kind": {"type": "string", "enum": ["huggingface", "local", "managed"]},
                        "ref": {
                            "type": "string",
                            "x-ui": {
                                "options_provider": "models.hub",
                                "options_provider_by": {
                                    "field": "source_kind",
                                    "huggingface": "models.hub",
                                    "local": "models.local",
                                    "managed": "models.managed"
                                },
                                "options_context": {"kind": "model"}
                            }
                        }
                    }
                },
                "ui_schema": {"order": ["source_kind", "ref"]},
                "default_params": {"source_kind": "huggingface", "ref": ""},
                "inputs": [],
                "outputs": [{"name": "model", "type": "Model"}]
            }]
        });
        let definition = definition_for(&capabilities, "load_model");
        let field = field(&definition, "ref").unwrap();
        assert_eq!(field.options_context.get("kind"), Some(&json!("model")));
        for (source, provider) in [
            ("huggingface", "models.hub"),
            ("local", "models.local"),
            ("managed", "models.managed"),
        ] {
            let params = Map::from_iter([("source_kind".into(), json!(source))]);
            assert_eq!(options_provider(&field, &params).as_deref(), Some(provider));
        }
    }

    #[test]
    fn nullable_and_mixed_v2_types_keep_their_wire_types() {
        let capabilities = json!({"blocks": [{
            "type_id": "evaluate_lm",
            "param_schema": {"type": "object", "properties": {
                "limit": {"type": ["integer", "null"], "minimum": 1},
                "batch_size": {"oneOf": [{"const": "auto"}, {"type": "integer", "minimum": 1, "maximum": 512}]}
            }},
            "ui_schema": {"order": ["limit", "batch_size"]},
            "default_params": {"limit": null, "batch_size": "auto"}
        }]});
        let definition = definition_for(&capabilities, "evaluate_lm");
        let limit = field(&definition, "limit").unwrap();
        let batch = field(&definition, "batch_size").unwrap();
        assert!(matches!(limit.kind, ParamKind::Integer));
        assert!(limit.nullable);
        assert_eq!(parse_field_text(&limit, ""), Ok(Value::Null));
        assert!(matches!(batch.kind, ParamKind::IntegerOrString(_)));
        assert_eq!(parse_field_text(&batch, "auto"), Ok(json!("auto")));
        assert_eq!(parse_field_text(&batch, "8"), Ok(json!(8)));
    }
}
