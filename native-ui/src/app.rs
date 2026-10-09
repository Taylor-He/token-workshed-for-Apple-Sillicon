use crate::api::{
    AgentProfileConfigurePayload, AgentProfileJob, AgentProfilePayload, ApiClient, ChatRequest,
    ChatStreamEvent, CommunitySearchPayload, ConfigPayload, IdeConnectionsPayload,
    ManagerModelsPayload, QuantizationCapabilities, QuantizationJob, RuntimeConfig, StatusPayload,
    string_at,
};
use crate::appearance::{
    ALUMINUM_FRAME_WIDTH, APP_PADDING, Accent, CONTENT_RADIUS, MaterialMode,
    NUMBER_ALUMINUM_RULE_HEIGHT, PAGE_FRAME_INSET, PAGE_SECTION_GAP, Page, SECTION_DIVIDER_INSET,
    SIDEBAR_ALUMINUM_CORE_ALPHA, SIDEBAR_ALUMINUM_HIGHLIGHT_ALPHA, SIDEBAR_ALUMINUM_LEFT_OFFSET,
    SIDEBAR_ALUMINUM_SHADE_ALPHA, SIDEBAR_ALUMINUM_WIDTH, SIDEBAR_BUTTON_SIZE,
    SIDEBAR_HOVER_PILL_HEIGHT, SIDEBAR_HOVER_PILL_WIDTH, SIDEBAR_ICON_HOVER_SCALE,
    SIDEBAR_ICON_SLOT_SIZE, SIDEBAR_ITEM_GAP, SIDEBAR_ITEM_INSET, SIDEBAR_SEPARATOR_GUTTER_WIDTH,
    SIDEBAR_WIDTH, palette, rgba,
};
use crate::glass::{
    GlassState, GlassVariant, glass_button_style, glass_container_style, state_from_status,
};
use crate::persistence::{
    NativeSettings, NativeState, NativeStateStore, StoredConversation, StoredMessage,
};
use crate::window_effects::{PlatformWindowEffects, WindowEffectOp, WindowEffectsBackend};
use crate::workshed::{self, BlockDefinition, InputBinding, ParamField, ParamKind, WorkshedIssue};
use cosmic::app::Core;
use cosmic::iced::widget::{button, container, slider, text, text_editor};
use cosmic::iced::{
    Alignment, Background, Border, Color, Font, Length, Shadow, Vector, keyboard, window,
};
use cosmic::{
    Core as CosmicCore, Element, Renderer, Theme, executor, iced, prelude::*, theme, widget,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[path = "app/terminal_view.rs"]
mod terminal_view;
#[path = "app/workshed_view.rs"]
mod workshed_view;

#[derive(Debug, Clone)]
pub struct AppFlags {
    pub api_base: String,
    pub server_url: String,
    pub manager_token: String,
    pub initial_page: Page,
    pub ide_companion: bool,
}

const AGENT_RUNTIME_GROUP_WIDTH: f32 = 178.0;
const AGENT_RUNTIME_GROUP_HEIGHT: f32 = 38.0;
const AGENT_RUNTIME_BUTTON_WIDTH: f32 = 174.0;
const AGENT_RUNTIME_BUTTON_HEIGHT: f32 = 32.0;
const WORKSHED_PREFLIGHT_DEFAULT_OPEN: bool = false;
const WORKSHED_CATEGORY_STRIPE_WIDTH: f32 = 3.0;
// A terminal window is a first-class Caligo surface, so its radius follows
// the app's content radius instead of introducing a separate visual scale.
const TERMINAL_WINDOW_RADIUS: f32 = CONTENT_RADIUS + 4.0;

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkshedCardPort {
    label: String,
    type_name: String,
    edge_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkshedPortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub enum Message {
    PagePressed(Page),
    SidebarHover(Page),
    SidebarHoverExit(Page),
    AnimationTick(Instant),
    PollTick(Instant),
    QuantizationJobTick(Instant),
    ConfigLoaded(Result<ConfigPayload, String>),
    ManagerLoaded {
        generation: u64,
        result: Result<ManagerModelsPayload, String>,
    },
    StatusLoaded(Result<StatusPayload, String>),
    RuntimeLoaded(Result<RuntimeConfig, String>),
    DeveloperLogsTick(Instant),
    DeveloperLogsLoaded(Result<Vec<String>, String>),
    DeveloperPromptSaved(Result<String, String>),
    DeveloperTerminalChanged(String),
    DeveloperTerminalSubmit,
    DeveloperTerminalSent(Result<String, String>),
    CommunitySearchLoaded(Result<CommunitySearchPayload, String>),
    CommunityJobLoaded(Result<Option<Value>, String>),
    CommunityDeployLoaded(Result<Option<Value>, String>),
    CommunityJobActionLoaded(Result<Option<Value>, String>),
    QuantizationCapabilitiesLoaded(Result<QuantizationCapabilities, String>),
    QuantizationPreflightLoaded(Result<Value, String>),
    QuantizationStarted(Result<QuantizationJob, String>),
    QuantizationJobLoaded(Result<QuantizationJob, String>),
    QuantizationActionLoaded(Result<QuantizationJob, String>),
    QuantizationDeliveryLoaded(Result<QuantizationJob, String>),
    WorkshedCapabilitiesLoaded(Result<Value, String>),
    WorkshedWorkOrdersLoaded(Result<Vec<Value>, String>),
    WorkshedWorkOrderLoaded(Result<Value, String>),
    WorkshedPreflightLoaded {
        generation: u64,
        result: Result<Value, String>,
    },
    WorkshedRunStarted(Result<Value, String>),
    WorkshedRunLoaded(Result<Value, String>),
    WorkshedActionLoaded(Result<Value, String>),
    WorkshedNewPressed,
    WorkshedWorkOrderSelectPressed(String),
    WorkshedCategoryPressed(String),
    WorkshedSearchChanged(String),
    WorkshedBlockAddPressed(String),
    WorkshedBlockSelectPressed(String),
    WorkshedBlockDeletePressed(String),
    WorkshedDeleteSelectedPressed,
    WorkshedBlockMovePressed(i8),
    WorkshedBlockDuplicatePressed,
    WorkshedModelNameChanged(String),
    WorkshedParamDraftChanged {
        block_id: String,
        field: String,
        value: String,
    },
    WorkshedParamCommit {
        block_id: String,
        field: String,
    },
    WorkshedParamSet {
        block_id: String,
        field: String,
        value: Value,
    },
    WorkshedParamReset {
        block_id: String,
        field: String,
    },
    WorkshedParamStep {
        block_id: String,
        field: String,
        direction: i8,
    },
    WorkshedParamBindingPressed {
        block_id: String,
        field: String,
    },
    WorkshedInputDraftChanged {
        block_id: String,
        port: String,
        value: String,
    },
    WorkshedInputCommit {
        block_id: String,
        port: String,
    },
    WorkshedInputModeChanged {
        block_id: String,
        port: String,
        mode: String,
    },
    WorkshedAdvancedToggled(String),
    WorkshedOptionQueryChanged {
        block_id: String,
        field: String,
        value: String,
    },
    WorkshedOptionSearchPressed {
        block_id: String,
        field: String,
    },
    WorkshedOptionSelected {
        block_id: String,
        field: String,
        value: String,
    },
    WorkshedOptionsLoaded {
        request_id: u64,
        block_id: String,
        field: String,
        result: Result<Value, String>,
    },
    WorkshedFieldFocused {
        block_id: String,
        field: String,
        focused: bool,
    },
    WorkshedIssuePressed {
        block_id: String,
        field: String,
        port: String,
    },
    WorkshedSaveTick(Instant),
    WorkshedSaveLoaded {
        generation: u64,
        result: Result<Value, String>,
    },
    WorkshedPreflightPressed,
    WorkshedRunPressed,
    WorkshedConfirmRunPressed,
    WorkshedModelPickerToggled(String),
    WorkshedActionPressed(String),
    WorkshedPanelToggled(String),
    IdeConnectionsLoaded(Result<IdeConnectionsPayload, String>),
    IdePairingDecisionLoaded {
        approved: bool,
        result: Result<(), String>,
    },
    IdeAuthorizationRevoked(Result<bool, String>),
    CopyJetBrainsMcpCommandPressed,
    ChatStreamUpdate {
        request_id: u64,
        event: ChatStreamEvent,
    },
    ChatRevealTick(Instant),
    AgentProfileLoaded {
        model: String,
        result: Result<AgentProfilePayload, String>,
    },
    AgentProfileConfigureStarted {
        model: String,
        result: Result<AgentProfileConfigurePayload, String>,
    },
    AgentProfileJobLoaded {
        model: String,
        result: Result<AgentProfileJob, String>,
    },
    AgentProfilePollTick(Instant),
    AgentProfileDeleted {
        model: String,
        result: Result<bool, String>,
    },
    SwitchModelLoaded {
        model: String,
        result: Result<ManagerModelsPayload, String>,
    },
    DeleteModelLoaded {
        model: String,
        result: Result<ManagerModelsPayload, String>,
    },
    RuntimeSaved(Result<RuntimeConfig, String>),
    ConcurrentSaved(Result<ManagerModelsPayload, String>),
    HfBindingSaved(Result<String, String>),
    AttachMediaLoaded(Result<Vec<MediaAttachment>, String>),
    MessageInputAction(text_editor::Action),
    InputFocusChanged(InputField, bool),
    ChatModelMenuToggled,
    ChatModelOptionPressed(String),
    ServerUrlChanged(String),
    ManagerTokenChanged(String),
    MaxTokensChanged(String),
    TemperatureChanged(String),
    TemperatureSliderChanged(f32),
    SystemPromptChanged(String),
    SystemPromptAction(text_editor::Action),
    ApplySystemPromptPressed,
    ClearSystemPromptPressed,
    CommunityQueryChanged(String),
    QuantizationSourceKindChanged(String),
    QuantizationSourceChanged(String),
    QuantizationEngineChanged(String),
    QuantizationPresetChanged(String),
    QuantizationOutputChanged(String),
    QuantizationRevisionChanged(String),
    QuantizationBitsChanged(String),
    QuantizationGroupSizeChanged(String),
    QuantizationModeChanged(String),
    QuantizationAlgorithmChanged(String),
    QuantizationAdvancedToggled,
    HfUsernameChanged(String),
    HfTokenChanged(String),
    RuntimeBoolChanged(RuntimeBoolField, bool),
    RuntimeTextChanged(RuntimeTextField, String),
    ToggleConcurrentModel(String),
    SendPressed,
    AttachMediaPressed,
    ClearMediaPressed,
    NewConversationPressed,
    DeleteConversationPressed(String),
    SelectConversationPressed(String),
    SaveServerSettingsPressed,
    ResetServerSettingsPressed,
    SaveGenerationPressed,
    ResetGenerationPressed,
    ClearStoragePressed,
    SearchCommunityPressed,
    DeployCommunityPressed(String),
    CommunityJobActionPressed(String),
    QuantizationPreflightPressed,
    QuantizationStartPressed,
    QuantizationActionPressed(String),
    QuantizationDeliveryPressed(String),
    QuantizationDeliveryDismissed,
    ApproveIdePairingPressed(String),
    ApproveIdePairingAllPressed(String),
    RejectIdePairingPressed(String),
    RevokeIdeAuthorizationPressed(String),
    CopyJetBrainsProviderUrlPressed,
    CopyJetBrainsProviderKeyPressed,
    SelectLocalModelPressed(String),
    SwitchModelPressed(String),
    DeleteSelectedModelPressed,
    ResetRuntimePressed,
    SaveRuntimePressed,
    StopExtrasPressed,
    ApplyConcurrentPressed,
    SaveHfBindingPressed,
    ToggleAgentRuntime(String),
    ConfigureAgentProfilePressed,
    DeleteAgentProfilePressed,
    RecalibrateAgentProfilePressed,
    DispatchModeChanged(String),
    LanguageSelected(String),
    CloseRequested(window::Id),
    QuitPressed,
}

#[derive(Debug, Clone, Copy)]
pub enum RuntimeBoolField {
    ContinuousBatching,
    PagedCache,
    KvCacheQuantization,
    EnableMtp,
}

#[derive(Debug, Clone, Copy)]
pub enum RuntimeTextField {
    ChunkedPrefill,
    MtpDraft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputField {
    RuntimeChunkedPrefill,
    RuntimeMtpDraft,
    ServerUrl,
    ManagerToken,
    MaxTokens,
    CommunityQuery,
    QuantizationSource,
    QuantizationOutput,
    QuantizationRevision,
    QuantizationBits,
    QuantizationGroupSize,
    QuantizationMode,
    QuantizationAlgorithm,
    WorkshedModelName,
    WorkshedSearch,
    HfUsername,
    HfToken,
    Temperature,
    DeveloperTerminal,
}

#[derive(Debug, Clone)]
pub struct MediaAttachment {
    name: String,
    kind: String,
    size: u64,
    data_url: String,
}

#[derive(Debug, Clone, Copy)]
struct SidebarBounceState {
    page: Page,
    started_at: Instant,
}

impl SidebarBounceState {
    const DURATION_MS: u64 = 220;

    fn progress(self, now: Instant) -> f32 {
        let elapsed = now.saturating_duration_since(self.started_at).as_millis() as f32;
        (elapsed / Self::DURATION_MS as f32).clamp(0.0, 1.0)
    }

    fn finished(self, now: Instant) -> bool {
        self.progress(now) >= 1.0
    }
}

#[derive(Debug, Clone)]
struct ModelSwitchNotice {
    message: String,
    expires_at: Option<Instant>,
}

#[derive(Debug, Clone)]
struct ModelSwitchSnapshot {
    selected_model: String,
    selected_local_model: String,
    profile_state: AgentProfileState,
    openclaw_enabled: bool,
    agent_runtime: String,
}

#[derive(Debug, Clone)]
enum AgentProfileState {
    Unconfigured,
    Configuring {
        job_id: String,
        progress: f32,
        phase: String,
    },
    Configured {
        profile: Value,
        completed_at: Option<Instant>,
    },
    Failed {
        message: String,
    },
}

impl Default for AgentProfileState {
    fn default() -> Self {
        Self::Unconfigured
    }
}

impl AgentProfileState {
    fn is_configured(&self) -> bool {
        matches!(
            self,
            Self::Configured {
                completed_at: None,
                ..
            }
        )
    }

    fn is_configuring(&self) -> bool {
        matches!(self, Self::Configuring { .. })
    }

    fn completion_active(&self) -> bool {
        matches!(
            self,
            Self::Configured {
                completed_at: Some(_),
                ..
            }
        )
    }
}

#[derive(Debug, Clone)]
struct StreamingReplyState {
    request_id: u64,
    request_model: String,
    conversation_id: String,
    message_index: usize,
    thinking_message_index: Option<usize>,
    thinking_received: String,
    thinking_started_at: Option<Instant>,
    thinking_duration: Option<Duration>,
    received: String,
    visible_chars: usize,
    done: bool,
    started_at: Instant,
}

enum MarkdownBlock {
    Paragraph(String),
    Heading { level: usize, text: String },
    Bullet(String),
    Ordered { number: String, text: String },
    Quote(String),
    Code(String),
    Space,
}

pub struct TokenWorkshedApp {
    core: CosmicCore,
    api_base: String,
    store: NativeStateStore,
    native_state: NativeState,
    current_page: Page,
    ide_companion: bool,
    hovered_page: Option<Page>,
    sidebar_bounce: Option<SidebarBounceState>,
    focused_input: Option<InputField>,
    chat_model_menu_open: bool,
    model_switch_notice: Option<ModelSwitchNotice>,
    pending_model_switch: Option<ModelSwitchSnapshot>,
    agent_profile_state: AgentProfileState,
    streaming_reply: Option<StreamingReplyState>,
    next_chat_request_id: u64,
    manager_refresh_generation: u64,
    minimum_manager_response_generation: u64,
    manager_refresh_in_flight: bool,
    status_refresh_in_flight: bool,
    agent_profile_refresh_in_flight: bool,
    developer_logs_refresh_in_flight: bool,
    community_job_refresh_in_flight: bool,
    quantization_capabilities_refresh_in_flight: bool,
    quantization_job_refresh_in_flight: bool,
    workshed_refresh_in_flight: bool,
    ide_connections_refresh_in_flight: bool,
    window_effects: PlatformWindowEffects,
    allow_close: bool,
    status_text: String,
    server_badge: String,
    token_workshed_version: String,
    vllm_mlx_version: String,
    message_content: text_editor::Content<Renderer>,
    prompt_content: text_editor::Content<Renderer>,
    developer_logs: Vec<String>,
    developer_terminal_input: String,
    max_tokens_input: String,
    temperature_input: String,
    community_query: String,
    hf_username: String,
    hf_token: String,
    models: Vec<String>,
    manager_models: ManagerModelsPayload,
    runtime_config: RuntimeConfig,
    status_payload: StatusPayload,
    community_results: Vec<Value>,
    community_job: Option<Value>,
    quantization_capabilities: QuantizationCapabilities,
    quantization_preflight: Value,
    quantization_job: Option<QuantizationJob>,
    quantization_delivery_open: bool,
    quantization_source_kind: String,
    quantization_source: String,
    quantization_engine: String,
    quantization_preset: String,
    quantization_output: String,
    quantization_revision: String,
    quantization_bits: String,
    quantization_group_size: String,
    quantization_mode: String,
    quantization_algorithm: String,
    quantization_advanced_open: bool,
    workshed_capabilities: Value,
    workshed_work_orders: Vec<Value>,
    workshed_order: Value,
    workshed_preflight: Value,
    workshed_preflight_generation: u64,
    workshed_run_after_preflight: bool,
    workshed_run: Option<Value>,
    workshed_category: String,
    workshed_search: String,
    workshed_selected_block: String,
    workshed_model_picker: Option<String>,
    workshed_model_name_input: String,
    workshed_field_drafts: BTreeMap<String, String>,
    workshed_field_errors: BTreeMap<String, String>,
    workshed_option_queries: BTreeMap<String, String>,
    workshed_option_results: BTreeMap<String, Value>,
    workshed_option_requests: BTreeMap<String, u64>,
    workshed_option_loading: BTreeSet<String>,
    workshed_next_option_request: u64,
    workshed_advanced_blocks: BTreeSet<String>,
    workshed_focused_field: Option<String>,
    workshed_save_due: Option<Instant>,
    workshed_save_generation: u64,
    workshed_save_in_flight: bool,
    workshed_preflight_open: bool,
    workshed_library_open: bool,
    workshed_inspector_open: bool,
    ide_connections: IdeConnectionsPayload,
    pending_media: Vec<MediaAttachment>,
    selected_concurrent_models: Vec<String>,
    request_count: u64,
    success_count: u64,
    error_count: u64,
    total_tokens: u64,
    last_latency_ms: Option<f64>,
    last_tps: Option<f64>,
}

impl cosmic::Application for TokenWorkshedApp {
    type Executor = executor::Default;
    type Flags = AppFlags;
    type Message = Message;

    const APP_ID: &'static str = "com.taylorhe.token-workshed.native-ui";

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(core: Core, flags: Self::Flags) -> (Self, cosmic::app::Task<Self::Message>) {
        let mut core = core;
        core.window.show_headerbar = false;
        crate::macos_tray::install();

        let store = NativeStateStore::new();
        let mut native_state = store.load_or_default();
        if !flags.server_url.trim().is_empty() {
            native_state.settings.server_url = flags.server_url.trim().to_string();
        }
        if !flags.manager_token.trim().is_empty() {
            native_state.settings.manager_token = flags.manager_token.trim().to_string();
        }
        normalize_settings(&mut native_state.settings);
        native_state.settings.openclaw_enabled = false;
        native_state.settings.agent_runtime = "auto".into();
        let _ = store.save(&native_state);

        let max_tokens_input = native_state.settings.max_tokens.to_string();
        let temperature_input = format!("{:.2}", native_state.settings.temperature);
        let message_content = text_editor::Content::new();
        let prompt_content = text_editor::Content::with_text(&native_state.settings.system_prompt);
        let current_page = flags.initial_page;

        let mut app = Self {
            core,
            api_base: flags.api_base,
            store,
            native_state,
            current_page,
            ide_companion: flags.ide_companion,
            hovered_page: None,
            sidebar_bounce: None,
            focused_input: None,
            chat_model_menu_open: false,
            model_switch_notice: None,
            pending_model_switch: None,
            agent_profile_state: AgentProfileState::Unconfigured,
            streaming_reply: None,
            next_chat_request_id: 0,
            manager_refresh_generation: 0,
            minimum_manager_response_generation: 0,
            manager_refresh_in_flight: false,
            status_refresh_in_flight: false,
            agent_profile_refresh_in_flight: false,
            developer_logs_refresh_in_flight: false,
            community_job_refresh_in_flight: false,
            quantization_capabilities_refresh_in_flight: false,
            quantization_job_refresh_in_flight: false,
            workshed_refresh_in_flight: false,
            ide_connections_refresh_in_flight: false,
            window_effects: PlatformWindowEffects::new(),
            allow_close: false,
            status_text: "Starting token-workshed native UI...".into(),
            server_badge: "Server: checking...".into(),
            token_workshed_version: "0.1.0".into(),
            vllm_mlx_version: String::new(),
            message_content,
            prompt_content,
            developer_logs: Vec::new(),
            developer_terminal_input: String::new(),
            max_tokens_input,
            temperature_input,
            community_query: String::new(),
            hf_username: String::new(),
            hf_token: String::new(),
            models: Vec::new(),
            manager_models: ManagerModelsPayload::default(),
            runtime_config: RuntimeConfig::default(),
            status_payload: StatusPayload::default(),
            community_results: Vec::new(),
            community_job: None,
            quantization_capabilities: QuantizationCapabilities::default(),
            quantization_preflight: Value::Null,
            quantization_job: None,
            quantization_delivery_open: false,
            quantization_source_kind: "huggingface".into(),
            quantization_source: String::new(),
            quantization_engine: "mlx_local".into(),
            quantization_preset: "mlx-balanced".into(),
            quantization_output: String::new(),
            quantization_revision: String::new(),
            quantization_bits: "4".into(),
            quantization_group_size: "128".into(),
            quantization_mode: "affine".into(),
            quantization_algorithm: "awq".into(),
            quantization_advanced_open: false,
            workshed_capabilities: Value::Null,
            workshed_work_orders: Vec::new(),
            workshed_order: workshed_demo_order(),
            workshed_preflight: Value::Null,
            workshed_preflight_generation: 0,
            workshed_run_after_preflight: false,
            workshed_run: None,
            workshed_category: "models".into(),
            workshed_search: String::new(),
            workshed_selected_block: "load_model_demo".into(),
            workshed_model_picker: None,
            workshed_model_name_input: String::new(),
            workshed_field_drafts: BTreeMap::new(),
            workshed_field_errors: BTreeMap::new(),
            workshed_option_queries: BTreeMap::new(),
            workshed_option_results: BTreeMap::new(),
            workshed_option_requests: BTreeMap::new(),
            workshed_option_loading: BTreeSet::new(),
            workshed_next_option_request: 0,
            workshed_advanced_blocks: BTreeSet::new(),
            workshed_focused_field: None,
            workshed_save_due: None,
            workshed_save_generation: 0,
            workshed_save_in_flight: false,
            workshed_preflight_open: WORKSHED_PREFLIGHT_DEFAULT_OPEN,
            workshed_library_open: false,
            workshed_inspector_open: false,
            ide_connections: IdeConnectionsPayload::default(),
            pending_media: Vec::new(),
            selected_concurrent_models: Vec::new(),
            request_count: 0,
            success_count: 0,
            error_count: 0,
            total_tokens: 0,
            last_latency_ms: None,
            last_tps: None,
        };

        let mut tasks = Vec::new();
        if let Some(id) = app.core.main_window_id() {
            let title = if app.ide_companion {
                "Token Workshed · IDE Companion"
            } else {
                "token-workshed"
            };
            tasks.push(app.set_window_title(title.into()));
            let init_ops = app.window_effects.init(id);
            let material_ops = app.window_effects.set_material(MaterialMode::Frosted);
            tasks.push(app.window_ops_task(init_ops));
            tasks.push(app.window_ops_task(material_ops));
        }
        tasks.push(app.refresh_config_task());
        tasks.push(app.refresh_manager_task());
        tasks.push(app.refresh_status_task());
        tasks.push(app.refresh_ide_connections_task());
        if app.current_page == Page::Community {
            tasks.push(app.refresh_community_job_task());
        }
        if app.current_page == Page::Workshed {
            tasks.push(app.refresh_workshed_task());
        }

        (app, cosmic::Task::batch(tasks))
    }

    fn update(&mut self, message: Self::Message) -> cosmic::app::Task<Self::Message> {
        match message {
            Message::PagePressed(page) => {
                self.chat_model_menu_open = false;
                self.sidebar_bounce = Some(SidebarBounceState {
                    page,
                    started_at: Instant::now(),
                });
                self.current_page = page;
                self.hovered_page = None;
                match page {
                    Page::Chat => {
                        return cosmic::Task::batch(vec![
                            self.refresh_status_task(),
                            self.refresh_manager_task(),
                        ]);
                    }
                    Page::Models => return self.refresh_manager_task(),
                    Page::Workshed => return self.refresh_workshed_task(),
                    Page::Server => {
                        return cosmic::Task::batch(vec![
                            self.refresh_manager_task(),
                            self.refresh_runtime_task(),
                        ]);
                    }
                    Page::Community => {
                        return cosmic::Task::batch(vec![
                            self.refresh_community_job_task(),
                            self.refresh_manager_task(),
                        ]);
                    }
                    Page::Logs => {
                        self.status_text = "Terminal ready.".into();
                        return cosmic::Task::batch(vec![
                            self.refresh_developer_logs_task(),
                            wrap_app_task(widget::text_input::focus(
                                terminal_view::command_input_id(),
                            )),
                        ]);
                    }
                    Page::Settings => return self.refresh_ide_connections_task(),
                    _ => {}
                }
            }
            Message::SidebarHover(page) => self.hovered_page = Some(page),
            Message::SidebarHoverExit(page) => {
                if self.hovered_page == Some(page) {
                    self.hovered_page = None;
                }
            }
            Message::AnimationTick(now) => {
                if let Some(bounce) = self.sidebar_bounce {
                    if bounce.finished(now) {
                        self.sidebar_bounce = None;
                    }
                }
                if let Some(notice) = &self.model_switch_notice {
                    if let Some(expires_at) = notice.expires_at {
                        if now >= expires_at {
                            self.model_switch_notice = None;
                        }
                    }
                }
                if let AgentProfileState::Configured { completed_at, .. } =
                    &mut self.agent_profile_state
                {
                    if completed_at.as_ref().is_some_and(|at| {
                        now.saturating_duration_since(*at) >= Duration::from_millis(420)
                    }) {
                        *completed_at = None;
                    }
                }
            }
            Message::PollTick(_) => {
                return match self.current_page {
                    Page::Chat | Page::Server => cosmic::Task::batch(vec![
                        self.refresh_status_task(),
                        self.refresh_manager_task(),
                    ]),
                    Page::Models => self.refresh_manager_task(),
                    Page::Workshed => self.refresh_workshed_task(),
                    Page::Community => cosmic::Task::batch(vec![
                        self.refresh_manager_task(),
                        self.refresh_community_job_task(),
                    ]),
                    Page::Settings => self.refresh_ide_connections_task(),
                    Page::Logs | Page::About => cosmic::Task::none(),
                };
            }
            Message::QuantizationJobTick(_) => {
                if self.current_page == Page::Workshed {
                    return self.refresh_workshed_run_task();
                }
            }
            Message::ConfigLoaded(result) => match result {
                Ok(config) => {
                    let mut persistent_state_changed = false;
                    if !config.server_url.is_empty()
                        && self.native_state.settings.server_url.is_empty()
                    {
                        self.native_state.settings.server_url = config.server_url;
                        persistent_state_changed = true;
                    }
                    if self.native_state.settings.max_tokens == 0 {
                        self.native_state.settings.max_tokens = config.max_tokens.max(1);
                        persistent_state_changed = true;
                    }
                    if self.native_state.settings.temperature <= 0.0 {
                        self.native_state.settings.temperature = config.temperature;
                        persistent_state_changed = true;
                    }
                    if !config.token_workshed_version.is_empty() {
                        self.token_workshed_version = config.token_workshed_version;
                    }
                    if !config.vllm_mlx_version.is_empty() {
                        self.vllm_mlx_version = config.vllm_mlx_version;
                    }
                    self.status_text = if config.manager_auth_required
                        && self.native_state.settings.manager_token.is_empty()
                    {
                        "Manager token is required for remote manager endpoints.".into()
                    } else {
                        "Backend config loaded.".into()
                    };
                    if persistent_state_changed {
                        self.save_state();
                    }
                }
                Err(error) => self.status_text = format!("Config load failed: {error}"),
            },
            Message::ManagerLoaded { generation, result } => {
                self.manager_refresh_in_flight = false;
                if generation < self.minimum_manager_response_generation {
                    return cosmic::Task::none();
                }
                match result {
                    Ok(data) => {
                        self.apply_manager_payload(data);
                        if matches!(self.agent_profile_state, AgentProfileState::Unconfigured)
                            && !self.current_profile_model().is_empty()
                        {
                            return self.refresh_agent_profile_task();
                        }
                    }
                    Err(error) => {
                        self.manager_models = ManagerModelsPayload::default();
                        self.status_text = format!("Desktop manager unavailable: {error}");
                    }
                }
            }
            Message::StatusLoaded(result) => {
                self.status_refresh_in_flight = false;
                match result {
                    Ok(data) => {
                        let mut persistent_state_changed = false;
                        self.server_badge = if data.ok {
                            format!("Server: online ({})", data.server_url)
                        } else {
                            format!("Server: offline ({})", data.server_url)
                        };
                        if !data.models.is_empty() && self.models.is_empty() {
                            self.models = data.models.clone();
                        }
                        if self.native_state.selected_model.is_empty() && !data.model.is_empty() {
                            self.native_state.selected_model = data.model.clone();
                            persistent_state_changed = true;
                        }
                        self.status_payload = data;
                        if persistent_state_changed {
                            self.save_state();
                        }
                    }
                    Err(error) => {
                        self.server_badge = format!("Server: status error ({error})");
                    }
                }
            }
            Message::RuntimeLoaded(result) | Message::RuntimeSaved(result) => match result {
                Ok(config) => {
                    self.runtime_config = config;
                    self.status_text = "Runtime config loaded.".into();
                }
                Err(error) => self.status_text = format!("Runtime config error: {error}"),
            },
            Message::DeveloperLogsTick(_) => {
                if self.current_page == Page::Logs {
                    return self.refresh_developer_logs_task();
                }
            }
            Message::DeveloperLogsLoaded(result) => {
                self.developer_logs_refresh_in_flight = false;
                match result {
                    Ok(logs) => {
                        if logs != self.developer_logs {
                            self.developer_logs = logs;
                        }
                    }
                    Err(error) => {
                        self.status_text = format!("Terminal refresh failed: {error}");
                    }
                }
            }
            Message::DeveloperPromptSaved(result) => match result {
                Ok(message) => {
                    self.status_text = if message.is_empty() {
                        "System prompt applied.".into()
                    } else {
                        message
                    };
                    return self.refresh_developer_logs_task();
                }
                Err(error) => self.status_text = format!("System prompt apply failed: {error}"),
            },
            Message::DeveloperTerminalChanged(value) => self.developer_terminal_input = value,
            Message::DeveloperTerminalSubmit => {
                let command = self.developer_terminal_input.trim().to_string();
                if command.is_empty() {
                    self.developer_logs
                        .push("vllm-mlx: enter a command first.".into());
                    self.status_text = "Enter a terminal command first.".into();
                } else {
                    self.developer_terminal_input.clear();
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client().developer_terminal(command),
                        Message::DeveloperTerminalSent,
                    ));
                }
            }
            Message::DeveloperTerminalSent(result) => match result {
                Ok(message) => {
                    self.status_text = if message.is_empty() {
                        "Command sent to mapped terminal.".into()
                    } else {
                        message
                    };
                    return self.refresh_developer_logs_task();
                }
                Err(error) => {
                    self.developer_logs.push(format!("vllm-mlx: {error}"));
                    self.status_text = format!("Terminal command failed: {error}");
                }
            },
            Message::CommunitySearchLoaded(result) => match result {
                Ok(payload) => {
                    self.community_results = payload.results;
                    self.status_text = format!("Found {} model(s).", payload.count);
                }
                Err(error) => self.status_text = format!("Community search failed: {error}"),
            },
            Message::CommunityJobLoaded(result) => {
                self.community_job_refresh_in_flight = false;
                match result {
                    Ok(job) => {
                        self.community_job = normalize_community_job(job);
                        self.status_text = "Community job updated.".into();
                    }
                    Err(error) => self.status_text = format!("Community job error: {error}"),
                }
            }
            Message::CommunityDeployLoaded(result) | Message::CommunityJobActionLoaded(result) => {
                match result {
                    Ok(job) => {
                        self.community_job = normalize_community_job(job);
                        self.status_text = "Community job updated.".into();
                        return self.refresh_manager_task();
                    }
                    Err(error) => self.status_text = format!("Community job error: {error}"),
                }
            }
            Message::QuantizationCapabilitiesLoaded(result) => {
                self.quantization_capabilities_refresh_in_flight = false;
                match result {
                    Ok(capabilities) => {
                        self.quantization_capabilities = capabilities;
                        self.status_text = "Quantization capabilities loaded.".into();
                    }
                    Err(error) => {
                        self.status_text = format!("Quantization capabilities unavailable: {error}")
                    }
                }
            }
            Message::QuantizationPreflightLoaded(result) => match result {
                Ok(value) => {
                    let ok = value.get("ok").and_then(Value::as_bool).unwrap_or(false);
                    self.quantization_preflight = value;
                    self.status_text = if ok {
                        "Preflight passed. The quantization job can start.".into()
                    } else {
                        "Preflight found issues. Review the warnings on the right.".into()
                    };
                }
                Err(error) => {
                    self.quantization_preflight = Value::Null;
                    self.status_text = format!("Quantization preflight failed: {error}");
                }
            },
            Message::QuantizationStarted(result) => match result {
                Ok(job) => {
                    self.quantization_job = Some(job);
                    self.quantization_delivery_open = false;
                    self.status_text = "Quantization job started.".into();
                    self.quantization_preflight = Value::Null;
                }
                Err(error) => self.status_text = format!("Could not start quantization: {error}"),
            },
            Message::QuantizationJobLoaded(result) => {
                self.quantization_job_refresh_in_flight = false;
                match result {
                    Ok(job) => {
                        let was_awaiting_delivery = self
                            .quantization_job
                            .as_ref()
                            .is_some_and(|current| current.status == "awaiting_delivery");
                        if job.status == "awaiting_delivery" && !was_awaiting_delivery {
                            self.quantization_delivery_open = true;
                        }
                        self.quantization_job = Some(job);
                    }
                    Err(error) => {
                        self.status_text = format!("Quantization job update failed: {error}")
                    }
                }
            }
            Message::QuantizationActionLoaded(result) => match result {
                Ok(job) => {
                    self.quantization_job = Some(job);
                    self.status_text = "Quantization job action sent.".into();
                }
                Err(error) => self.status_text = format!("Quantization action failed: {error}"),
            },
            Message::QuantizationDeliveryLoaded(result) => match result {
                Ok(job) => {
                    self.quantization_delivery_open = job.status != "awaiting_delivery";
                    self.quantization_job = Some(job);
                    self.status_text = "Delivery choice applied.".into();
                    return self.refresh_manager_task();
                }
                Err(error) => self.status_text = format!("Delivery failed: {error}"),
            },
            Message::WorkshedCapabilitiesLoaded(result) => match result {
                Ok(value) => {
                    self.workshed_capabilities = value;
                    self.status_text = "Workshed block capabilities loaded.".into();
                    if !self.workshed_selected_block.is_empty() {
                        return self.refresh_workshed_options_for_block(
                            &self.workshed_selected_block.clone(),
                            false,
                        );
                    }
                }
                Err(error) => {
                    self.status_text = format!("Workshed capabilities unavailable: {error}")
                }
            },
            Message::WorkshedWorkOrdersLoaded(result) => {
                self.workshed_refresh_in_flight = false;
                match result {
                    Ok(orders) => {
                        self.workshed_work_orders = orders;
                        if let Some(first) = self.workshed_work_orders.first().cloned() {
                            if self
                                .workshed_order
                                .get("id")
                                .and_then(Value::as_str)
                                .is_none()
                                || (self.workshed_order.get("id").and_then(Value::as_str)
                                    == Some("local-draft")
                                    && self.workshed_save_generation == 0)
                            {
                                self.workshed_order = first;
                                self.workshed_model_name_input = self.workshed_model_name();
                                self.workshed_ensure_selected_block();
                            }
                        } else if self.workshed_order.get("id").and_then(Value::as_str)
                            == Some("local-draft")
                        {
                            return wrap_app_task(cosmic::Task::perform(
                                self.api_client()
                                    .workshed_create_work_order(self.workshed_order.clone()),
                                Message::WorkshedWorkOrderLoaded,
                            ));
                        }
                    }
                    Err(error) => {
                        self.status_text = format!("Workshed work orders unavailable: {error}")
                    }
                }
            }
            Message::WorkshedWorkOrderLoaded(result) => match result {
                Ok(value) => {
                    workshed_remember_order(&mut self.workshed_work_orders, &value);
                    self.workshed_order = value;
                    self.workshed_model_name_input = self.workshed_model_name();
                    self.workshed_ensure_selected_block();
                }
                Err(error) => {
                    self.status_text = format!("Workshed work order unavailable: {error}")
                }
            },
            Message::WorkshedSaveLoaded { generation, result } => {
                self.workshed_save_in_flight = false;
                match result {
                    Ok(value) => {
                        workshed_remember_order(&mut self.workshed_work_orders, &value);
                        if generation == self.workshed_save_generation {
                            self.workshed_order = value;
                            self.workshed_model_name_input = self.workshed_model_name();
                            self.status_text = "Workshed changes saved.".into();
                        } else {
                            // A response may arrive after a newer local edit. Preserve the
                            // newer graph and only adopt the server identity/revision so the
                            // next serialized save can use optimistic concurrency correctly.
                            if let Some(id) = value.get("id").cloned() {
                                self.workshed_order["id"] = id;
                            }
                            if let Some(revision) = value.get("revision").cloned() {
                                self.workshed_order["revision"] = revision;
                            }
                            self.workshed_save_due = Some(Instant::now());
                        }
                    }
                    Err(error) => {
                        self.status_text = format!("Workshed changes could not be saved: {error}");
                    }
                }
            }
            Message::WorkshedPreflightLoaded { generation, result } => {
                if generation != self.workshed_preflight_generation {
                    return cosmic::Task::none();
                }
                match result {
                    Ok(value) => {
                        let ok = value.get("ok").and_then(Value::as_bool).unwrap_or(false);
                        let consents = value
                            .get("required_consents")
                            .and_then(Value::as_array)
                            .is_some_and(|items| !items.is_empty());
                        self.workshed_preflight = value;
                        self.status_text = if ok && consents {
                            self.workshed_preflight_open = true;
                            self.workshed_run_after_preflight = false;
                            "Review the required local tool preparation before continuing.".into()
                        } else if ok && self.workshed_run_after_preflight {
                            self.workshed_run_after_preflight = false;
                            return self.workshed_start_task(Vec::new());
                        } else if ok {
                            "Preflight passed. Ready to run on this Mac.".into()
                        } else {
                            self.workshed_run_after_preflight = false;
                            self.workshed_preflight_open = true;
                            "Fix the highlighted items before starting.".into()
                        };
                    }
                    Err(error) => {
                        self.workshed_preflight = Value::Null;
                        self.workshed_run_after_preflight = false;
                        self.workshed_preflight_open = true;
                        self.status_text = format!("Preflight failed: {error}");
                    }
                }
            }
            Message::WorkshedRunStarted(result) => match result {
                Ok(value) => {
                    self.workshed_run = Some(value);
                    self.status_text = "Workshed run queued on this Mac.".into();
                    self.workshed_preflight = Value::Null;
                }
                Err(error) => self.status_text = format!("Could not start Workshed run: {error}"),
            },
            Message::WorkshedRunLoaded(result) => match result {
                Ok(value) => self.workshed_run = Some(value),
                Err(error) => self.status_text = format!("Workshed run update failed: {error}"),
            },
            Message::WorkshedActionLoaded(result) => match result {
                Ok(value) => {
                    self.workshed_run = Some(value);
                    self.status_text = "Workshed run action sent.".into();
                }
                Err(error) => self.status_text = format!("Workshed action failed: {error}"),
            },
            Message::WorkshedNewPressed => {
                // Do not discard an unsaved edit or adopt an in-flight save's
                // identity into a new order. The short autosave must finish first.
                if self.workshed_save_in_flight || self.workshed_save_due.is_some() {
                    self.status_text =
                        "Saving current changes. Try New work order again once saved.".into();
                    return cosmic::Task::none();
                }
                workshed_remember_order(&mut self.workshed_work_orders, &self.workshed_order);
                self.workshed_order = workshed_new_order();
                self.workshed_reset_editor();
                self.status_text = "New local model development work order.".into();
                self.workshed_mark_dirty();
            }
            Message::WorkshedWorkOrderSelectPressed(id) => {
                if id == string_at(&self.workshed_order, "id") {
                    return cosmic::Task::none();
                }
                if self.workshed_save_in_flight || self.workshed_save_due.is_some() {
                    self.status_text =
                        "Saving current changes. Select the work order again once saved.".into();
                    return cosmic::Task::none();
                }
                let Some(order) = self
                    .workshed_work_orders
                    .iter()
                    .find(|order| string_at(order, "id") == id)
                    .cloned()
                else {
                    self.status_text =
                        "Work order unavailable. Refresh Workshed and try again.".into();
                    return cosmic::Task::none();
                };
                workshed_remember_order(&mut self.workshed_work_orders, &self.workshed_order);
                self.workshed_order = order;
                self.workshed_reset_editor();
                self.workshed_save_generation = self.workshed_save_generation.saturating_add(1);
                self.status_text = "Work order selected.".into();
            }
            Message::WorkshedCategoryPressed(category) => self.workshed_category = category,
            Message::WorkshedSearchChanged(value) => self.workshed_search = value,
            Message::WorkshedBlockSelectPressed(block_id) => {
                self.workshed_selected_block = block_id.clone();
                self.workshed_inspector_open = true;
                self.workshed_library_open = false;
                self.workshed_focused_field = None;
                return self.refresh_workshed_options_for_block(&block_id, false);
            }
            Message::WorkshedModelPickerToggled(block_id) => {
                if self.workshed_model_picker.as_deref() == Some(block_id.as_str()) {
                    self.workshed_model_picker = None;
                } else {
                    self.workshed_model_picker = Some(block_id.clone());
                    return self.load_workshed_options(&block_id, "ref", true);
                }
            }
            Message::WorkshedBlockAddPressed(type_id) => {
                workshed_add_block(
                    &mut self.workshed_order,
                    &self.workshed_capabilities,
                    &type_id,
                );
                self.workshed_selected_block = self
                    .workshed_order
                    .get("blocks")
                    .and_then(Value::as_array)
                    .and_then(|blocks| blocks.last())
                    .map(|block| string_at(block, "id"))
                    .unwrap_or_default();
                self.status_text = format!("Added {type_id} block to the work order.");
                self.workshed_library_open = false;
                self.workshed_inspector_open = true;
                self.workshed_mark_dirty();
                let selected = self.workshed_selected_block.clone();
                return self.refresh_workshed_options_for_block(&selected, false);
            }
            Message::WorkshedBlockDeletePressed(block_id) => {
                workshed_delete_block(&mut self.workshed_order, &block_id);
                if self.workshed_selected_block == block_id {
                    self.workshed_selected_block.clear();
                }
                self.workshed_field_drafts
                    .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                self.workshed_field_errors
                    .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                self.workshed_option_queries
                    .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                self.workshed_option_results
                    .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                self.workshed_option_requests
                    .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                self.workshed_option_loading
                    .retain(|key| !key.starts_with(&format!("{block_id}\u{1f}")));
                self.workshed_mark_dirty();
            }
            Message::WorkshedDeleteSelectedPressed => {
                let block_id = self.workshed_selected_block.clone();
                if !block_id.is_empty() {
                    workshed_delete_block(&mut self.workshed_order, &block_id);
                    self.workshed_selected_block.clear();
                    self.workshed_field_drafts
                        .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                    self.workshed_field_errors
                        .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                    self.workshed_option_queries
                        .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                    self.workshed_option_results
                        .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                    self.workshed_option_requests
                        .retain(|key, _| !key.starts_with(&format!("{block_id}\u{1f}")));
                    self.workshed_option_loading
                        .retain(|key| !key.starts_with(&format!("{block_id}\u{1f}")));
                    self.workshed_mark_dirty();
                }
            }
            Message::WorkshedBlockMovePressed(direction) => {
                if !self.workshed_selected_block.is_empty()
                    && workshed_move_block(
                        &mut self.workshed_order,
                        &self.workshed_selected_block,
                        direction,
                    )
                {
                    self.workshed_mark_dirty();
                }
            }
            Message::WorkshedBlockDuplicatePressed => {
                if let Some(block_id) = workshed_duplicate_block(
                    &mut self.workshed_order,
                    &self.workshed_selected_block,
                ) {
                    self.workshed_selected_block = block_id.clone();
                    self.workshed_mark_dirty();
                    return self.refresh_workshed_options_for_block(&block_id, false);
                }
            }
            Message::WorkshedModelNameChanged(value) => {
                self.workshed_model_name_input = value.clone();
                self.workshed_model_name_changed(value);
                self.workshed_mark_dirty();
            }
            Message::WorkshedParamDraftChanged {
                block_id,
                field,
                value,
            } => {
                self.workshed_param_draft_changed(block_id, field, value, false);
            }
            Message::WorkshedParamCommit { block_id, field } => {
                self.workshed_commit_param_draft(&block_id, &field);
            }
            Message::WorkshedParamSet {
                block_id,
                field,
                value,
            } => {
                self.workshed_set_param_value(&block_id, &field, value);
                return self.refresh_workshed_options_for_block(&block_id, true);
            }
            Message::WorkshedParamReset { block_id, field } => {
                self.workshed_reset_param(&block_id, &field);
            }
            Message::WorkshedParamStep {
                block_id,
                field,
                direction,
            } => {
                self.workshed_step_param(&block_id, &field, direction);
            }
            Message::WorkshedParamBindingPressed { block_id, field } => {
                self.workshed_toggle_param_binding(&block_id, &field);
            }
            Message::WorkshedInputDraftChanged {
                block_id,
                port,
                value,
            } => {
                self.workshed_input_draft_changed(block_id, port, value, false);
            }
            Message::WorkshedInputCommit { block_id, port } => {
                self.workshed_commit_input_draft(&block_id, &port);
                return self.refresh_workshed_options_for_block(&block_id, true);
            }
            Message::WorkshedInputModeChanged {
                block_id,
                port,
                mode,
            } => {
                self.workshed_set_input_mode(&block_id, &port, &mode);
                return self.refresh_workshed_options_for_block(&block_id, true);
            }
            Message::WorkshedAdvancedToggled(block_id) => {
                if !self.workshed_advanced_blocks.remove(&block_id) {
                    self.workshed_advanced_blocks.insert(block_id);
                }
            }
            Message::WorkshedOptionQueryChanged {
                block_id,
                field,
                value,
            } => {
                self.workshed_option_queries
                    .insert(workshed_option_key(&block_id, &field), value);
            }
            Message::WorkshedOptionSearchPressed { block_id, field } => {
                return self.load_workshed_options(&block_id, &field, true);
            }
            Message::WorkshedOptionSelected {
                block_id,
                field,
                value,
            } => {
                self.workshed_select_option(&block_id, &field, &value);
                if field == "ref"
                    && self.workshed_model_picker.as_deref() == Some(block_id.as_str())
                {
                    self.workshed_model_picker = None;
                }
                return self.refresh_workshed_options_for_block(&block_id, true);
            }
            Message::WorkshedOptionsLoaded {
                request_id,
                block_id,
                field,
                result,
            } => {
                let key = workshed_option_key(&block_id, &field);
                if self.workshed_option_requests.get(&key).copied() != Some(request_id) {
                    return cosmic::Task::none();
                }
                self.workshed_option_loading.remove(&key);
                match result {
                    Ok(value) => {
                        self.workshed_option_results.insert(key, value);
                    }
                    Err(error) => {
                        self.workshed_option_results
                            .insert(key, json!({"ok": false, "items": [], "message": error}));
                    }
                }
            }
            Message::WorkshedFieldFocused {
                block_id,
                field,
                focused,
            } => {
                let key = workshed_draft_key(&block_id, &field);
                if focused {
                    self.workshed_focused_field = Some(key);
                    return self.load_workshed_options(&block_id, &field, false);
                } else if self.workshed_focused_field.as_deref() == Some(key.as_str()) {
                    self.workshed_focused_field = None;
                    if field.starts_with("input:") {
                        self.workshed_commit_input_draft(
                            &block_id,
                            field.trim_start_matches("input:"),
                        );
                        return self.refresh_workshed_options_for_block(&block_id, true);
                    } else {
                        self.workshed_commit_param_draft(&block_id, &field);
                    }
                }
            }
            Message::WorkshedIssuePressed {
                block_id,
                field,
                port,
            } => {
                if !block_id.is_empty() {
                    self.workshed_selected_block = block_id.clone();
                    self.workshed_inspector_open = true;
                    self.workshed_library_open = false;
                }
                let focus_field = if !field.is_empty() {
                    field
                } else if !port.is_empty() {
                    format!("input:{port}")
                } else {
                    String::new()
                };
                if !focus_field.is_empty() && !block_id.is_empty() {
                    let definition = self.workshed_definition(&block_id);
                    if definition
                        .as_ref()
                        .and_then(|definition| workshed::field(definition, &focus_field))
                        .is_some_and(|field| field.advanced)
                    {
                        self.workshed_advanced_blocks.insert(block_id.clone());
                    }
                    self.workshed_focused_field = Some(workshed_draft_key(&block_id, &focus_field));
                    return wrap_app_task(widget::text_input::focus(workshed_input_id(
                        &block_id,
                        &focus_field,
                    )));
                }
            }
            Message::WorkshedSaveTick(now) => {
                if self.workshed_save_due.is_some_and(|due| now >= due)
                    && !self.workshed_save_in_flight
                {
                    self.workshed_save_due = None;
                    self.workshed_save_in_flight = true;
                    return self.queue_workshed_save(self.workshed_save_generation);
                }
            }
            Message::WorkshedPreflightPressed => return self.workshed_preflight_task(false),
            Message::WorkshedRunPressed => {
                let required_consents = self
                    .workshed_preflight
                    .get("required_consents")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if self.workshed_preflight_ok() && !required_consents.is_empty() {
                    self.workshed_preflight_open = true;
                    self.status_text =
                        "Review the required local tool preparation before continuing.".into();
                } else if self.workshed_preflight_ok() {
                    return self.workshed_start_task(Vec::new());
                } else {
                    self.workshed_run_after_preflight = true;
                    return self.workshed_preflight_task(true);
                }
            }
            Message::WorkshedConfirmRunPressed => {
                let consents = self
                    .workshed_preflight
                    .get("required_consents")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if self.workshed_preflight_ok() && !consents.is_empty() {
                    self.workshed_preflight_open = false;
                    return self.workshed_start_task(consents);
                }
            }
            Message::WorkshedActionPressed(action) => {
                if let Some(run_id) = self
                    .workshed_run
                    .as_ref()
                    .map(|run| string_at(run, "id"))
                    .filter(|id| !id.is_empty())
                {
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client().workshed_action(run_id, action),
                        Message::WorkshedActionLoaded,
                    ));
                }
            }
            Message::WorkshedPanelToggled(panel) => match panel.as_str() {
                "library" => {
                    self.workshed_library_open = !self.workshed_library_open;
                    if self.workshed_library_open {
                        self.workshed_inspector_open = false;
                    }
                }
                "inspector" => {
                    self.workshed_inspector_open = !self.workshed_inspector_open;
                    if self.workshed_inspector_open {
                        self.workshed_library_open = false;
                    }
                }
                "preflight" => self.workshed_preflight_open = !self.workshed_preflight_open,
                _ => {}
            },
            Message::IdeConnectionsLoaded(result) => {
                self.ide_connections_refresh_in_flight = false;
                match result {
                    Ok(payload) => self.ide_connections = payload,
                    Err(error) => {
                        self.status_text = format!("IDE connections unavailable: {error}")
                    }
                }
            }
            Message::IdePairingDecisionLoaded { approved, result } => match result {
                Ok(()) => {
                    self.status_text = if approved {
                        "JetBrains pairing approved.".into()
                    } else {
                        "JetBrains pairing rejected.".into()
                    };
                    return self.refresh_ide_connections_task();
                }
                Err(error) => self.status_text = format!("IDE pairing update failed: {error}"),
            },
            Message::IdeAuthorizationRevoked(result) => match result {
                Ok(revoked) => {
                    self.status_text = if revoked {
                        "JetBrains authorization revoked.".into()
                    } else {
                        "JetBrains authorization was already absent.".into()
                    };
                    return self.refresh_ide_connections_task();
                }
                Err(error) => {
                    self.status_text = format!("IDE authorization revoke failed: {error}")
                }
            },
            Message::AgentProfileLoaded { model, result } => {
                self.agent_profile_refresh_in_flight = false;
                if !same_agent_profile_model(&model, &self.current_profile_model()) {
                    return cosmic::Task::none();
                }
                match result {
                    Ok(payload) => {
                        if payload.configured {
                            self.agent_profile_state = AgentProfileState::Configured {
                                profile: payload.profile.unwrap_or(Value::Null),
                                completed_at: None,
                            };
                        } else if !self.agent_profile_state.is_configuring() {
                            self.agent_profile_state = AgentProfileState::Unconfigured;
                        }
                    }
                    Err(error) => {
                        if !self.agent_profile_state.is_configuring() {
                            self.agent_profile_state = AgentProfileState::Failed {
                                message: error.clone(),
                            };
                        }
                    }
                }
            }
            Message::AgentProfileConfigureStarted { model, result } => {
                if !same_agent_profile_model(&model, &self.current_profile_model()) {
                    return cosmic::Task::none();
                }
                match result {
                    Ok(payload) => {
                        if payload.reused {
                            self.agent_profile_state = AgentProfileState::Configured {
                                profile: payload.profile.unwrap_or(Value::Null),
                                completed_at: Some(Instant::now()),
                            };
                            self.status_text = "Agent Profile already configured.".into();
                        } else if let Some(job) = payload.job {
                            self.agent_profile_state = AgentProfileState::Configuring {
                                job_id: if payload.job_id.is_empty() {
                                    job.job_id
                                } else {
                                    payload.job_id
                                },
                                progress: job.progress,
                                phase: job.phase,
                            };
                            self.status_text = "Configuring OpenClaw and Hermes...".into();
                        } else if !payload.job_id.is_empty() {
                            self.agent_profile_state = AgentProfileState::Configuring {
                                job_id: payload.job_id,
                                progress: 0.0,
                                phase: "metadata".into(),
                            };
                            self.status_text = "Configuring OpenClaw and Hermes...".into();
                        } else {
                            self.agent_profile_state = AgentProfileState::Failed {
                                message: "Backend returned no Agent Profile job.".into(),
                            };
                        }
                    }
                    Err(error) => {
                        self.agent_profile_state = AgentProfileState::Failed {
                            message: error.clone(),
                        };
                        self.status_text = format!("Agent Profile configuration failed: {error}");
                    }
                }
            }
            Message::AgentProfilePollTick(_) => {
                let job_id = match &self.agent_profile_state {
                    AgentProfileState::Configuring { job_id, .. } if !job_id.trim().is_empty() => {
                        Some(job_id.clone())
                    }
                    _ => None,
                };
                if let Some(job_id) = job_id {
                    return self.refresh_agent_profile_job_task(job_id);
                }
            }
            Message::AgentProfileJobLoaded { model, result } => {
                self.agent_profile_refresh_in_flight = false;
                if !same_agent_profile_model(&model, &self.current_profile_model()) {
                    return cosmic::Task::none();
                }
                let AgentProfileState::Configuring { job_id, .. } = &self.agent_profile_state
                else {
                    return cosmic::Task::none();
                };
                if let Ok(job) = &result {
                    if (!job.model.is_empty()
                        && !same_agent_profile_model(&job.model, &self.current_profile_model()))
                        || job_id.is_empty()
                        || job_id != &job.job_id
                    {
                        return cosmic::Task::none();
                    }
                } else if job_id.is_empty() {
                    return cosmic::Task::none();
                }
                match result {
                    Ok(job) => match job.state.as_str() {
                        "configured" => {
                            let profile = job.profile.unwrap_or(Value::Null);
                            let limited_tools = profile_has_degraded_tools(&profile);
                            self.agent_profile_state = AgentProfileState::Configured {
                                profile,
                                completed_at: Some(Instant::now()),
                            };
                            self.status_text = if limited_tools {
                                "AI agents configured; structured tool calls are limited for this model.".into()
                            } else {
                                "AI agents configured for this model.".into()
                            };
                        }
                        "failed" => {
                            let message = if job.error.trim().is_empty() {
                                "OpenClaw or Hermes calibration failed. Retry Configure AI agents."
                                    .into()
                            } else {
                                job.error
                            };
                            self.agent_profile_state = AgentProfileState::Failed {
                                message: message.clone(),
                            };
                            self.status_text = format!("Agent Profile failed: {message}");
                        }
                        _ => {
                            let phase = job.phase;
                            let progress = job.progress;
                            self.agent_profile_state = AgentProfileState::Configuring {
                                job_id: job.job_id,
                                progress,
                                phase: phase.clone(),
                            };
                            self.status_text = format!(
                                "Configuring Agent Profile · {} ({:.0}%)",
                                phase, progress,
                            );
                        }
                    },
                    Err(error) => {
                        self.agent_profile_state = AgentProfileState::Failed {
                            message: error.clone(),
                        };
                        self.status_text = format!("Agent Profile status failed: {error}");
                    }
                }
            }
            Message::AgentProfileDeleted { model, result } => {
                if !same_agent_profile_model(&model, &self.current_profile_model()) {
                    return cosmic::Task::none();
                }
                match result {
                    Ok(_) => {
                        self.native_state.settings.openclaw_enabled = false;
                        self.native_state.settings.agent_runtime = "auto".into();
                        self.agent_profile_state = AgentProfileState::Unconfigured;
                        self.status_text = "Agent Profile deleted; model weights were kept.".into();
                        self.save_state();
                    }
                    Err(error) => {
                        self.status_text = format!("Agent Profile delete failed: {error}")
                    }
                }
            }
            Message::ChatStreamUpdate { request_id, event } => {
                if self
                    .streaming_reply
                    .as_ref()
                    .map(|stream| stream.request_id)
                    != Some(request_id)
                {
                    return cosmic::Task::none();
                }
                match event {
                    ChatStreamEvent::Start => {
                        self.status_text = "Model is responding...".into();
                        self.advance_stream_reveal(Instant::now());
                    }
                    ChatStreamEvent::ThinkingDelta(text) => {
                        if !text.trim().is_empty() {
                            self.mark_thinking_started(Instant::now());
                            self.append_stream_thinking(&text);
                            self.status_text = format!(
                                "Thinking: {}",
                                text.lines()
                                    .next()
                                    .unwrap_or("")
                                    .chars()
                                    .take(80)
                                    .collect::<String>()
                            );
                        }
                        self.advance_stream_reveal(Instant::now());
                    }
                    ChatStreamEvent::AnswerDelta(text) => {
                        self.mark_thinking_finished(Instant::now());
                        if let Some(stream) = &mut self.streaming_reply {
                            stream.received.push_str(&text);
                        }
                        self.status_text = "Streaming response...".into();
                        self.advance_stream_reveal(Instant::now());
                    }
                    ChatStreamEvent::Done(result) => {
                        let request_model_is_current =
                            self.streaming_reply.as_ref().is_some_and(|stream| {
                                same_agent_profile_model(
                                    &stream.request_model,
                                    &self.current_profile_model(),
                                )
                            });
                        self.mark_thinking_finished(Instant::now());
                        self.success_count += 1;
                        if let Some(stream) = &mut self.streaming_reply {
                            if stream.received.trim().is_empty() && !result.reply.trim().is_empty()
                            {
                                stream.received = result.reply.clone();
                            } else if stream.received.trim().is_empty() {
                                stream.received = "No response returned.".into();
                            }
                            stream.done = true;
                        } else if !result.reply.trim().is_empty() {
                            self.push_chat_message("assistant", &result.reply, None);
                        }
                        self.apply_chat_metrics(&result.metrics);
                        if request_model_is_current
                            && self.pending_model_switch.is_none()
                            && !result.model.is_empty()
                        {
                            self.native_state.selected_model = result.model;
                        }
                        self.status_text = "Chat response received.".into();
                        if self.advance_stream_reveal(Instant::now()) {
                            self.save_state();
                        }
                        return self.refresh_status_task();
                    }
                    ChatStreamEvent::Error(error) => {
                        self.mark_thinking_finished(Instant::now());
                        self.error_count += 1;
                        if let Some(stream) = &mut self.streaming_reply {
                            if !stream.received.trim().is_empty() {
                                stream.received.push_str("\n\n");
                            }
                            stream.received.push_str(&format!("Error: {error}"));
                            stream.done = true;
                        } else {
                            self.push_chat_message("assistant", &format!("Error: {error}"), None);
                        }
                        self.status_text = format!("Chat failed: {error}");
                        if self.advance_stream_reveal(Instant::now()) {
                            self.save_state();
                        }
                    }
                }
            }
            Message::ChatRevealTick(now) => {
                if self.advance_stream_reveal(now) {
                    self.save_state();
                }
            }
            Message::SwitchModelLoaded { model, result } => {
                // A model switch request may finish after a user picked
                // another model. Do not let its manager result restore the
                // old model/profile/runtime selection over the newer choice.
                if !same_agent_profile_model(&model, &self.current_profile_model()) {
                    return cosmic::Task::none();
                }
                match result {
                    Ok(data) => {
                        let active_model = if data.active_model.trim().is_empty() {
                            self.native_state.selected_model.clone()
                        } else {
                            data.active_model.clone()
                        };
                        let switch_error = if !data.ok && !data.last_error.trim().is_empty() {
                            Some(data.last_error.clone())
                        } else {
                            None
                        };
                        let switch_succeeded = switch_error.is_none();
                        self.apply_manager_payload(data);
                        if switch_error.is_some() {
                            self.restore_pending_model_switch();
                        } else {
                            self.pending_model_switch = None;
                            if !active_model.trim().is_empty()
                                && self.native_state.selected_model != active_model
                            {
                                self.native_state.selected_model = active_model.clone();
                                self.save_state();
                            }
                        }
                        let notice = if let Some(error) = switch_error {
                            format!("Model switch failed: {error}")
                        } else {
                            format!("Switched to {}.", empty_dash(&active_model))
                        };
                        self.status_text = notice.clone();
                        self.set_model_switch_notice(notice, Some(Duration::from_millis(1000)));
                        if switch_succeeded {
                            return self.refresh_agent_profile_task();
                        }
                    }
                    Err(error) => {
                        self.restore_pending_model_switch();
                        let notice = format!("Model switch failed: {error}");
                        self.status_text = notice.clone();
                        self.set_model_switch_notice(notice, Some(Duration::from_millis(1600)));
                    }
                }
            }
            Message::DeleteModelLoaded { model, result } => match result {
                Ok(data) => {
                    self.apply_manager_payload(data);
                    if !self.models.iter().any(|candidate| candidate == &model) {
                        if self.native_state.selected_local_model == model {
                            self.native_state.selected_local_model.clear();
                        }
                        if self.native_state.selected_model == model {
                            self.native_state.selected_model =
                                self.models.first().cloned().unwrap_or_default();
                        }
                        self.agent_profile_state = AgentProfileState::Unconfigured;
                        self.status_text = format!("Deleted local model '{model}'.");
                        self.save_state();
                    }
                }
                Err(error) => self.status_text = format!("Manager operation failed: {error}"),
            },
            Message::ConcurrentSaved(result) => match result {
                Ok(data) => self.apply_manager_payload(data),
                Err(error) => self.status_text = format!("Manager operation failed: {error}"),
            },
            Message::HfBindingSaved(result) => match result {
                Ok(message) => {
                    self.hf_token.clear();
                    self.status_text = if message.is_empty() {
                        "Hugging Face binding saved.".into()
                    } else {
                        message
                    };
                }
                Err(error) => self.status_text = format!("Hugging Face binding failed: {error}"),
            },
            Message::AttachMediaLoaded(result) => match result {
                Ok(items) => {
                    let added = items.len();
                    self.pending_media.extend(items);
                    self.status_text = format!("Attached {added} media file(s).");
                }
                Err(error) => self.status_text = format!("Attach media failed: {error}"),
            },
            Message::MessageInputAction(action) => {
                self.message_content.perform(action);
            }
            Message::InputFocusChanged(field, focused) => {
                if focused {
                    self.focused_input = Some(field);
                } else if self.focused_input == Some(field) {
                    self.focused_input = None;
                }
            }
            Message::ChatModelMenuToggled => {
                self.chat_model_menu_open = !self.chat_model_menu_open;
            }
            Message::ChatModelOptionPressed(model) => {
                if self.streaming_reply.is_some() {
                    self.chat_model_menu_open = false;
                    self.status_text =
                        "Wait for the current response to finish before switching models.".into();
                    return cosmic::Task::none();
                }
                self.begin_model_switch(&model);
                let request_model = model.clone();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().switch_model(model),
                    move |result| Message::SwitchModelLoaded {
                        model: request_model,
                        result,
                    },
                ));
            }
            Message::ServerUrlChanged(value) => self.native_state.settings.server_url = value,
            Message::ManagerTokenChanged(value) => self.native_state.settings.manager_token = value,
            Message::MaxTokensChanged(value) => self.max_tokens_input = value,
            Message::TemperatureChanged(value) => self.temperature_input = value,
            Message::TemperatureSliderChanged(value) => {
                let value = value.clamp(0.0, 2.0);
                self.native_state.settings.temperature = value;
                self.temperature_input = format!("{value:.2}");
            }
            Message::SystemPromptChanged(value) => {
                self.native_state.settings.system_prompt = value;
                self.save_state();
            }
            Message::SystemPromptAction(action) => {
                let edited = action.is_edit();
                self.prompt_content.perform(action);
                if edited {
                    self.status_text = "System prompt edited. Press Save to apply.".into();
                }
            }
            Message::ApplySystemPromptPressed => {
                let prompt = self.prompt_content.text();
                self.native_state.settings.system_prompt = prompt.clone();
                self.save_state();
                self.status_text = "System prompt applied.".into();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().developer_prompt(prompt),
                    Message::DeveloperPromptSaved,
                ));
            }
            Message::ClearSystemPromptPressed => {
                self.prompt_content = text_editor::Content::new();
                self.native_state.settings.system_prompt.clear();
                self.save_state();
                self.status_text = "System prompt cleared.".into();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().developer_prompt(String::new()),
                    Message::DeveloperPromptSaved,
                ));
            }
            Message::CommunityQueryChanged(value) => self.community_query = value,
            Message::QuantizationSourceKindChanged(value) => {
                if self.quantization_engine == "vllm_cuda" && value == "local_path" {
                    self.status_text = "CUDA Worker accepts Hub sources only.".into();
                    return cosmic::Task::none();
                }
                self.quantization_source_kind = if value == "local_path" {
                    "local_path".into()
                } else {
                    "huggingface".into()
                };
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationSourceChanged(value) => {
                self.quantization_source = value;
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationEngineChanged(value) => {
                self.quantization_engine = if value == "vllm_cuda" {
                    "vllm_cuda".into()
                } else {
                    "mlx_local".into()
                };
                if self.quantization_engine == "vllm_cuda" {
                    self.quantization_source_kind = "huggingface".into();
                }
                self.quantization_preset = if self.quantization_engine == "vllm_cuda" {
                    "cuda-balanced".into()
                } else {
                    "mlx-balanced".into()
                };
                self.quantization_algorithm = if self.quantization_engine == "vllm_cuda" {
                    "awq".into()
                } else {
                    String::new()
                };
                self.quantization_mode = if self.quantization_engine == "vllm_cuda" {
                    "compressed-tensors".into()
                } else {
                    "affine".into()
                };
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationPresetChanged(value) => {
                self.quantization_preset = value.clone();
                let selected_preset = self
                    .quantization_capabilities
                    .engines
                    .iter()
                    .find(|engine| {
                        engine.get("id").and_then(Value::as_str)
                            == Some(self.quantization_engine.as_str())
                    })
                    .and_then(|engine| engine.get("presets"))
                    .and_then(Value::as_array)
                    .and_then(|presets| {
                        presets.iter().find(|preset| {
                            preset.get("id").and_then(Value::as_str) == Some(value.as_str())
                        })
                    })
                    .cloned();
                if let Some(preset) = selected_preset {
                    if let Some(bits) = preset.get("bits").and_then(Value::as_u64) {
                        self.quantization_bits = bits.to_string();
                    }
                    if let Some(group_size) = preset.get("group_size").and_then(Value::as_u64) {
                        self.quantization_group_size = group_size.to_string();
                    }
                    if let Some(mode) = preset.get("mode").and_then(Value::as_str) {
                        self.quantization_mode = mode.to_string();
                    }
                    if let Some(algorithm) = preset.get("algorithm").and_then(Value::as_str) {
                        self.quantization_algorithm = algorithm.to_string();
                    }
                }
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationOutputChanged(value) => {
                self.quantization_output = value;
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationRevisionChanged(value) => {
                self.quantization_revision = value;
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationBitsChanged(value) => {
                self.quantization_bits = value;
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationGroupSizeChanged(value) => {
                self.quantization_group_size = value;
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationModeChanged(value) => {
                self.quantization_mode = value;
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationAlgorithmChanged(value) => {
                self.quantization_algorithm = value;
                self.quantization_preflight = Value::Null;
            }
            Message::QuantizationAdvancedToggled => {
                self.quantization_advanced_open = !self.quantization_advanced_open;
            }
            Message::HfUsernameChanged(value) => self.hf_username = value,
            Message::HfTokenChanged(value) => self.hf_token = value,
            Message::RuntimeBoolChanged(field, value) => match field {
                RuntimeBoolField::ContinuousBatching => {
                    self.runtime_config.continuous_batching = value
                }
                RuntimeBoolField::PagedCache => self.runtime_config.use_paged_cache = value,
                RuntimeBoolField::KvCacheQuantization => {
                    self.runtime_config.kv_cache_quantization = value
                }
                RuntimeBoolField::EnableMtp => self.runtime_config.enable_mtp = value,
            },
            Message::RuntimeTextChanged(field, value) => match field {
                RuntimeTextField::ChunkedPrefill => {
                    self.runtime_config.chunked_prefill_tokens = value
                }
                RuntimeTextField::MtpDraft => self.runtime_config.mtp_draft_tokens = value,
            },
            Message::ToggleConcurrentModel(model) => {
                if self
                    .selected_concurrent_models
                    .iter()
                    .any(|item| item == &model)
                {
                    self.selected_concurrent_models
                        .retain(|item| item != &model);
                } else {
                    self.selected_concurrent_models.push(model);
                }
            }
            Message::SendPressed => return self.send_chat_task(),
            Message::AttachMediaPressed => {
                return wrap_app_task(cosmic::Task::perform(
                    async { pick_media_files() },
                    Message::AttachMediaLoaded,
                ));
            }
            Message::ClearMediaPressed => {
                self.pending_media.clear();
                self.status_text = "Media cleared.".into();
            }
            Message::NewConversationPressed => {
                let id = format!("conv-{}", now_millis());
                self.native_state.active_conversation_id = id.clone();
                self.native_state.conversations.insert(
                    0,
                    StoredConversation {
                        id,
                        title: "New Conversation".into(),
                        messages: Vec::new(),
                    },
                );
                self.save_state();
            }
            Message::DeleteConversationPressed(id) => {
                self.native_state.conversations.retain(|item| item.id != id);
                if self.native_state.conversations.is_empty() {
                    let id = "main".to_string();
                    self.native_state.active_conversation_id = id.clone();
                    self.native_state.conversations.push(StoredConversation {
                        id,
                        title: "New Conversation".into(),
                        messages: Vec::new(),
                    });
                } else if !self
                    .native_state
                    .conversations
                    .iter()
                    .any(|item| item.id == self.native_state.active_conversation_id)
                {
                    self.native_state.active_conversation_id =
                        self.native_state.conversations[0].id.clone();
                }
                self.save_state();
            }
            Message::SelectConversationPressed(id) => {
                if self
                    .native_state
                    .conversations
                    .iter()
                    .any(|item| item.id == id)
                {
                    self.native_state.active_conversation_id = id;
                    self.save_state();
                }
            }
            Message::SaveServerSettingsPressed => {
                normalize_settings(&mut self.native_state.settings);
                self.save_state();
                self.status_text = "Server settings saved.".into();
                return cosmic::Task::batch(vec![
                    self.refresh_status_task(),
                    self.refresh_manager_task(),
                ]);
            }
            Message::ResetServerSettingsPressed => {
                let defaults = NativeSettings::default();
                self.native_state.settings.server_url = defaults.server_url;
                self.native_state.settings.manager_token = defaults.manager_token;
                self.native_state.settings.chat_dispatch_mode = defaults.chat_dispatch_mode;
                self.selected_concurrent_models.clear();
                self.save_state();
                self.status_text = "Server settings reset.".into();
                return cosmic::Task::batch(vec![
                    self.refresh_status_task(),
                    self.refresh_manager_task(),
                ]);
            }
            Message::SaveGenerationPressed => {
                self.apply_settings_inputs();
                self.save_state();
                self.status_text = "Generation settings saved.".into();
            }
            Message::ResetGenerationPressed => {
                let defaults = NativeSettings::default();
                self.native_state.settings.max_tokens = defaults.max_tokens;
                self.native_state.settings.temperature = defaults.temperature;
                self.max_tokens_input = self.native_state.settings.max_tokens.to_string();
                self.temperature_input = format!("{:.2}", self.native_state.settings.temperature);
                self.save_state();
                self.status_text = "Generation settings reset.".into();
            }
            Message::ClearStoragePressed => {
                let _ = self.store.clear();
                self.native_state = NativeState::default();
                self.max_tokens_input = self.native_state.settings.max_tokens.to_string();
                self.temperature_input = format!("{:.2}", self.native_state.settings.temperature);
                self.prompt_content =
                    text_editor::Content::with_text(&self.native_state.settings.system_prompt);
                self.status_text = "Native UI storage cleared.".into();
            }
            Message::SearchCommunityPressed => {
                let query = self.community_query.trim().to_string();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().community_search(query),
                    Message::CommunitySearchLoaded,
                ));
            }
            Message::DeployCommunityPressed(model) => {
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().community_deploy(model),
                    Message::CommunityDeployLoaded,
                ));
            }
            Message::CommunityJobActionPressed(action) => {
                let allowed = self.community_job.as_ref().is_some_and(|job| {
                    community_job_allowed_actions(job).contains(&action.as_str())
                });
                if !allowed {
                    self.status_text = "No active community job supports this action.".into();
                    return cosmic::Task::none();
                }
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().community_job_action(action),
                    Message::CommunityJobActionLoaded,
                ));
            }
            Message::QuantizationPreflightPressed => {
                self.status_text = "Running quantization preflight...".into();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client()
                        .quantization_preflight(self.quantization_request()),
                    Message::QuantizationPreflightLoaded,
                ));
            }
            Message::QuantizationStartPressed => {
                if !self.quantization_preflight_ok() {
                    self.status_text = "Run a successful preflight before starting.".into();
                } else {
                    self.status_text = "Starting quantization...".into();
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client()
                            .quantization_start(self.quantization_request()),
                        Message::QuantizationStarted,
                    ));
                }
            }
            Message::QuantizationActionPressed(action) => {
                if let Some(job) = &self.quantization_job {
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client()
                            .quantization_action(job.id.clone(), action),
                        Message::QuantizationActionLoaded,
                    ));
                }
            }
            Message::QuantizationDeliveryPressed(action) => {
                self.status_text = "Applying delivery choice...".into();
                if let Some(job) = &self.quantization_job {
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client()
                            .quantization_delivery(job.id.clone(), action),
                        Message::QuantizationDeliveryLoaded,
                    ));
                }
            }
            Message::QuantizationDeliveryDismissed => {
                self.quantization_delivery_open = false;
                self.status_text =
                    "Delivery choice deferred; the artifact remains available.".into();
            }
            Message::ApproveIdePairingPressed(pairing_id) => {
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client()
                        .decide_ide_pairing(pairing_id, true, false),
                    |result| Message::IdePairingDecisionLoaded {
                        approved: true,
                        result,
                    },
                ));
            }
            Message::ApproveIdePairingAllPressed(pairing_id) => {
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().decide_ide_pairing(pairing_id, true, true),
                    |result| Message::IdePairingDecisionLoaded {
                        approved: true,
                        result,
                    },
                ));
            }
            Message::RejectIdePairingPressed(pairing_id) => {
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client()
                        .decide_ide_pairing(pairing_id, false, false),
                    |result| Message::IdePairingDecisionLoaded {
                        approved: false,
                        result,
                    },
                ));
            }
            Message::RevokeIdeAuthorizationPressed(client_id) => {
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().revoke_ide_authorization(client_id),
                    Message::IdeAuthorizationRevoked,
                ));
            }
            Message::CopyJetBrainsProviderUrlPressed => {
                let value = self.ide_connections.integration.provider_url.clone();
                self.status_text = if copy_to_clipboard(&value) {
                    "Token Workshed Provider URL copied.".into()
                } else {
                    "Could not access the macOS clipboard.".into()
                };
            }
            Message::CopyJetBrainsProviderKeyPressed => {
                let value = self.ide_connections.integration.provider_api_key.clone();
                self.status_text = if copy_to_clipboard(&value) {
                    "Token Workshed Provider key copied.".into()
                } else {
                    "Could not access the macOS clipboard.".into()
                };
            }
            Message::CopyJetBrainsMcpCommandPressed => {
                let integration = &self.ide_connections.integration;
                let value = std::iter::once(integration.mcp_command.as_str())
                    .chain(integration.mcp_args.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.status_text = if copy_to_clipboard(&value) {
                    "Token Workshed MCP command copied.".into()
                } else {
                    "Could not access the macOS clipboard.".into()
                };
            }
            Message::SelectLocalModelPressed(model) => {
                if self.models.iter().any(|candidate| candidate == &model) {
                    self.native_state.selected_local_model = model;
                    self.save_state();
                }
            }
            Message::SwitchModelPressed(model) => {
                if self.streaming_reply.is_some() {
                    self.status_text =
                        "Wait for the current response to finish before switching models.".into();
                    return cosmic::Task::none();
                }
                self.begin_model_switch(&model);
                let request_model = model.clone();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().switch_model(model),
                    move |result| Message::SwitchModelLoaded {
                        model: request_model,
                        result,
                    },
                ));
            }
            Message::DeleteSelectedModelPressed => {
                let model = if !self.native_state.selected_local_model.trim().is_empty() {
                    self.native_state.selected_local_model.trim().to_string()
                } else {
                    self.native_state.selected_model.trim().to_string()
                };
                if model.is_empty() {
                    self.status_text = "Select a local model first.".into();
                } else {
                    self.status_text = format!("Deleting local model '{model}'...");
                    let request_model = model.clone();
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client().delete_model(model),
                        move |result| Message::DeleteModelLoaded {
                            model: request_model,
                            result,
                        },
                    ));
                }
            }
            Message::ResetRuntimePressed => {
                self.runtime_config = RuntimeConfig::default();
            }
            Message::SaveRuntimePressed => {
                let cfg = self.runtime_config.clone();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().update_runtime_config(cfg),
                    Message::RuntimeSaved,
                ));
            }
            Message::StopExtrasPressed => {
                self.selected_concurrent_models.clear();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().set_concurrent_models(Vec::new()),
                    Message::ConcurrentSaved,
                ));
            }
            Message::ApplyConcurrentPressed => {
                let models = self.selected_concurrent_models.clone();
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client().set_concurrent_models(models),
                    Message::ConcurrentSaved,
                ));
            }
            Message::SaveHfBindingPressed => {
                return wrap_app_task(cosmic::Task::perform(
                    self.api_client()
                        .save_hf_binding(self.hf_username.clone(), self.hf_token.clone()),
                    Message::HfBindingSaved,
                ));
            }
            Message::ConfigureAgentProfilePressed => {
                let model = self.current_profile_model();
                if model.is_empty() {
                    self.agent_profile_state = AgentProfileState::Failed {
                        message: "Select a model before configuring AI agents.".into(),
                    };
                } else if !self.agent_profile_state.is_configuring() {
                    self.native_state.settings.openclaw_enabled = false;
                    self.native_state.settings.agent_runtime = "auto".into();
                    self.agent_profile_state = AgentProfileState::Configuring {
                        job_id: String::new(),
                        progress: 0.0,
                        phase: "metadata".into(),
                    };
                    self.save_state();
                    let request_model = model.clone();
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client().configure_agent_profile(
                            model,
                            self.native_state.settings.server_url.clone(),
                            false,
                        ),
                        move |result| Message::AgentProfileConfigureStarted {
                            model: request_model,
                            result,
                        },
                    ));
                }
            }
            Message::RecalibrateAgentProfilePressed => {
                let model = self.current_profile_model();
                if model.is_empty() {
                    self.status_text = "Select a model before recalibrating its profile.".into();
                } else if !self.agent_profile_state.is_configuring() {
                    self.native_state.settings.openclaw_enabled = false;
                    self.native_state.settings.agent_runtime = "auto".into();
                    self.agent_profile_state = AgentProfileState::Configuring {
                        job_id: String::new(),
                        progress: 0.0,
                        phase: "metadata".into(),
                    };
                    self.save_state();
                    let request_model = model.clone();
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client().configure_agent_profile(
                            model,
                            self.native_state.settings.server_url.clone(),
                            true,
                        ),
                        move |result| Message::AgentProfileConfigureStarted {
                            model: request_model,
                            result,
                        },
                    ));
                }
            }
            Message::DeleteAgentProfilePressed => {
                let model = self.current_profile_model();
                if model.is_empty() {
                    self.status_text = "Select a model before deleting its profile.".into();
                } else if self.agent_profile_state.is_configuring() {
                    self.status_text =
                        "Wait for the active Agent Profile configuration to finish or fail.".into();
                } else {
                    let request_model = model.clone();
                    return wrap_app_task(cosmic::Task::perform(
                        self.api_client().delete_agent_profile(model),
                        move |result| Message::AgentProfileDeleted {
                            model: request_model,
                            result,
                        },
                    ));
                }
            }
            Message::ToggleAgentRuntime(runtime) => {
                if runtime != "off" && !self.agent_profile_state.is_configured() {
                    self.status_text =
                        "Configure AI agents for this model before enabling a runtime.".into();
                } else if runtime == "off" || self.agent_mode() == runtime {
                    self.native_state.settings.openclaw_enabled = false;
                    self.native_state.settings.agent_runtime = "auto".into();
                } else {
                    self.native_state.settings.openclaw_enabled = true;
                    self.native_state.settings.agent_runtime = runtime;
                }
                self.save_state();
            }
            Message::DispatchModeChanged(value) => {
                self.native_state.settings.chat_dispatch_mode = if value == "parallel" {
                    "parallel".into()
                } else {
                    "single".into()
                };
                self.save_state();
            }
            Message::LanguageSelected(value) => {
                self.native_state.settings.language = if value == "zh" {
                    "zh".into()
                } else {
                    "en".into()
                };
                self.save_state();
            }
            Message::CloseRequested(id) => {
                self.status_text = "Window minimized; backend remains running.".into();
                return wrap_app_task(window::minimize::<Message>(id, true));
            }
            Message::QuitPressed => {
                self.allow_close = true;
                if let Some(id) = self.core.main_window_id() {
                    return wrap_app_task(window::close::<Message>(id));
                }
            }
        }

        cosmic::Task::none()
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        let mut subscriptions: Vec<iced::Subscription<Message>> =
            vec![iced::time::every(Duration::from_secs(6)).map(Message::PollTick)];
        let notice_expires = self
            .model_switch_notice
            .as_ref()
            .and_then(|notice| notice.expires_at)
            .is_some();
        if self.sidebar_bounce.is_some()
            || notice_expires
            || self.agent_profile_state.completion_active()
        {
            subscriptions
                .push(iced::time::every(Duration::from_millis(33)).map(Message::AnimationTick));
        }
        if self.agent_profile_state.is_configuring() {
            subscriptions.push(
                iced::time::every(Duration::from_millis(600)).map(Message::AgentProfilePollTick),
            );
        }
        if self.current_page == Page::Logs {
            subscriptions.push(
                iced::time::every(Duration::from_millis(500)).map(Message::DeveloperLogsTick),
            );
        }
        if self.current_page == Page::Workshed
            && self.workshed_run.as_ref().is_some_and(|run| {
                !matches!(
                    string_at(run, "status").as_str(),
                    "completed" | "failed" | "cancelled" | "interrupted"
                )
            })
        {
            subscriptions.push(
                iced::time::every(Duration::from_millis(700)).map(Message::QuantizationJobTick),
            );
        }
        if self.workshed_save_due.is_some() {
            subscriptions
                .push(iced::time::every(Duration::from_millis(100)).map(Message::WorkshedSaveTick));
        }
        if self.current_page == Page::Workshed && !self.workshed_selected_block.is_empty() {
            subscriptions.push(iced::event::listen_with(|event, status, _window| {
                if status != iced::event::Status::Ignored {
                    return None;
                }
                let iced::Event::Keyboard(keyboard::Event::KeyPressed { key, modifiers, .. }) =
                    event
                else {
                    return None;
                };
                match key.as_ref() {
                    keyboard::Key::Named(keyboard::key::Named::ArrowLeft)
                    | keyboard::Key::Named(keyboard::key::Named::ArrowUp) => {
                        Some(Message::WorkshedBlockMovePressed(-1))
                    }
                    keyboard::Key::Named(keyboard::key::Named::ArrowRight)
                    | keyboard::Key::Named(keyboard::key::Named::ArrowDown) => {
                        Some(Message::WorkshedBlockMovePressed(1))
                    }
                    keyboard::Key::Named(keyboard::key::Named::Backspace)
                    | keyboard::Key::Named(keyboard::key::Named::Delete) => {
                        Some(Message::WorkshedDeleteSelectedPressed)
                    }
                    keyboard::Key::Character(value)
                        if modifiers.command() && value.eq_ignore_ascii_case("d") =>
                    {
                        Some(Message::WorkshedBlockDuplicatePressed)
                    }
                    _ => None,
                }
            }));
        }
        if self.streaming_reply.is_some() {
            subscriptions
                .push(iced::time::every(Duration::from_millis(33)).map(Message::ChatRevealTick));
        }
        iced::Subscription::batch(subscriptions)
    }

    fn on_close_requested(&self, id: window::Id) -> Option<Self::Message> {
        if self.allow_close {
            None
        } else {
            Some(Message::CloseRequested(id))
        }
    }

    fn view(&self) -> Element<'_, Self::Message> {
        let palette = self.active_palette();
        let shell = if self.ide_companion {
            self.ide_companion_shell()
        } else {
            self.app_shell()
        };
        let base = widget::container(shell)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(APP_PADDING)
            .style(move |_| palette.window_style());

        Element::from(base)
    }
}

impl TokenWorkshedApp {
    fn active_palette(&self) -> crate::appearance::Palette {
        palette(MaterialMode::Frosted, self.window_effects.supports_blur())
    }

    fn glass_layer<'a>(
        &self,
        content: Element<'a, Message>,
        variant: GlassVariant,
        state: GlassState,
    ) -> Element<'a, Message> {
        match variant {
            GlassVariant::Panel => self.panel_surface(content, state),
            GlassVariant::Card => self.aluminum_card_frame(content),
            _ => content,
        }
    }

    fn aluminum_frame<'a>(&self, content: Element<'a, Message>) -> Element<'a, Message> {
        widget::container(content).width(Length::Fill).into()
    }

    fn panel_surface<'a>(
        &self,
        content: Element<'a, Message>,
        _state: GlassState,
    ) -> Element<'a, Message> {
        let palette = self.active_palette();
        widget::container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_| flat_frame_host_style(palette))
            .into()
    }

    fn aluminum_card_frame<'a>(&self, content: Element<'a, Message>) -> Element<'a, Message> {
        // Aluminum rules are reserved for real section boundaries. A card
        // wrapper must not add decorative top and bottom rails to every page.
        widget::container(content).width(Length::Fill).into()
    }

    fn api_client(&self) -> ApiClient {
        ApiClient::new(
            self.api_base.clone(),
            self.native_state.settings.manager_token.clone(),
        )
    }

    fn begin_model_switch(&mut self, model: &str) {
        self.chat_model_menu_open = false;
        self.pending_model_switch = Some(ModelSwitchSnapshot {
            selected_model: self.native_state.selected_model.clone(),
            selected_local_model: self.native_state.selected_local_model.clone(),
            profile_state: self.agent_profile_state.clone(),
            openclaw_enabled: self.native_state.settings.openclaw_enabled,
            agent_runtime: self.native_state.settings.agent_runtime.clone(),
        });
        self.native_state.selected_local_model = model.to_string();
        self.native_state.selected_model = model.to_string();
        self.native_state.settings.openclaw_enabled = false;
        self.native_state.settings.agent_runtime = "auto".into();
        self.agent_profile_state = AgentProfileState::Unconfigured;
        self.minimum_manager_response_generation =
            self.manager_refresh_generation.saturating_add(1);
        self.save_state();
        let notice = format!("Switching to {model}...");
        self.status_text = notice.clone();
        self.set_model_switch_notice(notice, None);
    }

    fn restore_pending_model_switch(&mut self) {
        let Some(snapshot) = self.pending_model_switch.take() else {
            return;
        };
        self.native_state.selected_model = snapshot.selected_model;
        self.native_state.selected_local_model = snapshot.selected_local_model;
        self.agent_profile_state = snapshot.profile_state;
        self.native_state.settings.openclaw_enabled = snapshot.openclaw_enabled;
        self.native_state.settings.agent_runtime = snapshot.agent_runtime;
        self.save_state();
    }

    fn current_profile_model(&self) -> String {
        if !self.native_state.selected_model.trim().is_empty() {
            self.native_state.selected_model.trim().to_string()
        } else {
            self.native_state.selected_local_model.trim().to_string()
        }
    }

    fn profile_status_label(&self) -> String {
        match &self.agent_profile_state {
            AgentProfileState::Unconfigured => "Not configured".into(),
            AgentProfileState::Configuring {
                phase, progress, ..
            } => format!("Configuring {phase} ({progress:.0}%)"),
            AgentProfileState::Configured { profile, .. } => {
                if profile_has_degraded_tools(profile) {
                    "Configured · limited tools".into()
                } else {
                    "Configured".into()
                }
            }
            AgentProfileState::Failed { message } => {
                format!("Failed: {}", message.chars().take(88).collect::<String>())
            }
        }
    }

    fn model_integrity_status_label(&self) -> String {
        let report = &self.manager_models.integrity_report;
        if report.is_null() {
            return "Not checked".into();
        }
        let errors = report
            .get("errors")
            .and_then(Value::as_array)
            .map(|items| items.len())
            .unwrap_or(0);
        if errors > 0 {
            return format!("Check issue ({errors})");
        }
        let pending = report
            .get("pending")
            .and_then(Value::as_array)
            .map(|items| items.len())
            .unwrap_or(0);
        if pending > 0 {
            return format!("Retry cleanup ({pending})");
        }
        let removed = report
            .get("removed")
            .and_then(Value::as_array)
            .map(|items| items.len())
            .unwrap_or(0);
        if removed > 0 {
            return format!("Auto-cleaned {removed}");
        }
        if report
            .get("checked")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            "Healthy".into()
        } else {
            "Not checked".into()
        }
    }

    fn set_model_switch_notice(&mut self, message: String, expires_after: Option<Duration>) {
        self.model_switch_notice = Some(ModelSwitchNotice {
            message,
            expires_at: expires_after.map(|duration| Instant::now() + duration),
        });
    }

    fn refresh_config_task(&self) -> cosmic::app::Task<Message> {
        wrap_app_task(cosmic::Task::perform(
            self.api_client().config(),
            Message::ConfigLoaded,
        ))
    }

    fn refresh_manager_task(&mut self) -> cosmic::app::Task<Message> {
        if self.manager_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.manager_refresh_in_flight = true;
        self.manager_refresh_generation = self.manager_refresh_generation.wrapping_add(1).max(1);
        let generation = self.manager_refresh_generation;
        wrap_app_task(cosmic::Task::perform(
            self.api_client().manager_models(),
            move |result| Message::ManagerLoaded { generation, result },
        ))
    }

    fn refresh_agent_profile_task(&mut self) -> cosmic::app::Task<Message> {
        let model = self.current_profile_model();
        if model.is_empty() || self.agent_profile_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.agent_profile_refresh_in_flight = true;
        let request_model = model.clone();
        wrap_app_task(cosmic::Task::perform(
            self.api_client().agent_profile(model),
            move |result| Message::AgentProfileLoaded {
                model: request_model,
                result,
            },
        ))
    }

    fn refresh_agent_profile_job_task(&mut self, job_id: String) -> cosmic::app::Task<Message> {
        if self.agent_profile_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.agent_profile_refresh_in_flight = true;
        let model = self.current_profile_model();
        wrap_app_task(cosmic::Task::perform(
            self.api_client().agent_profile_job(job_id),
            move |result| Message::AgentProfileJobLoaded { model, result },
        ))
    }

    fn refresh_status_task(&mut self) -> cosmic::app::Task<Message> {
        if self.status_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.status_refresh_in_flight = true;
        wrap_app_task(cosmic::Task::perform(
            self.api_client()
                .status(self.native_state.settings.server_url.clone()),
            Message::StatusLoaded,
        ))
    }

    fn refresh_runtime_task(&self) -> cosmic::app::Task<Message> {
        wrap_app_task(cosmic::Task::perform(
            self.api_client().runtime_config(),
            Message::RuntimeLoaded,
        ))
    }

    fn refresh_developer_logs_task(&mut self) -> cosmic::app::Task<Message> {
        if self.developer_logs_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.developer_logs_refresh_in_flight = true;
        wrap_app_task(cosmic::Task::perform(
            self.api_client().developer_logs(),
            Message::DeveloperLogsLoaded,
        ))
    }

    fn refresh_community_job_task(&mut self) -> cosmic::app::Task<Message> {
        if self.community_job_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.community_job_refresh_in_flight = true;
        wrap_app_task(cosmic::Task::perform(
            self.api_client().community_job(),
            Message::CommunityJobLoaded,
        ))
    }

    fn refresh_quantization_capabilities_task(&mut self) -> cosmic::app::Task<Message> {
        if self.quantization_capabilities_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.quantization_capabilities_refresh_in_flight = true;
        wrap_app_task(cosmic::Task::perform(
            self.api_client().quantization_capabilities(),
            Message::QuantizationCapabilitiesLoaded,
        ))
    }

    fn refresh_quantization_job_task(&mut self) -> cosmic::app::Task<Message> {
        let Some(job_id) = self
            .quantization_job
            .as_ref()
            .map(|job| job.id.clone())
            .filter(|value| !value.trim().is_empty())
        else {
            return cosmic::Task::none();
        };
        if self.quantization_job_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.quantization_job_refresh_in_flight = true;
        wrap_app_task(cosmic::Task::perform(
            self.api_client().quantization_job(job_id),
            Message::QuantizationJobLoaded,
        ))
    }

    fn refresh_workshed_task(&mut self) -> cosmic::app::Task<Message> {
        if self.workshed_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.workshed_refresh_in_flight = true;
        cosmic::Task::batch(vec![
            wrap_app_task(cosmic::Task::perform(
                self.api_client().workshed_capabilities(),
                Message::WorkshedCapabilitiesLoaded,
            )),
            wrap_app_task(cosmic::Task::perform(
                self.api_client().workshed_work_orders(),
                Message::WorkshedWorkOrdersLoaded,
            )),
            self.refresh_workshed_run_task(),
        ])
    }

    fn queue_workshed_save(&self, generation: u64) -> cosmic::app::Task<Message> {
        let order = self.workshed_order.clone();
        let id = string_at(&order, "id");
        if workshed_order_needs_create(&order, &self.workshed_work_orders) {
            wrap_app_task(cosmic::Task::perform(
                self.api_client().workshed_create_work_order(order),
                move |result| Message::WorkshedSaveLoaded { generation, result },
            ))
        } else {
            let revision = order.get("revision").and_then(Value::as_u64).unwrap_or(1);
            wrap_app_task(cosmic::Task::perform(
                self.api_client()
                    .workshed_update_work_order(id, order, revision),
                move |result| Message::WorkshedSaveLoaded { generation, result },
            ))
        }
    }

    fn refresh_workshed_run_task(&mut self) -> cosmic::app::Task<Message> {
        let Some(run_id) = self
            .workshed_run
            .as_ref()
            .map(|run| string_at(run, "id"))
            .filter(|id| !id.is_empty())
        else {
            return cosmic::Task::none();
        };
        let after = self
            .workshed_run
            .as_ref()
            .and_then(|run| run.get("event_cursor"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        wrap_app_task(cosmic::Task::perform(
            self.api_client().workshed_run_poll(run_id, after),
            Message::WorkshedRunLoaded,
        ))
    }

    fn workshed_preflight_ok(&self) -> bool {
        Self::workshed_preflight_is_current(&self.workshed_preflight, &self.workshed_field_errors)
    }

    fn workshed_preflight_is_current(
        preflight: &Value,
        field_errors: &BTreeMap<String, String>,
    ) -> bool {
        if !field_errors.is_empty() {
            return false;
        }
        let expires_at = preflight
            .get("expires_at")
            .and_then(Value::as_f64)
            .unwrap_or(f64::INFINITY);
        preflight
            .get("ok")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && !preflight
                .get("stale")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            && expires_at
                > SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64()
    }

    fn workshed_preflight_task(&mut self, run_after: bool) -> cosmic::app::Task<Message> {
        if !self.workshed_field_errors.is_empty() {
            self.workshed_run_after_preflight = false;
            self.status_text =
                "Fix the highlighted parameter errors before checking or running the work order."
                    .into();
            return cosmic::Task::none();
        }
        self.workshed_preflight_generation = self.workshed_preflight_generation.saturating_add(1);
        self.workshed_run_after_preflight = run_after;
        self.status_text = "Checking the model, available space, and local tools…".into();
        let generation = self.workshed_preflight_generation;
        wrap_app_task(cosmic::Task::perform(
            self.api_client()
                .workshed_preflight(self.workshed_order.clone()),
            move |result| Message::WorkshedPreflightLoaded { generation, result },
        ))
    }

    fn workshed_start_task(&self, consents: Vec<String>) -> cosmic::app::Task<Message> {
        let preflight_id = string_at(&self.workshed_preflight, "preflight_id");
        if !self.workshed_preflight_ok() || preflight_id.is_empty() {
            return wrap_app_task(cosmic::Task::none());
        }
        wrap_app_task(cosmic::Task::perform(
            self.api_client().workshed_start(preflight_id, consents),
            Message::WorkshedRunStarted,
        ))
    }

    fn workshed_model_name(&self) -> String {
        self.workshed_order
            .get("blocks")
            .and_then(Value::as_array)
            .and_then(|blocks| {
                blocks.iter().find(|block| {
                    block
                        .get("locked")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
            })
            .and_then(|root| root.get("params"))
            .and_then(|params| params.get("model_name"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }

    fn workshed_model_name_changed(&mut self, value: String) {
        if let Some(root) = self
            .workshed_order
            .get_mut("blocks")
            .and_then(Value::as_array_mut)
            .and_then(|blocks| {
                blocks.iter_mut().find(|block| {
                    block
                        .get("locked")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
            })
        {
            root["params"]["model_name"] = Value::String(value.clone());
        }
        self.workshed_order["name"] = Value::String(if value.trim().is_empty() {
            "New model".into()
        } else {
            value
        });
    }

    fn workshed_mark_dirty(&mut self) {
        self.workshed_save_generation = self.workshed_save_generation.saturating_add(1);
        self.workshed_save_due = Some(Instant::now() + Duration::from_millis(500));
        self.workshed_run_after_preflight = false;
        self.workshed_preflight_open = false;
        Self::workshed_invalidate_preflight(
            &mut self.workshed_preflight,
            &mut self.workshed_preflight_generation,
        );
    }

    fn workshed_invalidate_preflight(preflight: &mut Value, generation: &mut u64) {
        // An edit invalidates both a displayed plan and any still-in-flight
        // response. Otherwise a late response can approve the previous draft.
        *generation = generation.saturating_add(1);
        if !preflight.is_null() {
            preflight["ok"] = Value::Bool(false);
            preflight["stale"] = Value::Bool(true);
        }
    }

    fn workshed_reset_editor(&mut self) {
        self.workshed_model_name_input = self.workshed_model_name();
        self.workshed_preflight = Value::Null;
        self.workshed_preflight_generation = self.workshed_preflight_generation.saturating_add(1);
        self.workshed_run = None;
        self.workshed_model_picker = None;
        self.workshed_run_after_preflight = false;
        self.workshed_selected_block.clear();
        self.workshed_focused_field = None;
        self.workshed_inspector_open = false;
        self.workshed_field_drafts.clear();
        self.workshed_field_errors.clear();
        self.workshed_option_queries.clear();
        self.workshed_option_results.clear();
        self.workshed_option_requests.clear();
        self.workshed_option_loading.clear();
        self.workshed_advanced_blocks.clear();
        self.workshed_ensure_selected_block();
    }

    fn workshed_definition(&self, block_id: &str) -> Option<BlockDefinition> {
        let block = workshed::block(&self.workshed_order, block_id)?;
        Some(workshed::definition_for(
            &self.workshed_capabilities,
            &string_at(block, "type_id"),
        ))
    }

    fn workshed_ensure_selected_block(&mut self) {
        let selected_exists = self
            .workshed_order
            .get("blocks")
            .and_then(Value::as_array)
            .is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|block| string_at(block, "id") == self.workshed_selected_block.as_str())
            });
        if selected_exists {
            return;
        }
        self.workshed_selected_block = self
            .workshed_order
            .get("blocks")
            .and_then(Value::as_array)
            .and_then(|blocks| {
                blocks.iter().find(|block| {
                    !block
                        .get("locked")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                        && string_at(block, "type_id") != "model_development"
                })
            })
            .map(|block| string_at(block, "id"))
            .unwrap_or_default();
    }

    fn workshed_param_draft_changed(
        &mut self,
        block_id: String,
        field_key: String,
        value: String,
        force: bool,
    ) {
        let draft_key = workshed_draft_key(&block_id, &field_key);
        self.workshed_field_drafts
            .insert(draft_key.clone(), value.clone());
        let Some(definition) = self.workshed_definition(&block_id) else {
            return;
        };
        let Some(field) = workshed::field(&definition, &field_key) else {
            return;
        };
        match workshed::parse_field_text(&field, &value) {
            Ok(parsed) => {
                self.workshed_field_errors.remove(&draft_key);
                self.workshed_apply_param_value(&block_id, &field, parsed);
            }
            Err(error) => {
                if force
                    || matches!(
                        field.kind,
                        ParamKind::Integer | ParamKind::Number | ParamKind::Boolean
                    )
                {
                    self.workshed_field_errors.insert(draft_key, error);
                }
            }
        }
    }

    fn workshed_commit_param_draft(&mut self, block_id: &str, field_key: &str) {
        let key = workshed_draft_key(block_id, field_key);
        let Some(value) = self.workshed_field_drafts.get(&key).cloned() else {
            return;
        };
        self.workshed_param_draft_changed(block_id.to_string(), field_key.to_string(), value, true);
    }

    fn workshed_set_param_value(&mut self, block_id: &str, field_key: &str, value: Value) {
        let Some(definition) = self.workshed_definition(block_id) else {
            return;
        };
        let Some(field) = workshed::field(&definition, field_key) else {
            return;
        };
        let draft_key = workshed_draft_key(block_id, field_key);
        self.workshed_field_drafts.insert(
            draft_key.clone(),
            workshed::field_text(Some(&value), &field),
        );
        self.workshed_field_errors.remove(&draft_key);
        self.workshed_apply_param_value(block_id, &field, value);
    }

    fn workshed_apply_param_value(&mut self, block_id: &str, field: &ParamField, value: Value) {
        let current = workshed::block(&self.workshed_order, block_id)
            .and_then(|block| block.get("params"))
            .and_then(|params| params.get(&field.key))
            .cloned();
        let normalized = (!value.is_null()).then_some(value);
        if current == normalized {
            return;
        }
        if field.bindable {
            let _ = workshed::set_input_binding(
                &mut self.workshed_order,
                block_id,
                &format!("param.{}", field.key),
                InputBinding::Missing,
            );
        }
        if workshed::set_param(&mut self.workshed_order, block_id, &field.key, normalized) {
            self.workshed_mark_dirty();
        }
    }

    fn workshed_reset_param(&mut self, block_id: &str, field_key: &str) {
        let Some(definition) = self.workshed_definition(block_id) else {
            return;
        };
        let Some(field) = workshed::field(&definition, field_key) else {
            return;
        };
        let key = workshed_draft_key(block_id, field_key);
        self.workshed_field_drafts.remove(&key);
        self.workshed_field_errors.remove(&key);
        if workshed::set_param(&mut self.workshed_order, block_id, field_key, None) {
            // Reset removes only the explicit override. A connected bindable
            // parameter remains connected; otherwise its schema/default_params
            // value becomes effective immediately.
            let _ = field;
            self.workshed_mark_dirty();
        }
    }

    fn workshed_step_param(&mut self, block_id: &str, field_key: &str, direction: i8) {
        let Some(definition) = self.workshed_definition(block_id) else {
            return;
        };
        let Some(field) = workshed::field(&definition, field_key) else {
            return;
        };
        let current = workshed::block(&self.workshed_order, block_id)
            .map(|block| workshed::effective_params(block, &definition))
            .and_then(|params| params.get(field_key).cloned())
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0);
        let step = field.step.unwrap_or(match field.kind {
            ParamKind::Integer => 1.0,
            _ => 0.1,
        });
        let mut next = current + step * f64::from(direction);
        if let Some(minimum) = field.minimum {
            next = next.max(minimum);
        }
        if let Some(maximum) = field.maximum {
            next = next.min(maximum);
        }
        let value = if matches!(field.kind, ParamKind::Integer) {
            Value::from(next.round() as i64)
        } else {
            Value::from(next)
        };
        self.workshed_set_param_value(block_id, field_key, value);
    }

    fn workshed_toggle_param_binding(&mut self, block_id: &str, field_key: &str) {
        let Some(definition) = self.workshed_definition(block_id) else {
            return;
        };
        let Some(field) = workshed::field(&definition, field_key) else {
            return;
        };
        let target_port = format!("param.{field_key}");
        let binding = workshed::block(&self.workshed_order, block_id)
            .map(|block| workshed::input_binding(&self.workshed_order, block, &target_port))
            .unwrap_or(InputBinding::Missing);
        let changed = if matches!(binding, InputBinding::Edge { .. }) {
            workshed::set_input_binding(
                &mut self.workshed_order,
                block_id,
                &target_port,
                InputBinding::Missing,
            )
        } else {
            workshed::auto_connect(
                &mut self.workshed_order,
                &self.workshed_capabilities,
                block_id,
                &target_port,
                workshed::value_type_for_field(&field),
            )
        };
        if changed {
            self.workshed_mark_dirty();
        } else {
            self.status_text = format!(
                "No compatible upstream value is available for {}.",
                field.label
            );
        }
    }

    fn workshed_input_draft_changed(
        &mut self,
        block_id: String,
        port_name: String,
        value: String,
        force: bool,
    ) {
        let field_key = format!("input:{port_name}");
        let draft_key = workshed_draft_key(&block_id, &field_key);
        self.workshed_field_drafts
            .insert(draft_key.clone(), value.clone());
        let Some(definition) = self.workshed_definition(&block_id) else {
            return;
        };
        let Some(port) = definition.inputs.iter().find(|port| port.name == port_name) else {
            return;
        };
        let field = workshed::literal_field(port);
        match workshed::parse_field_text(&field, &value) {
            Ok(parsed) => {
                self.workshed_field_errors.remove(&draft_key);
                let current = workshed::block(&self.workshed_order, &block_id)
                    .map(|block| workshed::input_binding(&self.workshed_order, block, &port_name));
                let next = InputBinding::Literal(parsed);
                if current.as_ref() != Some(&next)
                    && workshed::set_input_binding(
                        &mut self.workshed_order,
                        &block_id,
                        &port_name,
                        next,
                    )
                {
                    self.workshed_mark_dirty();
                }
            }
            Err(error) => {
                if force
                    || matches!(
                        field.kind,
                        ParamKind::Integer | ParamKind::Number | ParamKind::Boolean
                    )
                {
                    self.workshed_field_errors.insert(draft_key, error);
                }
            }
        }
    }

    fn workshed_commit_input_draft(&mut self, block_id: &str, port_name: &str) {
        let key = workshed_draft_key(block_id, &format!("input:{port_name}"));
        let Some(value) = self.workshed_field_drafts.get(&key).cloned() else {
            return;
        };
        self.workshed_input_draft_changed(block_id.to_string(), port_name.to_string(), value, true);
    }

    fn workshed_set_input_mode(&mut self, block_id: &str, port_name: &str, mode: &str) {
        let Some(definition) = self.workshed_definition(block_id) else {
            return;
        };
        let Some(port) = definition.inputs.iter().find(|port| port.name == port_name) else {
            return;
        };
        let changed = match mode {
            "edge" => workshed::auto_connect(
                &mut self.workshed_order,
                &self.workshed_capabilities,
                block_id,
                port_name,
                &port.type_name,
            ),
            "auto" => workshed::set_input_binding(
                &mut self.workshed_order,
                block_id,
                port_name,
                InputBinding::Auto,
            ),
            "inherited" => workshed::set_input_binding(
                &mut self.workshed_order,
                block_id,
                port_name,
                InputBinding::Inherited(port_name.to_string()),
            ),
            "literal" => {
                let field = workshed::literal_field(port);
                let value = field.default.clone().unwrap_or_else(|| match field.kind {
                    ParamKind::Boolean => Value::Bool(false),
                    ParamKind::Integer => Value::from(0),
                    ParamKind::Number => Value::from(0.0),
                    ParamKind::StringArray => Value::Array(Vec::new()),
                    ParamKind::Enum(ref choices) => choices
                        .first()
                        .map(|choice| choice.value.clone())
                        .unwrap_or(Value::Null),
                    ParamKind::IntegerOrString(ref choices) => choices
                        .first()
                        .map(|choice| choice.value.clone())
                        .unwrap_or_else(|| Value::from(1)),
                    ParamKind::String => Value::String(String::new()),
                });
                workshed::set_input_binding(
                    &mut self.workshed_order,
                    block_id,
                    port_name,
                    InputBinding::Literal(value),
                )
            }
            _ => workshed::set_input_binding(
                &mut self.workshed_order,
                block_id,
                port_name,
                InputBinding::Missing,
            ),
        };
        if changed {
            self.workshed_mark_dirty();
        } else if mode == "edge" {
            self.status_text = format!(
                "No compatible upstream {} output is available for {}.",
                port.type_name, port.label
            );
        }
    }

    fn refresh_workshed_options_for_block(
        &mut self,
        block_id: &str,
        force: bool,
    ) -> cosmic::app::Task<Message> {
        let Some(block) = workshed::block(&self.workshed_order, block_id).cloned() else {
            return cosmic::Task::none();
        };
        let definition =
            workshed::definition_for(&self.workshed_capabilities, &string_at(&block, "type_id"));
        let params = workshed::effective_params(&block, &definition);
        let fields = definition
            .fields
            .iter()
            .filter(|field| workshed::options_provider(field, &params).is_some())
            .map(|field| field.key.clone())
            .collect::<Vec<_>>();
        let tasks = fields
            .into_iter()
            .map(|field| self.load_workshed_options(block_id, &field, force))
            .collect::<Vec<_>>();
        cosmic::Task::batch(tasks)
    }

    fn load_workshed_options(
        &mut self,
        block_id: &str,
        field_key: &str,
        force: bool,
    ) -> cosmic::app::Task<Message> {
        let Some(block) = workshed::block(&self.workshed_order, block_id).cloned() else {
            return cosmic::Task::none();
        };
        let definition =
            workshed::definition_for(&self.workshed_capabilities, &string_at(&block, "type_id"));
        let Some(field) = workshed::field(&definition, field_key) else {
            return cosmic::Task::none();
        };
        let params = workshed::effective_params(&block, &definition);
        let Some(provider) = workshed::options_provider(&field, &params) else {
            return cosmic::Task::none();
        };
        let key = workshed_option_key(block_id, field_key);
        if !force
            && (self.workshed_option_loading.contains(&key)
                || self.workshed_option_results.contains_key(&key))
        {
            return cosmic::Task::none();
        }
        self.workshed_next_option_request = self.workshed_next_option_request.saturating_add(1);
        let request_id = self.workshed_next_option_request;
        self.workshed_option_requests
            .insert(key.clone(), request_id);
        self.workshed_option_loading.insert(key.clone());
        if force {
            self.workshed_option_results.remove(&key);
        }
        let query = self
            .workshed_option_queries
            .get(&key)
            .cloned()
            .unwrap_or_default();
        let context = self.workshed_option_context(&block, &definition, &field);
        let response_block = block_id.to_string();
        let response_field = field_key.to_string();
        wrap_app_task(cosmic::Task::perform(
            self.api_client()
                .workshed_options(provider, query, 24, context),
            move |result| Message::WorkshedOptionsLoaded {
                request_id,
                block_id: response_block.clone(),
                field: response_field.clone(),
                result,
            },
        ))
    }

    fn workshed_option_context(
        &self,
        block: &Value,
        definition: &BlockDefinition,
        field: &ParamField,
    ) -> Value {
        let params = workshed::effective_params(block, definition);
        let mut context = serde_json::Map::new();
        context.insert("type_id".into(), Value::String(definition.type_id.clone()));
        for key in ["source_kind", "ref", "revision", "engine", "runner"] {
            if let Some(value) = params.get(key).cloned() {
                context.insert(key.into(), value);
            }
        }
        if let Some(source) = params.get("source_kind").cloned() {
            context.insert("source".into(), source);
        }
        for (key, value) in &field.options_context {
            if key != "model_input" {
                context.insert(key.clone(), value.clone());
            }
        }

        let model_port = field
            .options_context
            .get("model_input")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                definition
                    .inputs
                    .iter()
                    .find(|port| port.type_name == "Model")
                    .map(|port| port.name.clone())
            });
        let model_binding = model_port
            .as_deref()
            .map(|port| workshed::input_binding(&self.workshed_order, block, port));
        let upstream_model = match &model_binding {
            Some(InputBinding::Edge {
                block_id: source_id,
                ..
            }) => workshed::block(&self.workshed_order, &source_id).cloned(),
            _ => None,
        };
        let mut model = serde_json::Map::new();
        if let Some(source_block) = upstream_model {
            let source_definition = workshed::definition_for(
                &self.workshed_capabilities,
                &string_at(&source_block, "type_id"),
            );
            let source_params = workshed::effective_params(&source_block, &source_definition);
            if let Some(source) = source_params.get("source_kind").cloned() {
                model.insert("source".into(), source);
            }
            for key in ["ref", "revision"] {
                if let Some(value) = source_params.get(key).cloned() {
                    model.insert(key.into(), value);
                }
            }
        } else if let Some(InputBinding::Literal(value)) = &model_binding {
            if let Some(reference) = value.as_str().filter(|value| !value.trim().is_empty()) {
                model.insert("ref".into(), Value::String(reference.to_string()));
            }
        }
        if !model.is_empty() {
            context.insert("model".into(), Value::Object(model));
        }
        Value::Object(context)
    }

    fn workshed_select_option(&mut self, block_id: &str, field_key: &str, option: &str) {
        let Some(block) = workshed::block(&self.workshed_order, block_id).cloned() else {
            return;
        };
        let definition =
            workshed::definition_for(&self.workshed_capabilities, &string_at(&block, "type_id"));
        let Some(field) = workshed::field(&definition, field_key) else {
            return;
        };
        let value = if matches!(field.kind, ParamKind::StringArray) {
            let mut items = workshed::effective_params(&block, &definition)
                .get(field_key)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let selected = Value::String(option.to_string());
            if items.contains(&selected) {
                items.retain(|item| item != &selected);
            } else {
                items.push(selected);
            }
            Value::Array(items)
        } else {
            workshed::parse_field_text(&field, option)
                .unwrap_or_else(|_| Value::String(option.to_string()))
        };
        self.workshed_set_param_value(block_id, field_key, value);
    }

    fn quantization_request(&self) -> Value {
        let worker_id = self
            .quantization_capabilities
            .engines
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some("vllm_cuda"))
            .and_then(|item| item.get("worker_id"))
            .and_then(Value::as_str)
            .or_else(|| {
                self.quantization_capabilities
                    .workers
                    .first()
                    .and_then(|item| item.get("id"))
                    .and_then(Value::as_str)
            })
            .unwrap_or("");
        let mut overrides = json!({});
        if self.quantization_advanced_open {
            overrides = json!({
                "bits": self.quantization_bits.trim().parse::<u8>().unwrap_or(4),
                "group_size": self.quantization_group_size.trim().parse::<u16>().unwrap_or(128),
                "mode": self.quantization_mode.trim(),
                "algorithm": self.quantization_algorithm.trim(),
            });
        }
        json!({
            "source": {
                "kind": self.quantization_source_kind,
                "ref": self.quantization_source.trim(),
                "revision": self.quantization_revision.trim(),
            },
            "engine": self.quantization_engine,
            "worker_id": worker_id,
            "preset_id": self.quantization_preset,
            "overrides": overrides,
            "output_name": self.quantization_output.trim(),
        })
    }

    fn quantization_preflight_ok(&self) -> bool {
        self.quantization_preflight
            .get("ok")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    fn refresh_ide_connections_task(&mut self) -> cosmic::app::Task<Message> {
        if self.ide_connections_refresh_in_flight {
            return cosmic::Task::none();
        }
        self.ide_connections_refresh_in_flight = true;
        wrap_app_task(cosmic::Task::perform(
            self.api_client().ide_connections(),
            Message::IdeConnectionsLoaded,
        ))
    }

    fn apply_manager_payload(&mut self, data: ManagerModelsPayload) {
        // A successful delete can legitimately return an empty model list;
        // retaining the old list made the deleted model appear to survive in
        // Model Settings until the next full refresh.
        if data.ok || !data.available_models.is_empty() {
            self.models = data.available_models.clone();
        }
        let mut persistent_state_changed = false;
        // Passive manager polling can finish after an optimistic switch has
        // started. Keep that stale payload from rolling the picker back and
        // causing the real switch response to be rejected as stale.
        if self.pending_model_switch.is_none()
            && !data.active_model.is_empty()
            && self.native_state.selected_model != data.active_model
        {
            self.native_state.selected_model = data.active_model.clone();
            self.native_state.settings.openclaw_enabled = false;
            self.native_state.settings.agent_runtime = "auto".into();
            self.agent_profile_state = AgentProfileState::Unconfigured;
            persistent_state_changed = true;
        }
        if !data.concurrent_models.is_empty() {
            self.selected_concurrent_models = data.concurrent_models.clone();
        }
        self.runtime_config = data.runtime_config.clone();
        self.community_job = normalize_community_job(data.community_job.clone());
        self.status_text = if data.ok {
            "Desktop manager connected.".into()
        } else if !data.last_error.is_empty() {
            data.last_error.clone()
        } else {
            "Desktop manager response received.".into()
        };
        self.manager_models = data;
        if persistent_state_changed {
            self.save_state();
        }
    }

    fn save_state(&mut self) {
        normalize_settings(&mut self.native_state.settings);
        if let Err(error) = self.store.save(&self.native_state) {
            #[cfg(debug_assertions)]
            eprintln!("failed to save native UI state: {error}");
            #[cfg(not(debug_assertions))]
            let _ = error;
        }
    }

    fn apply_settings_inputs(&mut self) {
        let max_tokens = self.max_tokens_input.trim().parse::<u32>().unwrap_or(1024);
        let temperature = self.temperature_input.trim().parse::<f32>().unwrap_or(0.7);
        self.native_state.settings.max_tokens = max_tokens.clamp(1, 16_384);
        self.native_state.settings.temperature = temperature.clamp(0.0, 2.0);
        normalize_settings(&mut self.native_state.settings);
        self.max_tokens_input = self.native_state.settings.max_tokens.to_string();
        self.temperature_input = format!("{:.2}", self.native_state.settings.temperature);
    }

    fn apply_chat_metrics(&mut self, metrics: &Value) {
        if let Some(total) = metrics.get("total_tokens").and_then(Value::as_u64) {
            self.total_tokens = self.total_tokens.saturating_add(total);
        }
        self.last_latency_ms = metrics.get("latency_ms").and_then(Value::as_f64);
        self.last_tps = metrics.get("tokens_per_second").and_then(Value::as_f64);
    }

    fn active_conversation_mut(&mut self) -> &mut StoredConversation {
        let active_id = self.native_state.active_conversation_id.clone();
        if let Some(index) = self
            .native_state
            .conversations
            .iter()
            .position(|item| item.id == active_id)
        {
            &mut self.native_state.conversations[index]
        } else {
            self.native_state.conversations.push(StoredConversation {
                id: active_id.clone(),
                title: "New Conversation".into(),
                messages: Vec::new(),
            });
            let last_index = self.native_state.conversations.len().saturating_sub(1);
            &mut self.native_state.conversations[last_index]
        }
    }

    fn active_conversation(&self) -> Option<&StoredConversation> {
        self.native_state
            .conversations
            .iter()
            .find(|item| item.id == self.native_state.active_conversation_id)
    }

    fn push_chat_message(
        &mut self,
        role: &str,
        content: &str,
        api_content: Option<Value>,
    ) -> usize {
        let conversation = self.active_conversation_mut();
        if conversation.messages.is_empty() && role == "user" {
            conversation.title = content
                .lines()
                .next()
                .unwrap_or("New Conversation")
                .chars()
                .take(48)
                .collect();
        }
        let index = conversation.messages.len();
        conversation.messages.push(StoredMessage {
            role: role.into(),
            content: content.into(),
            api_content,
        });
        index
    }

    fn set_chat_message_content(
        &mut self,
        conversation_id: &str,
        message_index: usize,
        content: String,
    ) {
        if let Some(conversation) = self
            .native_state
            .conversations
            .iter_mut()
            .find(|item| item.id == conversation_id)
        {
            if let Some(message) = conversation.messages.get_mut(message_index) {
                message.content = content;
            }
        }
    }

    fn insert_chat_message_at(
        &mut self,
        conversation_id: &str,
        message_index: usize,
        role: &str,
        content: &str,
        api_content: Option<Value>,
    ) -> usize {
        if let Some(conversation) = self
            .native_state
            .conversations
            .iter_mut()
            .find(|item| item.id == conversation_id)
        {
            let index = message_index.min(conversation.messages.len());
            conversation.messages.insert(
                index,
                StoredMessage {
                    role: role.into(),
                    content: content.into(),
                    api_content,
                },
            );
            return index;
        }
        message_index
    }

    fn mark_thinking_started(&mut self, now: Instant) {
        if let Some(stream) = &mut self.streaming_reply {
            if stream.thinking_started_at.is_none() {
                stream.thinking_started_at = Some(now);
            }
        }
    }

    fn mark_thinking_finished(&mut self, now: Instant) {
        if let Some(stream) = &mut self.streaming_reply {
            if stream.thinking_duration.is_none() {
                if let Some(started_at) = stream.thinking_started_at {
                    stream.thinking_duration = Some(now.saturating_duration_since(started_at));
                }
            }
        }
    }

    fn collapse_stream_thinking(&mut self, now: Instant) {
        let Some(stream) = self.streaming_reply.as_ref() else {
            return;
        };
        let Some(message_index) = stream.thinking_message_index else {
            return;
        };
        let conversation_id = stream.conversation_id.clone();
        let duration = stream.thinking_duration.or_else(|| {
            stream
                .thinking_started_at
                .map(|started_at| now.saturating_duration_since(started_at))
        });
        let Some(duration) = duration else {
            return;
        };
        self.set_chat_message_content(
            &conversation_id,
            message_index,
            format_thought_duration(duration),
        );
    }

    fn append_stream_thinking(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        let Some(stream) = self.streaming_reply.as_mut() else {
            return;
        };

        stream.thinking_received.push_str(text);
        let conversation_id = stream.conversation_id.clone();
        let content = stream.thinking_received.clone();
        let existing_index = stream.thinking_message_index;
        let insert_index = existing_index.is_none().then_some(stream.message_index);

        if let Some(index) = insert_index {
            stream.thinking_message_index = Some(index);
            stream.message_index += 1;
            let inserted_index =
                self.insert_chat_message_at(&conversation_id, index, "thinking", &content, None);
            if inserted_index != index {
                if let Some(stream) = self.streaming_reply.as_mut() {
                    stream.thinking_message_index = Some(inserted_index);
                    stream.message_index = inserted_index + 1;
                }
            }
        } else if let Some(index) = existing_index {
            self.set_chat_message_content(&conversation_id, index, content);
        }
    }

    fn advance_stream_reveal(&mut self, now: Instant) -> bool {
        let Some(stream) = self.streaming_reply.as_mut() else {
            return false;
        };

        let total_chars = stream.received.chars().count();
        let was_done = stream.done;
        let mut should_save = false;
        let content = if total_chars == 0 && !stream.done {
            thinking_ellipsis(stream.started_at, now)
        } else {
            if stream.visible_chars < total_chars {
                let backlog = total_chars.saturating_sub(stream.visible_chars);
                let batch = if stream.done {
                    48
                } else {
                    backlog.clamp(8, 28)
                };
                stream.visible_chars = (stream.visible_chars + batch).min(total_chars);
            }
            let mut visible = take_chars(&stream.received, stream.visible_chars);
            if !stream.done && stream.visible_chars >= total_chars {
                visible.push_str(powder_cursor(stream.started_at, now));
            }
            visible
        };

        let conversation_id = stream.conversation_id.clone();
        let message_index = stream.message_index;
        let finished = stream.done && stream.visible_chars >= total_chars;
        if finished {
            should_save = true;
        }

        self.set_chat_message_content(&conversation_id, message_index, content);
        if was_done {
            self.collapse_stream_thinking(now);
        }
        if finished || (was_done && total_chars == 0) {
            self.streaming_reply = None;
        }
        should_save
    }

    fn send_chat_task(&mut self) -> cosmic::app::Task<Message> {
        if self.streaming_reply.is_some() {
            self.status_text =
                "Wait for the current response to finish before sending again.".into();
            return cosmic::Task::none();
        }
        let text = self.message_content.text().trim().to_string();
        if text.is_empty() && self.pending_media.is_empty() {
            self.status_text = "Enter a message or attach media first.".into();
            return cosmic::Task::none();
        }
        if self.native_state.selected_model.is_empty() {
            self.status_text = "No model selected. Install or select a model first.".into();
            return cosmic::Task::none();
        }

        self.apply_settings_inputs();
        self.request_count += 1;

        let user_content = build_user_content(&text, &self.pending_media);
        let display_text = build_display_text(&text, &self.pending_media);
        let history = self
            .active_conversation()
            .map(|conversation| {
                conversation
                    .messages
                    .iter()
                    .filter(|item| item.role != "thinking")
                    .map(|item| {
                        json!({
                            "role": item.role,
                            "content": item.api_content.clone().unwrap_or_else(|| Value::String(item.content.clone())),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        self.push_chat_message("user", &display_text, Some(user_content.clone()));
        let assistant_index = self.push_chat_message("assistant", "", None);
        self.next_chat_request_id = self.next_chat_request_id.wrapping_add(1).max(1);
        let request_id = self.next_chat_request_id;
        let request_model = self.native_state.selected_model.clone();
        let stream_conversation_id = self.native_state.active_conversation_id.clone();
        self.streaming_reply = Some(StreamingReplyState {
            request_id,
            request_model: request_model.clone(),
            conversation_id: stream_conversation_id,
            message_index: assistant_index,
            thinking_message_index: None,
            thinking_received: String::new(),
            thinking_started_at: None,
            thinking_duration: None,
            received: String::new(),
            visible_chars: 0,
            done: false,
            started_at: Instant::now(),
        });
        self.advance_stream_reveal(Instant::now());
        self.message_content = text_editor::Content::new();
        self.pending_media.clear();
        self.status_text = "Model is responding...".into();
        self.save_state();

        let settings = self.native_state.settings.clone();
        let conversation_id = self.native_state.active_conversation_id.clone();
        let payload = ChatRequest {
            message: if text.is_empty() {
                "Please analyze the attached media.".into()
            } else {
                text
            },
            user_content,
            history,
            server_url: settings.server_url,
            max_tokens: settings.max_tokens,
            temperature: settings.temperature,
            system_prompt: settings.system_prompt,
            model: request_model,
            openclaw_enabled: settings.openclaw_enabled,
            cypherclaw_enabled: settings.openclaw_enabled,
            agent_runtime: if settings.openclaw_enabled {
                if settings.agent_runtime == "hermes" {
                    "hermes".into()
                } else {
                    "openclaw".into()
                }
            } else {
                "auto".into()
            },
            conversation_id,
            // Tool schemas are selected automatically by the backend from the
            // configured profile and the user's intent. The model decides
            // whether to emit a tool call, Ollama-style.
            tool_mode: None,
        };

        let client = self.api_client();
        let (sender, receiver) = iced::futures::channel::mpsc::unbounded();
        std::thread::spawn(move || {
            let mut pending_delta: Option<ChatStreamEvent> = None;
            let mut last_emit = Instant::now()
                .checked_sub(Duration::from_millis(33))
                .unwrap_or_else(Instant::now);
            let result = client.chat_stream(payload, |event| {
                let is_delta = matches!(
                    &event,
                    ChatStreamEvent::ThinkingDelta(_) | ChatStreamEvent::AnswerDelta(_)
                );
                if is_delta {
                    match (&mut pending_delta, event) {
                        (
                            Some(ChatStreamEvent::ThinkingDelta(buffer)),
                            ChatStreamEvent::ThinkingDelta(delta),
                        )
                        | (
                            Some(ChatStreamEvent::AnswerDelta(buffer)),
                            ChatStreamEvent::AnswerDelta(delta),
                        ) => buffer.push_str(&delta),
                        (pending, delta) => {
                            if let Some(queued) = pending.take() {
                                let _ = sender.unbounded_send(Message::ChatStreamUpdate {
                                    request_id,
                                    event: queued,
                                });
                            }
                            *pending = Some(delta);
                        }
                    }
                    if last_emit.elapsed() >= Duration::from_millis(33) {
                        if let Some(queued) = pending_delta.take() {
                            let _ = sender.unbounded_send(Message::ChatStreamUpdate {
                                request_id,
                                event: queued,
                            });
                        }
                        last_emit = Instant::now();
                    }
                } else {
                    if let Some(queued) = pending_delta.take() {
                        let _ = sender.unbounded_send(Message::ChatStreamUpdate {
                            request_id,
                            event: queued,
                        });
                    }
                    let _ = sender.unbounded_send(Message::ChatStreamUpdate { request_id, event });
                    last_emit = Instant::now();
                }
            });
            if let Some(queued) = pending_delta.take() {
                let _ = sender.unbounded_send(Message::ChatStreamUpdate {
                    request_id,
                    event: queued,
                });
            }
            if let Err(error) = result {
                let _ = sender.unbounded_send(Message::ChatStreamUpdate {
                    request_id,
                    event: ChatStreamEvent::Error(error),
                });
            }
        });
        wrap_app_task(cosmic::task::stream(receiver))
    }

    fn app_shell(&self) -> Element<'_, Message> {
        let content = widget::row::with_capacity(3)
            .push(self.sidebar())
            .push(self.sidebar_separator_gutter())
            .push(
                widget::container(self.page_content())
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill);

        let mut layers: Vec<Element<'_, Message>> = vec![content.into()];
        if let Some(notice) = &self.model_switch_notice {
            layers.push(self.model_switch_toast(&notice.message));
        }

        widget::container(
            iced::widget::stack(layers)
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn ide_companion_shell(&self) -> Element<'_, Message> {
        let palette = self.active_palette();
        let rail = widget::container(
            widget::column::with_capacity(3)
                .push(
                    widget::container(self.sidebar_icon(
                        Page::Chat,
                        sidebar_active_icon_color(),
                        1.0,
                    ))
                    .width(Length::Fixed(SIDEBAR_BUTTON_SIZE))
                    .height(Length::Fixed(SIDEBAR_BUTTON_SIZE))
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center),
                )
                .push(widget::space().height(Length::Fill))
                .push(
                    widget::container(self.prompt_dot(self.accent().display_color()))
                        .width(Length::Fixed(SIDEBAR_BUTTON_SIZE))
                        .height(Length::Fixed(SIDEBAR_BUTTON_SIZE))
                        .align_x(Alignment::Center)
                        .align_y(Alignment::Center),
                )
                .spacing(SIDEBAR_ITEM_GAP as u16)
                .padding([SIDEBAR_ITEM_INSET as u16, 0])
                .align_x(Alignment::Center),
        )
        .width(Length::Fixed(SIDEBAR_WIDTH))
        .height(Length::Fill)
        .style(move |_| frameless_sidebar_style(palette.text_main));

        widget::container(
            widget::row::with_capacity(3)
                .push(rail)
                .push(self.sidebar_separator_gutter())
                .push(
                    widget::container(self.ide_companion_page())
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .spacing(0)
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn model_switch_toast(&self, message: &str) -> Element<'_, Message> {
        let palette = self.active_palette();
        let toast = widget::container(
            self.glass_layer(
                widget::container(
                    widget::text(message.to_string())
                        .size(12)
                        .wrapping(text::Wrapping::Word),
                )
                .padding([7, 12])
                .height(Length::Fill)
                .into(),
                GlassVariant::Toast,
                GlassState::Rest,
            ),
        )
        .height(Length::Fixed(34.0))
        .width(Length::Fixed(360.0))
        .style(move |_| model_switch_toast_style(palette));

        widget::container(
            widget::column::with_capacity(3)
                .push(widget::space().height(Length::Fill))
                .push(
                    widget::row::with_capacity(3)
                        .push(widget::space().width(Length::Fill))
                        .push(toast)
                        .push(widget::space().width(Length::Fill))
                        .align_y(Alignment::Center),
                )
                .push(widget::space().height(Length::Fixed(96.0))),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn page_content(&self) -> Element<'_, Message> {
        match self.current_page {
            Page::Chat => self.chat_page(),
            Page::Models => self.models_page(),
            Page::Workshed => self.workshed_page(),
            Page::Server => self.server_page(),
            Page::Logs => self.logs_page(),
            Page::Community => self.community_page(),
            Page::Settings => self.settings_page(),
            Page::About => self.about_page(),
        }
    }

    fn sidebar(&self) -> Element<'_, Message> {
        let palette = self.active_palette();
        let mut column = widget::column::with_capacity(Page::ALL.len() + 1)
            .spacing(SIDEBAR_ITEM_GAP as u16)
            .padding([SIDEBAR_ITEM_INSET as u16, 0])
            .align_x(Alignment::Center);

        for page in [
            Page::Chat,
            Page::Models,
            Page::Workshed,
            Page::Server,
            Page::Logs,
            Page::Community,
            Page::About,
        ] {
            column = column.push(self.sidebar_button(page));
        }
        column = column
            .push(
                widget::container(widget::text(""))
                    .width(Length::Fixed(SIDEBAR_WIDTH))
                    .height(Length::Fill),
            )
            .push(self.sidebar_button(Page::Settings));

        let base = widget::container(column)
            .width(Length::Fixed(SIDEBAR_WIDTH))
            .height(Length::Fill)
            .style(move |_| frameless_sidebar_style(palette.text_main))
            .into();

        let mut layers = vec![base];
        if let Some(target) = self.hovered_page {
            layers.push(self.sidebar_hover_pill(target).into());
        }

        iced::widget::stack(layers)
            .width(Length::Fixed(SIDEBAR_WIDTH))
            .height(Length::Fill)
            .into()
    }

    fn sidebar_separator_gutter(&self) -> Element<'_, Message> {
        let rail = widget::row::with_capacity(3)
            .push(self.aluminum_sidebar_bar(rgba(0x6f, 0x78, 0x85, SIDEBAR_ALUMINUM_SHADE_ALPHA)))
            .push(self.aluminum_sidebar_bar(rgba(0xb9, 0xc1, 0xcb, SIDEBAR_ALUMINUM_CORE_ALPHA)))
            .push(self.aluminum_sidebar_bar(rgba(
                0xff,
                0xff,
                0xff,
                SIDEBAR_ALUMINUM_HIGHLIGHT_ALPHA,
            )))
            .spacing(0)
            .width(Length::Fixed(SIDEBAR_ALUMINUM_WIDTH))
            .height(Length::Fill);

        let offset_rail = widget::row::with_capacity(3)
            .push(widget::space().width(Length::Fixed(SIDEBAR_ALUMINUM_LEFT_OFFSET)))
            .push(rail)
            .push(widget::space().width(Length::Fill))
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill);

        widget::container(
            widget::column::with_capacity(3)
                .push(widget::space().height(Length::Fixed(SECTION_DIVIDER_INSET)))
                .push(
                    widget::container(offset_rail)
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .push(widget::space().height(Length::Fixed(SECTION_DIVIDER_INSET))),
        )
        .width(Length::Fixed(SIDEBAR_SEPARATOR_GUTTER_WIDTH))
        .height(Length::Fill)
        .into()
    }

    fn aluminum_sidebar_bar(&self, color: Color) -> Element<'static, Message> {
        widget::container(widget::text(""))
            .width(Length::Fixed(1.0))
            .height(Length::Fill)
            .style(move |_| aluminum_bar_style(color))
            .into()
    }

    fn sidebar_button(&self, page: Page) -> Element<'_, Message> {
        let palette = self.active_palette();
        let is_active = self.current_page == page;
        let is_hovered = self.hovered_page == Some(page);
        let scale = self.sidebar_icon_scale(page);
        let base_color = if is_active {
            sidebar_active_icon_color()
        } else {
            palette.icon_inactive
        };
        let color = Color::from_rgba(
            base_color.r,
            base_color.g,
            base_color.b,
            if is_hovered { 0.0 } else { 1.0 },
        );
        let inner = widget::container(self.sidebar_icon(page, color, scale))
            .width(Length::Fixed(SIDEBAR_BUTTON_SIZE))
            .height(Length::Fixed(SIDEBAR_ICON_SLOT_SIZE))
            .align_x(Alignment::Center)
            .align_y(Alignment::Center);

        let button = iced::widget::button(inner)
            .padding(0)
            .width(Length::Fixed(SIDEBAR_BUTTON_SIZE))
            .height(Length::Fixed(SIDEBAR_BUTTON_SIZE))
            .class(theme::iced::Button::Transparent)
            .on_press(Message::PagePressed(page));

        widget::mouse_area(button)
            .on_enter(Message::SidebarHover(page))
            .on_exit(Message::SidebarHoverExit(page))
            .interaction(iced::mouse::Interaction::Pointer)
            .into()
    }

    fn sidebar_hover_pill(&self, page: Page) -> Element<'_, Message> {
        let palette = self.active_palette();
        let scale = self.sidebar_icon_scale(page);
        let color = if self.current_page == page {
            sidebar_active_icon_color()
        } else {
            palette.icon_inactive
        };
        let top = self.sidebar_button_top(page);
        let pill = widget::container(self.sidebar_icon(page, color, scale))
            .width(Length::Fixed(SIDEBAR_HOVER_PILL_WIDTH))
            .height(Length::Fixed(SIDEBAR_HOVER_PILL_HEIGHT))
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(move |_| palette.pill_style(self.accent(), false));

        widget::container(
            widget::column::with_capacity(3)
                .push(widget::container(widget::text("")).height(Length::Fixed(top)))
                .push(
                    widget::container(pill)
                        .width(Length::Fixed(SIDEBAR_WIDTH))
                        .height(Length::Fixed(SIDEBAR_BUTTON_SIZE))
                        .align_x(Alignment::Center)
                        .align_y(Alignment::Center),
                )
                .push(widget::container(widget::text("")).height(Length::Fill)),
        )
        .width(Length::Fixed(SIDEBAR_WIDTH))
        .height(Length::Fill)
        .into()
    }

    fn sidebar_button_top(&self, page: Page) -> f32 {
        let index = match page {
            Page::Chat => 0.0,
            Page::Models => 1.0,
            Page::Workshed => 2.0,
            Page::Server => 3.0,
            Page::Logs => 4.0,
            Page::Community => 5.0,
            Page::About => 6.0,
            Page::Settings => 7.0,
        };
        SIDEBAR_ITEM_INSET + index * (SIDEBAR_BUTTON_SIZE + SIDEBAR_ITEM_GAP)
    }

    fn sidebar_icon_scale(&self, page: Page) -> f32 {
        if let Some(bounce) = self.sidebar_bounce {
            if bounce.page == page {
                return bounce_scale(bounce.progress(Instant::now()));
            }
        }
        if self.hovered_page == Some(page) {
            SIDEBAR_ICON_HOVER_SCALE
        } else {
            1.0
        }
    }

    fn chat_page(&self) -> Element<'_, Message> {
        self.chat_page_layout(true)
    }

    fn ide_companion_page(&self) -> Element<'_, Message> {
        self.chat_page_layout(false)
    }

    fn chat_page_layout(&self, include_dashboard: bool) -> Element<'_, Message> {
        let palette = self.active_palette();
        let top_identity: Element<'_, Message> = if include_dashboard {
            self.small_label(&self.server_badge)
        } else {
            widget::row::with_capacity(2)
                .push(self.small_label("IDE Companion"))
                .push(self.soft_text(&self.server_badge))
                .spacing(8)
                .align_y(Alignment::Center)
                .into()
        };
        let top = widget::container(
            self.glass_layer(
                widget::container(
                    widget::row::with_capacity(2)
                        .push(widget::container(top_identity).width(Length::Fill))
                        .push(self.chat_model_picker())
                        .spacing(8)
                        .align_y(Alignment::Center),
                )
                .height(Length::Fill)
                .padding([0, 10])
                .into(),
                GlassVariant::Control,
                GlassState::Rest,
            ),
        )
        .height(Length::Fixed(38.0))
        .style(move |_| glass_container_style(palette, GlassVariant::Control, GlassState::Rest));

        let palette = self.active_palette();
        let no_model_banner = if self.native_state.selected_model.trim().is_empty() {
            widget::container(
                self.label_text("To start using Token Workshed, please install a model."),
            )
            .height(Length::Fixed(36.0))
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(move |_| glass_container_style(palette, GlassVariant::Toast, GlassState::Rest))
        } else {
            widget::container(widget::text(""))
                .height(Length::Fixed(0.0))
                .style(|_| container::Style::default())
        };

        let messages = self
            .active_conversation()
            .map(|conversation| conversation.messages.as_slice())
            .unwrap_or(&[]);
        let feed_column = if messages.is_empty() {
            widget::column::with_capacity(1).push(
                widget::container(self.soft_text(
                    "Start a conversation with token-workshed.\nYou can tune settings in the Settings page.",
                ))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center),
            )
        } else {
            messages.iter().fold(
                widget::column::with_capacity(messages.len()),
                |column, msg| column.push(self.chat_bubble(&msg.role, &msg.content)),
            )
        };

        let feed_area = widget::container(
            widget::scrollable(feed_column.spacing(8))
                .height(Length::Fill)
                .class(theme::iced::Scrollable::Transient),
        )
        .padding(10)
        .height(Length::Fill);

        let media_line = if self.pending_media.is_empty() {
            None
        } else {
            Some(
                self.pending_media
                    .iter()
                    .map(|item| format!("{} ({})", item.name, format_bytes(item.size)))
                    .collect::<Vec<_>>()
                    .join(" | "),
            )
        };

        let message_editor = widget::text_editor(&self.message_content)
            .placeholder("Type your message... (Enter to send, Shift+Enter newline)")
            .on_action(Message::MessageInputAction)
            .wrapping(text::Wrapping::WordOrGlyph)
            .padding([8, 98, 56, 10])
            .size(13)
            .height(Length::Fixed(84.0))
            .class(theme::iced::TextEditor::Custom(Box::new(
                transparent_text_editor_style(self.accent().display_color()),
            )))
            .key_binding(|key_press| match key_press.key.as_ref() {
                keyboard::Key::Named(keyboard::key::Named::Enter)
                    if !key_press.modifiers.shift() =>
                {
                    Some(text_editor::Binding::Custom(Message::SendPressed))
                }
                _ => text_editor::Binding::from_key_press(key_press),
            });

        let composer_leading: Element<'_, Message> = if include_dashboard {
            self.agent_runtime_or_configure_group()
        } else {
            self.soft_pill("IDE context")
        };

        let composer_controls = widget::row::with_capacity(4)
            .push(composer_leading)
            .push(widget::space().width(Length::Fill))
            .push(self.media_fab(Message::AttachMediaPressed))
            .push(self.send_fab(Message::SendPressed))
            .spacing(8)
            .align_y(Alignment::Center)
            .width(Length::Fill);

        let composer_stack = iced::widget::stack(vec![
            widget::container(message_editor)
                .height(Length::Fixed(84.0))
                .width(Length::Fill)
                .into(),
            widget::container(composer_controls)
                .height(Length::Fixed(84.0))
                .width(Length::Fill)
                .padding([0, 10, 10, 10])
                .align_y(Alignment::End)
                .into(),
        ])
        .width(Length::Fill)
        .height(Length::Fixed(84.0));

        let palette = self.active_palette();
        let composer_frame = widget::container(
            self.glass_layer(
                widget::container(composer_stack)
                    .padding(8)
                    .height(Length::Fill)
                    .width(Length::Fill)
                    .into(),
                GlassVariant::Input,
                GlassState::Focused,
            ),
        )
        .height(Length::Fixed(100.0))
        .width(Length::Fill)
        .style(move |_| frosted_composer_style(palette));

        let mut composer_content = widget::column::with_capacity(2)
            .push(composer_frame)
            .spacing(6);
        if let Some(media_line) = &media_line {
            composer_content = composer_content.push(self.soft_text(media_line));
        }
        let composer = widget::container(composer_content)
            .padding([0, 8, 8, 8])
            .height(Length::Shrink);
        let palette = self.active_palette();
        let chat_surface_content = widget::column::with_capacity(2)
            .push(feed_area)
            .push(composer)
            .spacing(0);
        let chat_surface = widget::container(self.glass_layer(
            chat_surface_content.into(),
            GlassVariant::Panel,
            GlassState::Rest,
        ))
        .height(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let mut center_column = widget::column::with_capacity(4).push(top);
        if self.native_state.selected_model.trim().is_empty() {
            center_column = center_column.push(no_model_banner);
        }
        let center = center_column
            .push(chat_surface)
            .spacing(8)
            .height(Length::Fill)
            .width(Length::FillPortion(3));
        let center: Element<'_, Message> = if self.chat_model_menu_open {
            iced::widget::stack(vec![center.into(), self.chat_model_dropdown_overlay()])
                .width(Length::FillPortion(3))
                .height(Length::Fill)
                .into()
        } else {
            center.into()
        };
        let content: Element<'_, Message> = if include_dashboard {
            let palette = self.active_palette();
            let right = widget::container(
                self.glass_layer(
                    widget::container(
                        widget::column::with_capacity(3)
                            .push(self.dashboard_panel())
                            .push(self.settings_divider_fill())
                            .push(self.history_panel())
                            .spacing(10)
                            .height(Length::Fill),
                    )
                    .padding(10)
                    .height(Length::Fill)
                    .into(),
                    GlassVariant::Panel,
                    GlassState::Rest,
                ),
            )
            .width(Length::Fixed(340.0))
            .height(Length::Fill)
            .style(move |_| flat_frame_host_style(palette));

            widget::row::with_capacity(3)
                .push(center)
                .push(self.caligo_vertical_rule())
                .push(right)
                .spacing(6)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            center
        };

        self.page_shell(
            "Chat",
            "Send prompts, switch models, and watch local serving metrics.",
            content.into(),
        )
    }

    fn workshed_model_name_control(&self) -> Element<'_, Message> {
        let input = widget::text_input(
            "input the name of the new model",
            &self.workshed_model_name_input,
        )
        .padding([8, 10])
        .size(13)
        .font(cosmic::font::default())
        .style(app_input_style(self.accent().display_color()))
        .on_input(Message::WorkshedModelNameChanged)
        .on_focus(Message::InputFocusChanged(
            InputField::WorkshedModelName,
            true,
        ))
        .on_unfocus(Message::InputFocusChanged(
            InputField::WorkshedModelName,
            false,
        ))
        .width(Length::Fill);
        let palette = self.active_palette();
        let focused = self.focused_input == Some(InputField::WorkshedModelName);
        let state = if focused {
            GlassState::Focused
        } else {
            GlassState::Rest
        };
        widget::container(
            self.glass_layer(
                widget::container(input)
                    .height(Length::Fill)
                    .align_y(Alignment::Center)
                    .into(),
                GlassVariant::Input,
                state,
            ),
        )
        .height(Length::Fixed(38.0))
        .align_y(Alignment::Center)
        .style(move |_| input_shell_style(palette, focused))
        .into()
    }

    fn workshed_page(&self) -> Element<'_, Message> {
        workshed_view::render(self)
    }

    fn workshed_library_panel(&self) -> Element<'_, Message> {
        let categories = [
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
        let mut category_column = widget::column::with_capacity(categories.len()).spacing(4);
        for (id, label) in categories {
            category_column = category_column.push(self.workshed_category_button(
                id,
                label,
                self.workshed_category == id,
            ));
        }
        let query = self.workshed_search.trim().to_ascii_lowercase();
        let items = workshed_catalog_items(&self.workshed_capabilities, &self.workshed_category)
            .into_iter()
            .filter(|item| query.is_empty() || item.1.to_ascii_lowercase().contains(&query))
            .collect::<Vec<_>>();
        let mut palette = widget::column::with_capacity(items.len().max(1)).spacing(5);
        if items.is_empty() {
            palette = palette.push(self.workshed_help_text("No blocks match this search."));
        } else {
            for (type_id, label, available, reason) in items {
                let add = if available {
                    self.workshed_catalog_add_button(
                        format!("+ {label}"),
                        Message::WorkshedBlockAddPressed(type_id),
                    )
                } else {
                    widget::container(self.workshed_help_text(format!("{label} · unavailable")))
                        .padding([7, 6])
                        .style(|_| workshed_unavailable_style())
                        .into()
                };
                palette = palette.push(add);
                if !available && !reason.is_empty() {
                    palette = palette.push(self.workshed_help_text(reason));
                }
            }
        }
        let body = widget::column::with_capacity(6)
            .push(self.workshed_library_search())
            .push(category_column)
            .push(self.caligo_horizontal_rule(false))
            .push(widget::scrollable(palette).height(Length::Fill))
            .push(self.secondary_button("New work order", Message::WorkshedNewPressed))
            .spacing(8);
        widget::container(body)
            .padding(10)
            .width(Length::Fixed(210.0))
            .height(Length::Fill)
            .style(|_| workshed_panel_style())
            .into()
    }

    fn workshed_catalog_add_button(&self, label: String, message: Message) -> Element<'_, Message> {
        iced::widget::button(
            widget::container(
                widget::text(label)
                    .size(12)
                    .class(workshed_color(""))
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph)
                    .center(),
            )
            .width(Length::Fill)
            .height(Length::Shrink),
        )
        .padding([6, 7])
        .width(Length::Fill)
        .height(Length::Shrink)
        .class(iced_button_class(secondary_button_style(
            self.active_palette(),
            self.accent(),
            false,
        )))
        .on_press(message)
        .into()
    }

    fn workshed_library_search(&self) -> Element<'_, Message> {
        // libcosmic's `label` is a visible floating label. The library search
        // deliberately uses only one visible prompt: its placeholder.
        let input = widget::text_input("Search blocks", &self.workshed_search)
            .padding([8, 10])
            .size(13)
            .style(app_input_style(self.accent().display_color()))
            .on_input(Message::WorkshedSearchChanged)
            .on_focus(Message::InputFocusChanged(InputField::WorkshedSearch, true))
            .on_unfocus(Message::InputFocusChanged(
                InputField::WorkshedSearch,
                false,
            ))
            .width(Length::Fill);
        let palette = self.active_palette();
        let focused = self.focused_input == Some(InputField::WorkshedSearch);
        widget::container(input)
            .height(Length::Fixed(38.0))
            .align_y(Alignment::Center)
            .style(move |_| input_shell_style(palette, focused))
            .into()
    }

    fn workshed_category_button(
        &self,
        category: &str,
        label: &str,
        selected: bool,
    ) -> Element<'_, Message> {
        let stripe_category = category.to_string();
        let content = widget::row::with_capacity(2)
            .push(
                widget::container(widget::space())
                    .width(Length::Fixed(WORKSHED_CATEGORY_STRIPE_WIDTH))
                    .height(Length::Fixed(18.0))
                    .style(move |_| workshed_category_stripe_style(&stripe_category, selected)),
            )
            .push(widget::text(label.to_string()).size(12))
            .spacing(8)
            .align_y(Alignment::Center);
        iced::widget::button(content)
            .padding([0, 8])
            .width(Length::Fill)
            .height(Length::Fixed(32.0))
            .class(iced_button_class(workshed_category_button_style(selected)))
            .name(format!(
                "{} category{}",
                label,
                if selected { ", selected" } else { "" }
            ))
            .on_press(Message::WorkshedCategoryPressed(category.into()))
            .into()
    }

    fn workshed_block_card(&self, block: &Value) -> Element<'_, Message> {
        let type_id = string_at(block, "type_id");
        let definition = workshed::definition_for(&self.workshed_capabilities, &type_id);
        let label = definition.label.clone();
        let block_id = string_at(block, "id");
        let selected = self.workshed_selected_block == block_id;
        let lane = string_at(block, "lane");
        let block_issues = workshed::issues(&self.workshed_preflight)
            .into_iter()
            .filter(|issue| issue.block_id == block_id)
            .collect::<Vec<_>>();
        let summary = workshed::summary_parts(block, &definition, 3);
        let summary_text = if summary.is_empty() {
            "Configure in Inspector".into()
        } else {
            summary.join(" · ")
        };
        let issue_text = if block_issues.is_empty() {
            String::new()
        } else {
            format!(
                "{} issue{}",
                block_issues.len(),
                if block_issues.len() == 1 { "" } else { "s" }
            )
        };
        let input_ports = workshed_card_ports(
            &self.workshed_order,
            &block_id,
            &definition,
            WorkshedPortDirection::Input,
        );
        let output_ports = workshed_card_ports(
            &self.workshed_order,
            &block_id,
            &definition,
            WorkshedPortDirection::Output,
        );
        let connected_inputs = input_ports
            .iter()
            .filter(|port| port.edge_count > 0)
            .count();
        let outgoing_edges = output_ports
            .iter()
            .map(|port| port.edge_count)
            .sum::<usize>();
        let input_count = input_ports.len();
        let connections = widget::row::with_capacity(3)
            .push(self.workshed_card_port_group(
                format!("IN {connected_inputs}/{input_count}"),
                input_ports,
                true,
                &definition.category,
            ))
            .push(widget::space().width(Length::Fill))
            .push(self.workshed_card_port_group(
                workshed_edge_summary(outgoing_edges),
                output_ports,
                false,
                &definition.category,
            ))
            .spacing(6)
            .align_y(Alignment::Start);
        let content = widget::column::with_capacity(5)
            .push(
                widget::row::with_capacity(4)
                    .push(
                        widget::icon::from_name(workshed_category_icon_name(&definition.category))
                            .size(14)
                            .icon(),
                    )
                    .push(
                        widget::text(label.clone())
                            .size(13)
                            .width(Length::FillPortion(3))
                            .wrapping(text::Wrapping::WordOrGlyph),
                    )
                    .push(widget::space().width(Length::Fill))
                    .push(
                        widget::container(self.workshed_help_text_sized(issue_text.clone(), 10))
                            .width(Length::FillPortion(2)),
                    )
                    .align_y(Alignment::Center),
            )
            .push(self.workshed_help_text(summary_text.clone()))
            .push(connections)
            .push(self.workshed_help_text_sized(lane.clone(), 10))
            .spacing(5);
        let a11y_name = format!(
            "{}, {} stage, {}, {connected_inputs} of {input_count} inputs connected, {outgoing_edges} outgoing edges{}",
            label,
            lane,
            summary_text,
            if issue_text.is_empty() {
                String::new()
            } else {
                format!(", {issue_text}")
            }
        );
        let select = iced::widget::button(content)
            .padding([9, 10])
            .width(Length::Fill)
            .height(Length::Shrink)
            .class(iced_button_class(workshed_block_button_style(
                self.active_palette(),
                selected,
            )))
            .name(a11y_name)
            .on_press(Message::WorkshedBlockSelectPressed(block_id.clone()));
        let effective = workshed::effective_params(block, &definition);
        let quick = workshed_card_allows_quick_control(&type_id)
            .then(|| {
                definition
                    .fields
                    .iter()
                    .find(|field| field.quick)
                    .and_then(|field| {
                        self.workshed_quick_control(&block_id, field, effective.get(&field.key))
                    })
            })
            .flatten();
        let mut card = widget::column::with_capacity(2).push(select).spacing(4);
        if let Some(quick) = quick {
            card = card.push(quick);
        }
        widget::container(card)
            .padding(0)
            .width(Length::Fixed(224.0))
            .style({
                let category = definition.category;
                move |_| workshed_block_style(&category, selected)
            })
            .into()
    }

    fn workshed_card_port_group(
        &self,
        title: String,
        ports: Vec<WorkshedCardPort>,
        input: bool,
        category: &str,
    ) -> Element<'_, Message> {
        let mut content = widget::column::with_capacity(ports.len() + 1)
            .push(widget::text(title).size(9).class(workshed_help_color()))
            .spacing(3);
        for port in ports {
            let connected = port.edge_count > 0;
            let category_for_dot = category.to_string();
            let category_for_line = category.to_string();
            let dot = widget::container(widget::space())
                .width(Length::Fixed(7.0))
                .height(Length::Fixed(7.0))
                .style(move |_| workshed_port_dot_style(&category_for_dot, connected));
            let line = widget::container(widget::space())
                .width(Length::Fixed(9.0))
                .height(Length::Fixed(1.0))
                .style(move |_| workshed_port_line_style(&category_for_line, connected));
            let connector: Element<'_, Message> = if input {
                widget::row::with_capacity(2)
                    .push(dot)
                    .push(line)
                    .spacing(0)
                    .align_y(Alignment::Center)
                    .into()
            } else {
                widget::row::with_capacity(2)
                    .push(line)
                    .push(dot)
                    .spacing(0)
                    .align_y(Alignment::Center)
                    .into()
            };
            let port_text = workshed_port_display(&port);
            let row = if input {
                widget::row::with_capacity(2)
                    .push(connector)
                    .push(self.workshed_card_port_text(port_text, false))
            } else {
                widget::row::with_capacity(2)
                    .push(self.workshed_card_port_text(port_text, true))
                    .push(connector)
            };
            content = content.push(row.spacing(3).align_y(Alignment::Center));
        }
        widget::container(content)
            .width(Length::Shrink)
            .height(Length::Shrink)
            .into()
    }

    fn workshed_card_port_text(&self, text: String, align_right: bool) -> Element<'_, Message> {
        let label = widget::text(text)
            .size(9)
            .class(workshed_help_color())
            .width(Length::Fixed(72.0))
            .wrapping(text::Wrapping::WordOrGlyph);
        let label = if align_right {
            label.align_x(iced::alignment::Horizontal::Right)
        } else {
            label
        };
        label.into()
    }

    fn workshed_inspector_heading(&self) -> Element<'_, Message> {
        let close_icon = widget::icon::from_name("window-close-symbolic")
            .size(13)
            .icon()
            .class(theme::Svg::custom(|_| cosmic::iced::widget::svg::Style {
                color: Some(Color::from_rgb8(0x11, 0x11, 0x11)),
            }));
        let close = iced::widget::button(close_icon)
            .padding(6)
            .width(Length::Fixed(28.0))
            .height(Length::Fixed(28.0))
            .class(iced_button_class(|_, status| {
                let mut style = button::Style::default();
                style.text_color = Color::from_rgb8(0x11, 0x11, 0x11);
                style.border.radius = CONTENT_RADIUS.into();
                if matches!(status, button::Status::Hovered | button::Status::Pressed) {
                    style.background =
                        Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.04)));
                }
                style
            }))
            .name("Close block settings")
            .on_press(Message::WorkshedPanelToggled("inspector".into()));
        widget::row::with_capacity(3)
            .push(self.settings_subtitle("Block settings"))
            .push(widget::space().width(Length::Fill))
            .push(close)
            .align_y(Alignment::Center)
            .width(Length::Fill)
            .into()
    }

    fn workshed_inspector_panel(&self) -> Element<'_, Message> {
        let selected = self
            .workshed_order
            .get("blocks")
            .and_then(Value::as_array)
            .and_then(|blocks| {
                blocks
                    .iter()
                    .find(|block| string_at(block, "id") == self.workshed_selected_block)
                    .cloned()
            });
        let Some(selected) = selected else {
            return widget::container(
                widget::column::with_capacity(2)
                    .push(self.workshed_inspector_heading())
                    .push(self.workshed_help_text("Select a block to configure it."))
                    .spacing(16),
            )
            .padding([0, 0, 0, 16])
            .width(Length::Fixed(240.0))
            .height(Length::Fill)
            .style(|_| container::Style::default().color(workshed_color("")))
            .into();
        };
        let type_id = string_at(&selected, "type_id");
        let definition = workshed::definition_for(&self.workshed_capabilities, &type_id);
        let block_id = string_at(&selected, "id");
        let is_root = selected
            .get("locked")
            .and_then(Value::as_bool)
            .unwrap_or(type_id == "model_development");
        let effective =
            Self::workshed_inspector_values(&self.workshed_capabilities, &selected, &definition);
        let issues = workshed::issues(&self.workshed_preflight)
            .into_iter()
            .filter(|issue| issue.block_id.is_empty() || issue.block_id == block_id)
            .collect::<Vec<_>>();
        let input_names = definition
            .inputs
            .iter()
            .map(|port| port.name.as_str())
            .collect::<BTreeSet<_>>();
        let basic_fields = if type_id == "quantize_mlx" {
            ["preset_id", "preset", "bits", "group_size"]
                .into_iter()
                .filter_map(|key| definition.fields.iter().find(|field| field.key == key))
                .collect::<Vec<_>>()
        } else {
            definition
                .fields
                .iter()
                .filter(|field| !field.advanced && !input_names.contains(field.key.as_str()))
                .collect::<Vec<_>>()
        };
        let basic_field_keys = basic_fields
            .iter()
            .map(|field| field.key.as_str())
            .collect::<BTreeSet<_>>();
        let description = if type_id == "quantize_mlx" {
            "Configure quantization parameters for this block.".to_string()
        } else {
            format!("Configure {} for this block.", definition.label)
        };
        let mut body = widget::column::with_capacity(20)
            .push(self.workshed_help_text(description))
            .push(widget::space().height(Length::Fixed(2.0)))
            .spacing(14);
        if basic_fields.is_empty() {
            body = body.push(self.workshed_help_text("This block has no basic parameters."));
        } else {
            for field in &basic_fields {
                body = body.push(self.workshed_inspector_simple_field(
                    &selected,
                    &definition,
                    field,
                    effective.get(&field.key),
                    &issues,
                ));
            }
        }
        let advanced_open = self.workshed_advanced_blocks.contains(&block_id);
        let disclosure = widget::icon::from_name(if advanced_open {
            "pan-down-symbolic"
        } else {
            "pan-end-symbolic"
        })
        .size(12)
        .icon()
        .class(theme::Svg::custom(|_| cosmic::iced::widget::svg::Style {
            color: Some(Color::from_rgb8(0x11, 0x11, 0x11)),
        }));
        let advanced = iced::widget::button(
            widget::row::with_capacity(2)
                .push(disclosure)
                .push(widget::text("Advanced").size(13))
                .spacing(8)
                .align_y(Alignment::Center),
        )
        .padding([8, 0])
        .width(Length::Fill)
        .class(iced_button_class(|_, status| {
            let mut style = button::Style::default();
            style.text_color = Color::from_rgb8(0x6b, 0x6b, 0x6b);
            style.border.radius = CONTENT_RADIUS.into();
            if matches!(status, button::Status::Hovered | button::Status::Pressed) {
                style.background = Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.025)));
            }
            style
        }))
        .name(if advanced_open {
            "Collapse advanced block settings"
        } else {
            "Expand advanced block settings"
        })
        .on_press(Message::WorkshedAdvancedToggled(block_id.clone()));
        body = body.push(self.caligo_horizontal_rule(false)).push(advanced);
        if advanced_open {
            body = body
                .push(self.workshed_dashboard_row("Block", &definition.label))
                .push(self.workshed_dashboard_row("Type", &type_id))
                .push(self.settings_subtitle("Inputs"));
            if definition.inputs.is_empty() {
                body = body.push(self.workshed_help_text("This block has no typed inputs."));
            } else {
                for port in &definition.inputs {
                    body = body.push(self.workshed_input_control(&selected, port, &issues));
                }
            }
            // The quiet summary above never replaces the full schema controls:
            // every override, Reset, typed binding, and option provider remains
            // available in the explicitly expanded advanced inspector.
            body = body.push(self.settings_subtitle("Parameters"));
            for field in definition.fields.iter().filter(|field| {
                !input_names.contains(field.key.as_str())
                    && !basic_field_keys.contains(field.key.as_str())
            }) {
                body =
                    body.push(self.workshed_param_control(&selected, &definition, field, &issues));
            }
            // Expose reset and binding affordances for the simple fields without
            // rendering a second input with the same native widget identity.
            for field in &basic_fields {
                body = body.push(self.workshed_inspector_field_actions(&selected, field));
            }
            body = body.push(self.settings_subtitle("Effective values"));
            for field in &definition.fields {
                let raw_value = workshed::effective_params(&selected, &definition)
                    .get(&field.key)
                    .cloned();
                let source = if raw_value.as_ref().is_none_or(Value::is_null)
                    && effective
                        .get(&field.key)
                        .is_some_and(|value| !value.is_null())
                {
                    "preset".into()
                } else {
                    selected
                        .get("param_origins")
                        .and_then(|origins| origins.get(&field.key))
                        .map(|origin| string_at(origin, "kind"))
                        .filter(|kind| !kind.is_empty())
                        .unwrap_or_else(|| {
                            if selected
                                .get("params")
                                .and_then(|params| params.get(&field.key))
                                .is_some_and(|value| !value.is_null())
                            {
                                "override".into()
                            } else {
                                "default".into()
                            }
                        })
                };
                body = body.push(self.workshed_dashboard_row(
                    &field.label,
                    format!(
                        "{} · {source}",
                        workshed::field_text(effective.get(&field.key), field)
                    ),
                ));
            }
            body = body.push(self.settings_subtitle("Issues"));
            if issues.is_empty() {
                body = body.push(self.workshed_help_text("No issues reported for this block."));
            } else {
                for issue in &issues {
                    body = body.push(self.workshed_issue_button(issue));
                }
            }
            body = body.push(self.settings_subtitle("Arrange block"));
            if is_root {
                body = body.push(self.workshed_help_text(
                    "The root block stays at the start of every work order and cannot be deleted.",
                ));
            } else {
                // Stack these actions in the narrow drawer rather than forcing
                // three fixed-label buttons onto one overflowing row.
                body = body
                    .push(self.small_button("Move earlier", Message::WorkshedBlockMovePressed(-1)))
                    .push(self.small_button("Move later", Message::WorkshedBlockMovePressed(1)))
                    .push(self.small_button("Duplicate", Message::WorkshedBlockDuplicatePressed))
                    .push(self.danger_button(
                        "Delete block",
                        Message::WorkshedBlockDeletePressed(block_id.clone()),
                    ));
            }
        }
        widget::container(
            widget::column::with_capacity(2)
                .push(self.workshed_inspector_heading())
                .push(
                    widget::scrollable(
                        widget::container(body)
                            .padding([0, 10, 0, 0])
                            .width(Length::Fill),
                    )
                    .width(Length::Fill)
                    .height(Length::Fill),
                )
                .spacing(14),
        )
        .padding([0, 0, 0, 16])
        .width(Length::Fixed(240.0))
        .height(Length::Fill)
        .style(|_| container::Style::default().color(workshed_color("")))
        .into()
    }

    fn workshed_inspector_values(
        capabilities: &Value,
        block: &Value,
        definition: &BlockDefinition,
    ) -> serde_json::Map<String, Value> {
        let mut values = workshed::effective_params(block, definition);
        let preset_id = values
            .get("preset_id")
            .or_else(|| values.get("preset"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if let Some(params) = capabilities
            .get("blocks")
            .and_then(Value::as_array)
            .and_then(|blocks| {
                blocks
                    .iter()
                    .find(|block| string_at(block, "type_id") == definition.type_id)
            })
            .and_then(|block| block.get("presets"))
            .and_then(Value::as_array)
            .and_then(|presets| {
                presets
                    .iter()
                    .find(|preset| string_at(preset, "id") == preset_id)
            })
            .and_then(|preset| preset.get("params"))
            .and_then(Value::as_object)
        {
            for (key, value) in params {
                if values.get(key).is_none_or(Value::is_null) {
                    values.insert(key.clone(), value.clone());
                }
            }
        }
        values
    }

    fn workshed_inspector_field_actions(
        &self,
        block: &Value,
        field: &ParamField,
    ) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let mut body = widget::column::with_capacity(3)
            .push(self.workshed_label_text(&field.label))
            .spacing(5);
        if !field.description.is_empty() {
            body = body.push(self.workshed_help_text(&field.description));
        }
        let mut actions = widget::column::with_capacity(2).spacing(4);
        if block
            .get("params")
            .and_then(|params| params.get(&field.key))
            .is_some()
        {
            actions = actions.push(self.small_button(
                "Reset to default",
                Message::WorkshedParamReset {
                    block_id: block_id.clone(),
                    field: field.key.clone(),
                },
            ));
        }
        if field.bindable {
            let connected = matches!(
                workshed::input_binding(
                    &self.workshed_order,
                    block,
                    &format!("param.{}", field.key)
                ),
                InputBinding::Edge { .. }
            );
            actions = actions.push(self.small_button(
                if connected {
                    "Use value"
                } else {
                    "Connect upstream value"
                },
                Message::WorkshedParamBindingPressed {
                    block_id: block_id.clone(),
                    field: field.key.clone(),
                },
            ));
        }
        body.push(actions).width(Length::Fill).into()
    }

    fn workshed_inspector_simple_field(
        &self,
        block: &Value,
        definition: &BlockDefinition,
        field: &ParamField,
        value: Option<&Value>,
        issues: &[WorkshedIssue],
    ) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        if matches!(
            workshed::input_binding(&self.workshed_order, block, &format!("param.{}", field.key)),
            InputBinding::Edge { .. }
        ) {
            return self.workshed_param_control(block, definition, field, issues);
        }
        let label = if matches!(field.key.as_str(), "preset_id" | "preset") {
            "Preset"
        } else {
            &field.label
        };
        let control: Element<'_, Message> = match &field.kind {
            ParamKind::Enum(choices) if !choices.is_empty() => {
                let options = choices
                    .iter()
                    .map(|choice| choice.label.clone())
                    .collect::<Vec<_>>();
                let selected = choices
                    .iter()
                    .find(|choice| Some(&choice.value) == value)
                    .map(|choice| choice.label.clone());
                let choices = choices.clone();
                let key = field.key.clone();
                let picker = iced::widget::pick_list(options, selected, move |label: String| {
                    let value = choices
                        .iter()
                        .find(|choice| choice.label == label)
                        .map(|choice| choice.value.clone())
                        .unwrap_or(Value::Null);
                    Message::WorkshedParamSet {
                        block_id: block_id.clone(),
                        field: key.clone(),
                        value,
                    }
                })
                .placeholder("Select")
                .padding([7, 9])
                .text_size(13)
                .text_wrap(text::Wrapping::WordOrGlyph)
                .width(Length::Fill);
                widget::container(picker)
                    .height(Length::Fixed(34.0))
                    .align_y(Alignment::Center)
                    .width(Length::Fill)
                    .style(|_| {
                        container::Style::default()
                            .background(Background::Color(Color::WHITE))
                            .color(workshed_color(""))
                            .border(Border {
                                width: 1.0,
                                radius: CONTENT_RADIUS.into(),
                                color: Color::from_rgb8(0xdc, 0xe1, 0xe9),
                            })
                    })
                    .into()
            }
            ParamKind::Boolean => {
                let key = field.key.clone();
                widget::checkbox(value.and_then(Value::as_bool).unwrap_or(false))
                    .label(if value.and_then(Value::as_bool).unwrap_or(false) {
                        "Enabled"
                    } else {
                        "Disabled"
                    })
                    .on_toggle(move |enabled| Message::WorkshedParamSet {
                        block_id: block_id.clone(),
                        field: key.clone(),
                        value: Value::Bool(enabled),
                    })
                    .into()
            }
            _ => self.workshed_param_text_input(&block_id, field, value),
        };
        let mut column = widget::column::with_capacity(4)
            .push(
                widget::text(label.to_string())
                    .size(13)
                    .class(workshed_color("")),
            )
            .push(control)
            .spacing(6);
        if let Some(error) = self
            .workshed_field_errors
            .get(&workshed_draft_key(&string_at(block, "id"), &field.key))
        {
            column = column.push(self.workshed_help_text(format!("Error · {error}")));
        }
        for issue in issues.iter().filter(|issue| issue.field == field.key) {
            column = column.push(self.workshed_issue_button(issue));
        }
        column.width(Length::Fill).into()
    }

    fn workshed_quick_control(
        &self,
        block_id: &str,
        field: &ParamField,
        value: Option<&Value>,
    ) -> Option<Element<'_, Message>> {
        let target_port = format!("param.{}", field.key);
        let connected = workshed::block(&self.workshed_order, block_id)
            .map(|block| workshed::input_binding(&self.workshed_order, block, &target_port))
            .is_some_and(|binding| matches!(binding, InputBinding::Edge { .. }));
        if connected {
            return Some(
                widget::container(self.workshed_help_text(format!("{} · connected", field.label)))
                    .padding([4, 7])
                    .width(Length::Fill)
                    .into(),
            );
        }
        match &field.kind {
            ParamKind::Enum(choices) if !choices.is_empty() => {
                let selected = value.cloned().unwrap_or(Value::Null);
                let mut row = widget::row::with_capacity(choices.len().min(3)).spacing(3);
                for choice in choices.iter().take(3) {
                    let is_selected = choice.value == selected;
                    let message = Message::WorkshedParamSet {
                        block_id: block_id.to_string(),
                        field: field.key.clone(),
                        value: choice.value.clone(),
                    };
                    row = row.push(
                        iced::widget::button(centered_text(choice.label.clone(), 10))
                            .padding([3, 5])
                            .height(Length::Fixed(25.0))
                            .width(Length::Fill)
                            .class(iced_button_class(secondary_button_style(
                                self.active_palette(),
                                self.accent(),
                                is_selected,
                            )))
                            .name(format!("Set {} to {}", field.label, choice.label))
                            .on_press(message),
                    );
                }
                Some(widget::container(row).width(Length::Fill).into())
            }
            ParamKind::Boolean => {
                let enabled = value.and_then(Value::as_bool).unwrap_or(false);
                Some(self.small_toggle(
                    format!("{} · {}", field.label, if enabled { "On" } else { "Off" }),
                    enabled,
                    Message::WorkshedParamSet {
                        block_id: block_id.to_string(),
                        field: field.key.clone(),
                        value: Value::Bool(!enabled),
                    },
                ))
            }
            ParamKind::Integer | ParamKind::Number => {
                let rendered = workshed::field_text(value, field);
                let minus = iced::widget::button(centered_text("−".into(), 12))
                    .padding(0)
                    .width(Length::Fixed(28.0))
                    .height(Length::Fixed(25.0))
                    .class(iced_button_class(secondary_button_style(
                        self.active_palette(),
                        self.accent(),
                        false,
                    )))
                    .name(format!("Decrease {}", field.label))
                    .on_press(Message::WorkshedParamStep {
                        block_id: block_id.to_string(),
                        field: field.key.clone(),
                        direction: -1,
                    });
                let plus = iced::widget::button(centered_text("+".into(), 12))
                    .padding(0)
                    .width(Length::Fixed(28.0))
                    .height(Length::Fixed(25.0))
                    .class(iced_button_class(secondary_button_style(
                        self.active_palette(),
                        self.accent(),
                        false,
                    )))
                    .name(format!("Increase {}", field.label))
                    .on_press(Message::WorkshedParamStep {
                        block_id: block_id.to_string(),
                        field: field.key.clone(),
                        direction: 1,
                    });
                Some(
                    widget::row::with_capacity(3)
                        .push(minus)
                        .push(
                            widget::container(self.workshed_help_text(format!(
                                "{} {}{}",
                                field.label,
                                rendered,
                                if field.unit.is_empty() {
                                    String::new()
                                } else {
                                    format!(" {}", field.unit)
                                }
                            )))
                            .width(Length::Fill)
                            .align_x(Alignment::Center),
                        )
                        .push(plus)
                        .spacing(4)
                        .align_y(Alignment::Center)
                        .into(),
                )
            }
            _ => None,
        }
    }

    fn workshed_param_control(
        &self,
        block: &Value,
        definition: &BlockDefinition,
        field: &ParamField,
        issues: &[WorkshedIssue],
    ) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let effective = workshed::effective_params(block, definition);
        let value = effective.get(&field.key);
        let explicit = block
            .get("params")
            .and_then(|params| params.get(&field.key))
            .is_some();
        let required = if field.required { " · required" } else { "" };
        let mut column = widget::column::with_capacity(7).push(
            widget::row::with_capacity(3)
                .push(self.workshed_label_text(format!("{}{}", field.label, required)))
                .push(widget::space().width(Length::Fill))
                .push(if explicit {
                    self.settings_select_button(
                        "Reset",
                        Message::WorkshedParamReset {
                            block_id: block_id.clone(),
                            field: field.key.clone(),
                        },
                        58.0,
                    )
                } else {
                    widget::space().width(Length::Fixed(58.0)).into()
                })
                .align_y(Alignment::Center),
        );
        if !field.description.is_empty() {
            column = column.push(self.workshed_help_text(&field.description));
        }

        let target_port = format!("param.{}", field.key);
        let binding = workshed::input_binding(&self.workshed_order, block, &target_port);
        if let InputBinding::Edge {
            block_id: source_id,
            port,
            ..
        } = &binding
        {
            column = column.push(
                widget::row::with_capacity(2)
                    .push(self.workshed_help_text(format!("Connected · {source_id}.{port}")))
                    .push(widget::space().width(Length::Fill))
                    .push(self.settings_select_button(
                        "Use value",
                        Message::WorkshedParamBindingPressed {
                            block_id: block_id.clone(),
                            field: field.key.clone(),
                        },
                        82.0,
                    ))
                    .align_y(Alignment::Center),
            );
        } else {
            column =
                column.push(self.workshed_param_value_control(&block_id, definition, field, value));
            if field.bindable {
                column = column.push(
                    widget::row::with_capacity(2)
                        .push(self.workshed_help_text(
                            "A compatible upstream value may drive this parameter.",
                        ))
                        .push(widget::space().width(Length::Fill))
                        .push(self.settings_select_button(
                            "Connect",
                            Message::WorkshedParamBindingPressed {
                                block_id: block_id.clone(),
                                field: field.key.clone(),
                            },
                            72.0,
                        ))
                        .align_y(Alignment::Center),
                );
            }
        }
        if workshed::options_provider(field, &effective).is_some() {
            column = column.push(self.workshed_option_picker(block, definition, field, value));
        }
        if let Some(error) = self
            .workshed_field_errors
            .get(&workshed_draft_key(&block_id, &field.key))
        {
            column = column.push(self.workshed_help_text(format!("Error · {error}")));
        }
        for issue in issues
            .iter()
            .filter(|issue| issue.field == field.key)
            .take(2)
        {
            column = column.push(self.workshed_issue_button(issue));
        }
        widget::container(column.spacing(5))
            .padding([7, 7])
            .width(Length::Fill)
            .style(|_| workshed_field_style())
            .into()
    }

    fn workshed_param_value_control(
        &self,
        block_id: &str,
        definition: &BlockDefinition,
        field: &ParamField,
        value: Option<&Value>,
    ) -> Element<'_, Message> {
        match &field.kind {
            ParamKind::Enum(choices) if !choices.is_empty() => {
                let selected = value.cloned().unwrap_or(Value::Null);
                let mut options = widget::column::with_capacity(choices.len()).spacing(3);
                for choice in choices {
                    options = options.push(self.small_toggle(
                        &choice.label,
                        choice.value == selected,
                        Message::WorkshedParamSet {
                            block_id: block_id.to_string(),
                            field: field.key.clone(),
                            value: choice.value.clone(),
                        },
                    ));
                }
                options.into()
            }
            ParamKind::Boolean => {
                let enabled = value.and_then(Value::as_bool).unwrap_or(false);
                widget::row::with_capacity(2)
                    .push(self.small_toggle(
                        "Off",
                        !enabled,
                        Message::WorkshedParamSet {
                            block_id: block_id.to_string(),
                            field: field.key.clone(),
                            value: Value::Bool(false),
                        },
                    ))
                    .push(self.small_toggle(
                        "On",
                        enabled,
                        Message::WorkshedParamSet {
                            block_id: block_id.to_string(),
                            field: field.key.clone(),
                            value: Value::Bool(true),
                        },
                    ))
                    .spacing(4)
                    .into()
            }
            _ => {
                let mut input_field = field.clone();
                if input_field.placeholder.is_empty() && input_field.key == "ref" {
                    let source_kind = workshed::block(&self.workshed_order, block_id)
                        .map(|block| workshed::effective_params(block, definition))
                        .and_then(|params| params.get("source_kind").cloned())
                        .and_then(|value| value.as_str().map(str::to_string))
                        .unwrap_or_default();
                    input_field.placeholder =
                        workshed_reference_placeholder(&definition.type_id, &source_kind);
                }
                let mut column = widget::column::with_capacity(2)
                    .push(self.workshed_param_text_input(block_id, &input_field, value))
                    .spacing(4);
                if workshed::options_provider(
                    field,
                    &workshed::block(&self.workshed_order, block_id)
                        .map(|block| workshed::effective_params(block, definition))
                        .unwrap_or_default(),
                )
                .is_none()
                    && matches!(field.key.as_str(), "ref" | "model")
                {
                    let options = self.chat_model_options();
                    if !options.is_empty() {
                        let mut model_buttons =
                            widget::column::with_capacity(options.len().min(4)).spacing(3);
                        for model in options.into_iter().take(4) {
                            model_buttons = model_buttons.push(self.small_button(
                                fitted_model_label_for_width(model.clone(), 27),
                                Message::WorkshedParamSet {
                                    block_id: block_id.to_string(),
                                    field: field.key.clone(),
                                    value: Value::String(model),
                                },
                            ));
                        }
                        column = column.push(model_buttons);
                    }
                }
                column.into()
            }
        }
    }

    fn workshed_option_picker(
        &self,
        block: &Value,
        definition: &BlockDefinition,
        field: &ParamField,
        value: Option<&Value>,
    ) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let key = workshed_option_key(&block_id, &field.key);
        let query = self
            .workshed_option_queries
            .get(&key)
            .cloned()
            .unwrap_or_default();
        let loading = self.workshed_option_loading.contains(&key);
        let result = self.workshed_option_results.get(&key);
        let items = result
            .and_then(|result| result.get("items"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let message = result
            .map(|result| string_at(result, "message"))
            .unwrap_or_default();
        let query_block = block_id.clone();
        let query_field = field.key.clone();
        let submit_block = block_id.clone();
        let submit_field = field.key.clone();
        let search = widget::text_input("Search available choices", query)
            .id(workshed_input_id(
                &block_id,
                &format!("option:{}", field.key),
            ))
            .padding([7, 9])
            .size(12)
            .font(cosmic::font::default())
            .style(app_input_style(self.accent().display_color()))
            .on_input(move |value| Message::WorkshedOptionQueryChanged {
                block_id: query_block.clone(),
                field: query_field.clone(),
                value,
            })
            .on_submit(move |_| Message::WorkshedOptionSearchPressed {
                block_id: submit_block.clone(),
                field: submit_field.clone(),
            })
            .width(Length::Fill);
        let mut column = widget::column::with_capacity(10).push(
            widget::row::with_capacity(2)
                .push(widget::container(search).width(Length::Fill))
                .push(self.settings_select_button(
                    if loading { "Loading…" } else { "Search" },
                    Message::WorkshedOptionSearchPressed {
                        block_id: block_id.clone(),
                        field: field.key.clone(),
                    },
                    74.0,
                ))
                .spacing(4)
                .align_y(Alignment::Center),
        );
        if items.is_empty() && !loading {
            column = column.push(self.workshed_help_text(if message.is_empty() {
                "Search or refresh to load choices from this Mac and the configured catalog."
            } else {
                &message
            }));
        } else {
            let selected_values = value.and_then(Value::as_array).cloned().unwrap_or_default();
            for item in items.iter().take(8) {
                let option_id = string_at(item, "id");
                if option_id.is_empty() {
                    continue;
                }
                let option_label = {
                    let label = string_at(item, "label");
                    if label.is_empty() {
                        option_id.clone()
                    } else {
                        label
                    }
                };
                let selected = if matches!(field.kind, ParamKind::StringArray) {
                    selected_values.contains(&Value::String(option_id.clone()))
                } else {
                    value.is_some_and(|value| {
                        value == &Value::String(option_id.clone())
                            || workshed::field_text(Some(value), field) == option_id
                    })
                };
                column = column.push(self.small_toggle(
                    option_label,
                    selected,
                    Message::WorkshedOptionSelected {
                        block_id: block_id.clone(),
                        field: field.key.clone(),
                        value: option_id,
                    },
                ));
                let description = string_at(item, "description");
                if !description.is_empty() {
                    column = column.push(self.workshed_help_text(description));
                }
            }
            if !message.is_empty() {
                column = column.push(self.workshed_help_text(message));
            }
        }
        let _ = definition;
        widget::container(column.spacing(4))
            .padding([6, 6])
            .width(Length::Fill)
            .style(|_| workshed_option_style())
            .into()
    }

    fn workshed_param_text_input(
        &self,
        block_id: &str,
        field: &ParamField,
        value: Option<&Value>,
    ) -> Element<'_, Message> {
        let key = workshed_draft_key(block_id, &field.key);
        let display = self
            .workshed_field_drafts
            .get(&key)
            .cloned()
            .unwrap_or_else(|| workshed::field_text(value, field));
        let placeholder = if field.placeholder.is_empty() {
            match field.kind {
                ParamKind::Integer => "0".into(),
                ParamKind::Number => "0.0".into(),
                ParamKind::StringArray => "item-one, item-two".into(),
                _ => field.label.clone(),
            }
        } else {
            field.placeholder.clone()
        };
        let input_block = block_id.to_string();
        let input_field = field.key.clone();
        let submit_block = block_id.to_string();
        let submit_field = field.key.clone();
        let focus_block = block_id.to_string();
        let focus_field = field.key.clone();
        let blur_block = block_id.to_string();
        let blur_field = field.key.clone();
        let input = widget::text_input(placeholder, display)
            .id(workshed_input_id(block_id, &field.key))
            .padding([8, 10])
            .size(13)
            .font(cosmic::font::default())
            .style(app_input_style(self.accent().display_color()))
            .on_input(move |value| Message::WorkshedParamDraftChanged {
                block_id: input_block.clone(),
                field: input_field.clone(),
                value,
            })
            .on_submit(move |_| Message::WorkshedParamCommit {
                block_id: submit_block.clone(),
                field: submit_field.clone(),
            })
            .on_focus(Message::WorkshedFieldFocused {
                block_id: focus_block,
                field: focus_field,
                focused: true,
            })
            .on_unfocus(Message::WorkshedFieldFocused {
                block_id: blur_block,
                field: blur_field,
                focused: false,
            })
            .width(Length::Fill);
        let focused = self.workshed_focused_field.as_deref() == Some(key.as_str());
        let palette = self.active_palette();
        widget::container(input)
            .height(Length::Fixed(38.0))
            .align_y(Alignment::Center)
            .style(move |_| input_shell_style(palette, focused))
            .into()
    }

    fn workshed_input_control(
        &self,
        block: &Value,
        port: &workshed::PortDefinition,
        issues: &[WorkshedIssue],
    ) -> Element<'_, Message> {
        let block_id = string_at(block, "id");
        let binding = workshed::input_binding(&self.workshed_order, block, &port.name);
        let current_mode = match binding {
            InputBinding::Literal(_) => "literal",
            InputBinding::Auto => "auto",
            InputBinding::Inherited(_) => "inherited",
            InputBinding::Edge { .. } => "edge",
            InputBinding::Missing => "",
        };
        let mut column = widget::column::with_capacity(8)
            .push(
                widget::row::with_capacity(2)
                    .push(self.workshed_label_text(format!(
                        "{} · {}{}",
                        port.label,
                        port.type_name,
                        if port.required { " · required" } else { "" }
                    )))
                    .push(widget::space().width(Length::Fill)),
            )
            .spacing(5);
        let mut modes = widget::row::with_capacity(port.binding_modes.len().max(1)).spacing(3);
        let normalized_modes = if port.binding_modes.is_empty() {
            vec!["edge".to_string(), "literal".to_string()]
        } else {
            port.binding_modes.clone()
        };
        for mode in normalized_modes {
            let normalized = if matches!(mode.as_str(), "ref" | "connection") {
                "edge"
            } else {
                mode.as_str()
            };
            let label = match normalized {
                "edge" => "Connect",
                "literal" => "Value",
                "auto" => "Auto",
                "inherited" => "Inherit",
                other => other,
            };
            modes = modes.push(self.small_toggle(
                label,
                current_mode == normalized,
                Message::WorkshedInputModeChanged {
                    block_id: block_id.clone(),
                    port: port.name.clone(),
                    mode: normalized.to_string(),
                },
            ));
        }
        column = column.push(modes);
        match &binding {
            InputBinding::Edge {
                block_id: source_id,
                port: source_port,
                ..
            } => {
                column = column.push(
                    self.workshed_help_text(format!("Connected from {source_id}.{source_port}")),
                );
            }
            InputBinding::Auto => {
                column = column
                    .push(self.workshed_help_text("Resolved automatically during preflight."));
            }
            InputBinding::Inherited(name) => {
                column = column.push(self.workshed_help_text(format!("Inherited from {name}.")));
            }
            InputBinding::Literal(value) => {
                column =
                    column.push(self.workshed_input_literal_control(port, &block_id, Some(value)));
            }
            InputBinding::Missing => {
                if port.binding_modes.iter().any(|mode| mode == "literal")
                    || port.binding_modes.is_empty()
                {
                    column =
                        column.push(self.workshed_input_literal_control(port, &block_id, None));
                }
            }
        }
        if let Some(error) = self.workshed_field_errors.get(&workshed_draft_key(
            &block_id,
            &format!("input:{}", port.name),
        )) {
            column = column.push(self.workshed_help_text(format!("Error · {error}")));
        }
        for issue in issues
            .iter()
            .filter(|issue| issue.port == port.name)
            .take(2)
        {
            column = column.push(self.workshed_issue_button(issue));
        }
        widget::container(column)
            .padding([7, 7])
            .width(Length::Fill)
            .style(|_| workshed_field_style())
            .into()
    }

    fn workshed_input_literal_control(
        &self,
        port: &workshed::PortDefinition,
        block_id: &str,
        value: Option<&Value>,
    ) -> Element<'_, Message> {
        let field = workshed::literal_field(port);
        match &field.kind {
            ParamKind::Enum(choices) if !choices.is_empty() => {
                let selected = value.cloned().unwrap_or(Value::Null);
                let mut options = widget::column::with_capacity(choices.len()).spacing(3);
                for choice in choices {
                    options = options.push(self.small_toggle(
                        &choice.label,
                        choice.value == selected,
                        Message::WorkshedInputDraftChanged {
                            block_id: block_id.to_string(),
                            port: port.name.clone(),
                            value: workshed::field_text(Some(&choice.value), &field),
                        },
                    ));
                }
                options.into()
            }
            ParamKind::Boolean => {
                let enabled = value.and_then(Value::as_bool).unwrap_or(false);
                self.small_toggle(
                    if enabled { "On" } else { "Off" },
                    enabled,
                    Message::WorkshedInputDraftChanged {
                        block_id: block_id.to_string(),
                        port: port.name.clone(),
                        value: (!enabled).to_string(),
                    },
                )
            }
            _ => {
                let field_key = format!("input:{}", port.name);
                let key = workshed_draft_key(block_id, &field_key);
                let display = self
                    .workshed_field_drafts
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| workshed::field_text(value, &field));
                let placeholder = if field.placeholder.is_empty() {
                    field.label.clone()
                } else {
                    field.placeholder.clone()
                };
                let input_block = block_id.to_string();
                let input_port = port.name.clone();
                let submit_block = block_id.to_string();
                let submit_port = port.name.clone();
                let focus_block = block_id.to_string();
                let blur_block = block_id.to_string();
                let mut column = widget::column::with_capacity(2).push(
                    widget::container(
                        widget::text_input(placeholder, display)
                            .id(workshed_input_id(block_id, &field_key))
                            .padding([8, 10])
                            .size(13)
                            .style(app_input_style(self.accent().display_color()))
                            .on_input(move |value| Message::WorkshedInputDraftChanged {
                                block_id: input_block.clone(),
                                port: input_port.clone(),
                                value,
                            })
                            .on_submit(move |_| Message::WorkshedInputCommit {
                                block_id: submit_block.clone(),
                                port: submit_port.clone(),
                            })
                            .on_focus(Message::WorkshedFieldFocused {
                                block_id: focus_block,
                                field: field_key.clone(),
                                focused: true,
                            })
                            .on_unfocus(Message::WorkshedFieldFocused {
                                block_id: blur_block,
                                field: field_key,
                                focused: false,
                            })
                            .width(Length::Fill),
                    )
                    .height(Length::Fixed(38.0))
                    .style({
                        let palette = self.active_palette();
                        let focused = self.workshed_focused_field.as_deref() == Some(key.as_str());
                        move |_| input_shell_style(palette, focused)
                    }),
                );
                if port.type_name == "Model" {
                    let options = self.chat_model_options();
                    if !options.is_empty() {
                        let mut models =
                            widget::column::with_capacity(options.len().min(3)).spacing(3);
                        for model in options.into_iter().take(3) {
                            models = models.push(self.small_button(
                                fitted_model_label_for_width(model.clone(), 27),
                                Message::WorkshedInputDraftChanged {
                                    block_id: block_id.to_string(),
                                    port: port.name.clone(),
                                    value: model,
                                },
                            ));
                        }
                        column = column.push(models);
                    }
                }
                column.into()
            }
        }
    }

    fn workshed_issue_button(&self, issue: &WorkshedIssue) -> Element<'_, Message> {
        let prefix = if issue.severity == "error" {
            "Error"
        } else {
            "Note"
        };
        iced::widget::button(
            widget::text(format!("{prefix} · {}", issue.message))
                .size(11)
                .wrapping(text::Wrapping::WordOrGlyph),
        )
        .padding([6, 7])
        .width(Length::Fill)
        .class(iced_button_class(secondary_button_style(
            self.active_palette(),
            self.accent(),
            false,
        )))
        .name(format!("{prefix}: {}", issue.message))
        .on_press(Message::WorkshedIssuePressed {
            block_id: issue.block_id.clone(),
            field: issue.field.clone(),
            port: issue.port.clone(),
        })
        .into()
    }

    fn workshed_preflight_panel(&self) -> Element<'_, Message> {
        let errors = workshed_messages(&self.workshed_preflight, "errors");
        let warnings = workshed_messages(&self.workshed_preflight, "warnings");
        let estimate = self
            .workshed_preflight
            .get("estimates")
            .unwrap_or(&Value::Null);
        let toolchains = self
            .workshed_preflight
            .get("toolchains")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        format!(
                            "{} · {}",
                            string_at(item, "label"),
                            string_at(item, "status")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("  •  ")
            })
            .unwrap_or_else(|| "Run preflight to resolve toolchains".into());
        let summary: String = if self.workshed_preflight.is_null() {
            "Run preflight to compile a frozen local plan.".into()
        } else if self.workshed_preflight_ok() {
            "Plan is valid. Required consent is shown before tool preparation.".into()
        } else {
            "Fix the highlighted issues before Run becomes available.".into()
        };
        let mut body = widget::column::with_capacity(8)
            .push(
                widget::row::with_capacity(4)
                    .push(self.settings_subtitle("Preflight plan"))
                    .push(widget::space().width(Length::Fill))
                    .push(self.workshed_help_text("Local only · no Worker requests"))
                    .push(self.small_button(
                        if self.workshed_preflight_open {
                            "Collapse"
                        } else {
                            "Expand"
                        },
                        Message::WorkshedPanelToggled("preflight".into()),
                    ))
                    .spacing(8),
            )
            .push(self.workshed_help_text(summary))
            .spacing(6);
        if self.workshed_preflight_open {
            body = body
                .push(self.dashboard_row("Toolchains", toolchains))
                .push(self.dashboard_row(
                    "Estimate",
                    format!(
                        "{} steps · {} MB output · {} GB memory",
                        estimate
                            .get("step_count")
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                        estimate
                            .get("estimated_output_bytes")
                            .and_then(Value::as_u64)
                            .map(|v| v / 1024 / 1024)
                            .unwrap_or(0),
                        estimate
                            .get("estimated_memory_bytes")
                            .and_then(Value::as_u64)
                            .map(|v| v / 1024 / 1024 / 1024)
                            .unwrap_or(0)
                    ),
                ));
            for error in errors.iter().take(2) {
                body = body.push(self.workshed_help_text(format!("Error · {error}")));
            }
            for warning in warnings.iter().take(2) {
                body = body.push(self.workshed_help_text(format!("Note · {warning}")));
            }
            body = body.push(
                widget::row::with_capacity(3)
                    .push(self.secondary_button("Run preflight", Message::WorkshedPreflightPressed))
                    .push(if self.workshed_preflight_ok() {
                        self.primary_button("Prepare & Run", Message::WorkshedRunPressed)
                    } else {
                        widget::container(centered_text("Prepare & Run".into(), 14))
                            .height(Length::Fixed(38.0))
                            .width(Length::Fill)
                            .style(|_| disabled_action_style())
                            .into()
                    })
                    .spacing(8),
            );
        }
        widget::container(body)
            .padding(10)
            .width(Length::Fill)
            .style(|_| workshed_panel_style())
            .into()
    }

    fn legacy_quantize_page(&self) -> Element<'_, Message> {
        let selected_engine = self.quantization_capabilities.engines.iter().find(|item| {
            item.get("id").and_then(Value::as_str) == Some(self.quantization_engine.as_str())
        });
        let presets = selected_engine
            .and_then(|item| item.get("presets"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let worker = self.quantization_capabilities.workers.first();
        let remote_ready = worker
            .and_then(|item| item.get("status"))
            .and_then(Value::as_str)
            .is_some_and(|status| matches!(status, "ready" | "configured"));
        let source_local = self.quantization_source_kind == "local_path";
        let worker_summary = selected_engine
            .and_then(|engine| engine.get("worker"))
            .map(|details| {
                let name = string_at(details, "name");
                let gpu = string_at(details, "gpu");
                let memory = string_at(details, "memory");
                let status = worker
                    .map(|item| string_at(item, "status"))
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "unknown".into());
                format!("{name} · {gpu} · {memory} · {status}")
            })
            .unwrap_or_else(|| "No HTTPS Worker configured".into());

        let source_card = self.section_card(
            "1 · Source",
            "Choose a Hub model or a local folder containing compatible weights.",
            widget::column::with_capacity(5)
                .push(
                    widget::row::with_capacity(2)
                        .push(self.small_toggle(
                            "Hugging Face",
                            !source_local,
                            Message::QuantizationSourceKindChanged("huggingface".into()),
                        ))
                        .push(if self.quantization_engine == "vllm_cuda" {
                            self.disabled_toggle("Local folder · Hub only")
                        } else {
                            self.small_toggle(
                                "Local folder",
                                source_local,
                                Message::QuantizationSourceKindChanged("local_path".into()),
                            )
                        })
                        .spacing(8),
                )
                .push(self.labeled_app_text_input(
                    "Model ID or local path",
                    if source_local {
                        "/Users/name/Models/MyModel"
                    } else {
                        "org/model-name"
                    },
                    &self.quantization_source,
                    Some(Message::QuantizationSourceChanged),
                    InputField::QuantizationSource,
                    false,
                ))
                .push(self.soft_text(if self.quantization_engine == "vllm_cuda" {
                    "CUDA Worker accepts Hub sources only; local folders are disabled for this engine."
                } else {
                    "The source is read-only. MLX writes only to its private quantization workspace."
                }))
                .spacing(8)
                .into(),
        );

        let engine_card = self.section_card(
            "2 · Executor",
            "Capabilities are supplied by the local manager and configured Worker.",
            widget::column::with_capacity(5)
                .push(
                    widget::row::with_capacity(2)
                        .push(self.small_toggle(
                            "MLX · This Mac",
                            self.quantization_engine == "mlx_local",
                            Message::QuantizationEngineChanged("mlx_local".into()),
                        ))
                        .push(self.small_toggle(
                            "vLLM/CUDA · Worker",
                            self.quantization_engine == "vllm_cuda",
                            Message::QuantizationEngineChanged("vllm_cuda".into()),
                        ))
                        .spacing(8),
                )
                .push(if self.quantization_engine == "vllm_cuda" {
                    self.dashboard_row("Worker", worker_summary)
                } else {
                    self.dashboard_row("Runtime", "MLX-LM · Apple Silicon")
                })
                .spacing(8)
                .into(),
        );

        let preset_buttons = if presets.is_empty() {
            self.soft_text("Waiting for executor capabilities...")
        } else {
            presets
                .iter()
                .fold(widget::row::with_capacity(presets.len()), |row, preset| {
                    let id = string_at(preset, "id");
                    let label = string_at(preset, "label");
                    row.push(self.small_toggle(
                        if label.is_empty() { id.clone() } else { label },
                        self.quantization_preset == id,
                        Message::QuantizationPresetChanged(id),
                    ))
                })
                .spacing(8)
                .into()
        };
        let advanced: Element<'_, Message> = if self.quantization_advanced_open {
            widget::column::with_capacity(4)
                .push(
                    widget::row::with_capacity(3)
                        .push(self.labeled_app_text_input(
                            "Bits",
                            "bits (4/8)",
                            &self.quantization_bits,
                            Some(Message::QuantizationBitsChanged),
                            InputField::QuantizationBits,
                            false,
                        ))
                        .push(self.labeled_app_text_input(
                            "Group size",
                            "group size",
                            &self.quantization_group_size,
                            Some(Message::QuantizationGroupSizeChanged),
                            InputField::QuantizationGroupSize,
                            false,
                        ))
                        .push(self.labeled_app_text_input(
                            "Mode",
                            "mode",
                            &self.quantization_mode,
                            Some(Message::QuantizationModeChanged),
                            InputField::QuantizationMode,
                            false,
                        ))
                        .spacing(8),
                )
                .push(self.labeled_app_text_input(
                    "Revision (optional)",
                    "revision (optional)",
                    &self.quantization_revision,
                    Some(Message::QuantizationRevisionChanged),
                    InputField::QuantizationRevision,
                    false,
                ))
                .push(self.labeled_app_text_input(
                    "Algorithm (Worker)",
                    "algorithm (Worker)",
                    &self.quantization_algorithm,
                    Some(Message::QuantizationAlgorithmChanged),
                    InputField::QuantizationAlgorithm,
                    false,
                ))
                .push(
                    self.soft_text("Only values advertised by the selected executor are accepted."),
                )
                .spacing(8)
                .into()
        } else {
            widget::column::with_capacity(1).into()
        };
        let preset_card = self.section_card(
            "3 · Preset",
            "Start with a guided profile. Advanced overrides stay collapsed until needed.",
            widget::column::with_capacity(5)
                .push(preset_buttons)
                .push(self.secondary_button(
                    if self.quantization_advanced_open {
                        "Hide advanced parameters"
                    } else {
                        "Show advanced parameters"
                    },
                    Message::QuantizationAdvancedToggled,
                ))
                .push(advanced)
                .spacing(8)
                .into(),
        );

        let output_card =
            self.section_card(
                "4 · Output",
                "Give the managed artifact a stable name before preflight.",
                widget::column::with_capacity(3)
                    .push(self.labeled_app_text_input(
                        "Output name",
                        "my-model-mlx-4bit",
                        &self.quantization_output,
                        Some(Message::QuantizationOutputChanged),
                        InputField::QuantizationOutput,
                        false,
                    ))
                    .push(self.soft_text(
                        "Output names use letters, numbers, dots, dashes, and underscores.",
                    ))
                    .spacing(8)
                    .into(),
            );

        let preflight = &self.quantization_preflight;
        let errors = preflight
            .get("errors")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let warnings = preflight
            .get("warnings")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let estimate = preflight.get("estimate").unwrap_or(&Value::Null);
        let preflight_summary = if self.quantization_preflight.is_null() {
            self.soft_text("Run preflight to check compatibility, space, and output conflicts.")
        } else if self.quantization_preflight_ok() {
            self.soft_text("Ready to start. The selected executor accepted this request.")
        } else {
            self.soft_text("Preflight needs attention before the job can start.")
        };
        let mut preflight_body = widget::column::with_capacity(8)
            .push(preflight_summary)
            .push(self.dashboard_row(
                "Engine",
                if self.quantization_engine == "vllm_cuda" {
                    "vLLM/CUDA Worker"
                } else {
                    "MLX · This Mac"
                },
            ))
            .push(
                self.dashboard_row(
                    "Compression",
                    estimate
                        .get("compression_ratio")
                        .map(|value| format!("{}×", value))
                        .unwrap_or_else(|| "-".into()),
                ),
            );
        preflight_body = preflight_body
            .push(
                self.dashboard_row(
                    "Disk estimate",
                    estimate
                        .get("estimated_output_bytes")
                        .and_then(Value::as_u64)
                        .map(format_bytes)
                        .unwrap_or_else(|| "-".into()),
                ),
            )
            .push(
                self.dashboard_row(
                    "Free space",
                    estimate
                        .get("free_bytes")
                        .and_then(Value::as_u64)
                        .map(format_bytes)
                        .unwrap_or_else(|| "-".into()),
                ),
            );
        for error in errors.iter().take(3) {
            preflight_body = preflight_body.push(self.soft_text(format!("Error · {error}")));
        }
        for warning in warnings.iter().take(3) {
            preflight_body = preflight_body.push(self.soft_text(format!("Note · {warning}")));
        }
        preflight_body = preflight_body
            .push(self.secondary_button("Run preflight", Message::QuantizationPreflightPressed))
            .push(if self.quantization_preflight_ok() {
                self.primary_button("Start quantization", Message::QuantizationStartPressed)
            } else {
                widget::container(centered_text("Start quantization".into(), 14))
                    .height(Length::Fixed(38.0))
                    .width(Length::Fill)
                    .style(|_| disabled_action_style())
                    .into()
            })
            .spacing(8);
        let preflight_card = self.section_card(
            "Preflight",
            "The start action is intentionally gated on this check.",
            preflight_body.into(),
        );

        let job_card = if let Some(job) = &self.quantization_job {
            let percent = (job.progress.clamp(0.0, 1.0) * 100.0).round();
            let speed = if job.speed.trim().is_empty() {
                "—".to_string()
            } else {
                job.speed.clone()
            };
            let eta = match job.eta_seconds {
                Some(seconds) if seconds >= 60 => {
                    format!("{}m {:02}s", seconds / 60, seconds % 60)
                }
                Some(seconds) => format!("{seconds}s"),
                None => "—".to_string(),
            };
            let mut body = widget::column::with_capacity(8)
                .push(self.dashboard_row("Stage", &job.stage))
                .push(self.dashboard_row("Progress", format!("{percent:.0}%")))
                .push(self.dashboard_row("Speed", speed))
                .push(self.dashboard_row("ETA", eta))
                .push(self.dashboard_row("Status", &job.status))
                .push(self.soft_text(&job.message));
            if let Some(last_log) = job
                .raw
                .get("logs")
                .and_then(Value::as_array)
                .and_then(|items| items.last())
                .and_then(Value::as_str)
            {
                body = body.push(self.soft_text(format!("Log · {last_log}")));
            }
            if !job.error.trim().is_empty() {
                body = body.push(self.soft_text(format!("Error · {}", job.error)));
            }
            if let Some(command) = job
                .raw
                .get("deployment")
                .and_then(|value| value.get("command"))
                .and_then(Value::as_str)
            {
                body = body.push(self.soft_text(format!("Deploy · {command}")));
            }
            if matches!(
                job.status.as_str(),
                "queued" | "running" | "pausing" | "cancelling" | "paused"
            ) {
                let mut actions = widget::row::with_capacity(3)
                    .push(self.danger_button(
                        "Cancel",
                        Message::QuantizationActionPressed("cancel".into()),
                    ))
                    .spacing(8);
                if job.allowed_actions.iter().any(|action| action == "pause")
                    && job.status == "running"
                {
                    actions = actions.push(self.secondary_button(
                        "Pause",
                        Message::QuantizationActionPressed("pause".into()),
                    ));
                } else if job.allowed_actions.iter().any(|action| action == "pause")
                    && job.status == "paused"
                {
                    actions = actions.push(self.secondary_button(
                        "Resume",
                        Message::QuantizationActionPressed("resume".into()),
                    ));
                }
                body = body.push(actions);
            }
            if matches!(job.status.as_str(), "awaiting_delivery" | "completed") {
                let options = if job.engine == "vllm_cuda" {
                    vec![
                        ("Retain on Worker", "retain"),
                        ("Download to Mac", "download"),
                    ]
                } else {
                    vec![
                        ("Register in Model Settings", "register"),
                        ("Reveal in Finder", "reveal"),
                    ]
                };
                body = body.push(
                    widget::row::with_capacity(options.len())
                        .push(self.secondary_button(
                            options[0].0,
                            Message::QuantizationDeliveryPressed(options[0].1.into()),
                        ))
                        .push(self.primary_button(
                            options[1].0,
                            Message::QuantizationDeliveryPressed(options[1].1.into()),
                        ))
                        .spacing(8),
                );
            }
            self.section_card(
                "5 · Live job",
                "The manager persists this record so it can be recovered after restart.",
                body.into(),
            )
        } else {
            self.section_card(
                "5 · Live job",
                "No quantization job is running.",
                self.soft_text("After start, this strip shows stage, progress, speed, ETA, and delivery actions."),
            )
        };

        let setup = widget::scrollable(
            widget::column::with_capacity(5)
                .push(source_card)
                .push(engine_card)
                .push(preset_card)
                .push(output_card)
                .spacing(12),
        )
        .height(Length::Fill)
        .class(theme::iced::Scrollable::Transient);
        // Keep the live job strip outside the setup scroller so it remains
        // visible at the bottom while source/preset details move underneath.
        let left = widget::column::with_capacity(2)
            .push(setup)
            .push(job_card)
            .spacing(12)
            .height(Length::Fill);
        let right = widget::column::with_capacity(3)
            .push(preflight_card)
            .push(
                self.section_card(
                    "Compatibility",
                    "The selected executor reports the available algorithms and hardware.",
                    widget::column::with_capacity(4)
                        .push(self.dashboard_row(
                            "Source",
                            if source_local {
                                "Local folder"
                            } else {
                                "Hugging Face Hub"
                            },
                        ))
                        .push(self.dashboard_row(
                            "Worker",
                            if remote_ready {
                                "Connected"
                            } else {
                                "Not configured"
                            },
                        ))
                        .push(self.dashboard_row("Preset", &self.quantization_preset))
                        .push(self.soft_text(
                            if self.quantization_engine == "vllm_cuda" && !source_local {
                                "Remote jobs never receive the Mac's Hugging Face token."
                            } else {
                                "Credentials stay within their owning runtime."
                            },
                        ))
                        .spacing(8)
                        .into(),
                ),
            )
            .spacing(12)
            .width(Length::Fixed(320.0));
        let content = widget::row::with_capacity(2)
            .push(
                widget::container(left)
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .push(right)
            .spacing(12)
            .height(Length::Fill);
        let page = self.page_shell(
            "Quantize",
            "A guided workbench for MLX on this Mac or vLLM/CUDA on a trusted Worker.",
            content.into(),
        );
        if self.quantization_delivery_open {
            iced::widget::stack(vec![page, self.quantization_delivery_dialog()])
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            page
        }
    }

    fn quantization_delivery_dialog(&self) -> Element<'_, Message> {
        let Some(job) = &self.quantization_job else {
            return widget::container(text("No quantization artifact is waiting for delivery."))
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
        };
        let palette = self.active_palette();
        let is_cuda = job.engine == "vllm_cuda";
        let output_name = job
            .raw
            .get("output_name")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("quantized-model");
        let (first_label, first_action, second_label, second_action) = if is_cuda {
            ("Retain on Worker", "retain", "Download to Mac", "download")
        } else {
            (
                "Register in Model Settings",
                "register",
                "Reveal in Finder",
                "reveal",
            )
        };
        let dialog = widget::container(
            widget::column::with_capacity(7)
                .push(self.settings_subtitle("Choose delivery"))
                .push(self.soft_text(if is_cuda {
                    "The CUDA Worker finished. Decide whether to keep the deployable artifact remote or bring a managed copy to this Mac."
                } else {
                    "The MLX artifact passed validation. Register it for model switching or reveal the managed folder in Finder."
                }))
                .push(self.caligo_horizontal_rule(false))
                .push(self.dashboard_row("Output", output_name))
                .push(self.dashboard_row("Status", &job.status))
                .push(
                    widget::row::with_capacity(2)
                        .push(self.secondary_button(
                            first_label,
                            Message::QuantizationDeliveryPressed(first_action.into()),
                        ))
                        .push(self.primary_button(
                            second_label,
                            Message::QuantizationDeliveryPressed(second_action.into()),
                        ))
                        .spacing(8),
                )
                .push(self.secondary_button(
                    "Choose later",
                    Message::QuantizationDeliveryDismissed,
                ))
                .spacing(10)
                .padding(18)
                .width(Length::Fixed(560.0)),
        )
        .style(move |_| glass_container_style(palette, GlassVariant::Floating, GlassState::Focused));
        widget::container(dialog)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(move |_| {
                container::Style::default().background(Background::Color(palette.overlay_backdrop))
            })
            .into()
    }

    fn models_page(&self) -> Element<'_, Message> {
        let palette = self.active_palette();
        let list = self.models.iter().fold(
            widget::column::with_capacity(self.models.len()),
            |column, model| {
                let selected = self.native_state.selected_local_model == *model
                    || self.native_state.selected_model == *model;
                column.push(self.model_row(model, selected))
            },
        );

        let selected_model = if !self.native_state.selected_local_model.trim().is_empty() {
            self.native_state.selected_local_model.clone()
        } else {
            self.native_state.selected_model.clone()
        };
        let running_model = self
            .manager_models
            .running_models
            .iter()
            .find(|running| running.model == selected_model);
        let selected_detail = self
            .manager_models
            .model_details
            .iter()
            .find(|detail| detail.id == selected_model);
        let is_active = !selected_model.trim().is_empty()
            && (self.manager_models.active_model == selected_model
                || self.native_state.selected_model == selected_model
                || selected_detail.map(|detail| detail.active).unwrap_or(false));
        let is_running = running_model.is_some()
            || selected_detail
                .map(|detail| detail.running)
                .unwrap_or(false);
        let state_label = selected_detail
            .map(|detail| detail.status.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                if selected_model.trim().is_empty() {
                    "none"
                } else if is_active {
                    "active"
                } else if is_running {
                    "running"
                } else {
                    "local"
                }
            });
        let server_label = selected_detail
            .map(|detail| detail.server_url.as_str())
            .filter(|server| !server.trim().is_empty())
            .or_else(|| running_model.map(|running| running.server_url.as_str()))
            .filter(|server| !server.trim().is_empty())
            .or_else(|| {
                if is_active && !self.manager_models.active_server_url.trim().is_empty() {
                    Some(self.manager_models.active_server_url.as_str())
                } else if !self.manager_models.server_url.trim().is_empty() {
                    Some(self.manager_models.server_url.as_str())
                } else {
                    None
                }
            })
            .unwrap_or("-");
        let role_label = selected_detail
            .map(|detail| detail.role.as_str())
            .filter(|role| !role.trim().is_empty())
            .or_else(|| {
                running_model
                    .map(|running| running.role.as_str())
                    .filter(|role| !role.trim().is_empty())
            })
            .unwrap_or(if is_active { "active" } else { "-" });
        let source_label = selected_detail
            .map(|detail| detail.source.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("-");
        let backend_label = selected_detail
            .map(|detail| detail.backend.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("-");
        let quantization_label = selected_detail
            .map(|detail| detail.quantization.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("-");
        let kind_label = selected_detail
            .map(|detail| detail.kind.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("-");
        let reasoning_label = selected_detail
            .map(|detail| detail.reasoning_parser.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("-");
        let tool_parser_label = selected_detail
            .map(|detail| detail.tool_call_parser.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("-");
        let agent_profile_label = self.profile_status_label();
        let integrity_label = self.model_integrity_status_label();
        let selected_display = selected_detail
            .map(|detail| detail.display_name.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&selected_model);
        let selected_label = fitted_model_label(empty_dash(selected_display).into());
        let selected_label_size = fitted_model_label_size(&selected_label);
        let model_list_body: Element<'_, Message> = if self.models.is_empty() {
            self.empty_state("No local models found. Use Community to download one.")
        } else {
            widget::scrollable(list.spacing(8))
                .height(Length::Fill)
                .width(Length::Fill)
                .class(theme::iced::Scrollable::Transient)
                .into()
        };

        let models_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(5)
                        .push(
                            widget::row::with_capacity(2)
                                .push(
                                    widget::container(self.settings_subtitle("Models"))
                                        .width(Length::Fill),
                                )
                                .push(self.models_count_pill(self.models.len()))
                                .padding([0, 0, 8, 0])
                                .align_y(Alignment::Center),
                        )
                        .push(self.settings_divider_fill())
                        .push(widget::space().height(Length::Fixed(9.0)))
                        .push(model_list_body),
                )
                .padding(12)
                .height(Length::Fill)
                .into(),
                GlassVariant::Panel,
                GlassState::Rest,
            ),
        )
        .width(Length::Fixed(380.0))
        .height(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let palette = self.active_palette();
        let detail_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(12)
                        .push(self.settings_subtitle("Model Details"))
                        .push(
                            widget::container(
                                widget::text(selected_label)
                                    .size(selected_label_size)
                                    .wrapping(text::Wrapping::None),
                            )
                            .height(Length::Fixed(34.0))
                            .width(Length::Fill)
                            .align_y(Alignment::Center),
                        )
                        .push(self.settings_divider_fill())
                        .push(
                            self.split_danger_action_group(
                                "Switch",
                                Message::SwitchModelPressed(selected_model.clone()),
                                "Delete model",
                                Message::DeleteSelectedModelPressed,
                                112.0,
                                112.0,
                            ),
                        )
                        .push(self.dashboard_row("State", state_label))
                        .push(self.dashboard_row("Source", source_label))
                        .push(self.dashboard_row("Backend", backend_label))
                        .push(self.dashboard_row("Quantization", quantization_label))
                        .push(self.dashboard_row("Server URL", server_label))
                        .push(self.dashboard_row("Kind", kind_label))
                        .push(self.dashboard_row("Role", role_label))
                        .push(self.dashboard_row("Reasoning", reasoning_label))
                        .push(self.dashboard_row("Tool Parser", tool_parser_label))
                        .push(self.dashboard_row("Agent Profile", agent_profile_label))
                        .push(self.dashboard_row("Cache Check", integrity_label))
                        .push(self.split_danger_action_group(
                            "Recalibrate Profile",
                            Message::RecalibrateAgentProfilePressed,
                            "Delete Agent Profile",
                            Message::DeleteAgentProfilePressed,
                            136.0,
                            136.0,
                        ))
                        .push(self.soft_text(
                            "Delete model removes local weights. Delete Profile keeps model weights; Recalibrate reruns both agent probes.",
                        ))
                        .spacing(8),
                )
                .padding(12)
                .height(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let palette = self.active_palette();
        let generation_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(8)
                        .push(self.settings_subtitle("Generation"))
                        .push(self.settings_row_fixed(
                            "Max Tokens",
                            self.app_text_input(
                                "1024",
                                &self.max_tokens_input,
                                Some(Message::MaxTokensChanged),
                                InputField::MaxTokens,
                                false,
                            ),
                            180.0,
                        ))
                        .push(self.settings_row_fixed(
                            "Temperature",
                            self.temperature_control(),
                            338.0,
                        ))
                        .push(self.split_action_group(
                            "Reset",
                            Message::ResetGenerationPressed,
                            "Save",
                            Message::SaveGenerationPressed,
                            96.0,
                            96.0,
                        ))
                        .push(self.soft_text("These request settings apply to new chat messages."))
                        .spacing(10),
                )
                .padding(12)
                .height(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let palette = self.active_palette();
        let runtime_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(10)
                        .push(self.settings_subtitle("Runtime"))
                        .push(self.runtime_checks())
                        .push(self.runtime_numbers())
                        .push(self.split_action_group(
                            "Reset",
                            Message::ResetRuntimePressed,
                            "Apply & Restart",
                            Message::SaveRuntimePressed,
                            96.0,
                            136.0,
                        ))
                        .push(self.soft_text(
                            "Changing runtime options restarts the active model process.",
                        ))
                        .push(self.soft_text(&self.status_text))
                        .spacing(10),
                )
                .padding(12)
                .height(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let settings_column = widget::container(
            widget::scrollable(
                widget::column::with_capacity(3)
                    .push(detail_card)
                    .push(generation_card)
                    .push(runtime_card)
                    .spacing(10)
                    .width(Length::Fill),
            )
            .height(Length::Fill)
            .width(Length::Fill)
            .class(theme::iced::Scrollable::Transient),
        )
        .width(Length::Fill)
        .height(Length::Fill);

        let palette = self.active_palette();
        let content = widget::container(
            self.glass_layer(
                widget::container(
                    widget::row::with_capacity(2)
                        .push(models_card)
                        .push(self.caligo_vertical_rule())
                        .push(settings_column)
                        .spacing(12)
                        .height(Length::Fill),
                )
                .padding(10)
                .height(Length::Fill)
                .into(),
                GlassVariant::Panel,
                GlassState::Rest,
            ),
        )
        .height(Length::Fill)
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        self.page_shell(
            "Model Settings",
            "Manage local models, inspect runtime metadata, and tune generation/runtime options.",
            content.into(),
        )
    }

    fn server_page(&self) -> Element<'_, Message> {
        let palette = self.active_palette();
        let connection_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(9)
                        .push(self.settings_subtitle("Connection"))
                        .push(self.soft_text_wrapped(
                            "Configure the backend URL and optional manager token.",
                        ))
                        .push(self.settings_divider_fill())
                        .push(self.settings_row(
                            "Server URL",
                            self.app_text_input(
                                "http://localhost:8000",
                                &self.native_state.settings.server_url,
                                Some(Message::ServerUrlChanged),
                                InputField::ServerUrl,
                                false,
                            ),
                        ))
                        .push(self.settings_row(
                            "Manager API Token",
                            self.app_text_input(
                                "optional-manager-token",
                                &self.native_state.settings.manager_token,
                                Some(Message::ManagerTokenChanged),
                                InputField::ManagerToken,
                                false,
                            ),
                        ))
                        .push(self.soft_text_wrapped(
                            "Only required when desktop manager auth is enabled.",
                        ))
                        .push(self.split_action_group(
                            "Reset Server",
                            Message::ResetServerSettingsPressed,
                            "Save Server",
                            Message::SaveServerSettingsPressed,
                            112.0,
                            112.0,
                        ))
                        .push(self.soft_text_wrapped(&self.status_text))
                        .spacing(10),
                )
                .padding(12)
                .width(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let palette = self.active_palette();
        let concurrent_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(7)
                        .push(self.settings_subtitle("Concurrent Models"))
                        .push(self.soft_text_wrapped(
                            "Keep additional local models running alongside the active model.",
                        ))
                        .push(self.settings_divider_fill())
                        .push(self.concurrent_models_panel())
                        .push(self.split_action_group(
                            "Stop Extras",
                            Message::StopExtrasPressed,
                            "Apply Concurrent Models",
                            Message::ApplyConcurrentPressed,
                            112.0,
                            178.0,
                        ))
                        .push(self.soft_text_wrapped(
                            "Extra models increase memory use and may reduce responsiveness.",
                        ))
                        .spacing(10),
                )
                .padding(12)
                .width(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let palette = self.active_palette();
        let dispatch_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(5)
                        .push(self.settings_subtitle("Dispatch"))
                        .push(self.soft_text_wrapped(
                            "Choose how chat requests target running models.",
                        ))
                        .push(self.settings_divider_fill())
                        .push(self.settings_subtitle("Chat Dispatch Mode"))
                        .push(self.settings_select_button_fill(
                            if self.native_state.settings.chat_dispatch_mode == "parallel" {
                                "Parallel (all running models)"
                            } else {
                                "Single (active model)"
                            },
                            Message::DispatchModeChanged(
                                if self.native_state.settings.chat_dispatch_mode == "parallel" {
                                    "single"
                                } else {
                                    "parallel"
                                }
                                .into(),
                            ),
                        ))
                        .push(self.soft_text_wrapped(
                            "Parallel mode sends the same prompt to all running models and merges replies.",
                        ))
                        .spacing(10),
                )
                .padding(12)
                .width(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let palette = self.active_palette();
        let content = widget::container(
            self.glass_layer(
                widget::container(
                    widget::row::with_capacity(2)
                        .push(
                            widget::column::with_capacity(2)
                                .push(connection_card)
                                .push(dispatch_card)
                                .spacing(12)
                                .width(Length::FillPortion(1)),
                        )
                        .push(self.caligo_vertical_rule())
                        .push(
                            widget::container(concurrent_card)
                                .width(Length::FillPortion(1))
                                .height(Length::Fill),
                        )
                        .spacing(12)
                        .height(Length::Fill),
                )
                .padding(10)
                .height(Length::Fill)
                .into(),
                GlassVariant::Panel,
                GlassState::Rest,
            ),
        )
        .height(Length::Fill)
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        self.page_shell(
            "Server Settings",
            "Configure connection, parallel model serving, and dispatch behavior.",
            content.into(),
        )
    }

    fn logs_page(&self) -> Element<'_, Message> {
        terminal_view::render(self)
    }

    fn community_page(&self) -> Element<'_, Message> {
        let palette = self.active_palette();
        let results: Element<'_, Message> = if self.community_results.is_empty() {
            self.empty_state("Search results will appear here.")
        } else {
            widget::scrollable(
                self.community_results
                    .iter()
                    .fold(
                        widget::column::with_capacity(self.community_results.len()),
                        |column, item| column.push(self.community_result_card(item)),
                    )
                    .spacing(8)
                    .width(Length::Fill),
            )
            .height(Length::Fill)
            .width(Length::Fill)
            .class(theme::iced::Scrollable::Transient)
            .into()
        };
        let job = self.community_job_card();
        let search_card = self.section_card(
            "Search",
            "Find Hugging Face and local community models.",
            widget::column::with_capacity(4)
                .push(self.small_label("Hugging Face model or repo ID"))
                .push(
                    widget::row::with_capacity(2)
                        .push(
                            widget::container(self.app_text_input(
                                "Search Hugging Face",
                                &self.community_query,
                                Some(Message::CommunityQueryChanged),
                                InputField::CommunityQuery,
                                false,
                            ))
                            .width(Length::Fill),
                        )
                        .push(
                            widget::container(
                                self.secondary_button("Search", Message::SearchCommunityPressed),
                            )
                            .width(Length::Fixed(88.0)),
                        )
                        .spacing(8)
                        .width(Length::Fill),
                )
                .push(self.soft_text_wrapped(
                    "Enter a model name or Hugging Face repo ID (for example, Qwen).",
                ))
                .spacing(8)
                .width(Length::Fill)
                .into(),
        );

        let results_card = widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(4)
                        .push(self.settings_subtitle("Search Results"))
                        .push(self.soft_text(format!("{} result(s)", self.community_results.len())))
                        .push(self.settings_divider_fill())
                        .push(results)
                        .spacing(8)
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .padding(12)
                .height(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .height(Length::Fill)
        .width(Length::Fill)
        .style(move |_| flat_frame_host_style(palette));

        let content = widget::container(
            widget::column::with_capacity(4)
                .push(search_card)
                .push(job)
                .push(results_card)
                .push(self.soft_text_wrapped(&self.status_text))
                .spacing(12)
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill);
        self.page_shell(
            "Community",
            "Search, download, deploy, and monitor community model jobs.",
            content.into(),
        )
    }

    fn settings_page(&self) -> Element<'_, Message> {
        let interface_card = self.section_card(
            "Interface",
            "Language and UI-level preferences.",
            self.settings_row(
                "Language",
                widget::row::with_capacity(2)
                    .push(self.small_toggle(
                        "English",
                        self.native_state.settings.language == "en",
                        Message::LanguageSelected("en".into()),
                    ))
                    .push(self.small_toggle(
                        "Chinese",
                        self.native_state.settings.language == "zh",
                        Message::LanguageSelected("zh".into()),
                    ))
                    .spacing(8)
                    .into(),
            ),
        );

        let storage_card = self.section_card(
            "Storage & Shortcuts",
            "Manage native UI state and jump to common configuration pages.",
            widget::column::with_capacity(3)
                .push(
                    widget::row::with_capacity(2)
                        .push(self.secondary_button(
                            "Open Model Settings",
                            Message::PagePressed(Page::Models),
                        ))
                        .push(self.secondary_button(
                            "Clear UI Storage",
                            Message::ClearStoragePressed,
                        ))
                        .spacing(8),
                )
                .push(self.soft_text(
                    "Clearing UI storage resets chat memory, model preferences, and request settings.",
                ))
                .spacing(10)
                .into(),
        );

        let ide_connections_card = self.ide_connections_card();

        let content = widget::scrollable(
            widget::column::with_capacity(3)
                .push(interface_card)
                .push(storage_card)
                .push(ide_connections_card)
                .spacing(12),
        )
        .height(Length::Fill)
        .class(theme::iced::Scrollable::Transient);

        self.page_shell(
            "App Settings",
            "Configure application-level behavior and UI storage.",
            content.into(),
        )
    }

    fn ide_connections_card(&self) -> Element<'_, Message> {
        let integration = &self.ide_connections.integration;
        let provider: Element<'_, Message> = if integration.provider_url.is_empty() {
            self.soft_text("Provider details are unavailable until the desktop backend is ready.")
        } else {
            let key = if integration.provider_api_key.is_empty() {
                "Unavailable (open the desktop App locally)".to_string()
            } else {
                integration.provider_api_key.clone()
            };
            let mcp_command = std::iter::once(integration.mcp_command.as_str())
                .chain(integration.mcp_args.iter().map(String::as_str))
                .collect::<Vec<_>>()
                .join(" ");
            self.card_block(
                "AI Assistant Provider",
                widget::column::with_capacity(11)
                    .push(self.soft_text("Enter these values in JetBrains AI Assistant's native OpenAI-compatible Provider settings."))
                    .push(self.settings_row_fixed(
                        "Base URL",
                        self.soft_text(&integration.provider_url),
                        420.0,
                    ))
                    .push(self.settings_row_fixed(
                        "Model",
                        self.soft_text(if integration.provider_model.is_empty() {
                            "Start a local model first"
                        } else {
                            &integration.provider_model
                        }),
                        420.0,
                    ))
                    .push(self.settings_row_fixed(
                        "Provider Key",
                        self.soft_text(&key),
                        420.0,
                    ))
                    .push(
                        widget::row::with_capacity(2)
                            .push(self.secondary_button(
                                "Copy Base URL",
                                Message::CopyJetBrainsProviderUrlPressed,
                            ))
                            .push(self.secondary_button(
                                "Copy Provider Key",
                                Message::CopyJetBrainsProviderKeyPressed,
                            ))
                            .spacing(8),
                    )
                    .push(self.soft_text(
                        if integration.completion_mode == "experimental_reuse_chat_model" {
                            "Completion: experimental reuse of the chat model; native FIM is attempted when the tokenizer supports it."
                        } else {
                            "Completion: configure the local model before testing IDE edit prediction."
                        },
                    ))
                    .push(self.settings_row_fixed(
                        "MCP command",
                        self.soft_text(if mcp_command.is_empty() {
                            "Unavailable (open the desktop App locally)"
                        } else {
                            &mcp_command
                        }),
                        420.0,
                    ))
                    .push(self.soft_text(
                        "Register this command as the native token-workshed-mcp MCP server. The App keeps its bearer credential, server policy, approvals, and audit log locally.",
                    ))
                    .push(self.secondary_button(
                        "Copy MCP Command",
                        Message::CopyJetBrainsMcpCommandPressed,
                    ))
                    .spacing(8)
                    .into(),
            )
        };
        let pending: Element<'_, Message> = if self.ide_connections.pairings.is_empty() {
            self.soft_text("No pending IDE pairing requests.")
        } else {
            self.ide_connections
                .pairings
                .iter()
                .fold(
                    widget::column::with_capacity(self.ide_connections.pairings.len()),
                    |column, pairing| {
                        let label = format!(
                            "{} · {} · plugin {}",
                            empty_dash(&pairing.client_name),
                            empty_dash(&pairing.ide_name),
                            empty_dash(&pairing.plugin_version),
                        );
                        let pending_since = if pairing.created_at > 0 {
                            format!("Requested at {}", pairing.created_at)
                        } else {
                            "Requested now".into()
                        };
                        let requested_scopes = if pairing.requested_scopes.is_empty() {
                            "Code assistance and model access".to_string()
                        } else {
                            format!(
                                "Requested (sensitive scopes require the explicit all button):\n• {}",
                                pairing.requested_scopes.join("\n• ")
                            )
                        };
                        column.push(
                            self.card_block(
                                label,
                                widget::column::with_capacity(2)
                                    .push(self.soft_text(pending_since))
                                    .push(self.soft_text(requested_scopes))
                                    .push(
                                            widget::row::with_capacity(3)
                                            .push(self.danger_button(
                                                "Reject",
                                                Message::RejectIdePairingPressed(
                                                    pairing.pairing_id.clone(),
                                                ),
                                            ))
                                            .push(self.primary_button(
                                                "Approve safe",
                                                Message::ApproveIdePairingPressed(
                                                    pairing.pairing_id.clone(),
                                                ),
                                            ))
                                            .push(self.secondary_button(
                                                "Approve all requested",
                                                Message::ApproveIdePairingAllPressed(
                                                    pairing.pairing_id.clone(),
                                                ),
                                            ))
                                            .spacing(8),
                                    )
                                    .spacing(8)
                                    .into(),
                            ),
                        )
                    },
                )
                .spacing(8)
                .into()
        };

        let authorized: Element<'_, Message> = if self.ide_connections.authorizations.is_empty() {
            self.soft_text("No JetBrains IDE is authorized yet.")
        } else {
            self.ide_connections
                .authorizations
                .iter()
                .fold(
                    widget::column::with_capacity(self.ide_connections.authorizations.len()),
                    |column, authorization| {
                        let label = format!(
                            "{} · {} · {} capability(s)",
                            empty_dash(&authorization.client_name),
                            empty_dash(&authorization.ide_name),
                            authorization.scopes.len(),
                        );
                        let approved_at = if authorization.approved_at > 0 {
                            format!("Approved at {}", authorization.approved_at)
                        } else {
                            "Approved".into()
                        };
                        column.push(
                            self.card_block(
                                label,
                                widget::column::with_capacity(2)
                                    .push(self.soft_text(approved_at))
                                    .push(self.danger_button(
                                        "Revoke",
                                        Message::RevokeIdeAuthorizationPressed(
                                            authorization.client_id.clone(),
                                        ),
                                    ))
                                    .spacing(8)
                                    .into(),
                            ),
                        )
                    },
                )
                .spacing(8)
                .into()
        };

        self.section_card(
            "JetBrains Connections",
            "Approve local IDE pairing here. Requested Agent/session scopes are shown before approval; manager and Hermes credentials are never shared.",
            widget::column::with_capacity(6)
                .push(self.settings_subtitle("Native AI Assistant setup"))
                .push(provider)
                .push(self.settings_subtitle("Pending Requests"))
                .push(pending)
                .push(self.settings_divider_fill())
                .push(self.settings_subtitle("Authorized IDEs"))
                .push(authorized)
                .spacing(8)
                .into(),
        )
    }

    fn about_page(&self) -> Element<'_, Message> {
        let native_version = env!("CARGO_PKG_VERSION");
        let token_version = empty_dash(&self.token_workshed_version);
        let vllm_version = empty_dash(&self.vllm_mlx_version);
        let selected_model = empty_dash(&self.native_state.selected_model);
        let about_card = self.section_card(
            "token-workshed",
            "Native Rust UI for local vllm-mlx serving, chat, model management, and community deployment.",
            widget::column::with_capacity(8)
                .push(self.dashboard_row("token-workshed", token_version))
                .push(self.dashboard_row("native-ui", native_version))
                .push(self.dashboard_row("vllm-mlx", vllm_version))
                .push(self.dashboard_row("Backend", "vllm-mlx desktop manager"))
                .push(self.dashboard_row("Active Model", selected_model))
                .push(self.dashboard_row("Developer", "Taylor He"))
                .spacing(8)
                .into(),
        );
        self.page_shell(
            "About",
            "Version, runtime, and project information.",
            widget::container(about_card)
                .width(Length::Fill)
                .height(Length::Shrink)
                .into(),
        )
    }

    fn page_frame<'a>(&self, content: Element<'a, Message>) -> Element<'a, Message> {
        widget::container(content)
            .padding(PAGE_FRAME_INSET)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn page_shell<'a, T: ToString, U: ToString>(
        &'a self,
        title: T,
        subtitle: U,
        content: Element<'a, Message>,
    ) -> Element<'a, Message> {
        self.page_frame(
            widget::column::with_capacity(2)
                .push(
                    widget::column::with_capacity(2)
                        .push(self.page_title(title))
                        .push(self.soft_text(subtitle))
                        .spacing(3)
                        .width(Length::Fill),
                )
                .push(widget::container(content).width(Length::Fill))
                .spacing(PAGE_SECTION_GAP)
                .width(Length::Fill)
                .height(Length::Fill)
                .into(),
        )
    }

    fn section_card<'a, T: ToString, U: ToString>(
        &'a self,
        title: T,
        subtitle: U,
        content: Element<'a, Message>,
    ) -> Element<'a, Message> {
        let palette = self.active_palette();
        widget::container(
            self.glass_layer(
                widget::container(
                    widget::column::with_capacity(4)
                        .push(self.settings_subtitle(title))
                        .push(self.soft_text(subtitle))
                        .push(self.caligo_horizontal_rule(false))
                        .push(widget::container(content).width(Length::Fill))
                        .spacing(8)
                        .width(Length::Fill),
                )
                .padding(12)
                .width(Length::Fill)
                .into(),
                GlassVariant::Card,
                GlassState::Rest,
            ),
        )
        .width(Length::Fill)
        .style(move |_| flat_section_style(palette))
        .into()
    }

    fn empty_state<T: ToString>(&self, text: T) -> Element<'_, Message> {
        widget::container(self.soft_text_wrapped(text))
            .padding(14)
            .width(Length::Fill)
            .height(Length::Fixed(64.0))
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(|_| transparent_panel_style())
            .into()
    }

    fn card_page<'a>(
        &'a self,
        content: widget::Column<'a, Message, Theme, Renderer>,
    ) -> Element<'a, Message> {
        let palette = self.active_palette();
        widget::container(content.width(Length::Fill).height(Length::Fill))
            .padding(12)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_| palette.card_style(true))
            .into()
    }

    fn dashboard_panel(&self) -> Element<'_, Message> {
        let success_rate = if self.request_count == 0 {
            "-".into()
        } else {
            format!(
                "{:.1}%",
                (self.success_count as f64 / self.request_count as f64) * 100.0
            )
        };
        let latency = self
            .last_latency_ms
            .map(|v| format!("{v:.0} ms"))
            .unwrap_or_else(|| "-".into());
        let tps = self
            .last_tps
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "-".into());
        let rows = [
            (
                "Requests",
                format!(
                    "{} (ok {} / err {})",
                    self.request_count, self.success_count, self.error_count
                ),
            ),
            ("Success Rate", success_rate),
            ("Avg Latency", latency),
            ("Tokens/s", tps),
            ("Total Tokens", format!("{} (p 0 / c 0)", self.total_tokens)),
            (
                "Model",
                fitted_model_label_for_width(
                    empty_dash(&self.native_state.selected_model).into(),
                    30,
                ),
            ),
        ];
        let column = rows.iter().fold(
            widget::column::with_capacity(rows.len()),
            |column, (label, value)| column.push(self.dashboard_row(label, value)),
        );
        widget::column::with_capacity(2)
            .push(
                widget::row::with_capacity(2)
                    .push(widget::container(self.history_title("Dashboard")).width(Length::Fill))
                    .push(
                        widget::container(
                            self.small_button("Refresh", Message::PollTick(Instant::now())),
                        )
                        .width(Length::Fixed(82.0)),
                    )
                    .spacing(8)
                    .width(Length::Fill)
                    .align_y(Alignment::Center),
            )
            .push(widget::container(column.spacing(5)).width(Length::Fill))
            .spacing(8)
            .width(Length::Fill)
            .into()
    }

    fn history_panel(&self) -> Element<'_, Message> {
        let list = self.native_state.conversations.iter().fold(
            widget::column::with_capacity(self.native_state.conversations.len()),
            |column, conversation| {
                let active = conversation.id == self.native_state.active_conversation_id;
                column.push(self.history_conversation_row(
                    &conversation.id,
                    &conversation.title,
                    active,
                ))
            },
        );
        widget::column::with_capacity(3)
            .push(
                widget::row::with_capacity(4)
                    .push(
                        widget::container(self.history_title("History Conversations"))
                            .width(Length::Fill),
                    )
                    .push(self.beta_tag())
                    .push(self.new_conversation_button())
                    .spacing(8)
                    .width(Length::Fill)
                    .align_y(Alignment::Center),
            )
            .push(
                widget::scrollable(list.spacing(6))
                    .height(Length::Fill)
                    .width(Length::Fill)
                    .class(theme::iced::Scrollable::Transient),
            )
            .spacing(8)
            .height(Length::Fill)
            .width(Length::Fill)
            .into()
    }

    fn card_block<'a, T: ToString>(
        &'a self,
        title: T,
        content: Element<'a, Message>,
    ) -> Element<'a, Message> {
        let palette = self.active_palette();
        widget::container(
            widget::column::with_capacity(2)
                .push(self.label_text(title))
                .push(content)
                .spacing(8),
        )
        .padding(10)
        .width(Length::Fill)
        .style(move |_| palette.card_style(true))
        .into()
    }

    fn chat_bubble<'a>(&'a self, role: &'a str, content: &'a str) -> Element<'a, Message> {
        let is_user = role == "user";
        let is_thinking = role == "thinking";
        let is_thinking_summary = is_thinking && content.trim_start().starts_with("thought for ");
        let text = if content.trim().is_empty() {
            "..."
        } else {
            content
        };
        let body: Element<'a, Message> = if is_thinking {
            if is_thinking_summary {
                widget::text(text)
                    .font(Font::MONOSPACE)
                    .size(11)
                    .wrapping(text::Wrapping::None)
                    .into()
            } else {
                widget::column::with_capacity(2)
                    .push(
                        widget::text("thinking")
                            .font(Font::MONOSPACE)
                            .size(11)
                            .wrapping(text::Wrapping::None),
                    )
                    .push(self.markdown_content(text, false))
                    .spacing(5)
                    .into()
            }
        } else {
            self.markdown_content(text, is_user)
        };
        let bubble = widget::container(body)
            .padding([8, 10])
            .width(Length::Shrink)
            .max_width(720)
            .style(move |_| {
                if is_thinking {
                    thinking_bubble_style()
                } else {
                    chat_bubble_style(is_user)
                }
            });

        let row = if is_user {
            widget::row::with_capacity(2)
                .push(widget::space().width(Length::Fill))
                .push(bubble)
        } else {
            widget::row::with_capacity(2)
                .push(bubble)
                .push(widget::space().width(Length::Fill))
        };

        row.width(Length::Fill).into()
    }

    fn markdown_content<'a>(&'a self, content: &'a str, is_user: bool) -> Element<'a, Message> {
        let blocks = parse_markdown_blocks(content);
        let column = blocks.into_iter().fold(
            widget::column::with_capacity(8).spacing(5),
            |column, block| column.push(self.markdown_block(block, is_user)),
        );
        column.into()
    }

    fn markdown_block(&self, block: MarkdownBlock, is_user: bool) -> Element<'_, Message> {
        match block {
            MarkdownBlock::Paragraph(text) => widget::text(clean_inline_markdown(&text))
                .size(13)
                .wrapping(text::Wrapping::WordOrGlyph)
                .into(),
            MarkdownBlock::Heading { level, text } => {
                let size = match level {
                    1 => 18,
                    2 => 16,
                    _ => 14,
                };
                widget::text(clean_inline_markdown(&text))
                    .size(size)
                    .wrapping(text::Wrapping::WordOrGlyph)
                    .into()
            }
            MarkdownBlock::Bullet(text) => widget::row::with_capacity(2)
                .push(widget::text("•").size(13))
                .push(
                    widget::text(clean_inline_markdown(&text))
                        .size(13)
                        .width(Length::Fill)
                        .wrapping(text::Wrapping::WordOrGlyph),
                )
                .spacing(6)
                .align_y(Alignment::Start)
                .into(),
            MarkdownBlock::Ordered { number, text } => widget::row::with_capacity(2)
                .push(widget::text(format!("{number}.")).size(13))
                .push(
                    widget::text(clean_inline_markdown(&text))
                        .size(13)
                        .width(Length::Fill)
                        .wrapping(text::Wrapping::WordOrGlyph),
                )
                .spacing(6)
                .align_y(Alignment::Start)
                .into(),
            MarkdownBlock::Quote(text) => widget::container(
                widget::text(clean_inline_markdown(&text))
                    .size(13)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph),
            )
            .padding([5, 8])
            .width(Length::Fill)
            .style(move |_| markdown_quote_style(is_user))
            .into(),
            MarkdownBlock::Code(code) => widget::container(
                widget::text(code)
                    .font(Font::MONOSPACE)
                    .size(12)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::WordOrGlyph),
            )
            .padding([7, 8])
            .width(Length::Fill)
            .style(move |_| markdown_code_style(is_user))
            .into(),
            MarkdownBlock::Space => widget::space().height(Length::Fixed(3.0)).into(),
        }
    }

    fn model_row<'a>(&'a self, model: &'a str, selected: bool) -> Element<'a, Message> {
        let active =
            self.manager_models.active_model == model || self.native_state.selected_model == model;
        let label = fitted_model_label(model.to_string());
        let label_size = fitted_model_label_size(&label);
        let marker = widget::container(widget::text(""))
            .width(Length::Fixed(3.0))
            .height(Length::Fill)
            .style(move |_| model_marker_style(selected));
        let body = widget::column::with_capacity(2)
            .push(
                widget::text(label)
                    .size(label_size)
                    .wrapping(text::Wrapping::None),
            )
            .push(widget::text(if active { "active" } else { "local" }).size(11))
            .spacing(4)
            .width(Length::Fill);
        let item = iced::widget::button(
            widget::row::with_capacity(2)
                .push(marker)
                .push(body)
                .spacing(8)
                .align_y(Alignment::Center),
        )
        .padding([8, 10])
        .width(Length::Fill)
        .class(iced_button_class(model_item_button_style(
            self.active_palette(),
            selected,
            active,
        )))
        .on_press(Message::SelectLocalModelPressed(model.to_string()));

        item.into()
    }

    fn runtime_checks(&self) -> Element<'_, Message> {
        widget::column::with_capacity(4)
            .push(self.check_line(
                "Continuous Batching",
                self.runtime_config.continuous_batching,
                RuntimeBoolField::ContinuousBatching,
            ))
            .push(self.check_line(
                "Paged Cache",
                self.runtime_config.use_paged_cache,
                RuntimeBoolField::PagedCache,
            ))
            .push(self.check_line(
                "KV Cache Quantization",
                self.runtime_config.kv_cache_quantization,
                RuntimeBoolField::KvCacheQuantization,
            ))
            .push(self.check_line(
                "MTP",
                self.runtime_config.enable_mtp,
                RuntimeBoolField::EnableMtp,
            ))
            .spacing(8)
            .into()
    }

    fn runtime_numbers(&self) -> Element<'_, Message> {
        widget::column::with_capacity(2)
            .push(
                self.settings_row(
                    "Chunked Prefill Tokens",
                    widget::container(self.app_text_input(
                        "0",
                        &self.runtime_config.chunked_prefill_tokens,
                        Some(|v| Message::RuntimeTextChanged(RuntimeTextField::ChunkedPrefill, v)),
                        InputField::RuntimeChunkedPrefill,
                        false,
                    ))
                    .width(Length::Fixed(150.0))
                    .into(),
                ),
            )
            .push(
                self.settings_row(
                    "MTP Draft Tokens",
                    widget::container(self.app_text_input(
                        "1",
                        &self.runtime_config.mtp_draft_tokens,
                        Some(|v| Message::RuntimeTextChanged(RuntimeTextField::MtpDraft, v)),
                        InputField::RuntimeMtpDraft,
                        false,
                    ))
                    .width(Length::Fixed(150.0))
                    .into(),
                ),
            )
            .spacing(8)
            .into()
    }

    fn check_line<'a>(
        &'a self,
        label: &'a str,
        value: bool,
        field: RuntimeBoolField,
    ) -> Element<'a, Message> {
        iced::widget::button(
            widget::row::with_capacity(2)
                .push(
                    widget::container(widget::text(""))
                        .width(Length::Fixed(18.0))
                        .height(Length::Fixed(18.0))
                        .style(move |_| check_dot_style(value)),
                )
                .push(
                    widget::container(widget::text(label).size(13))
                        .width(Length::Fill)
                        .style(|_| {
                            container::Style::default().color(Color::from_rgb8(0x0f, 0x17, 0x2a))
                        }),
                )
                .spacing(11)
                .width(Length::Fill)
                .align_y(Alignment::Center),
        )
        .padding([3, 0])
        .width(Length::Fill)
        .class(iced_button_class(transparent_text_button_style()))
        .on_press(Message::RuntimeBoolChanged(field, !value))
        .into()
    }

    fn concurrent_models_panel(&self) -> Element<'_, Message> {
        let list = self.models.iter().fold(
            widget::column::with_capacity(self.models.len()),
            |column, model| {
                let checked = self
                    .selected_concurrent_models
                    .iter()
                    .any(|item| item == model);
                let prefix = if checked { "[x]" } else { "[ ]" };
                column.push(self.concurrent_model_button(
                    format!("{prefix} {model}"),
                    Message::ToggleConcurrentModel(model.clone()),
                ))
            },
        );
        widget::container(
            widget::scrollable(list.spacing(8))
                .width(Length::Fill)
                .height(Length::Fixed(150.0))
                .class(theme::iced::Scrollable::Transient),
        )
        .padding([4, 3])
        .width(Length::Fill)
        .style(|_| concurrent_list_style())
        .into()
    }

    fn concurrent_model_button(&self, label: String, message: Message) -> Element<'_, Message> {
        let content = widget::container(
            widget::text(label)
                .size(12)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
        )
        .width(Length::Fill)
        .align_x(Alignment::Center)
        .align_y(Alignment::Center);
        iced::widget::button(content)
            .padding([9, 8])
            .width(Length::Fill)
            .height(Length::Shrink)
            .class(iced_button_class(secondary_button_style(
                self.active_palette(),
                self.accent(),
                false,
            )))
            .on_press(message)
            .into()
    }

    fn concurrent_models_empty_panel(&self) -> Element<'_, Message> {
        widget::container(self.soft_text("No local models."))
            .padding([8, 4])
            .style(|_| concurrent_list_style())
            .into()
    }

    fn community_job_card(&self) -> Element<'_, Message> {
        let job = self
            .community_job
            .as_ref()
            .filter(|job| community_job_is_valid(job));
        let status = job
            .map(job_summary)
            .unwrap_or_else(|| "No active job.".into());
        let mut body = widget::column::with_capacity(4)
            .push(self.settings_subtitle("Deploy Job"))
            .push(self.soft_text_wrapped(&status));
        if let Some(job) = job {
            let actions = community_job_allowed_actions(job);
            if !actions.is_empty() {
                let controls = actions.iter().fold(
                    widget::row::with_capacity(actions.len())
                        .spacing(8)
                        .width(Length::Fill),
                    |row, action| {
                        let message = Message::CommunityJobActionPressed((*action).into());
                        row.push(match *action {
                            "pause" => self.secondary_button("Pause", message),
                            "resume" => self.secondary_button("Resume", message),
                            _ => self.danger_button("Cancel", message),
                        })
                    },
                );
                body = body.push(controls);
            }
        } else {
            body = body
                .push(self.empty_state("Download and deploy a model to see job progress here."));
        }
        self.aluminum_frame(
            widget::container(body.spacing(8).width(Length::Fill))
                .padding(10)
                .width(Length::Fill)
                .into(),
        )
    }

    fn hf_binding_card(&self) -> Element<'_, Message> {
        self.aluminum_frame(
            widget::container(
                widget::column::with_capacity(4)
                    .push(self.page_title("Bind Hugging Face Account"))
                    .push(self.settings_row(
                        "Username",
                        self.app_text_input(
                            "your-hf-username",
                            &self.hf_username,
                            Some(Message::HfUsernameChanged),
                            InputField::HfUsername,
                            false,
                        ),
                    ))
                    .push(self.settings_row(
                        "Access Token",
                        self.app_text_input(
                            "hf_xxx...",
                            &self.hf_token,
                            Some(Message::HfTokenChanged),
                            InputField::HfToken,
                            false,
                        ),
                    ))
                    .push(self.primary_button("Save", Message::SaveHfBindingPressed))
                    .spacing(8),
            )
            .padding(10)
            .into(),
        )
    }

    fn community_result_card<'a>(&'a self, item: &'a Value) -> Element<'a, Message> {
        let id = string_at(item, "id")
            .if_empty_then(|| string_at(item, "model_id"))
            .if_empty_then(|| string_at(item, "repo_id"));
        let source = string_at(item, "source").if_empty_then(|| "community".into());
        let local = item.get("local").and_then(Value::as_bool).unwrap_or(false);
        let downloads = item.get("downloads").and_then(Value::as_u64).unwrap_or(0);
        let likes = item.get("likes").and_then(Value::as_u64).unwrap_or(0);
        let source_chip = if local { "Local" } else { source.as_str() };
        widget::container(
            widget::column::with_capacity(4)
                .push(
                    widget::row::with_capacity(2)
                        .push(
                            widget::row::with_capacity(2)
                                .push(
                                    widget::container(widget::text("HF").size(9))
                                        .width(Length::Fixed(24.0))
                                        .height(Length::Fixed(24.0))
                                        .align_x(Alignment::Center)
                                        .align_y(Alignment::Center)
                                        .style(|_| community_logo_style()),
                                )
                                .push(
                                    widget::container(
                                        widget::text(id.clone())
                                            .size(13)
                                            .wrapping(text::Wrapping::WordOrGlyph),
                                    )
                                    .width(Length::Fill),
                                )
                                .spacing(8)
                                .align_y(Alignment::Center)
                                .width(Length::Fill),
                        )
                        .push(self.community_chip(source_chip))
                        .spacing(8)
                        .align_y(Alignment::Center)
                        .width(Length::Fill),
                )
                .push(
                    widget::row::with_capacity(2)
                        .push(self.community_chip(format!("downloads {downloads}")))
                        .push(self.community_chip(format!("likes {likes}")))
                        .spacing(6),
                )
                .push(
                    widget::row::with_capacity(2)
                        .push(
                            widget::container(self.secondary_button(
                                "Switch Now",
                                Message::SwitchModelPressed(id.clone()),
                            ))
                            .width(Length::FillPortion(1)),
                        )
                        .push(
                            widget::container(self.primary_button(
                                "Download & Deploy",
                                Message::DeployCommunityPressed(id.clone()),
                            ))
                            .width(Length::FillPortion(1)),
                        )
                        .spacing(8)
                        .width(Length::Fill),
                )
                .spacing(8)
                .width(Length::Fill),
        )
        .padding(8)
        .width(Length::Fill)
        .style(|_| community_card_style())
        .into()
    }

    fn settings_row<'a, T: ToString>(
        &'a self,
        label: T,
        control: Element<'a, Message>,
    ) -> Element<'a, Message> {
        widget::row::with_capacity(2)
            .push(widget::container(self.label_text(label)).width(Length::Fixed(132.0)))
            .push(widget::container(control).width(Length::Fill))
            .spacing(10)
            .width(Length::Fill)
            .align_y(Alignment::Center)
            .into()
    }

    fn settings_row_fixed<'a, T: ToString>(
        &'a self,
        label: T,
        control: Element<'a, Message>,
        width: f32,
    ) -> Element<'a, Message> {
        widget::row::with_capacity(3)
            .push(widget::container(self.label_text(label)).width(Length::Fixed(132.0)))
            .push(widget::container(control).width(Length::Fixed(width)))
            .push(widget::space().width(Length::Fill))
            .spacing(10)
            .width(Length::Fill)
            .align_y(Alignment::Center)
            .into()
    }

    fn section_header<'a, T: ToString, U: ToString>(
        &'a self,
        title: T,
        meta: U,
    ) -> Element<'a, Message> {
        widget::row::with_capacity(2)
            .push(widget::container(self.page_title(title)).width(Length::Fill))
            .push(self.soft_pill(meta))
            .align_y(Alignment::Center)
            .width(Length::Fill)
            .into()
    }

    fn settings_subtitle<T: ToString>(&self, text: T) -> Element<'_, Message> {
        widget::container(widget::text(text.to_string()).size(13))
            .style(|_| container::Style::default().color(Color::from_rgb8(0x1f, 0x29, 0x37)))
            .into()
    }

    fn models_count_pill(&self, count: usize) -> Element<'_, Message> {
        widget::container(widget::text(count.to_string()).size(11))
            .padding([4, 9])
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(|_| models_count_style())
            .into()
    }

    fn settings_divider(&self) -> Element<'_, Message> {
        self.caligo_horizontal_rule(false)
    }

    fn settings_divider_fill(&self) -> Element<'_, Message> {
        self.caligo_horizontal_rule(false)
    }

    fn caligo_horizontal_rule(&self, top_edge: bool) -> Element<'static, Message> {
        let shade = rgba(0x6f, 0x78, 0x85, SIDEBAR_ALUMINUM_SHADE_ALPHA);
        let core = rgba(0xb9, 0xc1, 0xcb, SIDEBAR_ALUMINUM_CORE_ALPHA);
        let highlight = rgba(0xff, 0xff, 0xff, SIDEBAR_ALUMINUM_HIGHLIGHT_ALPHA);
        let (first, second, third) = if top_edge {
            (highlight, core, shade)
        } else {
            (shade, core, highlight)
        };

        widget::column::with_capacity(3)
            .push(caligo_horizontal_bar(first))
            .push(caligo_horizontal_bar(second))
            .push(caligo_horizontal_bar(third))
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fixed(NUMBER_ALUMINUM_RULE_HEIGHT))
            .into()
    }

    fn caligo_vertical_rule(&self) -> Element<'static, Message> {
        let shade = rgba(0x6f, 0x78, 0x85, SIDEBAR_ALUMINUM_SHADE_ALPHA);
        let core = rgba(0xb9, 0xc1, 0xcb, SIDEBAR_ALUMINUM_CORE_ALPHA);
        let highlight = rgba(0xff, 0xff, 0xff, SIDEBAR_ALUMINUM_HIGHLIGHT_ALPHA);
        let rail = widget::row::with_capacity(3)
            .push(self.aluminum_sidebar_bar(shade))
            .push(self.aluminum_sidebar_bar(core))
            .push(self.aluminum_sidebar_bar(highlight))
            .spacing(0)
            .width(Length::Fixed(ALUMINUM_FRAME_WIDTH))
            .height(Length::Fill);

        widget::container(
            widget::column::with_capacity(3)
                .push(widget::space().height(Length::Fixed(SECTION_DIVIDER_INSET)))
                .push(rail)
                .push(widget::space().height(Length::Fixed(SECTION_DIVIDER_INSET))),
        )
        .width(Length::Fixed(ALUMINUM_FRAME_WIDTH))
        .height(Length::Fill)
        .into()
    }

    fn kv_row<'a, T: ToString, U: ToString>(&'a self, label: T, value: U) -> Element<'a, Message> {
        widget::row::with_capacity(2)
            .push(widget::container(self.soft_text(label)).width(Length::FillPortion(1)))
            .push(widget::container(self.label_text(value)).width(Length::FillPortion(1)))
            .align_y(Alignment::Center)
            .into()
    }

    fn dashboard_row<'a, T: ToString, U: ToString>(
        &'a self,
        label: T,
        value: U,
    ) -> Element<'a, Message> {
        widget::container(
            widget::row::with_capacity(2)
                .push(
                    widget::container(widget::text(label.to_string()).size(12))
                        .width(Length::FillPortion(5)),
                )
                .push(
                    widget::container(
                        widget::text(value.to_string())
                            .size(12)
                            .wrapping(text::Wrapping::Word),
                    )
                    .width(Length::FillPortion(7))
                    .align_x(Alignment::End),
                )
                .spacing(8)
                .align_y(Alignment::Center),
        )
        .height(Length::Fixed(28.0))
        .padding([4, 8])
        .style(|_| dashboard_row_style())
        .into()
    }

    fn history_conversation_row<'a>(
        &'a self,
        id: &'a str,
        title: &'a str,
        active: bool,
    ) -> Element<'a, Message> {
        let label = if active {
            format!("> {title}")
        } else {
            title.to_string()
        };
        widget::container(
            widget::row::with_capacity(2)
                .push(
                    iced::widget::button(
                        widget::container(widget::text(label).size(12))
                            .width(Length::Fill)
                            .align_y(Alignment::Center),
                    )
                    .padding(0)
                    .width(Length::Fill)
                    .height(Length::Fixed(30.0))
                    .class(iced_button_class(history_title_button_style()))
                    .on_press(Message::SelectConversationPressed(id.to_string())),
                )
                .push(
                    iced::widget::button(
                        widget::container(widget::text("x").size(12))
                            .width(Length::Fill)
                            .height(Length::Fill)
                            .align_x(Alignment::Center)
                            .align_y(Alignment::Center),
                    )
                    .padding(0)
                    .width(Length::Fixed(22.0))
                    .height(Length::Fixed(22.0))
                    .class(iced_button_class(history_delete_button_style()))
                    .on_press(Message::DeleteConversationPressed(id.to_string())),
                )
                .spacing(8)
                .align_y(Alignment::Center),
        )
        .height(Length::Fixed(42.0))
        .padding([6, 8])
        .width(Length::Fill)
        .style(move |_| history_item_style(active))
        .into()
    }

    fn history_title<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(widget::text(text.to_string()).size(14))
            .style(move |_| container::Style::default().color(palette.text_main))
            .into()
    }

    fn beta_tag(&self) -> Element<'_, Message> {
        widget::container(widget::text("BETA").size(10))
            .height(Length::Fixed(18.0))
            .padding([0, 7])
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(|_| beta_tag_style())
            .into()
    }

    fn community_chip<T: ToString>(&self, text: T) -> Element<'_, Message> {
        widget::container(
            widget::text(text.to_string())
                .size(11)
                .wrapping(text::Wrapping::WordOrGlyph),
        )
        .padding([3, 8])
        .max_width(180.0)
        .align_x(Alignment::Center)
        .align_y(Alignment::Center)
        .style(|_| community_chip_style())
        .into()
    }

    fn prompt_dot(&self, color: Color) -> Element<'_, Message> {
        widget::container(widget::text(""))
            .width(Length::Fixed(7.0))
            .height(Length::Fixed(7.0))
            .style(move |_| prompt_dot_style(color))
            .into()
    }

    fn terminal_dot(&self, color: Color) -> Element<'_, Message> {
        widget::container(widget::text(""))
            .width(Length::Fixed(12.0))
            .height(Length::Fixed(12.0))
            .style(move |_| prompt_dot_style(color))
            .into()
    }

    fn new_conversation_button(&self) -> Element<'_, Message> {
        iced::widget::button(
            widget::container(widget::text("+").size(18))
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(Alignment::Center)
                .align_y(Alignment::Center),
        )
        .padding(0)
        .width(Length::Fixed(34.0))
        .height(Length::Fixed(30.0))
        .class(iced_button_class(new_conversation_button_style()))
        .on_press(Message::NewConversationPressed)
        .into()
    }

    fn soft_pill<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(widget::text(text.to_string()).size(12))
            .padding([8, 10])
            .style(move |_| palette.card_style(false))
            .into()
    }

    fn chat_model_picker(&self) -> Element<'_, Message> {
        let header = widget::row::with_capacity(2)
            .push(widget::container(self.small_label("Models")).width(Length::Fixed(48.0)))
            .push(self.model_select_button(
                empty_dash(&self.native_state.selected_model),
                Message::ChatModelMenuToggled,
            ))
            .spacing(6)
            .align_y(Alignment::Center)
            .width(Length::Fill);

        widget::container(header)
            .width(Length::Fixed(270.0))
            .height(Length::Fill)
            .align_y(Alignment::Center)
            .into()
    }

    fn chat_model_dropdown_overlay(&self) -> Element<'_, Message> {
        widget::container(
            widget::column::with_capacity(2)
                .push(widget::space().height(Length::Fixed(44.0)))
                .push(
                    widget::row::with_capacity(2)
                        .push(widget::space().width(Length::Fill))
                        .push(self.chat_model_dropdown_panel())
                        .align_y(Alignment::Start),
                ),
        )
        .padding([0, 10, 0, 10])
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn chat_model_dropdown_panel(&self) -> Element<'_, Message> {
        let options = self.chat_model_options();
        let selected_model = self.native_state.selected_model.as_str();
        let list =
            if options.is_empty() {
                widget::column::with_capacity(1).push(
                    widget::container(self.soft_text("No local models found."))
                        .padding([7, 8])
                        .width(Length::Fill),
                )
            } else {
                options.iter().fold(
                    widget::column::with_capacity(options.len()),
                    |column, model| {
                        column.push(self.chat_model_option_button(
                            model.clone(),
                            model.as_str() == selected_model,
                        ))
                    },
                )
            };
        let height = if options.is_empty() {
            38.0
        } else {
            options.len().min(5) as f32 * 34.0 + 8.0
        };
        let palette = self.active_palette();

        widget::container(
            self.glass_layer(
                widget::container(
                    widget::scrollable(list.spacing(4))
                        .height(Length::Fixed(height))
                        .width(Length::Fill)
                        .class(theme::iced::Scrollable::Transient),
                )
                .padding(4)
                .height(Length::Fill)
                .into(),
                GlassVariant::Floating,
                GlassState::Rest,
            ),
        )
        .width(Length::Fixed(270.0))
        .height(Length::Fixed(height + 8.0))
        .style(move |_| chat_model_dropdown_style(palette))
        .into()
    }

    fn chat_model_options(&self) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();
        for model in self
            .models
            .iter()
            .chain(self.manager_models.available_models.iter())
            .chain(std::iter::once(&self.manager_models.active_model))
            .chain(std::iter::once(&self.native_state.selected_model))
        {
            let model = model.trim();
            if !model.is_empty() && !options.iter().any(|item| item.as_str() == model) {
                options.push(model.to_string());
            }
        }
        options
    }

    fn chat_model_option_button(&self, model: String, selected: bool) -> Element<'_, Message> {
        let label = fitted_model_label_for_width(model.clone(), 30);
        let size = fitted_model_label_size(&label);
        iced::widget::button(
            widget::container(
                widget::text(label)
                    .size(size)
                    .width(Length::Fill)
                    .wrapping(text::Wrapping::None),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .align_y(Alignment::Center),
        )
        .padding([6, 8])
        .height(Length::Fixed(30.0))
        .width(Length::Fill)
        .class(iced_button_class(chat_model_option_style(selected)))
        .on_press(Message::ChatModelOptionPressed(model))
        .into()
    }

    fn model_select_chip<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(widget::text(text.to_string()).size(12))
            .padding([5, 9])
            .height(Length::Fixed(30.0))
            .align_y(Alignment::Center)
            .style(move |_| input_shell_style(palette, false))
            .into()
    }

    fn model_select_button<T: ToString>(&self, text: T, message: Message) -> Element<'_, Message> {
        let value = fitted_model_label_for_width(text.to_string(), 23);
        let size = fitted_model_label_size(&value);
        let disclosure = widget::icon::from_name("pan-down-symbolic")
            .size(12)
            .icon()
            .class(theme::Svg::custom(|_| cosmic::iced::widget::svg::Style {
                color: Some(Color::from_rgb8(0x11, 0x11, 0x11)),
            }));
        iced::widget::button(
            widget::container(
                widget::row::with_capacity(2)
                    .push(
                        widget::text(value)
                            .size(size)
                            .class(Color::from_rgb8(0x11, 0x11, 0x11))
                            .width(Length::Fill)
                            .wrapping(text::Wrapping::None),
                    )
                    .push(disclosure)
                    .spacing(7)
                    .align_y(Alignment::Center),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .align_y(Alignment::Center),
        )
        .padding([0, 10])
        .height(Length::Fixed(34.0))
        .width(Length::Fill)
        .class(iced_button_class(rounded_selector_button_style()))
        .on_press(message)
        .into()
    }

    fn small_label<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(widget::text(text.to_string()).size(12))
            .style(move |_| container::Style::default().color(palette.text_main))
            .into()
    }

    fn app_text_input<'a>(
        &'a self,
        placeholder: &'a str,
        value: &'a str,
        message: Option<fn(String) -> Message>,
        field: InputField,
        tall: bool,
    ) -> Element<'a, Message> {
        let mut input = widget::text_input(placeholder, value)
            .padding([8, 10])
            .size(13)
            .font(cosmic::font::default())
            .style(app_input_style(self.accent().display_color()))
            .on_focus(Message::InputFocusChanged(field, true))
            .on_unfocus(Message::InputFocusChanged(field, false))
            .width(Length::Fill);
        if let Some(on_input) = message {
            input = input.on_input(on_input);
        }
        let palette = self.active_palette();
        let focused = self.focused_input == Some(field);
        let state = if focused {
            GlassState::Focused
        } else {
            GlassState::Rest
        };
        widget::container(
            self.glass_layer(
                widget::container(input)
                    .height(Length::Fill)
                    .align_y(Alignment::Center)
                    .into(),
                GlassVariant::Input,
                state,
            ),
        )
        .height(Length::Fixed(if tall { 88.0 } else { 38.0 }))
        .align_y(Alignment::Center)
        .style(move |_| input_shell_style(palette, focused))
        .into()
    }

    fn labeled_app_text_input<'a>(
        &'a self,
        label: &'a str,
        placeholder: &'a str,
        value: &'a str,
        message: Option<fn(String) -> Message>,
        field: InputField,
        tall: bool,
    ) -> Element<'a, Message> {
        widget::column::with_capacity(2)
            .push(self.small_label(label))
            .push(self.app_text_input(placeholder, value, message, field, tall))
            .spacing(4)
            .width(Length::Fill)
            .into()
    }

    fn page_title<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(widget::text(text.to_string()).size(16))
            .style(move |_| container::Style::default().color(palette.text_main))
            .into()
    }

    fn label_text<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(widget::text(text.to_string()).size(14))
            .style(move |_| container::Style::default().color(palette.text_main))
            .into()
    }

    fn soft_text<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(widget::text(text.to_string()).size(12))
            .style(move |_| container::Style::default().color(palette.text_soft))
            .into()
    }

    fn soft_text_wrapped<T: ToString>(&self, text: T) -> Element<'_, Message> {
        let palette = self.active_palette();
        widget::container(
            widget::text(text.to_string())
                .size(12)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
        )
        .width(Length::Fill)
        .style(move |_| container::Style::default().color(palette.text_soft))
        .into()
    }

    fn workshed_help_text<T: ToString>(&self, text: T) -> Element<'_, Message> {
        self.workshed_help_text_sized(text, 12)
    }

    fn workshed_help_text_sized<T: ToString>(&self, text: T, size: u16) -> Element<'_, Message> {
        widget::container(
            widget::text(text.to_string())
                .size(size)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
        )
        .width(Length::Fill)
        .style(|_| container::Style::default().color(workshed_help_color()))
        .into()
    }

    fn workshed_label_text<T: ToString>(&self, text: T) -> Element<'_, Message> {
        widget::text(text.to_string())
            .size(14)
            .class(workshed_color(""))
            .width(Length::Fill)
            .wrapping(text::Wrapping::WordOrGlyph)
            .into()
    }

    fn workshed_dashboard_row<'a, T: ToString, U: ToString>(
        &'a self,
        label: T,
        value: U,
    ) -> Element<'a, Message> {
        widget::container(
            widget::row::with_capacity(2)
                .push(
                    widget::text(label.to_string())
                        .size(12)
                        .class(workshed_color(""))
                        .width(Length::FillPortion(5))
                        .wrapping(text::Wrapping::WordOrGlyph),
                )
                .push(
                    widget::text(value.to_string())
                        .size(12)
                        .class(workshed_color(""))
                        .width(Length::FillPortion(7))
                        .wrapping(text::Wrapping::WordOrGlyph)
                        .align_x(iced::alignment::Horizontal::Right),
                )
                .spacing(8)
                .align_y(Alignment::Start),
        )
        .padding([6, 8])
        .width(Length::Fill)
        .style(|_| workshed_field_style())
        .into()
    }

    fn primary_button<T: ToString>(&self, label: T, message: Message) -> Element<'_, Message> {
        iced::widget::button(centered_text(label.to_string(), 14))
            .padding(0)
            .height(Length::Fixed(38.0))
            .width(Length::Fill)
            .class(iced_button_class(primary_button_style(self.accent())))
            .on_press(message)
            .into()
    }

    fn secondary_button<T: ToString>(&self, label: T, message: Message) -> Element<'_, Message> {
        iced::widget::button(centered_text(label.to_string(), 14))
            .padding(0)
            .height(Length::Fixed(38.0))
            .width(Length::Fill)
            .class(iced_button_class(secondary_button_style(
                self.active_palette(),
                self.accent(),
                false,
            )))
            .on_press(message)
            .into()
    }

    fn danger_button<T: ToString>(&self, label: T, message: Message) -> Element<'_, Message> {
        iced::widget::button(centered_text(label.to_string(), 14))
            .padding(0)
            .height(Length::Fixed(38.0))
            .width(Length::Fill)
            .class(iced_button_class(danger_button_style()))
            .on_press(message)
            .into()
    }

    fn small_button<T: ToString>(&self, label: T, message: Message) -> Element<'_, Message> {
        iced::widget::button(centered_text(label.to_string(), 12))
            .padding(0)
            .width(Length::Fill)
            .height(Length::Fixed(32.0))
            .class(iced_button_class(secondary_button_style(
                self.active_palette(),
                self.accent(),
                false,
            )))
            .on_press(message)
            .into()
    }

    fn split_action_group<L: ToString, R: ToString>(
        &self,
        left_label: L,
        left_message: Message,
        right_label: R,
        right_message: Message,
        left_width: f32,
        right_width: f32,
    ) -> Element<'_, Message> {
        let left = iced::widget::button(centered_text(left_label.to_string(), 12))
            .padding(0)
            .width(Length::Fill)
            .height(Length::Fixed(32.0))
            .class(iced_button_class(segmented_secondary_button_style(
                self.active_palette(),
            )))
            .on_press(left_message);
        let right = iced::widget::button(centered_text(right_label.to_string(), 12))
            .padding(0)
            .width(Length::Fill)
            .height(Length::Fixed(32.0))
            .class(iced_button_class(segmented_primary_button_style(
                self.accent(),
            )))
            .on_press(right_message);

        widget::container(
            widget::row::with_capacity(2)
                .push(widget::container(left).width(Length::Fixed(left_width)))
                .push(widget::container(right).width(Length::Fixed(right_width)))
                .spacing(0)
                .height(Length::Fixed(32.0)),
        )
        .width(Length::Fill)
        .height(Length::Fixed(32.0))
        .align_x(Alignment::End)
        .into()
    }

    fn split_danger_action_group<L: ToString, R: ToString>(
        &self,
        left_label: L,
        left_message: Message,
        right_label: R,
        right_message: Message,
        left_width: f32,
        right_width: f32,
    ) -> Element<'_, Message> {
        let left = iced::widget::button(centered_text(left_label.to_string(), 12))
            .padding(0)
            .width(Length::Fill)
            .height(Length::Fixed(32.0))
            .class(iced_button_class(segmented_secondary_button_style(
                self.active_palette(),
            )))
            .on_press(left_message);
        let right = iced::widget::button(centered_text(right_label.to_string(), 12))
            .padding(0)
            .width(Length::Fill)
            .height(Length::Fixed(32.0))
            .class(iced_button_class(segmented_danger_button_style()))
            .on_press(right_message);

        widget::container(
            widget::row::with_capacity(2)
                .push(widget::container(left).width(Length::Fixed(left_width)))
                .push(widget::container(right).width(Length::Fixed(right_width)))
                .spacing(0)
                .height(Length::Fixed(32.0)),
        )
        .width(Length::Fill)
        .height(Length::Fixed(32.0))
        .align_x(Alignment::End)
        .into()
    }

    fn small_toggle<T: ToString>(
        &self,
        label: T,
        selected: bool,
        message: Message,
    ) -> Element<'_, Message> {
        let label = label.to_string();
        let text = if selected {
            format!("> {label}")
        } else {
            label
        };
        iced::widget::button(centered_text(text, 12))
            .padding(0)
            .height(Length::Fixed(32.0))
            .class(iced_button_class(secondary_button_style(
                self.active_palette(),
                self.accent(),
                selected,
            )))
            .on_press(message)
            .into()
    }

    fn disabled_toggle<T: ToString>(&self, label: T) -> Element<'_, Message> {
        widget::container(centered_text(label.to_string(), 12))
            .padding(0)
            .height(Length::Fixed(32.0))
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(|_| disabled_action_style())
            .into()
    }

    fn settings_select_button<T: ToString>(
        &self,
        label: T,
        message: Message,
        width: f32,
    ) -> Element<'_, Message> {
        iced::widget::button(centered_text(label.to_string(), 12))
            .padding(0)
            .width(Length::Fixed(width))
            .height(Length::Fixed(36.0))
            .class(iced_button_class(settings_select_button_style()))
            .on_press(message)
            .into()
    }

    fn settings_select_button_fill<T: ToString>(
        &self,
        label: T,
        message: Message,
    ) -> Element<'_, Message> {
        let label = widget::container(
            widget::text(label.to_string())
                .size(12)
                .width(Length::Fill)
                .wrapping(text::Wrapping::WordOrGlyph),
        )
        .width(Length::Fill)
        .align_x(Alignment::Center)
        .align_y(Alignment::Center);
        iced::widget::button(label)
            .padding([11, 8])
            .width(Length::Fill)
            .height(Length::Shrink)
            .class(iced_button_class(settings_select_button_style()))
            .on_press(message)
            .into()
    }

    fn temperature_control(&self) -> Element<'_, Message> {
        let temperature = self
            .temperature_input
            .trim()
            .parse::<f32>()
            .unwrap_or(self.native_state.settings.temperature)
            .clamp(0.0, 2.0);
        widget::row::with_capacity(2)
            .push(
                widget::slider(0.0..=2.0, temperature, Message::TemperatureSliderChanged)
                    .step(0.05)
                    .width(Length::Fixed(240.0))
                    .height(38.0)
                    .class(theme::iced::Slider::Custom {
                        active: Rc::new(|_| temperature_slider_style(slider::Status::Active)),
                        hovered: Rc::new(|_| temperature_slider_style(slider::Status::Hovered)),
                        dragging: Rc::new(|_| temperature_slider_style(slider::Status::Dragged)),
                    }),
            )
            .push(
                widget::container(self.app_text_input(
                    "0.7",
                    &self.temperature_input,
                    Some(Message::TemperatureChanged),
                    InputField::Temperature,
                    false,
                ))
                .width(Length::Fixed(90.0)),
            )
            .spacing(8)
            .align_y(Alignment::Center)
            .into()
    }

    fn agent_chip<T: ToString>(
        &self,
        label: T,
        selected: bool,
        message: Message,
    ) -> Element<'_, Message> {
        iced::widget::button(
            widget::container(widget::text(label.to_string()).size(12))
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(Alignment::Center)
                .align_y(Alignment::Center),
        )
        .padding(0)
        .width(Length::Fixed(86.0))
        .height(Length::Fixed(32.0))
        .class(iced_button_class(agent_chip_style(
            self.active_palette(),
            selected,
        )))
        .on_press(message)
        .into()
    }

    fn agent_runtime_or_configure_group(&self) -> Element<'_, Message> {
        if self.agent_profile_state.is_configured() {
            return widget::container(
                widget::row::with_capacity(2)
                    .push(self.agent_chip(
                        "openclaw",
                        self.agent_mode() == "openclaw",
                        Message::ToggleAgentRuntime("openclaw".into()),
                    ))
                    .push(self.agent_chip(
                        "hermes",
                        self.agent_mode() == "hermes",
                        Message::ToggleAgentRuntime("hermes".into()),
                    ))
                    .spacing(2)
                    .align_y(Alignment::Center),
            )
            .width(Length::Fixed(AGENT_RUNTIME_GROUP_WIDTH))
            .height(Length::Fixed(AGENT_RUNTIME_GROUP_HEIGHT))
            .padding(2)
            .style(|_| agent_group_style())
            .into();
        }

        let (label, fill, enabled) = match &self.agent_profile_state {
            AgentProfileState::Configuring { progress, .. } => (
                format!("Configuring · {progress:.0}%"),
                agent_profile_progress_color(*progress),
                false,
            ),
            AgentProfileState::Failed { .. } => (
                "Retry configure".into(),
                Color::from_rgb8(0x00, 0x00, 0x00),
                true,
            ),
            AgentProfileState::Configured { .. } => ("Ready".into(), Color::WHITE, false),
            _ => ("Configure".into(), Color::from_rgb8(0x00, 0x00, 0x00), true),
        };
        let text_color = if agent_profile_color_luminance(fill) > 0.72 {
            Color::from_rgb8(0x11, 0x11, 0x11)
        } else {
            Color::WHITE
        };
        let button_content = widget::container(widget::text(label).size(12))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center);
        let button = iced::widget::button(button_content)
            .padding(0)
            .width(Length::Fixed(AGENT_RUNTIME_BUTTON_WIDTH))
            .height(Length::Fixed(AGENT_RUNTIME_BUTTON_HEIGHT))
            .class(iced_button_class(agent_profile_button_style(
                fill, text_color,
            )));
        let button = if enabled {
            button.on_press(Message::ConfigureAgentProfilePressed)
        } else {
            button
        };
        widget::container(button)
            .width(Length::Fixed(AGENT_RUNTIME_GROUP_WIDTH))
            .height(Length::Fixed(AGENT_RUNTIME_GROUP_HEIGHT))
            .padding(2)
            // Keep the configure control at the same footprint without adding
            // the segmented runtime group's dark outer ring.
            .style(|_| transparent_panel_style())
            .into()
    }

    fn send_fab(&self, message: Message) -> Element<'_, Message> {
        let button = iced::widget::button(crate::icons::send_icon(Color::WHITE, 16.0))
            .padding(0)
            .width(Length::Fixed(32.0))
            .height(Length::Fixed(32.0))
            .class(iced_button_class(send_fab_style()));
        if self.streaming_reply.is_some() {
            button.into()
        } else {
            button.on_press(message).into()
        }
    }

    fn media_fab(&self, message: Message) -> Element<'_, Message> {
        iced::widget::button(crate::icons::plus_icon(
            Color::from_rgb8(0x11, 0x11, 0x11),
            16.0,
        ))
        .padding(0)
        .width(Length::Fixed(32.0))
        .height(Length::Fixed(32.0))
        .class(iced_button_class(media_fab_style()))
        .on_press(message)
        .into()
    }

    fn sidebar_icon(&self, page: Page, color: Color, scale: f32) -> Element<'static, Message> {
        crate::icons::sidebar_icon(page, color, scale)
    }

    fn window_ops_task(&self, ops: Vec<WindowEffectOp>) -> cosmic::app::Task<Message> {
        let mut tasks = Vec::new();
        for op in ops {
            match op {
                WindowEffectOp::EnableBlur(id) => {
                    tasks.push(wrap_app_task(window::enable_blur::<Message>(id)))
                }
                WindowEffectOp::DisableBlur(id) => {
                    tasks.push(wrap_app_task(window::disable_blur::<Message>(id)))
                }
            }
        }
        if tasks.is_empty() {
            cosmic::Task::none()
        } else {
            cosmic::Task::batch(tasks)
        }
    }

    fn accent(&self) -> Accent {
        Accent::Black
    }

    fn agent_mode(&self) -> &'static str {
        if !self.native_state.settings.openclaw_enabled {
            "off"
        } else if self.native_state.settings.agent_runtime == "hermes" {
            "hermes"
        } else {
            "openclaw"
        }
    }
}

fn wrap_app_task<Message: Send + 'static>(
    task: cosmic::Task<Message>,
) -> cosmic::app::Task<Message> {
    task.map(cosmic::Action::App)
}

fn normalized_agent_profile_model(value: &str) -> String {
    let normalized = value.trim().to_ascii_lowercase();
    for prefix in ["imstudio-community/", "1mstudio-community/"] {
        if let Some(suffix) = normalized.strip_prefix(prefix) {
            return format!("lmstudio-community/{suffix}");
        }
    }
    normalized
}

fn same_agent_profile_model(left: &str, right: &str) -> bool {
    let left = normalized_agent_profile_model(left);
    let right = normalized_agent_profile_model(right);
    !left.is_empty() && left == right
}

fn profile_has_degraded_tools(profile: &Value) -> bool {
    ["openclaw", "hermes"].iter().any(|runtime| {
        profile
            .get(*runtime)
            .and_then(Value::as_object)
            .and_then(|section| section.get("tool_calling"))
            .and_then(Value::as_str)
            .map(|value| value.eq_ignore_ascii_case("degraded"))
            .unwrap_or(false)
    })
}

fn centered_text(label: String, size: u16) -> Element<'static, Message> {
    widget::container(widget::text(label).size(size))
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(Alignment::Center)
        .align_y(Alignment::Center)
        .into()
}

fn thinking_ellipsis(started_at: Instant, now: Instant) -> String {
    let phase = now
        .saturating_duration_since(started_at)
        .as_millis()
        .saturating_div(260)
        % 3;
    match phase {
        0 => ".".into(),
        1 => "..".into(),
        _ => "...".into(),
    }
}

fn format_thought_duration(duration: Duration) -> String {
    format!("thought for {:.1} seconds", duration.as_secs_f64())
}

fn powder_cursor(started_at: Instant, now: Instant) -> &'static str {
    match now
        .saturating_duration_since(started_at)
        .as_millis()
        .saturating_div(120)
        % 4
    {
        0 => " ░",
        1 => " ▒",
        2 => " ▓",
        _ => "",
    }
}

fn take_chars(value: &str, count: usize) -> String {
    value.chars().take(count).collect()
}

fn normalize_settings(settings: &mut NativeSettings) {
    if settings.server_url.trim().is_empty() {
        settings.server_url = "http://127.0.0.1:8000".into();
    }
    settings.max_tokens = settings.max_tokens.clamp(1, 16_384);
    settings.temperature = settings.temperature.clamp(0.0, 2.0);
    if settings.agent_runtime != "openclaw" && settings.agent_runtime != "hermes" {
        settings.agent_runtime = "auto".into();
    }
    // Keep the legacy field readable for old state files, but tool exposure is
    // now always automatic and no longer user-selected in the composer.
    settings.openclaw_tool_profile = "auto".into();
    if settings.chat_dispatch_mode != "parallel" {
        settings.chat_dispatch_mode = "single".into();
    }
    if settings.language != "zh" {
        settings.language = "en".into();
    }
}

fn build_user_content(text: &str, media: &[MediaAttachment]) -> Value {
    if media.is_empty() {
        return Value::String(text.to_string());
    }
    let mut parts = Vec::new();
    let clean = text.trim();
    if !clean.is_empty() {
        parts.push(json!({ "type": "text", "text": clean }));
    } else {
        parts.push(json!({ "type": "text", "text": "Please analyze the attached media." }));
    }
    for item in media {
        if item.kind == "image" {
            parts.push(json!({ "type": "image_url", "image_url": { "url": item.data_url } }));
        } else if item.kind == "video" {
            parts.push(json!({ "type": "video_url", "video_url": { "url": item.data_url } }));
        }
    }
    Value::Array(parts)
}

fn build_display_text(text: &str, media: &[MediaAttachment]) -> String {
    let mut out = text.trim().to_string();
    for item in media {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!(
            "[{}] {} ({})",
            item.kind,
            item.name,
            format_bytes(item.size)
        ));
    }
    if out.is_empty() { "...".into() } else { out }
}

fn parse_markdown_blocks(content: &str) -> Vec<MarkdownBlock> {
    let mut blocks = Vec::new();
    let mut code_lines: Vec<String> = Vec::new();
    let mut in_code = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code {
                blocks.push(MarkdownBlock::Code(code_lines.join("\n")));
                code_lines.clear();
                in_code = false;
            } else {
                in_code = true;
            }
            continue;
        }

        if in_code {
            code_lines.push(line.to_string());
            continue;
        }

        if trimmed.is_empty() {
            blocks.push(MarkdownBlock::Space);
            continue;
        }

        if let Some((level, text)) = parse_heading(trimmed) {
            blocks.push(MarkdownBlock::Heading { level, text });
        } else if let Some(text) = parse_bullet(trimmed) {
            blocks.push(MarkdownBlock::Bullet(text));
        } else if let Some((number, text)) = parse_ordered(trimmed) {
            blocks.push(MarkdownBlock::Ordered { number, text });
        } else if let Some(text) = trimmed.strip_prefix("> ") {
            blocks.push(MarkdownBlock::Quote(text.trim().to_string()));
        } else {
            blocks.push(MarkdownBlock::Paragraph(trimmed.to_string()));
        }
    }

    if in_code || !code_lines.is_empty() {
        blocks.push(MarkdownBlock::Code(code_lines.join("\n")));
    }
    if blocks.is_empty() {
        blocks.push(MarkdownBlock::Paragraph("...".into()));
    }
    blocks
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let hashes = line.chars().take_while(|ch| *ch == '#').count();
    if (1..=6).contains(&hashes) && line.chars().nth(hashes) == Some(' ') {
        Some((hashes, line[hashes + 1..].trim().to_string()))
    } else {
        None
    }
}

fn parse_bullet(line: &str) -> Option<String> {
    for prefix in ["- ", "* ", "+ "] {
        if let Some(text) = line.strip_prefix(prefix) {
            return Some(text.trim().to_string());
        }
    }
    None
}

fn parse_ordered(line: &str) -> Option<(String, String)> {
    let (number, rest) = line.split_once(". ")?;
    if !number.is_empty() && number.chars().all(|ch| ch.is_ascii_digit()) {
        Some((number.to_string(), rest.trim().to_string()))
    } else {
        None
    }
}

fn clean_inline_markdown(text: &str) -> String {
    let mut out = text
        .replace("**", "")
        .replace("__", "")
        .replace('`', "")
        .replace("~~", "");

    while let Some(start) = out.find('[') {
        let Some(mid_rel) = out[start..].find("](") else {
            break;
        };
        let mid = start + mid_rel;
        let Some(end_rel) = out[mid + 2..].find(')') else {
            break;
        };
        let end = mid + 2 + end_rel;
        let label = out[start + 1..mid].to_string();
        let url = out[mid + 2..end].to_string();
        out.replace_range(start..=end, &format!("{label} ({url})"));
    }

    out
}

fn pick_media_files() -> Result<Vec<MediaAttachment>, String> {
    #[cfg(target_os = "macos")]
    let output = Command::new("osascript")
        .arg("-e")
        .arg("set pickedFiles to choose file with multiple selections allowed\nset outText to \"\"\nrepeat with f in pickedFiles\nset outText to outText & POSIX path of f & linefeed\nend repeat\nreturn outText")
        .output()
        .map_err(|e| format!("Failed to open file picker: {e}"))?;

    #[cfg(not(target_os = "macos"))]
    let output = Command::new("sh")
        .arg("-c")
        .arg("printf ''")
        .output()
        .map_err(|e| format!("File picker is unavailable: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            return Ok(Vec::new());
        }
        return Err(stderr);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut items = Vec::new();
    for path in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(4)
    {
        let path_buf = PathBuf::from(path);
        let name = path_buf
            .file_name()
            .and_then(|item| item.to_str())
            .unwrap_or("media")
            .to_string();
        let (kind, mime) = media_kind_and_mime(&path_buf)?;
        let max_bytes = if kind == "image" {
            10 * 1024 * 1024
        } else {
            35 * 1024 * 1024
        };
        let size = fs::metadata(&path_buf)
            .map_err(|e| format!("Failed to inspect {path}: {e}"))?
            .len();
        if size > max_bytes as u64 {
            return Err(format!("{name} is too large."));
        }
        let bytes = fs::read(&path_buf).map_err(|e| format!("Failed to read {path}: {e}"))?;
        if bytes.len() > max_bytes {
            return Err(format!(
                "{name} grew beyond the attachment limit while reading."
            ));
        }
        let data_url = format!("data:{mime};base64,{}", base64_encode(&bytes));
        items.push(MediaAttachment {
            name,
            kind,
            size,
            data_url,
        });
    }
    Ok(items)
}

fn media_kind_and_mime(path: &PathBuf) -> Result<(String, String), String> {
    let ext = path
        .extension()
        .and_then(|item| item.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => Ok(("image".into(), "image/png".into())),
        "jpg" | "jpeg" => Ok(("image".into(), "image/jpeg".into())),
        "gif" => Ok(("image".into(), "image/gif".into())),
        "webp" => Ok(("image".into(), "image/webp".into())),
        "mp4" => Ok(("video".into(), "video/mp4".into())),
        "mov" => Ok(("video".into(), "video/quicktime".into())),
        "webm" => Ok(("video".into(), "video/webm".into())),
        _ => Err(format!("Unsupported media extension: {ext}")),
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut index = 0;
    while index < bytes.len() {
        let b0 = bytes[index];
        let b1 = bytes.get(index + 1).copied().unwrap_or(0);
        let b2 = bytes.get(index + 2).copied().unwrap_or(0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if index + 1 < bytes.len() {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if index + 2 < bytes.len() {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
        index += 3;
    }
    out
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn empty_dash(value: &str) -> &str {
    if value.trim().is_empty() { "-" } else { value }
}

fn copy_to_clipboard(value: &str) -> bool {
    if value.trim().is_empty() || !cfg!(target_os = "macos") {
        return false;
    }
    let Ok(mut child) = Command::new("pbcopy")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let Some(mut stdin) = child.stdin.take() else {
        return false;
    };
    if stdin.write_all(value.as_bytes()).is_err() {
        return false;
    }
    drop(stdin);
    child.wait().map(|status| status.success()).unwrap_or(false)
}

fn fitted_model_label(value: String) -> String {
    let count = value.chars().count();
    if count <= 54 {
        return value;
    }
    let head: String = value.chars().take(28).collect();
    let tail: String = value
        .chars()
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}...{tail}")
}

fn fitted_model_label_for_width(value: String, max_chars: usize) -> String {
    let count = value.chars().count();
    if count <= max_chars || max_chars <= 6 {
        return value;
    }
    let head_len = (max_chars - 3) / 2;
    let tail_len = max_chars - 3 - head_len;
    let head: String = value.chars().take(head_len).collect();
    let tail: String = value
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}...{tail}")
}

fn fitted_model_label_size(value: &str) -> u16 {
    match value.chars().count() {
        0..=24 => 12,
        25..=34 => 11,
        35..=46 => 10,
        _ => 9,
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.2} GB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

fn job_summary(job: &Value) -> String {
    let status = string_at(job, "status").if_empty_then(|| "unknown".into());
    let model = string_at(job, "model");
    let message = string_at(job, "message");
    let percent = job
        .get("progress_percent")
        .or_else(|| job.get("percent"))
        .and_then(Value::as_f64)
        .filter(|percent| percent.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 100.0);
    if message.is_empty() {
        format!("[{status}] {model} ({percent:.1}%)")
    } else {
        format!("[{status}] {model}: {message} ({percent:.1}%)")
    }
}

fn community_job_is_valid(job: &Value) -> bool {
    job.is_object()
        && ["id", "model", "status"].iter().all(|field| {
            job.get(*field)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
        })
}

fn normalize_community_job(job: Option<Value>) -> Option<Value> {
    // Poll endpoints can return an explicit JSON null, not only an absent field.
    // Neither null nor an incomplete object represents an actionable deployment.
    job.filter(community_job_is_valid)
}

fn community_job_allowed_actions(job: &Value) -> &'static [&'static str] {
    if !community_job_is_valid(job) || job.get("done").and_then(Value::as_bool).unwrap_or(false) {
        return &[];
    }
    match string_at(job, "status")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "queued" | "downloading" => &["pause", "cancel"],
        "paused" => &["resume", "cancel"],
        "deploying" => &["cancel"],
        _ => &[],
    }
}

trait EmptyThen {
    fn if_empty_then<F: FnOnce() -> String>(self, fallback: F) -> String;
}

impl EmptyThen for String {
    fn if_empty_then<F: FnOnce() -> String>(self, fallback: F) -> String {
        if self.trim().is_empty() {
            fallback()
        } else {
            self
        }
    }
}

fn bounce_scale(progress: f32) -> f32 {
    if progress < 0.42 {
        1.06 - (progress / 0.42) * 0.18
    } else {
        0.88 + ((progress - 0.42) / 0.58) * 0.15
    }
}

fn iced_button_class<F>(style: F) -> theme::iced::Button
where
    F: Fn(&Theme, button::Status) -> button::Style + 'static,
{
    theme::iced::Button::Custom(Box::new(style))
}

fn frameless_sidebar_style(text_color: Color) -> container::Style {
    container::Style::default()
        .color(text_color)
        .border(Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        })
        .shadow(Shadow::default())
}

fn aluminum_bar_style(color: Color) -> container::Style {
    container::Style::default()
        .background(Background::Color(color))
        .border(Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        })
        .shadow(Shadow::default())
}

fn primary_button_style(
    accent: Accent,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, false),
            Some(accent.display_color()),
        )
    }
}

fn secondary_button_style(
    palette: crate::appearance::Palette,
    accent: Accent,
    selected: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        glass_button_style(
            palette,
            GlassVariant::Control,
            state_from_status(status, selected),
            if selected {
                Some(accent.display_color())
            } else {
                None
            },
        )
    }
}

fn segmented_primary_button_style(
    accent: Accent,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        let mut style = glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, false),
            Some(accent.display_color()),
        );
        style.border.radius = [0.0, CONTENT_RADIUS, CONTENT_RADIUS, 0.0].into();
        style
    }
}

fn segmented_secondary_button_style(
    palette: crate::appearance::Palette,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        let mut style = glass_button_style(
            palette,
            GlassVariant::Control,
            state_from_status(status, false),
            None,
        );
        style.border.radius = [CONTENT_RADIUS, 0.0, 0.0, CONTENT_RADIUS].into();
        style
    }
}

fn danger_button_style() -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, false),
            Some(Color::from_rgb8(0xef, 0x44, 0x44)),
        )
    }
}

fn segmented_danger_button_style()
-> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        let mut style = glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, false),
            Some(Color::from_rgb8(0xef, 0x44, 0x44)),
        );
        style.border.radius = [0.0, CONTENT_RADIUS, CONTENT_RADIUS, 0.0].into();
        style
    }
}

fn sidebar_active_icon_color() -> Color {
    Color::from_rgb8(0x11, 0x11, 0x11)
}

fn input_shell_style(palette: crate::appearance::Palette, focused: bool) -> container::Style {
    glass_container_style(
        palette,
        GlassVariant::Input,
        if focused {
            GlassState::Focused
        } else {
            GlassState::Rest
        },
    )
}

fn frosted_composer_style(palette: crate::appearance::Palette) -> container::Style {
    let mut style = glass_container_style(palette, GlassVariant::Floating, GlassState::Focused);
    style.background = Some(Background::Color(palette.card_background_strong));
    style.border = Border {
        width: 1.0,
        radius: (CONTENT_RADIUS + 2.0).into(),
        color: Color::from_rgba(1.0, 1.0, 1.0, 0.84),
    };
    style.shadow = Shadow {
        color: Color::from_rgba(0.08, 0.12, 0.18, 0.12),
        offset: Vector::new(0.0, 2.0),
        blur_radius: 12.0,
    };
    style
}

fn transparent_text_editor_style(
    accent: Color,
) -> impl Fn(&Theme, text_editor::Status) -> text_editor::Style + Clone + 'static {
    move |_theme, _status| text_editor::Style {
        background: Background::Color(Color::TRANSPARENT),
        border: Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        },
        placeholder: Color::from_rgba(0.07, 0.09, 0.13, 0.38),
        value: Color::from_rgb8(0x11, 0x18, 0x27),
        selection: Color::from_rgba(accent.r, accent.g, accent.b, 0.34),
    }
}

fn transparent_panel_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::TRANSPARENT))
        .border(Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        })
}

fn disabled_action_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, true);
    container::Style::default()
        .background(Background::Color(palette.inset_background))
        .color(palette.text_soft)
        .border(Border {
            width: 1.0,
            radius: CONTENT_RADIUS.into(),
            color: palette.card_border,
        })
}

fn agent_group_style() -> container::Style {
    glass_container_style(
        palette(MaterialMode::Frosted, true),
        GlassVariant::Control,
        GlassState::Focused,
    )
}

fn models_box_style(palette: crate::appearance::Palette) -> container::Style {
    glass_container_style(palette, GlassVariant::Panel, GlassState::Rest)
}

fn models_count_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, false);
    container::Style::default()
        .background(Background::Color(palette.inset_background))
        .color(palette.text_main)
        .border(Border {
            width: 1.0,
            radius: 999.0.into(),
            color: palette.card_border,
        })
}

fn flat_section_style(palette: crate::appearance::Palette) -> container::Style {
    container::Style::default()
        .background(Background::Color(palette.window_background))
        .color(palette.text_main)
        .border(Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        })
        .shadow(Shadow::default())
}

fn flat_frame_host_style(palette: crate::appearance::Palette) -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::TRANSPARENT))
        .color(palette.text_main)
        .border(Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        })
        .shadow(Shadow::default())
}

fn caligo_horizontal_bar(color: Color) -> Element<'static, Message> {
    widget::container(widget::text(""))
        .width(Length::Fill)
        .height(Length::Fixed(1.0))
        .style(move |_| aluminum_bar_style(color))
        .into()
}

fn model_marker_style(selected: bool) -> container::Style {
    container::Style::default()
        .background(Background::Color(if selected {
            Color::from_rgb8(0x11, 0x11, 0x11)
        } else {
            Color::TRANSPARENT
        }))
        .border(Border {
            width: 0.0,
            radius: 999.0.into(),
            color: Color::TRANSPARENT,
        })
}

fn model_item_button_style(
    palette: crate::appearance::Palette,
    selected: bool,
    _active: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let mut style = glass_button_style(
            palette,
            GlassVariant::Card,
            state_from_status(status, selected),
            None,
        );
        style.text_color = palette.text_main;
        style
    }
}

fn check_dot_style(checked: bool) -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::from_rgba(1.0, 1.0, 1.0, 0.34)))
        .border(Border {
            width: if checked { 3.0 } else { 1.0 },
            radius: 999.0.into(),
            color: if checked {
                Color::from_rgb8(0x11, 0x11, 0x11)
            } else {
                Color::from_rgba(0.07, 0.07, 0.07, 0.14)
            },
        })
}

fn concurrent_list_style() -> container::Style {
    glass_container_style(
        palette(MaterialMode::Frosted, true),
        GlassVariant::Card,
        GlassState::Rest,
    )
}

fn chat_model_dropdown_style(palette: crate::appearance::Palette) -> container::Style {
    glass_container_style(palette, GlassVariant::Floating, GlassState::Rest)
}

fn model_switch_toast_style(_palette: crate::appearance::Palette) -> container::Style {
    // System notices should read as a quiet, dark translucent surface instead
    // of inheriting the bright card palette used by the rest of the shell.
    container::Style::default()
        .background(Background::Color(Color::from_rgba(
            0.045, 0.055, 0.07, 0.80,
        )))
        .color(Color::from_rgba(0.96, 0.97, 0.99, 0.96))
        .border(Border {
            width: 1.0,
            radius: CONTENT_RADIUS.into(),
            color: Color::from_rgba(1.0, 1.0, 1.0, 0.15),
        })
        .shadow(Shadow::default())
}

fn chat_model_option_style(
    selected: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let palette = palette(MaterialMode::Frosted, true);
        let mut style = glass_button_style(
            palette,
            GlassVariant::Control,
            state_from_status(status, selected),
            None,
        );
        style.text_color = palette.text_main;
        style
    }
}

fn community_card_style() -> container::Style {
    glass_container_style(
        palette(MaterialMode::Frosted, true),
        GlassVariant::Card,
        GlassState::Rest,
    )
}

fn community_chip_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, false);
    container::Style::default()
        .background(Background::Color(palette.inset_background))
        .color(palette.text_main)
        .border(Border {
            width: 1.0,
            radius: 999.0.into(),
            color: palette.card_border,
        })
        .shadow(Shadow {
            color: Color::TRANSPARENT,
            offset: Vector::new(0.0, 0.0),
            blur_radius: 0.0,
        })
}

fn community_logo_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, false);
    container::Style::default()
        .background(Background::Color(palette.card_background_strong))
        .color(palette.text_main)
        .border(Border {
            width: 1.0,
            radius: 999.0.into(),
            color: palette.card_border,
        })
        .shadow(Shadow {
            color: Color::TRANSPARENT,
            offset: Vector::new(0.0, 0.0),
            blur_radius: 0.0,
        })
}

fn terminal_console_head_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, true);
    container::Style::default()
        // The title bar is part of the terminal glass surface. Keep it
        // transparent so the window reads as one continuous material rather
        // than three stacked cards.
        .background(Background::Color(Color::TRANSPARENT))
        .color(palette.text_main)
        .border(Border {
            width: 0.0,
            radius: [TERMINAL_WINDOW_RADIUS, TERMINAL_WINDOW_RADIUS, 0.0, 0.0].into(),
            color: Color::TRANSPARENT,
        })
}

fn terminal_window_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, true);
    // One Caligo panel owns the whole terminal: title bar, command line and
    // output are intentionally transparent children of this single frosted
    // surface. The only chrome is the thin perimeter that defines its edge.
    let mut style = glass_container_style(palette, GlassVariant::Panel, GlassState::Rest)
        .color(palette.text_main);
    style.border = Border {
        width: 1.0,
        radius: TERMINAL_WINDOW_RADIUS.into(),
        color: rgba(0x11, 0x11, 0x11, 0.10),
    };
    style.shadow = Shadow::default();
    style
}

fn terminal_body_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, true);
    container::Style::default()
        // Keep the prompt in the same material as the output area. The
        // contrast comes from the monospace prompt and spacing, not another
        // inset card.
        .background(Background::Color(Color::TRANSPARENT))
        .color(palette.text_main)
        .border(Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        })
}

fn terminal_output_style() -> container::Style {
    let palette = palette(MaterialMode::Frosted, true);
    container::Style::default()
        // Output is the same continuous glass surface as the title and
        // prompt. Do not introduce a separate white panel here.
        .background(Background::Color(Color::TRANSPARENT))
        .color(palette.text_main)
        .border(Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        })
}

fn prompt_dot_style(color: Color) -> container::Style {
    container::Style::default()
        .background(Background::Color(color))
        .border(Border {
            width: 1.0,
            radius: 999.0.into(),
            color: Color::from_rgba(
                0x0f as f32 / 255.0,
                0x17 as f32 / 255.0,
                0x2a as f32 / 255.0,
                0.12,
            ),
        })
}

fn send_fab_style() -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let fill = if matches!(status, button::Status::Pressed) {
            Color::from_rgb8(0x0a, 0x0a, 0x0a)
        } else {
            Color::from_rgb8(0x11, 0x11, 0x11)
        };
        let mut style = button::Style::default().with_background(fill);
        style.text_color = Color::WHITE;
        style.border = Border {
            width: 1.0,
            radius: 999.0.into(),
            color: Color::from_rgba(0.14, 0.14, 0.14, 0.68),
        };
        style.shadow = Shadow::default();
        style
    }
}

fn dashboard_row_style() -> container::Style {
    glass_container_style(
        palette(MaterialMode::Frosted, true),
        GlassVariant::Subtle,
        GlassState::Rest,
    )
}

fn beta_tag_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::from_rgb8(0x11, 0x18, 0x27)))
        .color(Color::WHITE)
        .border(Border {
            width: 1.0,
            radius: 999.0.into(),
            color: Color::from_rgba(0.07, 0.09, 0.13, 0.90),
        })
}

fn history_item_style(active: bool) -> container::Style {
    glass_container_style(
        palette(MaterialMode::Frosted, true),
        GlassVariant::Card,
        if active {
            GlassState::Selected
        } else {
            GlassState::Rest
        },
    )
}

fn history_title_button_style() -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static
{
    move |_theme, _status| {
        let mut style = button::Style::default();
        style.text_color = Color::from_rgb8(0x0f, 0x17, 0x2a);
        style.background = None;
        style.border = Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        };
        style
    }
}

fn history_delete_button_style()
-> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let mut style = glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, false),
            if hovered {
                Some(Color::from_rgb8(0xef, 0x44, 0x44))
            } else {
                None
            },
        );
        if !hovered {
            style.text_color = Color::from_rgb8(0x47, 0x55, 0x69);
        }
        style
    }
}

fn new_conversation_button_style()
-> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let mut style = glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, false),
            None,
        );
        style.text_color = Color::from_rgb8(0x11, 0x18, 0x27);
        style
    }
}

fn transparent_text_button_style()
-> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, _status| {
        let mut style = button::Style::default();
        style.text_color = Color::from_rgb8(0x0f, 0x17, 0x2a);
        style.background = None;
        style.border = Border {
            width: 0.0,
            radius: 0.0.into(),
            color: Color::TRANSPARENT,
        };
        style
    }
}

fn settings_select_button_style()
-> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let mut style = glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, false),
            None,
        );
        style.text_color = Color::from_rgb8(0x0f, 0x17, 0x2a);
        style
    }
}

fn rounded_selector_button_style()
-> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let (background, border) = match status {
            button::Status::Hovered => (
                Color::from_rgb8(0xfb, 0xfc, 0xfe),
                Color::from_rgb8(0xb7, 0xc0, 0xcc),
            ),
            button::Status::Pressed => (
                Color::from_rgb8(0xf3, 0xf5, 0xf8),
                Color::from_rgb8(0xa5, 0xaf, 0xbd),
            ),
            button::Status::Disabled => (
                Color::from_rgb8(0xf8, 0xf9, 0xfb),
                Color::from_rgb8(0xe4, 0xe7, 0xec),
            ),
            button::Status::Active => (Color::WHITE, Color::from_rgb8(0xdc, 0xe1, 0xe9)),
        };
        let mut style = button::Style::default();
        style.background = Some(Background::Color(background));
        style.text_color = if matches!(status, button::Status::Disabled) {
            Color::from_rgb8(0x8b, 0x93, 0x9f)
        } else {
            Color::from_rgb8(0x11, 0x11, 0x11)
        };
        style.border = Border {
            width: 1.0,
            radius: CONTENT_RADIUS.into(),
            color: border,
        };
        style.shadow = Shadow::default();
        style
    }
}

fn temperature_slider_style(status: slider::Status) -> slider::Style {
    let active = Color::from_rgb8(0x11, 0x11, 0x11);
    let handle = if matches!(status, slider::Status::Dragged) {
        Color::from_rgb8(0x0a, 0x0a, 0x0a)
    } else {
        active
    };
    slider::Style {
        rail: slider::Rail {
            backgrounds: (
                Background::Color(active),
                Background::Color(Color::from_rgba(1.0, 1.0, 1.0, 0.62)),
            ),
            width: 9.0,
            border: Border {
                width: 1.0,
                radius: 999.0.into(),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.80),
            },
        },
        handle: slider::Handle {
            shape: slider::HandleShape::Circle { radius: 8.0 },
            background: Background::Color(handle),
            border_width: 1.0,
            border_color: Color::from_rgba(1.0, 1.0, 1.0, 0.78),
        },
        breakpoint: slider::Breakpoint {
            color: Color::TRANSPARENT,
        },
    }
}

fn chat_bubble_style(is_user: bool) -> container::Style {
    if is_user {
        container::Style::default()
            .background(Background::Color(Color::from_rgba(0.04, 0.04, 0.04, 0.90)))
            .color(Color::WHITE)
            .border(Border {
                width: 1.0,
                radius: 10.0.into(),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.20),
            })
            .shadow(Shadow {
                color: Color::TRANSPARENT,
                offset: Vector::new(0.0, 0.0),
                blur_radius: 0.0,
            })
    } else {
        glass_container_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Card,
            GlassState::Rest,
        )
    }
}

fn thinking_bubble_style() -> container::Style {
    glass_container_style(
        palette(MaterialMode::Frosted, true),
        GlassVariant::Subtle,
        GlassState::Focused,
    )
    .color(Color::from_rgb8(0x3f, 0x49, 0x59))
}

fn markdown_code_style(is_user: bool) -> container::Style {
    let background = if is_user {
        Color::from_rgba(1.0, 1.0, 1.0, 0.10)
    } else {
        Color::from_rgba(1.0, 1.0, 1.0, 0.58)
    };
    let border = if is_user {
        Color::from_rgba(1.0, 1.0, 1.0, 0.18)
    } else {
        Color::from_rgba(0.67, 0.74, 0.84, 0.28)
    };
    container::Style::default()
        .background(Background::Color(background))
        .border(Border {
            width: 1.0,
            radius: 8.0.into(),
            color: border,
        })
}

fn markdown_quote_style(is_user: bool) -> container::Style {
    let background = if is_user {
        Color::from_rgba(1.0, 1.0, 1.0, 0.08)
    } else {
        Color::from_rgba(1.0, 1.0, 1.0, 0.40)
    };
    let border = if is_user {
        Color::from_rgba(1.0, 1.0, 1.0, 0.20)
    } else {
        Color::from_rgba(0.55, 0.62, 0.72, 0.24)
    };
    container::Style::default()
        .background(Background::Color(background))
        .border(Border {
            width: 1.0,
            radius: 8.0.into(),
            color: border,
        })
}

fn agent_chip_style(
    _palette: crate::appearance::Palette,
    selected: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        let mut style = glass_button_style(
            palette(MaterialMode::Frosted, true),
            GlassVariant::Control,
            state_from_status(status, selected),
            None,
        );
        if !selected && matches!(status, button::Status::Active) {
            style.background = Some(Background::Color(Color::TRANSPARENT));
            style.border.color = Color::TRANSPARENT;
            style.shadow = Shadow::default();
            style.text_color = Color::from_rgb8(0x7b, 0x87, 0x9a);
        } else {
            style.text_color = Color::from_rgb8(0x11, 0x11, 0x11);
        }
        style
    }
}

fn agent_profile_progress_color(progress: f32) -> Color {
    let t = (progress / 100.0).clamp(0.0, 1.0);
    let start = Color::from_rgb8(0x73, 0x73, 0x73);
    Color::from_rgba(
        start.r + (1.0 - start.r) * t,
        start.g + (1.0 - start.g) * t,
        start.b + (1.0 - start.b) * t,
        1.0,
    )
}

fn agent_profile_color_luminance(color: Color) -> f32 {
    0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
}

fn agent_profile_button_style(
    fill: Color,
    text_color: Color,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        let mut background = fill;
        if matches!(status, button::Status::Hovered) {
            background = Color::from_rgba(
                (fill.r + 0.04).min(1.0),
                (fill.g + 0.04).min(1.0),
                (fill.b + 0.04).min(1.0),
                1.0,
            );
        }
        let mut style = button::Style::default().with_background(background);
        style.text_color = text_color;
        style.border = Border {
            width: 0.0,
            radius: CONTENT_RADIUS.into(),
            color: Color::TRANSPARENT,
        };
        style.shadow = Shadow::default();
        style
    }
}

fn media_fab_style() -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme: &Theme, status| {
        let mut style = button::Style::default();
        style.text_color = if matches!(status, button::Status::Hovered | button::Status::Pressed) {
            Color::from_rgba(0.07, 0.07, 0.07, 0.72)
        } else {
            Color::from_rgb8(0x11, 0x11, 0x11)
        };
        style.background = None;
        style.border = Border {
            width: 0.0,
            radius: 999.0.into(),
            color: Color::TRANSPARENT,
        };
        style
    }
}

fn app_input_style(accent: Color) -> theme::TextInput {
    fn appearance(
        accent: Color,
        placeholder_alpha: f32,
        text_alpha: f32,
    ) -> widget::text_input::Appearance {
        widget::text_input::Appearance {
            background: Background::Color(Color::TRANSPARENT),
            border_radius: iced::border::Radius::from(0.0),
            border_offset: None,
            border_width: 0.0,
            border_color: Color::TRANSPARENT,
            label_color: Color::from_rgba(0.07, 0.09, 0.13, text_alpha),
            placeholder_color: Color::from_rgba(0.07, 0.09, 0.13, placeholder_alpha),
            selected_text_color: Color::WHITE,
            icon_color: Some(Color::from_rgba(0.07, 0.09, 0.13, text_alpha)),
            text_color: Some(Color::from_rgba(0.07, 0.09, 0.13, text_alpha)),
            selected_fill: Color::from_rgba(accent.r, accent.g, accent.b, 0.34),
        }
    }

    theme::TextInput::Custom {
        active: Box::new(move |_theme| appearance(accent, 0.45, 0.92)),
        error: Box::new(move |_theme| appearance(accent, 0.45, 0.92)),
        hovered: Box::new(move |_theme| appearance(accent, 0.4, 0.98)),
        focused: Box::new(move |_theme| appearance(accent, 0.38, 1.0)),
        disabled: Box::new(move |_theme| appearance(accent, 0.35, 0.52)),
    }
}

fn terminal_input_style() -> theme::TextInput {
    fn appearance(placeholder_alpha: f32, text_alpha: f32) -> widget::text_input::Appearance {
        widget::text_input::Appearance {
            background: Background::Color(Color::TRANSPARENT),
            border_radius: iced::border::Radius::from(0.0),
            border_offset: None,
            border_width: 0.0,
            border_color: Color::TRANSPARENT,
            label_color: Color::from_rgba(0.07, 0.09, 0.13, text_alpha),
            placeholder_color: Color::from_rgba(0.07, 0.09, 0.13, placeholder_alpha),
            selected_text_color: Color::WHITE,
            icon_color: Some(Color::from_rgba(0.07, 0.09, 0.13, text_alpha)),
            text_color: Some(Color::from_rgba(0.07, 0.09, 0.13, text_alpha)),
            selected_fill: Color::from_rgba(0.17, 0.42, 1.0, 0.24),
        }
    }

    theme::TextInput::Custom {
        active: Box::new(move |_theme| appearance(0.56, 0.98)),
        error: Box::new(move |_theme| appearance(0.56, 0.98)),
        hovered: Box::new(move |_theme| appearance(0.62, 1.0)),
        focused: Box::new(move |_theme| appearance(0.62, 1.0)),
        disabled: Box::new(move |_theme| appearance(0.42, 0.55)),
    }
}

fn workshed_demo_order() -> Value {
    json!({
        "schema_version": 2,
        "id": "local-draft",
        "revision": 1,
        "name": "",
        "root_block_id": "root_local",
        "blocks": [
            {"id": "root_local", "type_id": "model_development", "lane": "source", "order": 0, "params": {"model_name": ""}, "param_origins": {}, "input_bindings": {}, "locked": true},
            {"id": "load_model_demo", "type_id": "load_model", "lane": "source", "order": 1, "params": {"source_kind": "local", "ref": "", "revision": "", "trust_remote_code": false, "dtype": "auto"}, "param_origins": {}, "input_bindings": {}},
            {"id": "quantize_demo", "type_id": "quantize_mlx", "lane": "verify", "order": 2, "params": {"preset_id": "mlx-balanced", "bits": null, "group_size": null, "mode": null, "output_name": null}, "param_origins": {}, "input_bindings": {}},
            {"id": "register_demo", "type_id": "register_model", "lane": "deliver", "order": 3, "params": {"model_id": null, "collision_policy": "fail", "test_when_finished": true}, "param_origins": {}, "input_bindings": {}}
        ],
        "edges": [
            {"from": {"block_id": "load_model_demo", "port": "model", "type": "Model"}, "to": {"block_id": "quantize_demo", "port": "model", "type": "Model"}},
            {"from": {"block_id": "quantize_demo", "port": "model", "type": "Model"}, "to": {"block_id": "register_demo", "port": "model", "type": "Model"}}
        ]
    })
}

fn workshed_new_order() -> Value {
    static NEXT_ORDER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NEXT_ORDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut order = workshed_demo_order();
    order["id"] = Value::String(format!("work-order-{timestamp:x}-{nonce:x}"));
    // Revision zero is a local, not-yet-persisted draft. The manager assigns 1.
    order["revision"] = json!(0);
    order
}

fn workshed_remember_order(orders: &mut Vec<Value>, order: &Value) {
    let id = string_at(order, "id");
    if id.is_empty() {
        return;
    }
    if let Some(existing) = orders.iter_mut().find(|item| string_at(item, "id") == id) {
        *existing = order.clone();
    } else {
        orders.insert(0, order.clone());
    }
}

fn workshed_order_needs_create(order: &Value, orders: &[Value]) -> bool {
    let id = string_at(order, "id");
    id.is_empty()
        || order.get("revision").and_then(Value::as_u64) == Some(0)
        || (id == "local-draft" && !orders.iter().any(|item| string_at(item, "id") == id))
}

fn workshed_block_label(type_id: &str) -> String {
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
        other => other.replace('_', " "),
    }
}

fn workshed_catalog_items(
    capabilities: &Value,
    category: &str,
) -> Vec<(String, String, bool, String)> {
    let mut items = capabilities
        .get("blocks")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter(|block| string_at(block, "category") == category)
                .map(|block| {
                    let type_id = string_at(block, "type_id");
                    let label = string_at(block, "label");
                    let available = block
                        .get("available")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                    let reason = string_at(block, "unavailable_reason");
                    (
                        type_id,
                        if label.is_empty() {
                            workshed_block_label(&string_at(block, "type_id"))
                        } else {
                            label
                        },
                        available,
                        reason,
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if items.is_empty() {
        let fallback: &[(&str, &str, &str)] = match category {
            "models" => &[("load_model", "load base model", "")],
            "data" => &[("load_dataset", "prepare dataset", "")],
            "distill" => &[
                (
                    "generate_teacher_responses",
                    "generate teacher responses",
                    "",
                ),
                ("response_distill", "distill responses", ""),
                (
                    "logits_distill",
                    "distill compatible logits",
                    "Teacher/student tokenizers must match.",
                ),
            ],
            "train" => &[
                ("train_sft", "train with SFT", ""),
                ("train_lora", "train with LoRA", ""),
                ("train_qlora", "train with QLoRA", ""),
                ("train_dora", "train with DoRA", ""),
                ("train_full", "full fine-tune", ""),
                ("train_dpo", "train with DPO", ""),
            ],
            "quantize" => &[("quantize_mlx", "quantize with MLX", "")],
            "test_compare" => &[
                ("evaluate_loss", "evaluate loss", ""),
                ("evaluate_lm", "run reproducible lm-eval", ""),
                ("benchmark_model", "benchmark speed and memory", ""),
                ("compare_models", "compare candidates", ""),
                ("quality_gate", "quality gate", ""),
            ],
            "control" => &[
                ("repeat", "repeat N times", ""),
                ("for_each", "for each item", ""),
                ("retry", "retry N times", ""),
            ],
            "logic" => &[
                ("if_else", "if / else", ""),
                ("stop", "stop work order", ""),
            ],
            "deliver" => &[
                ("register_model", "register new model", ""),
                ("reveal_artifact", "show in Finder", ""),
                ("export_report", "export report", ""),
            ],
            _ => &[],
        };
        items = fallback
            .iter()
            .map(|(id, label, reason)| {
                (
                    (*id).into(),
                    (*label).into(),
                    reason.is_empty(),
                    (*reason).into(),
                )
            })
            .collect();
    }
    items
}

fn workshed_add_block(order: &mut Value, capabilities: &Value, type_id: &str) {
    let definition = workshed::definition_for(capabilities, type_id);
    let preferred_lane = match definition.category.as_str() {
        "models" | "data" => "source",
        "distill" | "train" | "control" | "logic" => "improve",
        "quantize" | "test_compare" => "verify",
        "deliver" => "deliver",
        _ => "improve",
    };
    let lane = if definition.allowed_lanes.is_empty()
        || definition
            .allowed_lanes
            .iter()
            .any(|candidate| candidate == preferred_lane)
    {
        preferred_lane.to_string()
    } else {
        definition.allowed_lanes[0].clone()
    };
    let id = format!("{}_{}", type_id, uuid_like_id());
    if let Some(blocks) = order.get_mut("blocks").and_then(Value::as_array_mut) {
        let order_index = blocks.len() as u64;
        blocks.push(json!({
            "id": id,
            "type_id": type_id,
            "lane": lane,
            "order": order_index,
            "definition_version": definition.definition_version,
            "params": {},
            "param_origins": {},
            "input_bindings": {}
        }));
    }
    let blocks = order
        .get("blocks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let Some(new_block) = blocks.iter().find(|block| string_at(block, "id") == id) else {
        return;
    };
    let mut auto_edges = Vec::new();
    for input in &definition.inputs {
        let upstream = blocks.iter().rev().skip(1).find_map(|candidate| {
            let candidate_id = string_at(candidate, "id");
            let candidate_type = string_at(candidate, "type_id");
            let candidate_definition = workshed::definition_for(capabilities, &candidate_type);
            candidate_definition
                .outputs
                .iter()
                .find(|output| workshed::compatible_types(&output.type_name, &input.type_name))
                .map(|output| (candidate_id, output.name.clone(), output.type_name.clone()))
        });
        if let Some((source_id, source_port, source_type)) = upstream {
            auto_edges.push(json!({
                "from": {"block_id": source_id, "port": source_port, "type": source_type},
                "to": {"block_id": string_at(new_block, "id"), "port": input.name, "type": input.type_name}
            }));
        }
    }
    if let Some(edges) = order.get_mut("edges").and_then(Value::as_array_mut) {
        edges.extend(auto_edges);
    }
}

fn workshed_move_block(order: &mut Value, block_id: &str, direction: i8) -> bool {
    let Some(blocks) = order.get_mut("blocks").and_then(Value::as_array_mut) else {
        return false;
    };
    let Some(index) = blocks
        .iter()
        .position(|block| string_at(block, "id") == block_id)
    else {
        return false;
    };
    if blocks[index]
        .get("locked")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return false;
    }
    let lane = string_at(&blocks[index], "lane");
    let target = if direction < 0 {
        (0..index)
            .rev()
            .find(|candidate| string_at(&blocks[*candidate], "lane") == lane)
    } else {
        ((index + 1)..blocks.len()).find(|candidate| string_at(&blocks[*candidate], "lane") == lane)
    };
    let Some(target) = target else {
        return false;
    };
    blocks.swap(index, target);
    for (order_index, block) in blocks.iter_mut().enumerate() {
        block["order"] = Value::from(order_index as u64);
    }
    true
}

fn workshed_duplicate_block(order: &mut Value, block_id: &str) -> Option<String> {
    let blocks = order.get_mut("blocks")?.as_array_mut()?;
    let index = blocks
        .iter()
        .position(|block| string_at(block, "id") == block_id)?;
    if blocks[index]
        .get("locked")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    let mut duplicate = blocks[index].clone();
    let type_id = string_at(&duplicate, "type_id");
    let new_id = format!("{}_{}", type_id, uuid_like_id());
    duplicate["id"] = Value::String(new_id.clone());
    duplicate["order"] = Value::from((index + 1) as u64);
    blocks.insert(index + 1, duplicate);
    for (order_index, block) in blocks.iter_mut().enumerate() {
        block["order"] = Value::from(order_index as u64);
    }
    Some(new_id)
}

fn workshed_delete_block(order: &mut Value, block_id: &str) {
    if let Some(blocks) = order.get_mut("blocks").and_then(Value::as_array_mut) {
        if blocks.iter().any(|block| {
            string_at(block, "id") == block_id
                && block
                    .get("locked")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        }) {
            return;
        }
        blocks.retain(|block| string_at(block, "id") != block_id);
    }
    if let Some(edges) = order.get_mut("edges").and_then(Value::as_array_mut) {
        edges.retain(|edge| {
            string_at(edge.get("from").unwrap_or(&Value::Null), "block_id") != block_id
                && string_at(edge.get("to").unwrap_or(&Value::Null), "block_id") != block_id
        });
    }
}

fn uuid_like_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", now & 0xffffffffff)
}

fn workshed_draft_key(block_id: &str, field: &str) -> String {
    format!("{block_id}\u{1f}{field}")
}

fn workshed_option_key(block_id: &str, field: &str) -> String {
    workshed_draft_key(block_id, &format!("option:{field}"))
}

fn workshed_reference_placeholder(type_id: &str, source_kind: &str) -> String {
    match (type_id, source_kind) {
        ("load_model", "huggingface") => {
            "Hugging Face model ID, for example mlx-community/Qwen3-4B".into()
        }
        ("load_model", "local") => "Local cached model ID or model folder".into(),
        ("load_model", "managed") => "Managed Workshed model ID".into(),
        ("load_dataset", "huggingface") => "Hugging Face dataset ID".into(),
        ("load_dataset", "local") => "Local JSON, JSONL, Parquet, Arrow, or text path".into(),
        ("load_dataset", "managed") => "Managed Workshed dataset ID".into(),
        _ => "Reference".into(),
    }
}

fn workshed_input_id(block_id: &str, field: &str) -> widget::Id {
    widget::Id::new(format!("workshed-field:{block_id}:{field}"))
}

fn workshed_messages(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    if item.is_string() {
                        string_at(&json!({"value": item}), "value")
                    } else {
                        string_at(item, "message")
                    }
                })
                .filter(|text| !text.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn workshed_card_allows_quick_control(type_id: &str) -> bool {
    !matches!(type_id, "load_model" | "load_dataset")
}

fn workshed_category_icon_name(category: &str) -> &'static str {
    match category {
        "models" => "package-x-generic-symbolic",
        "data" => "x-office-document-symbolic",
        "distill" => "network-transmit-receive-symbolic",
        "train" => "applications-engineering-symbolic",
        "quantize" => "preferences-system-symbolic",
        "test_compare" => "emblem-ok-symbolic",
        "control" => "media-playlist-repeat-symbolic",
        "logic" => "insert-link-symbolic",
        "deliver" => "document-save-symbolic",
        _ => "insert-object-symbolic",
    }
}

fn workshed_edge_summary(count: usize) -> String {
    format!(
        "OUT · {count} {}",
        if count == 1 { "edge" } else { "edges" }
    )
}

fn workshed_port_display(port: &WorkshedCardPort) -> String {
    let label = if port.label.eq_ignore_ascii_case(&port.type_name) {
        port.label.to_ascii_lowercase()
    } else {
        port.label.clone()
    };
    format!("{label} · {}", port.type_name)
}

fn workshed_card_ports(
    order: &Value,
    block_id: &str,
    definition: &BlockDefinition,
    direction: WorkshedPortDirection,
) -> Vec<WorkshedCardPort> {
    let ports = match direction {
        WorkshedPortDirection::Input => &definition.inputs,
        WorkshedPortDirection::Output => &definition.outputs,
    };
    ports
        .iter()
        .map(|port| {
            let edge_count = order
                .get("edges")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|edge| {
                    let endpoint = match direction {
                        WorkshedPortDirection::Input => edge.get("to"),
                        WorkshedPortDirection::Output => edge.get("from"),
                    };
                    endpoint.is_some_and(|endpoint| {
                        string_at(endpoint, "block_id") == block_id
                            && string_at(endpoint, "port") == port.name
                    })
                })
                .count();
            WorkshedCardPort {
                label: port.label.clone(),
                type_name: port.type_name.clone(),
                edge_count,
            }
        })
        .collect()
}

fn workshed_help_color() -> Color {
    Color::from_rgb8(0x6b, 0x6b, 0x6b)
}

fn workshed_color(category: &str) -> Color {
    let _ = category;
    Color::from_rgb8(0x11, 0x11, 0x11)
}

fn workshed_panel_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::from_rgb8(0xf8, 0xfa, 0xfc)))
        .color(workshed_color(""))
        .border(Border {
            width: 1.0,
            radius: CONTENT_RADIUS.into(),
            color: Color::from_rgba(0.07, 0.09, 0.13, 0.08),
        })
        .shadow(Shadow::default())
}

fn workshed_field_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::WHITE))
        .color(workshed_color(""))
        .border(Border {
            width: 1.0,
            radius: 6.0.into(),
            color: Color::from_rgba(0.07, 0.09, 0.13, 0.08),
        })
        .shadow(Shadow::default())
}

fn workshed_option_style() -> container::Style {
    container::Style::default()
        .background(Background::Color(Color::from_rgb8(0xf8, 0xfa, 0xfc)))
        .color(workshed_color(""))
        .border(Border {
            width: 1.0,
            radius: 6.0.into(),
            color: Color::from_rgba(0.07, 0.09, 0.13, 0.06),
        })
        .shadow(Shadow::default())
}

fn workshed_lane_style(category: &str) -> container::Style {
    let mut style = workshed_panel_style();
    let _ = category;
    style.border.color = Color::from_rgba(0.07, 0.09, 0.13, 0.08);
    style
}

fn workshed_block_style(category: &str, strong: bool) -> container::Style {
    let mut style = workshed_panel_style();
    style.background = Some(Background::Color(Color::from_rgb8(0xf9, 0xfa, 0xfc)));
    style.border.width = if strong { 1.5 } else { 1.0 };
    style.border.color = if strong {
        let accent = workshed_color(category);
        Color::from_rgba(accent.r, accent.g, accent.b, 0.62)
    } else {
        Color::from_rgba(0.07, 0.09, 0.13, 0.10)
    };
    style
}

fn workshed_accent_style(category: &str) -> container::Style {
    container::Style::default().background(Background::Color(workshed_color(category)))
}

fn workshed_category_stripe_style(category: &str, selected: bool) -> container::Style {
    container::Style::default()
        .background(Background::Color(if selected {
            workshed_color(category)
        } else {
            Color::TRANSPARENT
        }))
        .border(Border {
            width: 0.0,
            radius: 2.0.into(),
            color: Color::TRANSPARENT,
        })
}

fn workshed_category_button_style(
    selected: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let background = match status {
            button::Status::Pressed => Color::from_rgb8(0xe7, 0xeb, 0xf0),
            button::Status::Hovered => Color::from_rgb8(0xf0, 0xf3, 0xf7),
            _ if selected => Color::from_rgb8(0xed, 0xf1, 0xf5),
            _ => Color::TRANSPARENT,
        };
        let mut style = button::Style::default().with_background(background);
        style.text_color = Color::from_rgb8(0x11, 0x11, 0x11);
        style.border = Border {
            width: 0.0,
            radius: 6.0.into(),
            color: Color::TRANSPARENT,
        };
        style.shadow = Shadow::default();
        style
    }
}

fn workshed_port_dot_style(category: &str, connected: bool) -> container::Style {
    let accent = workshed_color(category);
    container::Style::default()
        .background(Background::Color(if connected {
            accent
        } else {
            Color::from_rgb8(0xf9, 0xfa, 0xfc)
        }))
        .border(Border {
            width: 1.0,
            radius: 999.0.into(),
            color: if connected {
                accent
            } else {
                Color::from_rgb8(0xa9, 0xb1, 0xbc)
            },
        })
}

fn workshed_port_line_style(category: &str, connected: bool) -> container::Style {
    let accent = workshed_color(category);
    container::Style::default().background(Background::Color(if connected {
        Color::from_rgba(accent.r, accent.g, accent.b, 0.72)
    } else {
        Color::from_rgb8(0xc6, 0xcc, 0xd4)
    }))
}

fn workshed_unavailable_style() -> container::Style {
    let mut style = workshed_panel_style();
    style.background = Some(Background::Color(Color::from_rgb8(0xf0, 0xf2, 0xf5)));
    style
}

fn workshed_run_bar_style() -> container::Style {
    let mut style = workshed_panel_style();
    style.background = Some(Background::Color(Color::from_rgb8(0xf9, 0xfa, 0xfc)));
    style
}

fn workshed_block_button_style(
    palette: crate::appearance::Palette,
    _selected: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style + Clone + 'static {
    move |_theme, status| {
        let background = match status {
            button::Status::Hovered => Color::from_rgb8(0xf3, 0xf5, 0xf8),
            button::Status::Pressed => Color::from_rgb8(0xec, 0xf0, 0xf4),
            _ => Color::TRANSPARENT,
        };
        let mut style = button::Style::default().with_background(background);
        style.text_color = palette.text_main;
        style.border = Border {
            width: 0.0,
            radius: CONTENT_RADIUS.into(),
            color: Color::TRANSPARENT,
        };
        style.shadow = Shadow::default();
        style
    }
}

#[cfg(test)]
mod community_job_tests {
    use super::*;

    fn job(status: &str) -> Value {
        json!({"id": "deploy-1", "model": "owner/model", "status": status, "done": false})
    }

    #[test]
    fn absent_null_and_invalid_community_jobs_do_not_create_a_job_card() {
        assert!(normalize_community_job(None).is_none());
        for invalid in [
            Value::Null,
            json!({}),
            json!([]),
            json!(false),
            json!("job"),
            json!({"status": "downloading"}),
            json!({"id": "", "model": "owner/model", "status": "downloading"}),
            json!({"id": "deploy-1", "model": "  ", "status": "downloading"}),
            json!({"id": "deploy-1", "model": "owner/model", "status": null}),
        ] {
            assert!(normalize_community_job(Some(invalid.clone())).is_none());
            assert!(community_job_allowed_actions(&invalid).is_empty());
        }
    }

    #[test]
    fn community_polling_null_clears_a_previously_active_job() {
        let active = normalize_community_job(Some(job("downloading")));
        assert!(active.is_some());
        let polled = normalize_community_job(Some(Value::Null));
        assert!(polled.is_none());
    }

    #[test]
    fn community_controls_match_the_backend_job_state() {
        assert_eq!(
            community_job_allowed_actions(&job("queued")),
            &["pause", "cancel"]
        );
        assert_eq!(
            community_job_allowed_actions(&job("downloading")),
            &["pause", "cancel"]
        );
        assert_eq!(
            community_job_allowed_actions(&job("paused")),
            &["resume", "cancel"]
        );
        assert_eq!(
            community_job_allowed_actions(&job("deploying")),
            &["cancel"]
        );
        for status in ["canceling", "completed", "failed", "canceled", "unknown"] {
            assert!(community_job_allowed_actions(&job(status)).is_empty());
        }
        let mut finished = job("downloading");
        finished["done"] = json!(true);
        assert!(community_job_allowed_actions(&finished).is_empty());
    }

    #[test]
    fn community_summary_uses_download_progress_not_an_unrelated_percent_field() {
        let mut deployment = job("downloading");
        deployment["progress_percent"] = json!(42.5);
        assert!(job_summary(&deployment).ends_with("(42.5%)"));
        deployment["progress_percent"] = json!(120.0);
        assert!(job_summary(&deployment).ends_with("(100.0%)"));
    }
}

#[cfg(test)]
mod agent_profile_tests {
    use super::*;

    #[test]
    fn logs_page_is_presented_as_terminal() {
        assert_eq!(Page::Logs.label(), "Terminal");
    }

    #[test]
    fn workshed_page_is_available_after_model_settings() {
        assert_eq!(Page::ALL[1], Page::Models);
        assert_eq!(Page::ALL[2], Page::Workshed);
        assert_eq!(Page::from_code("quantization"), Page::Workshed);
        assert_eq!(Page::Workshed.label(), "Workshed");
    }

    #[test]
    fn workshed_edit_invalidates_displayed_and_in_flight_preflight() {
        let response_generation = 7;
        let mut generation = response_generation;
        let mut displayed = json!({"ok": true, "preflight_id": "old-plan"});
        TokenWorkshedApp::workshed_invalidate_preflight(&mut displayed, &mut generation);
        assert_eq!(generation, response_generation + 1);
        assert_ne!(generation, response_generation);
        assert_eq!(displayed["ok"], false);
        assert_eq!(displayed["stale"], true);
        assert_eq!(displayed["preflight_id"], "old-plan");

        // A request may be in flight before any plan has been displayed.
        let mut empty = Value::Null;
        TokenWorkshedApp::workshed_invalidate_preflight(&mut empty, &mut generation);
        assert_eq!(generation, response_generation + 2);
        assert_eq!(empty, Value::Null);
    }

    #[test]
    fn workshed_inspector_resolves_preset_without_losing_user_override() {
        let capabilities = json!({"blocks": [{
            "type_id": "quantize_mlx",
            "default_params": {"preset_id": "balanced", "bits": null, "group_size": null},
            "presets": [{"id": "balanced", "params": {"bits": 4, "group_size": 128}}]
        }]});
        let definition = workshed::definition_for(&capabilities, "quantize_mlx");
        let mut block = json!({"id": "quant", "params": {"bits": null, "group_size": null}});
        let defaults =
            TokenWorkshedApp::workshed_inspector_values(&capabilities, &block, &definition);
        assert_eq!(defaults["bits"], 4);
        assert_eq!(defaults["group_size"], 128);
        assert_eq!(block["params"]["bits"], Value::Null);

        block["params"]["bits"] = json!(6);
        let overridden =
            TokenWorkshedApp::workshed_inspector_values(&capabilities, &block, &definition);
        assert_eq!(overridden["bits"], 6);
        assert_eq!(overridden["group_size"], 128);
    }

    #[test]
    fn workshed_local_draft_errors_block_even_a_valid_preflight() {
        let mut errors = BTreeMap::new();
        let mut plan = json!({"ok": true, "expires_at": 4_102_444_800.0});
        assert!(TokenWorkshedApp::workshed_preflight_is_current(
            &plan, &errors
        ));
        errors.insert("quant\u{1f}bits".into(), "Bits is not valid.".into());
        assert!(!TokenWorkshedApp::workshed_preflight_is_current(
            &plan, &errors
        ));
        errors.clear();
        plan["stale"] = json!(true);
        assert!(!TokenWorkshedApp::workshed_preflight_is_current(
            &plan, &errors
        ));
        plan["stale"] = json!(false);
        plan["expires_at"] = json!(0.0);
        assert!(!TokenWorkshedApp::workshed_preflight_is_current(
            &plan, &errors
        ));
    }

    #[test]
    fn workshed_new_orders_never_overwrite_the_existing_local_draft() {
        let original = workshed_demo_order();
        let new_order = workshed_new_order();
        let another = workshed_new_order();
        assert_ne!(string_at(&new_order, "id"), "local-draft");
        assert_ne!(string_at(&new_order, "id"), string_at(&another, "id"));
        assert!(workshed_order_needs_create(
            &new_order,
            std::slice::from_ref(&original)
        ));
        assert!(!workshed_order_needs_create(
            &original,
            std::slice::from_ref(&original)
        ));
        assert!(workshed_order_needs_create(&original, &[]));
        assert_eq!(string_at(&original, "name"), "");
    }

    #[test]
    fn workshed_order_history_preserves_original_and_updates_saved_revision() {
        let original = workshed_demo_order();
        let mut orders = vec![original.clone()];
        let mut new_order = workshed_new_order();
        workshed_remember_order(&mut orders, &new_order);
        new_order["revision"] = json!(1);
        new_order["name"] = json!("UI quantization test");
        workshed_remember_order(&mut orders, &new_order);
        assert_eq!(orders.len(), 2);
        assert_eq!(orders[1], original);
        assert_eq!(orders[0], new_order);
        assert!(!workshed_order_needs_create(&new_order, &orders));
    }

    #[test]
    fn workshed_visual_defaults_keep_preflight_and_source_quick_controls_quiet() {
        assert!(!WORKSHED_PREFLIGHT_DEFAULT_OPEN);
        assert!(!workshed_card_allows_quick_control("load_model"));
        assert!(!workshed_card_allows_quick_control("load_dataset"));
        assert!(workshed_card_allows_quick_control("train_lora"));
        assert_eq!(WORKSHED_CATEGORY_STRIPE_WIDTH, 3.0);
        assert_eq!(workshed_help_color(), Color::from_rgb8(0x6b, 0x6b, 0x6b));
        assert_eq!(
            workshed_category_icon_name("models"),
            "package-x-generic-symbolic"
        );
        assert_eq!(workshed_edge_summary(1), "OUT · 1 edge");
        assert_eq!(workshed_edge_summary(2), "OUT · 2 edges");
        assert_eq!(
            workshed_port_display(&WorkshedCardPort {
                label: "Model".into(),
                type_name: "Model".into(),
                edge_count: 1,
            }),
            "model · Model"
        );
    }

    #[test]
    fn workshed_card_ports_report_typed_dag_edges() {
        let capabilities = json!({"blocks": [{
            "type_id": "train_lora",
            "category": "train",
            "label": "train with LoRA",
            "param_schema": {"type": "object", "properties": {}},
            "inputs": [
                {"name": "model", "label": "Base", "type": "Model"},
                {"name": "dataset", "label": "Training data", "type": "Dataset"}
            ],
            "outputs": [{"name": "adapter", "label": "Adapter", "type": "Adapter"}]
        }]});
        let definition = workshed::definition_for(&capabilities, "train_lora");
        let order = json!({"edges": [
            {"from": {"block_id": "model", "port": "model"}, "to": {"block_id": "train", "port": "model"}},
            {"from": {"block_id": "train", "port": "adapter"}, "to": {"block_id": "fuse", "port": "adapter"}},
            {"from": {"block_id": "train", "port": "adapter"}, "to": {"block_id": "save", "port": "adapter"}}
        ]});

        assert_eq!(
            workshed_card_ports(&order, "train", &definition, WorkshedPortDirection::Input),
            vec![
                WorkshedCardPort {
                    label: "Base".into(),
                    type_name: "Model".into(),
                    edge_count: 1,
                },
                WorkshedCardPort {
                    label: "Training data".into(),
                    type_name: "Dataset".into(),
                    edge_count: 0,
                }
            ]
        );
        assert_eq!(
            workshed_card_ports(&order, "train", &definition, WorkshedPortDirection::Output),
            vec![WorkshedCardPort {
                label: "Adapter".into(),
                type_name: "Adapter".into(),
                edge_count: 2,
            }]
        );
    }

    #[test]
    fn workshed_selection_uses_one_light_caligo_border_layer() {
        let selected = workshed_block_style("train", true);
        assert_eq!(selected.border.width, 1.5);
        assert_ne!(selected.border.color, Color::from_rgb8(0x11, 0x11, 0x11));

        let category =
            workshed_category_button_style(true)(&Theme::default(), button::Status::Active);
        assert_eq!(category.border.width, 0.0);
        assert_eq!(
            category.background,
            Some(Background::Color(Color::from_rgb8(0xed, 0xf1, 0xf5)))
        );

        let inner = workshed_block_button_style(palette(MaterialMode::Frosted, true), true)(
            &Theme::default(),
            button::Status::Active,
        );
        assert_eq!(inner.border.width, 0.0);
        assert_eq!(inner.border.color, Color::TRANSPARENT);
    }

    #[test]
    fn profile_progress_interpolates_from_gray_to_white() {
        let start = agent_profile_progress_color(0.0);
        let midpoint = agent_profile_progress_color(50.0);
        let complete = agent_profile_progress_color(100.0);

        assert!((start.r - (0x73 as f32 / 255.0)).abs() < f32::EPSILON);
        assert!(midpoint.r > start.r);
        assert!(midpoint.g > start.g);
        assert!(midpoint.b > start.b);
        assert!((complete.r - 1.0).abs() < f32::EPSILON);
        assert!((complete.g - 1.0).abs() < f32::EPSILON);
        assert!((complete.b - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn profile_state_only_enables_runtime_after_completion_transition() {
        let pending = AgentProfileState::Configured {
            profile: Value::Null,
            completed_at: Some(Instant::now()),
        };
        let ready = AgentProfileState::Configured {
            profile: Value::Null,
            completed_at: None,
        };
        let configuring = AgentProfileState::Configuring {
            job_id: "job-1".into(),
            progress: 55.0,
            phase: "hermes probe".into(),
        };

        assert!(pending.completion_active());
        assert!(!pending.is_configured());
        assert!(ready.is_configured());
        assert!(!ready.completion_active());
        assert!(configuring.is_configuring());
    }

    #[test]
    fn profile_button_uses_the_runtime_groups_fixed_footprint() {
        assert_eq!(AGENT_RUNTIME_GROUP_WIDTH, 178.0);
        assert_eq!(AGENT_RUNTIME_GROUP_HEIGHT, 38.0);
        assert_eq!(AGENT_RUNTIME_BUTTON_WIDTH + 4.0, AGENT_RUNTIME_GROUP_WIDTH);
        assert_eq!(
            AGENT_RUNTIME_BUTTON_HEIGHT + 6.0,
            AGENT_RUNTIME_GROUP_HEIGHT
        );
    }

    #[test]
    fn profile_results_follow_the_current_model_across_aliases() {
        assert!(same_agent_profile_model(
            "lmstudio-community/Qwen3.5-2B",
            "IMStudio-Community/qwen3.5-2b",
        ));
        assert!(!same_agent_profile_model("owner/one", "owner/two"));
    }

    #[test]
    fn degraded_profile_is_marked_as_limited_tools() {
        let profile = serde_json::json!({
            "openclaw": {"tool_calling": "passed"},
            "hermes": {"tool_calling": "degraded"},
        });
        assert!(profile_has_degraded_tools(&profile));
    }

    #[test]
    fn thought_duration_is_compact_and_stable() {
        assert_eq!(
            format_thought_duration(Duration::from_millis(1530)),
            "thought for 1.5 seconds"
        );
    }
}
