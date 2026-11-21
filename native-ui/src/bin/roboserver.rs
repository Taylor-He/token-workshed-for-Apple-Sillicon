#[path = "../api.rs"]
mod api;

use api::{ApiClient, CommunitySearchPayload, ManagerModelsPayload, StatusPayload, string_at};
use cosmic::app::Settings;
use cosmic::iced::{Alignment, Font, Length, Size, Subscription, widget};
use cosmic::{Core, Element, executor, iced};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

const MANAGER_URL: &str = "http://127.0.0.1:17872";

#[derive(Debug, Clone)]
enum Message {
    Poll,
    Refresh,
    ModelsLoaded(Result<ManagerModelsPayload, String>),
    StatusLoaded(Result<StatusPayload, String>),
    PerformanceLoaded(Result<Value, String>),
    JobLoaded(Result<Option<Value>, String>),
    SearchChanged(String),
    SearchPressed,
    SearchLoaded(Result<CommunitySearchPayload, String>),
    DownloadPressed(String),
    DownloadLoaded(Result<Option<Value>, String>),
    JobActionPressed(String),
    JobActionLoaded(Result<Option<Value>, String>),
    SwitchModelPressed(String),
    ModelSwitchLoaded(Result<ManagerModelsPayload, String>),
    ServicePressed(String),
    RunnerStatusLoaded(Result<String, String>),
    ServiceActionLoaded(Result<String, String>),
}

struct RoboServerApp {
    core: Core,
    runner: PathBuf,
    manager_token: String,
    agent_token: String,
    query: String,
    searching: bool,
    service_action: bool,
    service_loaded: bool,
    api_ready: bool,
    inference_ready: bool,
    manager: ManagerModelsPayload,
    status: StatusPayload,
    performance: Value,
    community: CommunitySearchPayload,
    job: Option<Value>,
    notice: String,
    logs: Vec<String>,
}

impl RoboServerApp {
    fn client(&self) -> ApiClient {
        ApiClient::new(MANAGER_URL, self.manager_token.clone())
            .with_agent_token(self.agent_token.clone())
    }

    fn poll_task(&self) -> cosmic::app::Task<Message> {
        if self.manager_token.is_empty() || (!self.service_loaded && !self.api_ready) {
            return cosmic::app::Task::none();
        }
        let client = self.client();
        let models = cosmic::Task::perform(client.clone().manager_models(), Message::ModelsLoaded);
        let status = cosmic::Task::perform(
            client.clone().status("http://127.0.0.1:18010".into()),
            Message::StatusLoaded,
        );
        let job = cosmic::Task::perform(client.clone().community_job(), Message::JobLoaded);
        let performance = cosmic::Task::perform(client.performance(), Message::PerformanceLoaded);
        cosmic::Task::batch(vec![models, status, job, performance]).map(cosmic::Action::App)
    }

    fn refresh_task(&self) -> cosmic::app::Task<Message> {
        let runner = self.runner.clone();
        let service = cosmic::Task::perform(
            run_runner_async(runner, vec!["--status".into()]),
            Message::RunnerStatusLoaded,
        )
        .map(cosmic::Action::App);
        cosmic::app::Task::batch(vec![service, self.poll_task()])
    }

    fn service_status_text(&self) -> String {
        if self.service_action {
            return "Preparing server runtime…".into();
        }
        if self.api_ready {
            "Online · private loopback API".into()
        } else if self.service_loaded {
            "Starting · API is not ready yet".into()
        } else {
            "Stopped".into()
        }
    }

    fn model_list(&self) -> Element<'_, Message> {
        let models = if self.manager.available_models.is_empty() {
            local_model_cache_ids()
        } else {
            self.manager.available_models.clone()
        };
        let mut rows = widget::column::with_capacity(models.len().max(1) + 1).spacing(6);
        if models.is_empty() {
            rows = rows.push(widget::text("No downloaded models on this server yet."));
        } else if !self.api_ready {
            rows = rows.push(
                widget::text("Cached on this server · start the service to run a model.").size(11),
            );
        }
        for model in &models {
            let active = model == &self.manager.active_model;
            let caption = if active {
                format!("{model}  ·  active")
            } else {
                format!("{model}  ·  select")
            };
            let button = widget::button(widget::text(caption))
                .width(Length::Fill)
                .on_press_maybe(
                    (self.api_ready && !active).then(|| Message::SwitchModelPressed(model.clone())),
                );
            rows = rows.push(button);
        }
        widget::container(rows).width(Length::Fill).into()
    }

    fn community_results(&self) -> Element<'_, Message> {
        let mut rows =
            widget::column::with_capacity(self.community.results.len().min(10) + 1).spacing(8);
        for item in self.community.results.iter().take(10) {
            let id = string_at(item, "id");
            if id.is_empty() {
                continue;
            }
            let pullable = item
                .get("pullable")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let local = item.get("local").and_then(Value::as_bool).unwrap_or(false);
            let downloads = item
                .get("downloads")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let details = if local {
                "Already on this server".to_string()
            } else if !pullable {
                "Not available in a supported Transformers format".to_string()
            } else {
                format!("SGLang Runtime · {downloads} downloads")
            };
            let action = if local {
                "Installed"
            } else if pullable {
                "Download"
            } else {
                "Unavailable"
            };
            let title = widget::text(id.clone()).size(14);
            let subtitle = widget::text(details).size(12);
            let button = widget::button(widget::text(action)).on_press_maybe(
                (self.api_ready && pullable && !local).then(|| Message::DownloadPressed(id)),
            );
            rows = rows.push(
                widget::row::with_capacity(3)
                    .push(
                        widget::column::with_capacity(2)
                            .push(title)
                            .push(subtitle)
                            .spacing(3),
                    )
                    .push(widget::Space::new().width(Length::Fill))
                    .push(button)
                    .spacing(10)
                    .align_y(Alignment::Center),
            );
        }
        widget::container(rows).width(Length::Fill).into()
    }

    fn download_panel(&self) -> Element<'_, Message> {
        let Some(job) = self.job.as_ref() else {
            return widget::text("No active downloads.").size(12).into();
        };
        let title = string_at(job, "model");
        let status = [
            string_at(job, "progress_message"),
            string_at(job, "message"),
        ]
        .into_iter()
        .find(|message| !message.trim().is_empty())
        .unwrap_or_else(|| "Preparing download…".into());
        let percent = job
            .get("progress_percent")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 100.0) as f32;
        let running = !job.get("done").and_then(Value::as_bool).unwrap_or(true);
        let mut heading = widget::row::with_capacity(3)
            .push(widget::text(title).size(13))
            .push(widget::Space::new().width(Length::Fill))
            .push(widget::text(format!("{percent:.0}%")).size(12));
        if running {
            heading = heading.push(
                widget::button(widget::text("Cancel"))
                    .on_press(Message::JobActionPressed("cancel".into())),
            );
        }
        widget::column::with_capacity(3)
            .push(heading.spacing(8).align_y(Alignment::Center))
            .push(widget::progress_bar(0.0..=100.0, percent))
            .push(widget::text(status).size(12))
            .spacing(7)
            .into()
    }

    fn logs_panel(&self) -> Element<'_, Message> {
        let content = if self.logs.is_empty() {
            "Service log is empty.".to_string()
        } else {
            self.logs
                .iter()
                .rev()
                .take(16)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        };
        widget::scrollable(widget::text(content).size(11).font(Font::MONOSPACE))
            .height(Length::Fixed(220.0))
            .into()
    }

    fn performance_panel(&self) -> Element<'_, Message> {
        if !self.api_ready {
            return widget::text("Start RoboServer to measure live resource use.")
                .size(12)
                .into();
        }
        if self.performance.get("ok").and_then(Value::as_bool) != Some(true) {
            let error = string_at(&self.performance, "error");
            return widget::text(if error.is_empty() {
                "Waiting for the first resource sample…".to_string()
            } else {
                error
            })
            .size(12)
            .into();
        }

        let cpu = self
            .performance
            .get("cpu_utilization_percent")
            .and_then(Value::as_f64);
        let cpu_count = self
            .performance
            .get("logical_cpu_count")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let memory_used = self
            .performance
            .get("unified_memory_used_gb")
            .and_then(Value::as_f64);
        let memory_total = self
            .performance
            .get("unified_memory_total_gb")
            .and_then(Value::as_f64);
        let memory_percent = self
            .performance
            .get("unified_memory_percent")
            .and_then(Value::as_f64);

        let cpu_detail = cpu
            .map(|value| format!("{value:.1}% · {cpu_count} cores"))
            .unwrap_or_else(|| "Unavailable".into());
        let memory_detail = match (memory_used, memory_total, memory_percent) {
            (Some(used), Some(total), Some(percent)) => {
                format!("{used:.1} / {total:.1} GB · {percent:.1}%")
            }
            _ => "Unavailable".into(),
        };
        let mut rows = widget::column::with_capacity(4)
            .push(performance_metric("CPU use", cpu_detail, cpu))
            .push(performance_metric(
                "Unified memory",
                memory_detail,
                memory_percent,
            ));

        let model_memory = self
            .status
            .runtime
            .get("metal_active_memory_gb")
            .and_then(Value::as_f64);
        let model_percent = model_memory.zip(memory_total).and_then(|(used, total)| {
            (total > 0.0).then_some((used / total * 100.0).clamp(0.0, 100.0))
        });
        let model_detail = match (model_memory, model_percent) {
            (Some(used), Some(percent)) => format!("{used:.1} GB · {percent:.1}% of total"),
            _ => "Not reported by the active SGLang backend".into(),
        };
        rows = rows
            .push(performance_metric(
                "SGLang model memory",
                model_detail,
                model_percent,
            ))
            .push(
                widget::text(
                    "CPU and unified memory are host-wide. SGLang model memory appears when reported; GPU compute utilization is not exposed by this backend.",
                )
                .size(11),
            );
        rows.spacing(12).into()
    }
}

impl cosmic::Application for RoboServerApp {
    type Executor = executor::Default;
    type Flags = ();
    type Message = Message;

    const APP_ID: &'static str = "com.taylorhe.tokenworkshed.roboserver";

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(core: Core, _flags: Self::Flags) -> (Self, cosmic::app::Task<Self::Message>) {
        let runner = runner_path();
        let (manager_token, agent_token, credential_notice) = match local_connection_credentials() {
            Ok((manager_token, agent_token)) => (
                manager_token,
                agent_token,
                "Ready. Start the private server when you need it.".to_string(),
            ),
            Err(error) => (String::new(), String::new(), error),
        };
        let app = Self {
            core,
            runner,
            manager_token,
            agent_token,
            query: String::new(),
            searching: false,
            service_action: false,
            service_loaded: false,
            api_ready: false,
            inference_ready: false,
            manager: ManagerModelsPayload::default(),
            status: StatusPayload::default(),
            performance: Value::Null,
            community: CommunitySearchPayload::default(),
            job: None,
            notice: credential_notice,
            logs: Vec::new(),
        };
        let startup = app.refresh_task();
        (app, startup)
    }

    fn update(&mut self, message: Self::Message) -> cosmic::app::Task<Self::Message> {
        match message {
            Message::Poll => return self.refresh_task(),
            Message::Refresh => return self.refresh_task(),
            Message::ModelsLoaded(result) => match result {
                Ok(value) => {
                    self.manager = value;
                    self.api_ready = true;
                    self.load_logs();
                }
                Err(error) => {
                    self.api_ready = false;
                    if !self.service_action && !self.service_loaded {
                        self.notice = error;
                    }
                }
            },
            Message::StatusLoaded(result) => match result {
                Ok(value) => {
                    self.status = value;
                    self.inference_ready = self.status.ok;
                }
                Err(error) => {
                    self.status.error = error;
                }
            },
            Message::PerformanceLoaded(result) => match result {
                Ok(value) => self.performance = value,
                Err(error) => self.performance = serde_json::json!({"ok": false, "error": error}),
            },
            Message::JobLoaded(result) => match result {
                Ok(job) => self.job = job,
                Err(_) => {}
            },
            Message::SearchChanged(query) => self.query = query,
            Message::SearchPressed => {
                if !self.api_ready {
                    self.notice = "Start RoboServer before searching or downloading models.".into();
                    return cosmic::app::Task::none();
                }
                if self.query.trim().is_empty() || self.searching {
                    return cosmic::app::Task::none();
                }
                self.searching = true;
                self.notice = "Searching Hugging Face for SGLang-compatible checkpoints…".into();
                return wrap_app_task(cosmic::Task::perform(
                    self.client()
                        .community_search(self.query.trim().to_string()),
                    Message::SearchLoaded,
                ));
            }
            Message::SearchLoaded(result) => {
                self.searching = false;
                match result {
                    Ok(payload) => {
                        self.community = payload;
                        self.notice = format!("{} checkpoints found.", self.community.count);
                    }
                    Err(error) => self.notice = format!("Search failed: {error}"),
                }
            }
            Message::DownloadPressed(model) => {
                self.notice = format!("Starting server-side download for {model}…");
                return wrap_app_task(cosmic::Task::perform(
                    self.client().community_download_model(model),
                    Message::DownloadLoaded,
                ));
            }
            Message::DownloadLoaded(result) => match result {
                Ok(job) => {
                    self.job = job;
                    self.notice = "Download started on RoboServer.".into();
                    return wrap_app_task(cosmic::Task::perform(
                        self.client().community_job(),
                        Message::JobLoaded,
                    ));
                }
                Err(error) => self.notice = format!("Download could not start: {error}"),
            },
            Message::JobActionPressed(action) => {
                return wrap_app_task(cosmic::Task::perform(
                    self.client().community_job_action(action),
                    Message::JobActionLoaded,
                ));
            }
            Message::JobActionLoaded(result) => match result {
                Ok(job) => {
                    self.job = job;
                    self.notice = "Download cancellation requested.".into();
                }
                Err(error) => self.notice = error,
            },
            Message::SwitchModelPressed(model) => {
                self.notice = format!("Starting {model} on RoboServer…");
                return wrap_app_task(cosmic::Task::perform(
                    self.client().switch_model(model),
                    Message::ModelSwitchLoaded,
                ));
            }
            Message::ModelSwitchLoaded(result) => match result {
                Ok(value) => {
                    self.manager = value;
                    self.notice = "Model selection sent to SGLang.".into();
                }
                Err(error) => self.notice = format!("Model could not start: {error}"),
            },
            Message::ServicePressed(action) => {
                self.service_action = true;
                self.notice = if action == "--ensure-service" {
                    "Installing Python 3.12 and SGLang if needed, then starting the service…".into()
                } else {
                    "Stopping SGLang and shutting down the private server…".into()
                };
                return wrap_app_task(cosmic::Task::perform(
                    run_runner_async(self.runner.clone(), vec![action]),
                    Message::ServiceActionLoaded,
                ));
            }
            Message::RunnerStatusLoaded(result) => match result {
                Ok(output) => {
                    if let Ok(value) = serde_json::from_str::<Value>(&output) {
                        self.service_loaded = value
                            .get("service_loaded")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        self.api_ready = value
                            .get("api_ready")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                    }
                }
                Err(error) => {
                    self.service_loaded = false;
                    self.api_ready = false;
                    self.notice = format!("Could not inspect server status: {error}");
                }
            },
            Message::ServiceActionLoaded(result) => {
                self.service_action = false;
                match result {
                    Ok(output) => {
                        if let Ok(value) = serde_json::from_str::<Value>(&output) {
                            self.service_loaded = value
                                .get("service_loaded")
                                .and_then(Value::as_bool)
                                .unwrap_or(self.service_loaded);
                            self.api_ready = value
                                .get("api_ready")
                                .and_then(Value::as_bool)
                                .unwrap_or(self.api_ready);
                            if let Some(error) = value.get("error").and_then(Value::as_str) {
                                self.notice = error.to_string();
                            } else if value.get("ok").and_then(Value::as_bool) == Some(true) {
                                self.notice = "Server state updated.".into();
                            }
                        } else if !output.trim().is_empty() {
                            self.notice = output.trim().to_string();
                        }
                        self.load_logs();
                        return self.poll_task();
                    }
                    Err(error) => {
                        self.notice = format!("Server operation failed: {error}");
                    }
                }
            }
        }
        cosmic::app::Task::none()
    }

    fn subscription(&self) -> Subscription<Self::Message> {
        iced::time::every(Duration::from_secs(2)).map(|_| Message::Poll)
    }

    fn view(&self) -> Element<'_, Self::Message> {
        let header = widget::column::with_capacity(2)
            .push(
                widget::column::with_capacity(2)
                    .push(widget::text("token-workshed RoboServer").size(25))
                    .push(widget::text("Models and SGLang, running on this Mac.").size(13))
                    .spacing(4),
            )
            .push(widget::text("Apple Silicon · macOS 14+ · SGLang Runtime 0.5.22").size(11))
            .spacing(6);

        let service_actions = widget::row::with_capacity(3)
            .push(
                widget::button(widget::text("Start server")).on_press_maybe(
                    (!self.service_action && !self.service_loaded && !self.api_ready)
                        .then(|| Message::ServicePressed("--ensure-service".into())),
                ),
            )
            .push(
                widget::button(widget::text("Stop server")).on_press_maybe(
                    (!self.service_action && self.service_loaded)
                        .then(|| Message::ServicePressed("--stop-service".into())),
                ),
            )
            .push(widget::button(widget::text("Refresh")).on_press(Message::Refresh))
            .spacing(8);
        let status_title = if self.api_ready {
            "Server online".to_string()
        } else if self.service_loaded || self.service_action {
            "Server starting".to_string()
        } else {
            "Server stopped".to_string()
        };
        let inference_label = if self.inference_ready {
            format!("SGLang ready · {}", self.status.model)
        } else if self.api_ready {
            "SGLang is idle · select a model to load it".to_string()
        } else {
            "SGLang starts with the server".to_string()
        };
        let service = widget::row::with_capacity(4)
            .push(
                widget::column::with_capacity(2)
                    .push(widget::text(status_title).size(15))
                    .push(widget::text(inference_label).size(12))
                    .spacing(3),
            )
            .push(widget::Space::new().width(Length::Fill))
            .push(service_actions)
            .align_y(Alignment::Center);

        let models = section(
            "Model library",
            widget::column::with_capacity(2)
                .push(
                    widget::text(if self.manager.active_model.is_empty() {
                        "No model is running".to_string()
                    } else {
                        format!("Active model: {}", self.manager.active_model)
                    })
                    .size(13),
                )
                .push(self.model_list())
                .spacing(8),
        );

        let search = widget::row::with_capacity(2)
            .push(
                widget::text_input("Search Hugging Face model IDs", &self.query)
                    .on_input(Message::SearchChanged)
                    .on_submit(Message::SearchPressed)
                    .width(Length::Fill),
            )
            .push(
                widget::button(widget::text(if self.searching {
                    "Searching…"
                } else {
                    "Search"
                }))
                .on_press_maybe(
                    (!self.searching && self.api_ready).then_some(Message::SearchPressed),
                ),
            )
            .spacing(8);
        let community = section(
            "Download a model",
            widget::column::with_capacity(2)
                .push(search)
                .push(self.community_results())
                .spacing(10),
        );
        let downloads = section("Download progress", self.download_panel());
        let logs = section("Recent server log", self.logs_panel());
        let performance = section("Performance utilization", self.performance_panel());

        let library = widget::column::with_capacity(2)
            .push(models)
            .push(community)
            .spacing(24)
            .width(Length::FillPortion(7));
        let activity = widget::column::with_capacity(2)
            .push(performance)
            .push(downloads)
            .push(logs)
            .spacing(20)
            .width(Length::FillPortion(5));
        let content = widget::column::with_capacity(5)
            .push(header)
            .push(widget::rule::horizontal(1))
            .push(service)
            .push(widget::rule::horizontal(1))
            .push(
                widget::row::with_capacity(2)
                    .push(library)
                    .push(activity)
                    .spacing(28)
                    .align_y(Alignment::Start),
            )
            .spacing(16)
            .max_width(1180);
        widget::container(widget::scrollable(content).height(Length::Fill))
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .into()
    }
}

fn performance_metric(
    label: &'static str,
    detail: String,
    percent: Option<f64>,
) -> Element<'static, Message> {
    let mut content = widget::column::with_capacity(2).push(
        widget::row::with_capacity(3)
            .push(widget::text(label).size(12))
            .push(widget::Space::new().width(Length::Fill))
            .push(widget::text(detail).size(12))
            .spacing(8)
            .align_y(Alignment::Center),
    );
    if let Some(percent) = percent {
        content = content.push(widget::progress_bar(
            0.0..=100.0,
            percent.clamp(0.0, 100.0) as f32,
        ));
    }
    content.spacing(5).width(Length::Fill).into()
}

fn section<'a>(title: &'a str, body: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    widget::column::with_capacity(3)
        .push(widget::text(title).size(14))
        .push(widget::rule::horizontal(1))
        .push(body)
        .spacing(9)
        .width(Length::Fill)
        .into()
}

fn runner_path() -> PathBuf {
    std::env::var_os("TOKEN_WORKSHED_ROBOSERVER_RUNNER")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|path| path.parent().map(PathBuf::from))
                .map(|path| path.join("../Resources/roboserver-runner"))
                .unwrap_or_else(|| PathBuf::from("roboserver-runner"))
        })
}

fn local_connection_credentials() -> Result<(String, String), String> {
    let root = std::env::var_os("TOKEN_WORKSHED_ROBOSERVER_DATA_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| {
                PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
                    .join("token-workshed-roboserver")
            })
        })
        .ok_or_else(|| "Could not resolve the RoboServer data directory.".to_string())?;
    fs::create_dir_all(&root)
        .map_err(|error| format!("Could not create RoboServer data directory: {error}"))?;
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
        .map_err(|error| format!("Could not secure RoboServer data directory: {error}"))?;
    let manager = load_or_create_secret(&root.join(".manager-token"))?;
    let agent = load_or_create_secret(&root.join(".agent-token"))?;
    Ok((manager, agent))
}

fn load_or_create_secret(path: &std::path::Path) -> Result<String, String> {
    if let Ok(value) = fs::read_to_string(path) {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return Ok(value);
        }
    }
    let output = Command::new("/usr/bin/openssl")
        .args(["rand", "-hex", "48"])
        .output()
        .map_err(|error| format!("Could not create a secure RoboServer token: {error}"))?;
    if !output.status.success() {
        return Err("Could not create a secure RoboServer token with macOS OpenSSL.".into());
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.len() != 96 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("macOS OpenSSL returned an invalid RoboServer token.".into());
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    match options.open(path) {
        Ok(mut file) => {
            file.write_all(token.as_bytes())
                .and_then(|_| file.write_all(b"\n"))
                .map_err(|error| format!("Could not save RoboServer token: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("Could not flush RoboServer token: {error}"))?;
            Ok(token)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => fs::read_to_string(path)
            .map(|value| value.trim().to_string())
            .map_err(|read_error| format!("Could not read RoboServer token: {read_error}")),
        Err(error) => Err(format!("Could not save RoboServer token: {error}")),
    }
}

fn run_runner(runner: &PathBuf, args: &[String]) -> Result<String, String> {
    let result = Command::new(runner)
        .args(args)
        .output()
        .map_err(|error| format!("Could not launch RoboServer runner: {error}"))?;
    let stdout = String::from_utf8_lossy(&result.stdout).trim().to_string();
    if result.status.success() {
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&result.stderr).trim().to_string();
        Err(if stderr.is_empty() { stdout } else { stderr })
    }
}

fn wrap_app_task(task: cosmic::Task<Message>) -> cosmic::app::Task<Message> {
    task.map(cosmic::Action::App)
}

async fn run_runner_async(runner: PathBuf, args: Vec<String>) -> Result<String, String> {
    let (sender, receiver) = iced::futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let _ = sender.send(run_runner(&runner, &args));
    });
    receiver
        .await
        .unwrap_or_else(|_| Err("RoboServer operation worker stopped.".into()))
}

impl RoboServerApp {
    fn load_logs(&mut self) {
        let root = std::env::var_os("TOKEN_WORKSHED_ROBOSERVER_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs_path().join("Library/Application Support/token-workshed-roboserver")
            });
        let path = root.join("logs/roboserver.log");
        let Ok(contents) = std::fs::read_to_string(path) else {
            self.logs.clear();
            return;
        };
        self.logs = contents
            .lines()
            .rev()
            .take(40)
            .map(str::to_string)
            .collect();
        self.logs.reverse();
    }
}

fn dirs_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/Shared"))
}

fn local_model_cache_ids() -> Vec<String> {
    let data_root = std::env::var_os("TOKEN_WORKSHED_ROBOSERVER_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs_path().join("Library/Application Support/token-workshed-roboserver")
        });
    let hub = data_root.join("models/huggingface/hub");
    let Ok(entries) = fs::read_dir(hub) else {
        return Vec::new();
    };
    let mut models = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let cache_name = entry.file_name().to_string_lossy().into_owned();
            let model_id = cache_name.strip_prefix("models--")?.replace("--", "/");
            if model_id.is_empty() || !path.is_dir() {
                return None;
            }
            let snapshots = fs::read_dir(path.join("snapshots")).ok()?;
            let has_config = snapshots
                .filter_map(Result::ok)
                .any(|snapshot| snapshot.path().join("config.json").is_file());
            has_config.then_some(model_id)
        })
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    models
}

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
    let settings = Settings::default()
        .size(Size::new(1180.0, 820.0))
        .resizable(Some(8.0));
    cosmic::app::run::<RoboServerApp>(settings, ())?;
    Ok(())
}
