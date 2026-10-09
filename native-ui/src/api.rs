use serde::Serialize;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

const MAX_JSON_RESPONSE_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ApiClient {
    pub api_base: String,
    pub manager_token: String,
}

#[derive(Debug, Clone, Default)]
pub struct ConfigPayload {
    pub server_url: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub token_workshed_version: String,
    pub vllm_mlx_version: String,
    pub manager_auth_required: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeConfig {
    pub continuous_batching: bool,
    pub use_paged_cache: bool,
    pub kv_cache_quantization: bool,
    pub chunked_prefill_tokens: String,
    pub enable_mtp: bool,
    pub mtp_draft_tokens: String,
}

#[derive(Debug, Clone, Default)]
pub struct RunningModel {
    pub model: String,
    pub server_url: String,
    pub role: String,
}

#[derive(Debug, Clone, Default)]
pub struct ModelDetail {
    pub id: String,
    pub display_name: String,
    pub status: String,
    pub source: String,
    pub backend: String,
    pub quantization: String,
    pub active: bool,
    pub running: bool,
    pub server_url: String,
    pub role: String,
    pub kind: String,
    pub reasoning_parser: String,
    pub tool_call_parser: String,
}

#[derive(Debug, Clone, Default)]
pub struct ManagerModelsPayload {
    pub ok: bool,
    pub server_url: String,
    pub active_model: String,
    pub active_server_url: String,
    pub available_models: Vec<String>,
    pub model_details: Vec<ModelDetail>,
    pub running_models: Vec<RunningModel>,
    pub concurrent_models: Vec<String>,
    pub runtime_config: RuntimeConfig,
    pub community_job: Option<Value>,
    pub integrity_report: Value,
    pub last_error: String,
}

#[derive(Debug, Clone, Default)]
pub struct StatusPayload {
    pub ok: bool,
    pub server_url: String,
    pub model: String,
    pub models: Vec<String>,
    pub runtime: Value,
    pub error: String,
}

#[derive(Debug, Clone, Default)]
pub struct CommunitySearchPayload {
    pub results: Vec<Value>,
    pub count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct QuantizationCapabilities {
    pub ok: bool,
    pub engines: Vec<Value>,
    pub workers: Vec<Value>,
    pub warnings: Vec<String>,
    pub managed_model_root: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizationRequest {
    pub source: Value,
    pub engine: String,
    pub worker_id: String,
    pub preset_id: String,
    pub overrides: Value,
    pub output_name: String,
}

#[derive(Debug, Clone, Default)]
pub struct QuantizationJob {
    pub id: String,
    pub engine: String,
    pub status: String,
    pub stage: String,
    pub progress: f32,
    pub speed: String,
    pub eta_seconds: Option<u64>,
    pub message: String,
    pub error: String,
    pub artifact: Option<Value>,
    pub delivery_options: Vec<String>,
    pub allowed_actions: Vec<String>,
    pub raw: Value,
}

#[derive(Debug, Clone, Default)]
pub struct QuantizationDelivery {
    pub job_id: String,
    pub action: String,
}

#[derive(Debug, Clone, Default)]
pub struct ChatResult {
    pub reply: String,
    pub model: String,
    pub metrics: Value,
}

#[derive(Debug, Clone, Default)]
pub struct AgentProfilePayload {
    pub model: String,
    pub configured: bool,
    pub profile: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct AgentProfileJob {
    pub job_id: String,
    pub model: String,
    pub state: String,
    pub phase: String,
    pub progress: f32,
    pub error: String,
    pub profile: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct AgentProfileConfigurePayload {
    pub model: String,
    pub job_id: String,
    pub started: bool,
    pub reused: bool,
    pub profile: Option<Value>,
    pub job: Option<AgentProfileJob>,
}

#[derive(Debug, Clone, Default)]
pub struct IdePairingRequest {
    pub pairing_id: String,
    pub client_id: String,
    pub client_name: String,
    pub ide_name: String,
    pub plugin_version: String,
    pub requested_scopes: Vec<String>,
    pub granted_scopes: Vec<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Default)]
pub struct IdeAuthorization {
    pub client_id: String,
    pub client_name: String,
    pub ide_name: String,
    pub plugin_version: String,
    pub scopes: Vec<String>,
    pub requested_scopes: Vec<String>,
    pub granted_scopes: Vec<String>,
    pub approved_at: i64,
}

#[derive(Debug, Clone, Default)]
pub struct IdeConnectionsPayload {
    pub pairings: Vec<IdePairingRequest>,
    pub authorizations: Vec<IdeAuthorization>,
    pub integration: JetBrainsIntegrationPayload,
}

#[derive(Debug, Clone, Default)]
pub struct JetBrainsIntegrationPayload {
    pub provider_url: String,
    pub provider_model: String,
    pub provider_api_key: String,
    pub completion_mode: String,
    pub completion_suffix: bool,
    pub acp_command: String,
    pub acp_args: Vec<String>,
    pub mcp_command: String,
    pub mcp_args: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Start,
    ThinkingDelta(String),
    AnswerDelta(String),
    Done(ChatResult),
    Error(String),
}

impl ApiClient {
    pub fn new(api_base: impl Into<String>, manager_token: impl Into<String>) -> Self {
        Self {
            api_base: trim_slash(api_base.into()),
            manager_token: manager_token.into(),
        }
    }

    pub async fn config(self) -> Result<ConfigPayload, String> {
        let value = self.get_json("/api/config")?;
        Ok(ConfigPayload {
            server_url: string_at(&value, "server_url"),
            max_tokens: u32_at(&value, "max_tokens", 1024),
            temperature: f32_at(&value, "temperature", 0.7),
            token_workshed_version: string_at(&value, "token_workshed_version"),
            vllm_mlx_version: string_at(&value, "vllm_mlx_version"),
            manager_auth_required: bool_at(&value, "manager_auth_required"),
        })
    }

    pub async fn manager_models(self) -> Result<ManagerModelsPayload, String> {
        let value = self.get_json("/api/manager/models")?;
        Ok(parse_manager_models(value))
    }

    pub async fn status(self, server_url: String) -> Result<StatusPayload, String> {
        let path = format!("/api/status?server_url={}", percent_encode(&server_url));
        let value = self.get_json(&path)?;
        Ok(StatusPayload {
            ok: bool_at(&value, "ok"),
            server_url: string_at(&value, "server_url"),
            model: string_at(&value, "model"),
            models: string_array_at(&value, "models"),
            runtime: value.get("runtime").cloned().unwrap_or(Value::Null),
            error: string_at(&value, "error"),
        })
    }

    pub async fn switch_model(self, model: String) -> Result<ManagerModelsPayload, String> {
        let value = self.post_json("/api/manager/switch-model", json!({ "model": model }))?;
        Ok(parse_manager_models(value))
    }

    pub async fn delete_model(self, model: String) -> Result<ManagerModelsPayload, String> {
        let value = self.post_json("/api/manager/delete-model", json!({ "model": model }))?;
        Ok(parse_manager_models(value))
    }

    pub async fn agent_profile(self, model: String) -> Result<AgentProfilePayload, String> {
        let path = format!("/api/agent/profile?model={}", percent_encode(&model));
        let value = self.get_json(&path)?;
        Ok(AgentProfilePayload {
            model: string_at(&value, "model"),
            configured: bool_at(&value, "configured"),
            profile: value.get("profile").cloned().filter(|item| !item.is_null()),
        })
    }

    pub async fn configure_agent_profile(
        self,
        model: String,
        server_url: String,
        force: bool,
    ) -> Result<AgentProfileConfigurePayload, String> {
        let value = self.post_json(
            "/api/agent/profile/configure",
            json!({
                "model": model,
                "server_url": server_url,
                "force": force,
            }),
        )?;
        let job = value.get("job").map(parse_agent_profile_job);
        Ok(AgentProfileConfigurePayload {
            model: string_at(&value, "model"),
            job_id: string_at(&value, "job_id"),
            started: bool_at(&value, "started"),
            reused: bool_at(&value, "reused"),
            profile: value.get("profile").cloned().filter(|item| !item.is_null()),
            job,
        })
    }

    pub async fn agent_profile_job(self, job_id: String) -> Result<AgentProfileJob, String> {
        let path = format!("/api/agent/profile/configure/{}", percent_encode(&job_id));
        let value = self.get_json(&path)?;
        Ok(parse_agent_profile_job(value.get("job").unwrap_or(&value)))
    }

    pub async fn delete_agent_profile(self, model: String) -> Result<bool, String> {
        let path = format!("/api/agent/profile?model={}", percent_encode(&model));
        let value = self.delete_json(&path)?;
        Ok(bool_at(&value, "deleted"))
    }

    pub async fn ide_connections(self) -> Result<IdeConnectionsPayload, String> {
        let pending = self.get_json("/api/ide/v1/pairing/pending")?;
        let authorized = self.get_json("/api/ide/v1/authorizations")?;
        let integration = self
            .get_json("/api/jetbrains/integration")
            .ok()
            .map(parse_jetbrains_integration)
            .unwrap_or_default();
        Ok(IdeConnectionsPayload {
            pairings: parse_ide_pairings(pending.get("pairings").unwrap_or(&Value::Null)),
            authorizations: parse_ide_authorizations(
                authorized.get("authorizations").unwrap_or(&Value::Null),
            ),
            integration,
        })
    }

    pub async fn decide_ide_pairing(
        self,
        pairing_id: String,
        approved: bool,
        grant_all_requested: bool,
    ) -> Result<(), String> {
        let path = format!(
            "/api/ide/v1/pairing/requests/{}/decision",
            percent_encode(&pairing_id)
        );
        self.post_json(
            &path,
            json!({
                "approved": approved,
                "grant_all_requested": grant_all_requested,
            }),
        )?;
        Ok(())
    }

    pub async fn revoke_ide_authorization(self, client_id: String) -> Result<bool, String> {
        let path = format!(
            "/api/ide/v1/authorizations/{}/revoke",
            percent_encode(&client_id)
        );
        let value = self.post_json(&path, json!({}))?;
        Ok(bool_at(&value, "revoked"))
    }

    pub async fn runtime_config(self) -> Result<RuntimeConfig, String> {
        let value = self.get_json("/api/manager/runtime-config")?;
        Ok(parse_runtime_config(
            value.get("runtime_config").unwrap_or(&Value::Null),
        ))
    }

    pub async fn update_runtime_config(self, cfg: RuntimeConfig) -> Result<RuntimeConfig, String> {
        let value = self.post_json(
            "/api/manager/runtime-config",
            json!({
                "continuous_batching": cfg.continuous_batching,
                "use_paged_cache": cfg.use_paged_cache,
                "kv_cache_quantization": cfg.kv_cache_quantization,
                "chunked_prefill_tokens": cfg.chunked_prefill_tokens.parse::<u32>().unwrap_or(0),
                "enable_mtp": cfg.enable_mtp,
                "mtp_draft_tokens": cfg.mtp_draft_tokens.parse::<u32>().unwrap_or(1),
            }),
        )?;
        Ok(parse_runtime_config(
            value.get("runtime_config").unwrap_or(&Value::Null),
        ))
    }

    pub async fn developer_logs(self) -> Result<Vec<String>, String> {
        let value = self.get_json("/api/developer/terminal/output?limit=120")?;
        Ok(string_array_at(&value, "lines"))
    }

    pub async fn developer_prompt(self, prompt: String) -> Result<String, String> {
        let value = self.post_json("/api/developer/prompt", json!({ "prompt": prompt }))?;
        Ok(string_at(&value, "message"))
    }

    pub async fn developer_terminal(self, command: String) -> Result<String, String> {
        let value = self.post_json("/api/developer/terminal", json!({ "command": command }))?;
        Ok(string_at(&value, "message"))
    }

    pub async fn set_concurrent_models(
        self,
        models: Vec<String>,
    ) -> Result<ManagerModelsPayload, String> {
        let value = self.post_json(
            "/api/manager/concurrent-models",
            json!({ "models": models }),
        )?;
        Ok(parse_manager_models(value))
    }

    pub async fn community_search(self, query: String) -> Result<CommunitySearchPayload, String> {
        let path = format!(
            "/api/manager/community/search?q={}&limit=24",
            percent_encode(&query)
        );
        let value = self.get_json(&path)?;
        let results = value
            .get("results")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(CommunitySearchPayload {
            count: value
                .get("count")
                .and_then(Value::as_u64)
                .unwrap_or(results.len() as u64) as usize,
            results,
        })
    }

    pub async fn community_deploy(self, model: String) -> Result<Option<Value>, String> {
        let value = self.post_json(
            "/api/manager/community/download-deploy",
            json!({ "model": model }),
        )?;
        Ok(value.get("job").cloned())
    }

    pub async fn community_job(self) -> Result<Option<Value>, String> {
        let value = self.get_json("/api/manager/community/job")?;
        Ok(value.get("job").cloned())
    }

    pub async fn community_job_action(self, action: String) -> Result<Option<Value>, String> {
        let value = self.post_json(
            "/api/manager/community/job/action",
            json!({ "action": action }),
        )?;
        Ok(value.get("job").cloned())
    }

    pub async fn quantization_capabilities(self) -> Result<QuantizationCapabilities, String> {
        let value = self.get_json("/api/manager/quantization/capabilities")?;
        Ok(parse_quantization_capabilities(value))
    }

    pub async fn quantization_preflight(self, payload: Value) -> Result<Value, String> {
        self.post_json("/api/manager/quantization/preflight", payload)
    }

    pub async fn quantization_start(self, payload: Value) -> Result<QuantizationJob, String> {
        let value = self.post_json("/api/manager/quantization/jobs", payload)?;
        Ok(parse_quantization_job(value.get("job").unwrap_or(&value)))
    }

    pub async fn quantization_job(self, job_id: String) -> Result<QuantizationJob, String> {
        let path = format!("/api/manager/quantization/jobs/{}", percent_encode(&job_id));
        let value = self.get_json(&path)?;
        Ok(parse_quantization_job(value.get("job").unwrap_or(&value)))
    }

    pub async fn quantization_action(
        self,
        job_id: String,
        action: String,
    ) -> Result<QuantizationJob, String> {
        let path = format!(
            "/api/manager/quantization/jobs/{}/action",
            percent_encode(&job_id)
        );
        let value = self.post_json(&path, json!({ "action": action }))?;
        Ok(parse_quantization_job(value.get("job").unwrap_or(&value)))
    }

    pub async fn quantization_delivery(
        self,
        job_id: String,
        action: String,
    ) -> Result<QuantizationJob, String> {
        let path = format!(
            "/api/manager/quantization/jobs/{}/delivery",
            percent_encode(&job_id)
        );
        let value = self.post_json(&path, json!({ "action": action }))?;
        Ok(parse_quantization_job(value.get("job").unwrap_or(&value)))
    }

    pub async fn workshed_capabilities(self) -> Result<Value, String> {
        self.get_json("/api/manager/workshed/capabilities")
    }

    pub async fn workshed_options(
        self,
        provider: String,
        query: String,
        limit: u32,
        context: Value,
    ) -> Result<Value, String> {
        self.post_json(
            "/api/manager/workshed/options",
            json!({
                "provider": provider,
                "query": query,
                "limit": limit,
                "context": context,
            }),
        )
    }

    pub async fn workshed_work_orders(self) -> Result<Vec<Value>, String> {
        let value = self.get_json("/api/manager/workshed/work-orders")?;
        Ok(value
            .get("work_orders")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default())
    }

    pub async fn workshed_create_work_order(self, work_order: Value) -> Result<Value, String> {
        let value = self.post_json(
            "/api/manager/workshed/work-orders",
            json!({"work_order": work_order}),
        )?;
        Ok(value.get("work_order").cloned().unwrap_or(value))
    }

    pub async fn workshed_update_work_order(
        self,
        work_order_id: String,
        work_order: Value,
        expected_revision: u64,
    ) -> Result<Value, String> {
        let path = format!(
            "/api/manager/workshed/work-orders/{}",
            percent_encode(&work_order_id)
        );
        let value = self.put_json(
            &path,
            json!({"work_order": work_order, "expected_revision": expected_revision}),
        )?;
        Ok(value.get("work_order").cloned().unwrap_or(value))
    }

    pub async fn workshed_preflight(self, work_order: Value) -> Result<Value, String> {
        self.post_json(
            "/api/manager/workshed/preflight",
            json!({"work_order": work_order}),
        )
    }

    pub async fn workshed_start(
        self,
        preflight_id: String,
        consents: Vec<String>,
    ) -> Result<Value, String> {
        let value = self.post_json(
            "/api/manager/workshed/runs",
            json!({"preflight_id": preflight_id, "accepted_consent_ids": consents}),
        )?;
        Ok(value.get("run").cloned().unwrap_or(value))
    }

    pub async fn workshed_runs(self) -> Result<Vec<Value>, String> {
        let value = self.get_json("/api/manager/workshed/runs")?;
        Ok(value
            .get("runs")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default())
    }

    pub async fn workshed_run_poll(
        self,
        run_id: String,
        after_event_id: u64,
    ) -> Result<Value, String> {
        let path = format!(
            "/api/manager/workshed/runs/{}?after_event_id={}",
            percent_encode(&run_id),
            after_event_id
        );
        let value = self.get_json(&path)?;
        Ok(value.get("run").cloned().unwrap_or(value))
    }

    pub async fn workshed_run_receipt(self, run_id: String) -> Result<Value, String> {
        let path = format!(
            "/api/manager/workshed/runs/{}/receipt",
            percent_encode(&run_id)
        );
        let value = self.get_json(&path)?;
        Ok(value.get("receipt").cloned().unwrap_or(value))
    }

    pub async fn workshed_action(self, run_id: String, action: String) -> Result<Value, String> {
        let path = format!(
            "/api/manager/workshed/runs/{}/action",
            percent_encode(&run_id)
        );
        let value = self.post_json(&path, json!({"action": action}))?;
        Ok(value.get("run").cloned().unwrap_or(value))
    }

    pub async fn workshed_artifacts(self) -> Result<Vec<Value>, String> {
        let value = self.get_json("/api/manager/workshed/artifacts")?;
        Ok(value
            .get("artifacts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default())
    }

    pub async fn save_hf_binding(self, username: String, token: String) -> Result<String, String> {
        let value = self.post_json(
            "/api/manager/hf-binding",
            json!({ "username": username, "token": token, "seen": true }),
        )?;
        Ok(string_at(&value, "message"))
    }

    pub fn chat_stream<F>(&self, payload: ChatRequest, mut emit: F) -> Result<(), String>
    where
        F: FnMut(ChatStreamEvent),
    {
        let body = serde_json::to_string(&payload).map_err(|e| e.to_string())?;
        let mut reply = String::new();
        let mut model = payload.model.clone();
        let mut metrics = Value::Null;
        let mut terminal_event_seen = false;

        let read_result = self.post_json_lines("/api/chat/stream", body, |value| {
            if terminal_event_seen {
                return;
            }
            let event_type = string_at(&value, "type");
            match event_type.as_str() {
                "start" => {
                    emit(ChatStreamEvent::Start);
                }
                "thinking_delta" => {
                    let text = raw_string_at(&value, "text");
                    if !text.is_empty() {
                        emit(ChatStreamEvent::ThinkingDelta(text));
                    }
                }
                "answer_delta" => {
                    let text = raw_string_at(&value, "text");
                    if !text.is_empty() {
                        reply.push_str(&text);
                        emit(ChatStreamEvent::AnswerDelta(text));
                    }
                }
                "done" => {
                    terminal_event_seen = true;
                    let done_reply = raw_string_at(&value, "reply");
                    if !done_reply.trim().is_empty() {
                        reply = done_reply;
                    }
                    let done_model = string_at(&value, "model");
                    if !done_model.trim().is_empty() {
                        model = done_model;
                    }
                    metrics = value.get("metrics").cloned().unwrap_or(Value::Null);
                    emit(ChatStreamEvent::Done(ChatResult {
                        reply: reply.clone(),
                        model: model.clone(),
                        metrics: metrics.clone(),
                    }));
                }
                "error" => {
                    terminal_event_seen = true;
                    let message = string_at(&value, "message");
                    emit(ChatStreamEvent::Error(if message.is_empty() {
                        "Chat stream failed.".into()
                    } else {
                        message
                    }));
                }
                _ => {}
            }
        });

        finalize_chat_stream(terminal_event_seen, read_result)
    }

    fn get_json(&self, path: &str) -> Result<Value, String> {
        let body = self.request("GET", path, None)?;
        serde_json::from_str(&body).map_err(|e| format!("Invalid JSON from {path}: {e}"))
    }

    fn post_json(&self, path: &str, body: Value) -> Result<Value, String> {
        let body_text = serde_json::to_string(&body).map_err(|e| e.to_string())?;
        let response = self.request("POST", path, Some(body_text))?;
        serde_json::from_str(&response).map_err(|e| format!("Invalid JSON from {path}: {e}"))
    }

    fn put_json(&self, path: &str, body: Value) -> Result<Value, String> {
        let body_text = serde_json::to_string(&body).map_err(|e| e.to_string())?;
        let response = self.request("PUT", path, Some(body_text))?;
        serde_json::from_str(&response).map_err(|e| format!("Invalid JSON from {path}: {e}"))
    }

    fn delete_json(&self, path: &str) -> Result<Value, String> {
        let body = self.request("DELETE", path, None)?;
        serde_json::from_str(&body).map_err(|e| format!("Invalid JSON from {path}: {e}"))
    }

    fn request(&self, method: &str, path: &str, body: Option<String>) -> Result<String, String> {
        let target = parse_http_base(&self.api_base)?;
        let mut stream = connect_tcp(&target)?;
        let long_running = matches!(
            path,
            "/api/manager/switch-model"
                | "/api/manager/delete-model"
                | "/api/manager/concurrent-models"
        );
        stream
            .set_read_timeout(Some(Duration::from_secs(if long_running {
                300
            } else {
                20
            })))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| e.to_string())?;

        let request_path = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{path}")
        };
        let payload = body.unwrap_or_default();
        let mut request = format!(
            "{method} {request_path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAccept: application/json\r\n",
            target.host_header
        );
        if !self.manager_token.trim().is_empty() {
            request.push_str(&format!(
                "X-Token-Workshed-Manager-Token: {}\r\n",
                self.manager_token.trim()
            ));
        }
        if matches!(method, "POST" | "PUT") {
            request.push_str("Content-Type: application/json\r\n");
            request.push_str(&format!("Content-Length: {}\r\n", payload.as_bytes().len()));
        }
        request.push_str("\r\n");
        request.push_str(&payload);

        stream
            .write_all(request.as_bytes())
            .map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        stream
            .take(MAX_JSON_RESPONSE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_JSON_RESPONSE_BYTES {
            return Err("HTTP response exceeded the 32 MiB safety limit.".into());
        }
        let response = String::from_utf8_lossy(&bytes).to_string();
        let Some((head, body)) = response.split_once("\r\n\r\n") else {
            return Err("Malformed HTTP response.".into());
        };
        let status = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .unwrap_or(0);
        let decoded_body = if head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            decode_chunked_body(body)
        } else {
            body.to_string()
        };
        if !(200..300).contains(&status) {
            return Err(extract_error_message(&decoded_body, status));
        }
        Ok(decoded_body)
    }

    fn post_json_lines<F>(&self, path: &str, body: String, mut on_value: F) -> Result<(), String>
    where
        F: FnMut(Value),
    {
        let target = parse_http_base(&self.api_base)?;
        let mut stream = connect_tcp(&target)?;
        stream
            .set_read_timeout(Some(Duration::from_secs(600)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| e.to_string())?;

        let request_path = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{path}")
        };
        let mut request = format!(
            "POST {request_path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAccept: application/x-ndjson\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
            target.host_header,
            body.as_bytes().len()
        );
        if !self.manager_token.trim().is_empty() {
            request.push_str(&format!(
                "X-Token-Workshed-Manager-Token: {}\r\n",
                self.manager_token.trim()
            ));
        }
        request.push_str("\r\n");
        request.push_str(&body);

        stream
            .write_all(request.as_bytes())
            .map_err(|e| e.to_string())?;

        let mut reader = BufReader::new(stream);
        let mut header = String::new();
        loop {
            let mut line = String::new();
            let read = reader.read_line(&mut line).map_err(|e| e.to_string())?;
            if read == 0 {
                return Err("Malformed HTTP response.".into());
            }
            if line == "\r\n" || line == "\n" {
                break;
            }
            header.push_str(&line);
        }

        let status = header
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .unwrap_or(0);
        let chunked = header
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked");

        if !(200..300).contains(&status) {
            let mut body = String::new();
            reader
                .read_to_string(&mut body)
                .map_err(|e| e.to_string())?;
            let decoded = if chunked {
                decode_chunked_body(&body)
            } else {
                body
            };
            return Err(extract_error_message(&decoded, status));
        }

        if chunked {
            read_chunked_json_lines(&mut reader, &mut on_value)
        } else {
            read_plain_json_lines(&mut reader, &mut on_value)
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub message: String,
    pub user_content: Value,
    pub history: Vec<Value>,
    pub server_url: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub system_prompt: String,
    pub model: String,
    pub openclaw_enabled: bool,
    pub cypherclaw_enabled: bool,
    pub agent_runtime: String,
    pub conversation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_mode: Option<String>,
}

fn parse_manager_models(value: Value) -> ManagerModelsPayload {
    let active_model = string_at(&value, "active_model");
    let available_models = string_array_at(&value, "available_models");
    let running_models = parse_running_models(value.get("running_models").unwrap_or(&Value::Null));
    let runtime_config = parse_runtime_config(value.get("runtime_config").unwrap_or(&Value::Null));
    let mut model_details = parse_model_details(value.get("model_details").unwrap_or(&Value::Null));
    if model_details.is_empty() {
        model_details = fallback_model_details(
            &available_models,
            &running_models,
            &active_model,
            &runtime_config,
        );
    }
    ManagerModelsPayload {
        ok: bool_at(&value, "ok"),
        server_url: string_at(&value, "server_url"),
        active_model,
        active_server_url: string_at(&value, "active_server_url"),
        available_models,
        model_details,
        running_models,
        concurrent_models: string_array_at(&value, "concurrent_models"),
        runtime_config,
        community_job: value
            .get("community_job")
            .cloned()
            .or_else(|| value.get("job").cloned()),
        integrity_report: value
            .get("integrity_report")
            .cloned()
            .unwrap_or(Value::Null),
        last_error: string_at(&value, "last_error"),
    }
}

fn parse_quantization_capabilities(value: Value) -> QuantizationCapabilities {
    QuantizationCapabilities {
        ok: bool_at(&value, "ok"),
        engines: value
            .get("engines")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        workers: value
            .get("workers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        warnings: value
            .get("warnings")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        managed_model_root: string_at(&value, "managed_model_root"),
    }
}

fn parse_quantization_job(value: &Value) -> QuantizationJob {
    QuantizationJob {
        id: string_at(value, "id"),
        engine: string_at(value, "engine"),
        status: string_at(value, "status"),
        stage: string_at(value, "stage"),
        progress: value.get("progress").and_then(Value::as_f64).unwrap_or(0.0) as f32,
        speed: string_at(value, "speed"),
        eta_seconds: value.get("eta_seconds").and_then(Value::as_u64),
        message: string_at(value, "message"),
        error: string_at(value, "error"),
        artifact: value
            .get("artifact")
            .cloned()
            .filter(|item| !item.is_null()),
        delivery_options: value
            .get("delivery_options")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        allowed_actions: value
            .get("allowed_actions")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        raw: value.clone(),
    }
}

fn parse_ide_pairings(value: &Value) -> Vec<IdePairingRequest> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let pairing_id = string_at(item, "pairing_id");
                    if pairing_id.is_empty() {
                        None
                    } else {
                        Some(IdePairingRequest {
                            client_id: string_at(item, "client_id"),
                            client_name: string_at(item, "client_name"),
                            ide_name: string_at(item, "ide_name"),
                            plugin_version: string_at(item, "plugin_version"),
                            requested_scopes: string_array_at(item, "requested_scopes"),
                            granted_scopes: string_array_at(item, "granted_scopes"),
                            created_at: item.get("created_at").and_then(Value::as_i64).unwrap_or(0),
                            pairing_id,
                        })
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_jetbrains_integration(value: Value) -> JetBrainsIntegrationPayload {
    let provider = value.get("provider").unwrap_or(&Value::Null);
    let completion = value.get("completion").unwrap_or(&Value::Null);
    let acp = value.get("acp").unwrap_or(&Value::Null);
    let mcp = value.get("mcp").unwrap_or(&Value::Null);
    JetBrainsIntegrationPayload {
        provider_url: string_at(provider, "base_url"),
        provider_model: string_at(provider, "model"),
        provider_api_key: string_at(provider, "api_key"),
        completion_mode: string_at(completion, "mode"),
        completion_suffix: bool_at(completion, "suffix"),
        acp_command: string_at(acp, "command"),
        acp_args: string_array_at(acp, "args"),
        mcp_command: string_at(mcp, "command"),
        mcp_args: string_array_at(mcp, "args"),
    }
}

fn parse_ide_authorizations(value: &Value) -> Vec<IdeAuthorization> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let client_id = string_at(item, "client_id");
                    if client_id.is_empty() {
                        None
                    } else {
                        Some(IdeAuthorization {
                            client_name: string_at(item, "client_name"),
                            ide_name: string_at(item, "ide_name"),
                            plugin_version: string_at(item, "plugin_version"),
                            scopes: string_array_at(item, "scopes"),
                            requested_scopes: string_array_at(item, "requested_scopes"),
                            granted_scopes: string_array_at(item, "granted_scopes"),
                            approved_at: item
                                .get("approved_at")
                                .and_then(Value::as_i64)
                                .unwrap_or(0),
                            client_id,
                        })
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_agent_profile_job(value: &Value) -> AgentProfileJob {
    AgentProfileJob {
        job_id: string_at(value, "job_id"),
        model: string_at(value, "model"),
        state: string_at(value, "state"),
        phase: string_at(value, "phase"),
        progress: f32_at(value, "progress", 0.0).clamp(0.0, 100.0),
        error: string_at(value, "error"),
        profile: value.get("profile").cloned().filter(|item| !item.is_null()),
    }
}

fn parse_model_details(value: &Value) -> Vec<ModelDetail> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let id = string_at(item, "id");
                    if id.is_empty() {
                        None
                    } else {
                        Some(ModelDetail {
                            display_name: string_or(item, "display_name", &id),
                            status: string_or(item, "status", "local"),
                            source: string_or(item, "source", "Unknown"),
                            backend: string_or(item, "backend", "vllm-mlx"),
                            quantization: string_or(item, "quantization", "Unknown"),
                            active: bool_at(item, "active"),
                            running: bool_at(item, "running"),
                            server_url: string_at(item, "server_url"),
                            role: string_at(item, "role"),
                            kind: string_at(item, "kind"),
                            reasoning_parser: string_at(item, "reasoning_parser"),
                            tool_call_parser: string_at(item, "tool_call_parser"),
                            id,
                        })
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn fallback_model_details(
    models: &[String],
    running_models: &[RunningModel],
    active_model: &str,
    runtime_config: &RuntimeConfig,
) -> Vec<ModelDetail> {
    let mut ids: Vec<String> = Vec::new();
    for model in std::iter::once(active_model).chain(models.iter().map(String::as_str)) {
        if !model.trim().is_empty() && !ids.iter().any(|item| item == model) {
            ids.push(model.to_string());
        }
    }
    for running in running_models {
        if !running.model.trim().is_empty() && !ids.iter().any(|item| item == &running.model) {
            ids.push(running.model.clone());
        }
    }

    ids.into_iter()
        .map(|id| {
            let running = running_models.iter().find(|item| item.model == id);
            let active = !active_model.trim().is_empty() && active_model == id;
            let status = if active {
                "active"
            } else if running.is_some() {
                "running"
            } else {
                "local"
            };
            ModelDetail {
                display_name: id.clone(),
                status: status.into(),
                source: if id.contains('/') {
                    "Hugging Face".into()
                } else {
                    "Local cache".into()
                },
                backend: "vllm-mlx".into(),
                quantization: fallback_quantization_label(
                    &id,
                    runtime_config.kv_cache_quantization,
                ),
                active,
                running: running.is_some(),
                server_url: running
                    .map(|item| item.server_url.clone())
                    .unwrap_or_default(),
                role: running.map(|item| item.role.clone()).unwrap_or_default(),
                id,
                ..ModelDetail::default()
            }
        })
        .collect()
}

fn fallback_quantization_label(model: &str, kv_cache_quantization: bool) -> String {
    let lower = model.to_ascii_lowercase();
    let base = if lower.contains("4bit") || lower.contains("4-bit") || lower.contains("int4") {
        "4-bit"
    } else if lower.contains("8bit") || lower.contains("8-bit") || lower.contains("int8") {
        "8-bit"
    } else if lower.contains("bf16") {
        "BF16"
    } else if lower.contains("fp16") || lower.contains("f16") {
        "FP16"
    } else {
        "Unknown"
    };
    if kv_cache_quantization {
        if base == "Unknown" {
            "KV cache quantization".into()
        } else {
            format!("{base} + KV cache quantization")
        }
    } else {
        base.into()
    }
}

fn string_or(value: &Value, key: &str, fallback: &str) -> String {
    let value = string_at(value, key);
    if value.trim().is_empty() {
        fallback.to_string()
    } else {
        value
    }
}

fn parse_running_models(value: &Value) -> Vec<RunningModel> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let model = string_at(item, "model");
                    if model.is_empty() {
                        None
                    } else {
                        Some(RunningModel {
                            model,
                            server_url: string_at(item, "server_url"),
                            role: string_at(item, "role"),
                        })
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_runtime_config(value: &Value) -> RuntimeConfig {
    RuntimeConfig {
        continuous_batching: bool_at(value, "continuous_batching"),
        use_paged_cache: bool_at(value, "use_paged_cache"),
        kv_cache_quantization: bool_at(value, "kv_cache_quantization"),
        chunked_prefill_tokens: value
            .get("chunked_prefill_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .to_string(),
        enable_mtp: bool_at(value, "enable_mtp"),
        mtp_draft_tokens: value
            .get("mtp_draft_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .to_string(),
    }
}

fn finalize_chat_stream(
    terminal_event_seen: bool,
    read_result: Result<(), String>,
) -> Result<(), String> {
    if terminal_event_seen {
        return Ok(());
    }
    read_result?;
    Err("Chat stream ended before a final response was received.".into())
}

#[derive(Debug)]
struct ParsedBase {
    host: String,
    port: u16,
    host_header: String,
}

fn connect_tcp(target: &ParsedBase) -> Result<TcpStream, String> {
    let addresses = (target.host.as_str(), target.port)
        .to_socket_addrs()
        .map_err(|error| format!("Cannot resolve {}:{}: {error}", target.host, target.port))?;
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, Duration::from_secs(5)) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
    }
    Err(format!(
        "Cannot connect to {}:{}: {}",
        target.host,
        target.port,
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no address resolved".into())
    ))
}

fn parse_http_base(raw: &str) -> Result<ParsedBase, String> {
    let mut value = trim_slash(raw.to_string());
    value = value
        .strip_prefix("http://")
        .ok_or_else(|| "Only http:// API URLs are supported.".to_string())?
        .to_string();
    if let Some((host_port, _path)) = value.split_once('/') {
        value = host_port.to_string();
    }
    let (host, port) = if let Some((host, port)) = value.rsplit_once(':') {
        let port = port
            .parse::<u16>()
            .map_err(|_| format!("Invalid port in API URL: {raw}"))?;
        (host.to_string(), port)
    } else {
        (value, 80)
    };
    if host.trim().is_empty() {
        return Err("API URL host cannot be empty.".into());
    }
    Ok(ParsedBase {
        host_header: format!("{host}:{port}"),
        host,
        port,
    })
}

fn trim_slash(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}

pub fn percent_encode(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn decode_chunked_body(body: &str) -> String {
    let bytes = body.as_bytes();
    let mut cursor = 0;
    let mut out = Vec::new();
    loop {
        let Some(line_end) = bytes[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|offset| cursor + offset)
        else {
            break;
        };
        let size_line = String::from_utf8_lossy(&bytes[cursor..line_end]);
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let Ok(size) = usize::from_str_radix(size_hex, 16) else {
            break;
        };
        if size == 0 {
            break;
        }
        let data_start = line_end + 2;
        let Some(data_end) = data_start.checked_add(size) else {
            break;
        };
        if data_end > bytes.len() {
            break;
        }
        out.extend_from_slice(&bytes[data_start..data_end]);
        cursor = data_end.saturating_add(2);
        if cursor > bytes.len() {
            break;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_plain_json_lines<R, F>(reader: &mut R, on_value: &mut F) -> Result<(), String>
where
    R: BufRead,
    F: FnMut(Value),
{
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        parse_json_event_line(&line, on_value)?;
    }
    Ok(())
}

fn read_chunked_json_lines<R, F>(reader: &mut R, on_value: &mut F) -> Result<(), String>
where
    R: BufRead,
    F: FnMut(Value),
{
    let mut pending = Vec::new();
    loop {
        let mut size_line = String::new();
        let read = reader
            .read_line(&mut size_line)
            .map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        if size_hex.is_empty() {
            continue;
        }
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| format!("Invalid chunk size: {size_hex}"))?;
        if size == 0 {
            break;
        }

        let mut chunk = vec![0_u8; size];
        reader.read_exact(&mut chunk).map_err(|e| e.to_string())?;
        let mut trailer = [0_u8; 2];
        reader.read_exact(&mut trailer).map_err(|e| e.to_string())?;

        pending.extend_from_slice(&chunk);
        while let Some(index) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=index).collect::<Vec<_>>();
            let line = std::str::from_utf8(&line)
                .map_err(|error| format!("Invalid UTF-8 in chat stream: {error}"))?;
            parse_json_event_line(line, on_value)?;
        }
    }

    if !pending.iter().all(u8::is_ascii_whitespace) {
        let line = std::str::from_utf8(&pending)
            .map_err(|error| format!("Invalid UTF-8 in chat stream: {error}"))?;
        parse_json_event_line(line, on_value)?;
    }
    Ok(())
}

fn parse_json_event_line<F>(line: &str, on_value: &mut F) -> Result<(), String>
where
    F: FnMut(Value),
{
    let mut text = line.trim();
    if text.is_empty() {
        return Ok(());
    }
    if let Some(rest) = text.strip_prefix("data:") {
        text = rest.trim();
    }
    if text.is_empty() || text == "[DONE]" {
        return Ok(());
    }
    let value =
        serde_json::from_str::<Value>(text).map_err(|e| format!("Invalid stream JSON: {e}"))?;
    on_value(value);
    Ok(())
}

fn extract_error_message(body: &str, status: u16) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if let Some(detail) = value.get("detail") {
            if let Some(message) = detail.get("message").and_then(Value::as_str) {
                return message.to_string();
            }
            if let Some(message) = detail.as_str() {
                return message.to_string();
            }
            return detail.to_string();
        }
        if let Some(message) = value.get("message").and_then(Value::as_str) {
            return message.to_string();
        }
    }
    format!("HTTP {status}: {}", body.trim())
}

pub fn string_at(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

fn raw_string_at(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn string_array_at(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn bool_at(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn u32_at(value: &Value, key: &str, fallback: u32) -> u32 {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|item| u32::try_from(item).ok())
        .unwrap_or(fallback)
}

fn f32_at(value: &Value, key: &str, fallback: f32) -> f32 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|item| item as f32)
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn chunked_wire(parts: &[&[u8]]) -> Vec<u8> {
        let mut wire = Vec::new();
        for part in parts {
            wire.extend_from_slice(format!("{:X}\r\n", part.len()).as_bytes());
            wire.extend_from_slice(part);
            wire.extend_from_slice(b"\r\n");
        }
        wire.extend_from_slice(b"0\r\n\r\n");
        wire
    }

    #[test]
    fn chunked_json_keeps_utf8_split_across_chunks() {
        let payload = "{\"type\":\"answer_delta\",\"text\":\"你好\"}\n".as_bytes();
        let split = payload
            .windows(3)
            .position(|window| window == "你".as_bytes())
            .expect("test payload contains Chinese text")
            + 1;
        let wire = chunked_wire(&[&payload[..split], &payload[split..]]);
        let mut reader = BufReader::new(Cursor::new(wire));
        let mut values = Vec::new();

        read_chunked_json_lines(&mut reader, &mut |value| values.push(value))
            .expect("valid chunked JSON stream");

        assert_eq!(values.len(), 1);
        assert_eq!(raw_string_at(&values[0], "text"), "你好");
    }

    #[test]
    fn chunked_error_body_decodes_unicode_by_bytes() {
        let wire = chunked_wire(&["配置失败".as_bytes(), b" safely"]);
        let body = String::from_utf8(wire).expect("wire is valid UTF-8");

        assert_eq!(decode_chunked_body(&body), "配置失败 safely");
    }

    #[test]
    fn chat_stream_rejects_eof_without_terminal_event() {
        let error = finalize_chat_stream(false, Ok(())).expect_err("truncated stream must fail");
        assert!(error.contains("ended before a final response"));
        assert!(finalize_chat_stream(true, Ok(())).is_ok());
        assert!(finalize_chat_stream(true, Err("late socket error".into())).is_ok());
    }
}
