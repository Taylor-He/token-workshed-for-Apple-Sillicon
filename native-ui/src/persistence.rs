use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_content: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredConversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<StoredMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeSettings {
    pub server_url: String,
    pub manager_token: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub system_prompt: String,
    pub language: String,
    pub openclaw_enabled: bool,
    pub agent_runtime: String,
    pub openclaw_tool_profile: String,
    pub chat_dispatch_mode: String,
}

impl Default for NativeSettings {
    fn default() -> Self {
        Self {
            server_url: "http://127.0.0.1:8000".into(),
            manager_token: String::new(),
            max_tokens: 1024,
            temperature: 0.7,
            system_prompt: String::new(),
            language: "en".into(),
            openclaw_enabled: false,
            agent_runtime: "auto".into(),
            openclaw_tool_profile: "auto".into(),
            chat_dispatch_mode: "single".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeState {
    pub settings: NativeSettings,
    pub selected_model: String,
    pub selected_local_model: String,
    pub active_conversation_id: String,
    pub conversations: Vec<StoredConversation>,
}

impl Default for NativeState {
    fn default() -> Self {
        let id = "main".to_string();
        Self {
            settings: NativeSettings::default(),
            selected_model: String::new(),
            selected_local_model: String::new(),
            active_conversation_id: id.clone(),
            conversations: vec![StoredConversation {
                id,
                title: "New Conversation".into(),
                messages: Vec::new(),
            }],
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeStateStore {
    path: Option<PathBuf>,
}

impl NativeStateStore {
    pub fn new() -> Self {
        let path = ProjectDirs::from("com", "TaylorHe", "token-workshed")
            .map(|dirs| dirs.config_dir().join("native_ui_state.json"));
        Self { path }
    }

    pub fn load_or_default(&self) -> NativeState {
        let Some(path) = &self.path else {
            return NativeState::default();
        };
        let Ok(text) = fs::read_to_string(path) else {
            return NativeState::default();
        };
        match serde_json::from_str::<NativeState>(&text) {
            Ok(state) => state,
            Err(_) => {
                // Preserve an unreadable legacy or interrupted state for
                // diagnosis instead of silently destroying the only copy on
                // the next save.
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let backup = path.with_extension(format!("corrupt-{timestamp}.json"));
                let _ = fs::copy(path, backup);
                NativeState::default()
            }
        }
    }

    pub fn save(&self, state: &NativeState) -> io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let serialized = serde_json::to_string_pretty(state)?;
        if fs::read_to_string(path)
            .ok()
            .is_some_and(|current| current == serialized)
        {
            return Ok(());
        }

        // Write beside the destination and rename only after the complete
        // document is on disk. A crash can no longer leave a half-written
        // state file that erases settings and conversation history on launch.
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temporary = path.with_extension(format!("tmp-{}-{nonce}", std::process::id()));
        fs::write(&temporary, serialized)?;
        match fs::rename(&temporary, path) {
            Ok(()) => Ok(()),
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                Err(error)
            }
        }
    }

    pub fn clear(&self) -> io::Result<()> {
        if let Some(path) = &self.path {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_state_uses_defaults_for_new_settings_fields() {
        let state: NativeState = serde_json::from_str(
            r#"{
                "settings": {"server_url": "http://127.0.0.1:9000"},
                "selected_model": "example/model",
                "active_conversation_id": "main",
                "conversations": []
            }"#,
        )
        .expect("legacy state should remain readable");

        assert_eq!(state.settings.server_url, "http://127.0.0.1:9000");
        assert_eq!(state.settings.max_tokens, 1024);
        assert_eq!(state.settings.agent_runtime, "auto");
        assert_eq!(state.selected_model, "example/model");
        assert!(state.selected_local_model.is_empty());
    }

    #[test]
    fn state_store_round_trips_through_atomic_replace() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "token-workshed-native-state-{}-{unique}",
            std::process::id()
        ));
        let path = directory.join("native_ui_state.json");
        let store = NativeStateStore {
            path: Some(path.clone()),
        };
        let mut state = NativeState::default();
        state.selected_model = "example/model".into();

        store.save(&state).expect("atomic state save");
        assert_eq!(store.load_or_default().selected_model, "example/model");

        let _ = fs::remove_file(path);
        let _ = fs::remove_dir(directory);
    }
}
