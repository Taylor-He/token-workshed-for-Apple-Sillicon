use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_content: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<StoredAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredAttachment {
    pub kind: String,
    pub path: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredConversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<StoredMessage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StoredConversationProfile {
    pub active_conversation_id: String,
    pub conversations: Vec<StoredConversation>,
    pub selected_model: String,
    pub selected_local_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeSettings {
    pub server_url: String,
    pub manager_token: String,
    pub remote_server_host: String,
    pub remote_server_username: String,
    pub remote_server_port: String,
    pub remote_server_identity_file: String,
    pub local_server_mode_enabled: bool,
    pub max_tokens: u32,
    pub temperature: f32,
    pub system_prompt: String,
    pub language: String,
    pub chat_dispatch_mode: String,
}

impl Default for NativeSettings {
    fn default() -> Self {
        Self {
            server_url: "http://127.0.0.1:8010".into(),
            manager_token: String::new(),
            remote_server_host: String::new(),
            remote_server_username: String::new(),
            remote_server_port: "22".into(),
            remote_server_identity_file: String::new(),
            local_server_mode_enabled: false,
            max_tokens: 1024,
            temperature: 0.7,
            system_prompt: String::new(),
            language: "en".into(),
            chat_dispatch_mode: "single".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeState {
    pub settings: NativeSettings,
    /// `None` is reserved for legacy state so startup can adopt a saved RoboServer session.
    #[serde(default)]
    pub runtime_target: Option<String>,
    pub selected_model: String,
    pub selected_local_model: String,
    pub active_conversation_id: String,
    pub conversations: Vec<StoredConversation>,
    pub active_conversation_profile: String,
    pub conversation_profiles: BTreeMap<String, StoredConversationProfile>,
}

impl Default for NativeState {
    fn default() -> Self {
        let id = "main".to_string();
        Self {
            settings: NativeSettings::default(),
            runtime_target: None,
            selected_model: String::new(),
            selected_local_model: String::new(),
            active_conversation_id: id.clone(),
            conversations: vec![StoredConversation {
                id,
                title: "New Conversation".into(),
                messages: Vec::new(),
            }],
            active_conversation_profile: "local".into(),
            conversation_profiles: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeStateStore {
    path: Option<PathBuf>,
}

impl NativeStateStore {
    pub fn new() -> Self {
        let path = ProjectDirs::from("com", "TaylorHe", "token-workshed-nextcore")
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

    pub fn save_capture_image(
        &self,
        mime_type: &str,
        encoded: &str,
        label: &str,
    ) -> Result<StoredAttachment, String> {
        let directories = ProjectDirs::from("com", "TaylorHe", "token-workshed-nextcore")
            .ok_or_else(|| "Could not resolve the NextCore data directory.".to_string())?;
        let directory = directories.data_local_dir().join("agent").join("captures");
        save_capture_image_in(&directory, mime_type, encoded, label)
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

fn save_capture_image_in(
    directory: &Path,
    mime_type: &str,
    encoded: &str,
    label: &str,
) -> Result<StoredAttachment, String> {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use std::io::Write as _;

    const MAX_CAPTURE_BYTES: usize = 9 * 1024 * 1024;
    if encoded.len() > 12_000_000 {
        return Err("The captured image is too large to attach.".into());
    }
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|_| "The captured image could not be decoded.".to_string())?;
    if bytes.is_empty() || bytes.len() > MAX_CAPTURE_BYTES {
        return Err("The captured image is empty or too large to attach.".into());
    }

    let extension = match mime_type.trim().to_ascii_lowercase().as_str() {
        "image/jpeg" if bytes.starts_with(&[0xff, 0xd8, 0xff]) => "jpg",
        "image/png" if bytes.starts_with(b"\x89PNG\r\n\x1a\n") => "png",
        "image/webp"
            if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" =>
        {
            "webp"
        }
        _ => return Err("The captured image format is not supported.".into()),
    };
    fs::create_dir_all(directory)
        .map_err(|error| format!("Could not create the capture directory: {error}"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = directory.join(format!(
        "capture-{}-{nonce}.{extension}",
        std::process::id()
    ));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .map_err(|error| format!("Could not save the captured image: {error}"))?;
    if let Err(error) = file.write_all(&bytes) {
        let _ = fs::remove_file(&path);
        return Err(format!("Could not save the captured image: {error}"));
    }
    Ok(StoredAttachment {
        kind: "image".into(),
        path: path.to_string_lossy().into_owned(),
        label: label.trim().chars().take(300).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_agent_settings_are_ignored_and_removed() {
        let state: NativeState = serde_json::from_str(
            r#"{
                "settings": {"server_url": "http://127.0.0.1:9000", "agent_runtime": "hermes", "openclaw_enabled": true},
                "selected_model": "example/model",
                "active_conversation_id": "main",
                "conversations": []
            }"#,
        )
        .expect("legacy state should remain readable");

        assert_eq!(state.settings.server_url, "http://127.0.0.1:9000");
        assert_eq!(state.settings.max_tokens, 1024);
        let encoded = serde_json::to_string(&state).expect("state should serialize");
        assert!(!encoded.contains("agent_runtime"));
        assert!(!encoded.contains("openclaw_enabled"));
        assert_eq!(state.selected_model, "example/model");
        assert!(state.selected_local_model.is_empty());
        assert!(state.runtime_target.is_none());
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

    #[test]
    fn capture_attachment_is_saved_as_a_private_image_file() {
        use base64::Engine as _;
        use base64::engine::general_purpose::STANDARD;

        const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "nextcore-agent-capture-{}-{unique}",
            std::process::id()
        ));
        let attachment = save_capture_image_in(
            &directory,
            "image/png",
            PNG_BASE64,
            " Screen capture · NextCore ",
        )
        .expect("save screenshot attachment");
        let saved_path = PathBuf::from(&attachment.path);

        assert_eq!(attachment.kind, "image");
        assert_eq!(attachment.label, "Screen capture · NextCore");
        assert_eq!(
            fs::read(&saved_path).expect("read saved image"),
            STANDARD.decode(PNG_BASE64).unwrap()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&saved_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }

        fs::remove_file(saved_path).expect("remove test image");
        fs::remove_dir(directory).expect("remove test directory");
    }
}
