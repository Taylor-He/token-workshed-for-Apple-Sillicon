use crate::appearance::MaterialMode;
use cosmic::iced::window;

pub trait WindowEffectsBackend {
    fn init(&mut self, window_id: window::Id) -> Vec<WindowEffectOp>;
    fn set_material(&mut self, material: MaterialMode) -> Vec<WindowEffectOp>;
    fn set_blur(&mut self, enabled: bool) -> Vec<WindowEffectOp>;
    fn supports_blur(&self) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub enum WindowEffectOp {
    EnableBlur(window::Id),
    DisableBlur(window::Id),
}

#[derive(Debug)]
pub struct PlatformWindowEffects {
    window_id: Option<window::Id>,
    blur_enabled: bool,
}

impl PlatformWindowEffects {
    pub fn new() -> Self {
        Self {
            window_id: None,
            blur_enabled: false,
        }
    }
}

impl Default for PlatformWindowEffects {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowEffectsBackend for PlatformWindowEffects {
    fn init(&mut self, window_id: window::Id) -> Vec<WindowEffectOp> {
        self.window_id = Some(window_id);
        let mut ops = Vec::new();
        if self.blur_enabled && self.supports_blur() {
            ops.push(WindowEffectOp::EnableBlur(window_id));
        }
        ops
    }

    fn set_material(&mut self, material: MaterialMode) -> Vec<WindowEffectOp> {
        let _ = material;
        self.set_blur(false)
    }

    fn set_blur(&mut self, enabled: bool) -> Vec<WindowEffectOp> {
        self.blur_enabled = enabled && self.supports_blur();
        match self.window_id {
            Some(id) if self.blur_enabled => vec![WindowEffectOp::EnableBlur(id)],
            Some(id) => vec![WindowEffectOp::DisableBlur(id)],
            None => Vec::new(),
        }
    }

    fn supports_blur(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caligo_keeps_native_blur_disabled() {
        assert!(!PlatformWindowEffects::new().supports_blur());
    }
}
