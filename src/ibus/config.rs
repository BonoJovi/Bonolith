/// Bonolith configuration — user-configurable settings loaded from
/// `~/.config/bonolith/config.json`.
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::keymap::*;

/// User-facing config (serialized as JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonolithConfig {
    #[serde(default = "default_toggle_keys")]
    pub toggle_keys: Vec<ToggleKeyBinding>,
}

/// A single toggle key binding in human-readable form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToggleKeyBinding {
    pub keyval: String,
    #[serde(default)]
    pub modifiers: Vec<String>,
}

/// Pre-compiled toggle key for fast matching in process_key_event.
#[derive(Debug, Clone)]
pub struct CompiledToggleKey {
    pub keyval: u32,
    pub modifier_mask: u32,
}

fn default_toggle_keys() -> Vec<ToggleKeyBinding> {
    vec![ToggleKeyBinding {
        keyval: "space".to_string(),
        modifiers: vec!["ctrl".to_string(), "shift".to_string()],
    }]
}

impl Default for BonolithConfig {
    fn default() -> Self {
        Self {
            toggle_keys: default_toggle_keys(),
        }
    }
}

impl BonolithConfig {
    /// Resolve the config file path using XDG_CONFIG_HOME.
    fn config_path() -> PathBuf {
        let config_dir = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                PathBuf::from(home).join(".config")
            });
        config_dir.join("bonolith").join("config.json")
    }

    /// Load config from disk. Returns default if file doesn't exist or fails to parse.
    pub fn load() -> Self {
        let path = Self::config_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str::<BonolithConfig>(&contents) {
                Ok(config) => {
                    info!("Bonolith: Loaded config from {}", path.display());
                    config
                }
                Err(e) => {
                    warn!("Bonolith: Failed to parse {}: {}", path.display(), e);
                    Self::default()
                }
            },
            Err(_) => {
                info!(
                    "Bonolith: No config at {}, using defaults (Ctrl+Shift+Space)",
                    path.display()
                );
                Self::default()
            }
        }
    }

    /// Compile toggle key bindings into keysym/mask pairs for fast matching.
    pub fn compile_toggle_keys(&self) -> Vec<CompiledToggleKey> {
        self.toggle_keys
            .iter()
            .filter_map(|binding| {
                let keyval = parse_keyval(&binding.keyval)?;
                let modifier_mask = binding
                    .modifiers
                    .iter()
                    .filter_map(|m| parse_modifier(m))
                    .fold(0u32, |acc, mask| acc | mask);
                info!(
                    "Bonolith: Toggle key compiled: '{}' + {:?} → keyval=0x{:04X}, mask=0x{:04X}",
                    binding.keyval, binding.modifiers, keyval, modifier_mask
                );
                Some(CompiledToggleKey {
                    keyval,
                    modifier_mask,
                })
            })
            .collect()
    }
}

/// Map a human-readable key name to an X11 keysym value.
fn parse_keyval(name: &str) -> Option<u32> {
    match name.to_ascii_lowercase().as_str() {
        "space" => Some(IBUS_KEY_SPACE),
        "return" | "enter" => Some(IBUS_KEY_RETURN),
        "escape" | "esc" => Some(IBUS_KEY_ESCAPE),
        "tab" => Some(IBUS_KEY_TAB),
        "backspace" => Some(IBUS_KEY_BACKSPACE),
        "grave" | "backtick" => Some(0x0060),
        "zenkaku_hankaku" => Some(IBUS_KEY_ZENKAKU_HANKAKU),
        "henkan" | "henkan_mode" => Some(IBUS_KEY_HENKAN_MODE),
        "muhenkan" => Some(IBUS_KEY_MUHENKAN),
        // Single ASCII character
        s if s.len() == 1 => {
            let ch = s.chars().next().unwrap();
            if ch.is_ascii_graphic() {
                Some(ch as u32)
            } else {
                None
            }
        }
        // Hex keysym for advanced users: "0xff2a"
        s if s.starts_with("0x") => u32::from_str_radix(&s[2..], 16).ok(),
        other => {
            warn!("Bonolith: Unknown key name '{}', ignoring", other);
            None
        }
    }
}

/// Map a modifier name to an IBus modifier mask bit.
fn parse_modifier(name: &str) -> Option<u32> {
    match name.to_ascii_lowercase().as_str() {
        "ctrl" | "control" => Some(IBUS_CONTROL_MASK),
        "alt" | "mod1" => Some(IBUS_MOD1_MASK),
        "shift" => Some(IBUS_SHIFT_MASK),
        other => {
            warn!("Bonolith: Unknown modifier '{}', ignoring", other);
            None
        }
    }
}
