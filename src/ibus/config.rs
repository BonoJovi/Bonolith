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

impl CompiledToggleKey {
    /// Does this compiled binding fire for a physical key event with
    /// `keyval` + `active_modifiers` (already masked to the relevant bits
    /// by the caller)?
    ///
    /// `parse_keyval()` lowercases the incoming name, so a binding of
    /// "a" compiles to keyval=0x61. But the OS delivers a Shift+A press
    /// as the uppercase keysym 0x41, not 0x61 — comparing raw values
    /// misses every Shift+letter binding. Fold ASCII uppercase letters
    /// back to lowercase before comparing so a config of "a" + shift
    /// matches the physical Shift+A the user actually presses. Digits
    /// and punctuation are layout-dependent under Shift (Shift+2 → '@'
    /// on US, '"' on JIS) — users who want those must configure the
    /// shifted keyval directly ("exclam", "0x0021").
    pub fn matches(&self, keyval: u32, active_modifiers: u32) -> bool {
        if self.modifier_mask != active_modifiers {
            return false;
        }
        if keyval == self.keyval {
            return true;
        }
        let unshifted = if (0x41..=0x5A).contains(&keyval) {
            keyval + 0x20
        } else {
            keyval
        };
        unshifted == self.keyval
    }
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
        s if s.len() == 1 => s
            .chars()
            .next()
            .filter(|ch| ch.is_ascii_graphic())
            .map(|ch| ch as u32),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(keyval: &str, modifiers: &[&str]) -> CompiledToggleKey {
        let cfg = BonolithConfig {
            toggle_keys: vec![ToggleKeyBinding {
                keyval: keyval.to_string(),
                modifiers: modifiers.iter().map(|s| s.to_string()).collect(),
            }],
        };
        cfg.compile_toggle_keys()
            .into_iter()
            .next()
            .expect("binding compiled")
    }

    /// Regression [11]: a Shift+letter toggle binding must fire when the
    /// user actually presses Shift+<letter>. parse_keyval lowercases the
    /// name, so the compiled keyval is the lowercase form (0x61 for "a"),
    /// but the OS delivers a Shift+A press as 0x41. Without the
    /// uppercase→lowercase fold in `matches()`, the binding was silently
    /// dead for every letter.
    #[test]
    fn shift_letter_binding_matches_physical_uppercase_keysym() {
        let tk = binding("a", &["ctrl", "shift"]);
        // Physical Shift+Ctrl+A: keyval=0x41 ('A'), modifiers=CTRL|SHIFT.
        let modifiers = IBUS_CONTROL_MASK | IBUS_SHIFT_MASK;
        assert!(
            tk.matches(0x41, modifiers),
            "Shift+Ctrl+A must match binding 'a'+ctrl+shift",
        );
        // Same binding must still match the lowercase form when a
        // frontend somehow sends it.
        assert!(tk.matches(0x61, modifiers));
    }

    /// A binding without Shift must not fire for a physical Shift+letter
    /// press — the modifier mismatch trumps the keyval fold.
    #[test]
    fn shift_letter_press_does_not_leak_into_unshifted_binding() {
        let tk = binding("a", &["ctrl"]);
        // Physical Ctrl+Shift+A (user pressed Shift too, config didn't ask
        // for it). We should NOT match — the caller expects an exact
        // modifier equality.
        let modifiers = IBUS_CONTROL_MASK | IBUS_SHIFT_MASK;
        assert!(!tk.matches(0x41, modifiers));
    }

    /// Default ctrl+shift+space keeps working — Space (0x20) is shift-
    /// invariant so the fold is a no-op here.
    #[test]
    fn default_ctrl_shift_space_still_matches() {
        let tk = binding("space", &["ctrl", "shift"]);
        let modifiers = IBUS_CONTROL_MASK | IBUS_SHIFT_MASK;
        assert!(tk.matches(0x20, modifiers));
    }
}
