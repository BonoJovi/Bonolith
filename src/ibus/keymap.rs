/// IBus-specific key constants.
///
/// The shared X11 keysyms (Space, Tab, Return, arrows, F6-F10, etc.)
/// and modifier masks live in `bonolith::engine::dispatch` so the FFI
/// and IBus sides use one copy. `config.rs` still refers to Space /
/// Return / Escape / Tab / Backspace by name for toggle-key parsing, so
/// we re-export those under the historical `IBUS_KEY_*` names to keep
/// the config module unaware of the dispatch module. The Japanese-
/// keyboard 変換/無変換 keys and the modifier-mask helpers used by IBus
/// pre-dispatch stay IBus-side.

pub const IBUS_RELEASE_MASK: u32 = 1 << 30;
pub const IBUS_SHIFT_MASK: u32 = 1 << 0;
pub const IBUS_CONTROL_MASK: u32 = 1 << 2;
pub const IBUS_MOD1_MASK: u32 = 1 << 3; // Alt

// Toggle-key parser needs these names — one source of truth in dispatch.
pub use bonolith::engine::dispatch::{
    KEY_BACKSPACE as IBUS_KEY_BACKSPACE, KEY_ESCAPE as IBUS_KEY_ESCAPE,
    KEY_RETURN as IBUS_KEY_RETURN, KEY_SPACE as IBUS_KEY_SPACE, KEY_TAB as IBUS_KEY_TAB,
};

// Japanese keyboard absolute-on/off keys (handled in the IBus frontend
// before dispatch — they toggle `enabled` rather than compose text).
pub const IBUS_KEY_ZENKAKU_HANKAKU: u32 = 0xFF2A;
pub const IBUS_KEY_HENKAN_MODE: u32 = 0xFF23;
pub const IBUS_KEY_MUHENKAN: u32 = 0xFF22;

/// Check if modifier keys (Ctrl, Alt) are pressed — used to skip
/// dispatch on app shortcuts (Ctrl+C, Alt+Tab, …).
pub fn has_modifier(state: u32) -> bool {
    state & (IBUS_CONTROL_MASK | IBUS_MOD1_MASK) != 0
}

/// Check if this is a key release event.
pub fn is_release(state: u32) -> bool {
    state & IBUS_RELEASE_MASK != 0
}
