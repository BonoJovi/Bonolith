/// IBus key constants and modifier masks.
/// Based on IBus key event definitions from ibustypes.h.

// Modifier masks
pub const IBUS_RELEASE_MASK: u32 = 1 << 30;
pub const IBUS_SHIFT_MASK: u32 = 1 << 0;
pub const IBUS_CONTROL_MASK: u32 = 1 << 2;
pub const IBUS_MOD1_MASK: u32 = 1 << 3; // Alt

// Common key values (X11 keysym values)
pub const IBUS_KEY_SPACE: u32 = 0x0020;
pub const IBUS_KEY_RETURN: u32 = 0xFF0D;
pub const IBUS_KEY_ESCAPE: u32 = 0xFF1B;
pub const IBUS_KEY_BACKSPACE: u32 = 0xFF08;
pub const IBUS_KEY_TAB: u32 = 0xFF09;
pub const IBUS_KEY_UP: u32 = 0xFF52;
pub const IBUS_KEY_DOWN: u32 = 0xFF54;
pub const IBUS_KEY_LEFT: u32 = 0xFF51;
pub const IBUS_KEY_RIGHT: u32 = 0xFF53;
pub const IBUS_KEY_PAGE_UP: u32 = 0xFF55;
pub const IBUS_KEY_PAGE_DOWN: u32 = 0xFF56;

// Function keys
pub const IBUS_KEY_F6: u32 = 0xFFC3;
pub const IBUS_KEY_F7: u32 = 0xFFC4;
pub const IBUS_KEY_F8: u32 = 0xFFC5;
pub const IBUS_KEY_F9: u32 = 0xFFC6;
pub const IBUS_KEY_F10: u32 = 0xFFC7;

// Japanese keyboard toggle keys
pub const IBUS_KEY_ZENKAKU_HANKAKU: u32 = 0xFF2A;
pub const IBUS_KEY_HENKAN_MODE: u32 = 0xFF23;
pub const IBUS_KEY_MUHENKAN: u32 = 0xFF22;

/// Check if a keyval is a printable ASCII character (a-z, 0-9, punctuation).
pub fn is_printable_ascii(keyval: u32) -> bool {
    (0x0020..=0x007E).contains(&keyval)
}

/// Convert a keyval to a char (for printable ASCII).
pub fn keyval_to_char(keyval: u32) -> Option<char> {
    if is_printable_ascii(keyval) {
        char::from_u32(keyval)
    } else {
        None
    }
}

/// Check if modifier keys (Ctrl, Alt) are pressed.
pub fn has_modifier(state: u32) -> bool {
    state & (IBUS_CONTROL_MASK | IBUS_MOD1_MASK) != 0
}

/// Check if this is a key release event.
pub fn is_release(state: u32) -> bool {
    state & IBUS_RELEASE_MASK != 0
}
