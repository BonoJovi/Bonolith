//! Frontend-agnostic key dispatcher.
//!
//! Both the Fcitx5 (`src/ffi.rs`) and IBus (`src/ibus/engine_impl.rs`)
//! frontends previously carried near-identical ~360-line copies of the
//! same key-handling ladder (Tab commits, F6-F10 kana forms, arrow /
//! Space / Enter / Escape / Backspace during conversion, romaji /
//! fullwidth symbol otherwise). This module owns that ladder — each
//! frontend now handles only its framework-specific concerns (D-Bus
//! signal emission, release policy, IME on/off) and routes the actual
//! keystroke through [`dispatch_key`].
//!
//! The dispatcher operates on `&mut ConversionEngine` plus an out-of-
//! band `converting: &mut bool` flag (both frontends already mirror
//! `engine.conversion_state().is_some()` in a separate field for lock-
//! ordering reasons; we keep that shape rather than change it here) and
//! returns a [`KeyOutcome`] describing what the frontend should do.

use super::{ConversionEngine, KanaForm};

// X11 keysyms and modifier masks — the values IBus and Fcitx5 both
// forward straight from X11 / libinput. Kept `pub` so the frontends
// can reference the same constant when they need to guard release
// events (which don't go through the dispatcher).
pub const KEY_SPACE: u32 = 0x0020;
pub const KEY_TAB: u32 = 0xFF09;
pub const KEY_RETURN: u32 = 0xFF0D;
pub const KEY_ESCAPE: u32 = 0xFF1B;
pub const KEY_BACKSPACE: u32 = 0xFF08;
pub const KEY_UP: u32 = 0xFF52;
pub const KEY_DOWN: u32 = 0xFF54;
pub const KEY_LEFT: u32 = 0xFF51;
pub const KEY_RIGHT: u32 = 0xFF53;
pub const KEY_PAGE_UP: u32 = 0xFF55;
pub const KEY_PAGE_DOWN: u32 = 0xFF56;
pub const KEY_F6: u32 = 0xFFC3;
pub const KEY_F7: u32 = 0xFFC4;
pub const KEY_F8: u32 = 0xFFC5;
pub const KEY_F9: u32 = 0xFFC6;
pub const KEY_F10: u32 = 0xFFC7;

// Keypad keysyms — normalised to their main-keyboard equivalents in
// [`normalise_keypad`] before the dispatch ladder runs. Without this,
// preedit / conversion mode swallowed every keypad press as a silent
// consume (KP_Enter matched neither KEY_RETURN nor `printable_char`,
// so it couldn't commit the conversion — Mozc / ATOK both treat KP
// keys as their main-keyboard twins).
const KEY_KP_SPACE: u32 = 0xFF80;
const KEY_KP_TAB: u32 = 0xFF89;
const KEY_KP_ENTER: u32 = 0xFF8D;
const KEY_KP_HOME: u32 = 0xFF95;
const KEY_KP_LEFT: u32 = 0xFF96;
const KEY_KP_UP: u32 = 0xFF97;
const KEY_KP_RIGHT: u32 = 0xFF98;
const KEY_KP_DOWN: u32 = 0xFF99;
const KEY_KP_PAGE_UP: u32 = 0xFF9A;
const KEY_KP_PAGE_DOWN: u32 = 0xFF9B;
const KEY_KP_END: u32 = 0xFF9C;
const KEY_KP_DELETE: u32 = 0xFF9F;
const KEY_KP_MULTIPLY: u32 = 0xFFAA;
const KEY_KP_ADD: u32 = 0xFFAB;
const KEY_KP_SEPARATOR: u32 = 0xFFAC;
const KEY_KP_SUBTRACT: u32 = 0xFFAD;
const KEY_KP_DECIMAL: u32 = 0xFFAE;
const KEY_KP_DIVIDE: u32 = 0xFFAF;
const KEY_KP_0: u32 = 0xFFB0;
const KEY_KP_9: u32 = 0xFFB9;
const KEY_KP_EQUAL: u32 = 0xFFBD;
const KEY_HOME: u32 = 0xFF50;
const KEY_END: u32 = 0xFF57;
const KEY_DELETE: u32 = 0xFFFF;

pub const SHIFT_MASK: u32 = 1 << 0;
pub const CONTROL_MASK: u32 = 1 << 2;
pub const MOD1_MASK: u32 = 1 << 3;
// Super / Hyper / Meta: usually grabbed by the compositor, but any press
// that leaks through must be treated as a passthrough — otherwise
// Super+e romaji-ifies into "え" and hijacks the user's shortcut.
// Both IBus and Fcitx5 use these bit positions.
pub const MOD4_MASK: u32 = 1 << 6; // Super
pub const HYPER_MASK: u32 = 1 << 5; // Hyper
pub const META_MASK: u32 = 1 << 28; // Meta
pub const RELEASE_MASK: u32 = 1 << 30;

#[derive(Copy, Clone, Debug)]
pub struct KeyEvent {
    pub keyval: u32,
    pub state: u32,
}

impl KeyEvent {
    pub fn is_release(&self) -> bool {
        self.state & RELEASE_MASK != 0
    }
    pub fn has_shift(&self) -> bool {
        self.state & SHIFT_MASK != 0
    }
    /// Any non-Shift modifier — Ctrl, Alt, Super, Hyper, Meta. A press
    /// with any of these bits is a shortcut the app owns; the dispatcher
    /// returns passthrough so the char never becomes preedit.
    pub fn has_ctrl_alt(&self) -> bool {
        self.state & (CONTROL_MASK | MOD1_MASK | MOD4_MASK | HYPER_MASK | META_MASK) != 0
    }
}

/// UI-repaint hint returned by [`dispatch_key`]. The frontend interprets
/// this into whatever its framework expects (D-Bus signals for IBus, a
/// polled snapshot for the Fcitx5 FFI).
#[derive(Debug, Default, PartialEq, Eq)]
pub enum DisplayUpdate {
    /// No UI change; do not emit any signal.
    #[default]
    Unchanged,
    /// Hide preedit / lookup / auxiliary. Preedit and conversion are both
    /// empty in the engine now.
    Cleared,
    /// Render a plain (single-underline) preedit with this text; hide
    /// lookup / auxiliary. Never empty (the dispatcher promotes an empty
    /// preedit to [`DisplayUpdate::Cleared`]).
    Preedit(String),
    /// Render the engine's current `conversion_state()` — segmented
    /// preedit + lookup for the focused segment.
    Conversion,
}

/// What the frontend should do after [`dispatch_key`] returns.
#[derive(Debug, Default)]
pub struct KeyOutcome {
    /// Whether the IME consumed the key (do not forward to the app).
    pub consumed: bool,
    /// Text to commit into the app before the display update, if any.
    pub commit: Option<String>,
    /// UI state to render.
    pub display: DisplayUpdate,
    /// Hint that a background LLM rerank was (re)triggered — the frontend
    /// should start polling and re-render conversion when the result
    /// lands. IBus consumes this via `spawn_rerank_refresh`; the Fcitx5
    /// FFI ignores it (its client polls `bonolith_poll_apply_rerank`).
    pub schedule_rerank_refresh: bool,
}

impl KeyOutcome {
    fn consumed_noop() -> Self {
        Self {
            consumed: true,
            ..Self::default()
        }
    }
    fn conversion() -> Self {
        Self {
            consumed: true,
            display: DisplayUpdate::Conversion,
            ..Self::default()
        }
    }
    fn cleared() -> Self {
        Self {
            consumed: true,
            display: DisplayUpdate::Cleared,
            ..Self::default()
        }
    }
    fn commit(text: String) -> Self {
        Self {
            consumed: true,
            commit: Some(text),
            display: DisplayUpdate::Cleared,
            ..Self::default()
        }
    }
    fn preedit_or_cleared(text: String) -> Self {
        Self {
            consumed: true,
            display: if text.is_empty() {
                DisplayUpdate::Cleared
            } else {
                DisplayUpdate::Preedit(text)
            },
            ..Self::default()
        }
    }
}

/// Dispatch a key press through the engine.
///
/// The caller has already:
/// - filtered releases (they never reach here);
/// - handled framework-specific overrides (IBus toggle key, Henkan /
///   Muhenkan, force-enable, `enabled` gate);
///
/// so this function only sees "a real press that the IME owns." It
/// mutates the engine in place, flips `converting` when the engine
/// enters or leaves conversion mode, and returns a [`KeyOutcome`]
/// describing what the frontend should render / commit / return.
pub fn dispatch_key(
    engine: &mut ConversionEngine,
    converting: &mut bool,
    event: KeyEvent,
) -> KeyOutcome {
    // Defensive: releases are the frontend's responsibility (each side
    // has its own consume policy). If one slips through we ignore it
    // rather than mis-process.
    if event.is_release() {
        return KeyOutcome::default();
    }
    // Ctrl / Alt combos → passthrough (Ctrl+C, Alt+Tab, etc.)
    if event.has_ctrl_alt() {
        return KeyOutcome::default();
    }

    let keyval = normalise_keypad(event.keyval);
    let has_shift = event.has_shift();

    // Shift+Space → full-width space (Mozc / Google IME / ATOK
    // convention). Always inserts a full-width `　` directly, even at
    // an empty preedit (no conversion is triggered). If a conversion
    // is active, the current text is committed first — same pattern
    // as a printable char during conversion.
    if keyval == KEY_SPACE && has_shift {
        let commit = if *converting {
            let text = engine.commit_conversion();
            *converting = false;
            text
        } else {
            None
        };
        engine.append_raw("\u{3000}");
        return KeyOutcome {
            consumed: true,
            commit,
            display: DisplayUpdate::Preedit(engine.preedit()),
            ..KeyOutcome::default()
        };
    }

    // Tab while a preedit / conversion is active → commit current text
    // and consume the key. Focus does NOT move — matches the standard
    // Japanese IME convention (Mozc / Google IME / ATOK). When nothing
    // is composing we return passthrough so Tab still navigates focus.
    if keyval == KEY_TAB {
        if *converting {
            if let Some(text) = engine.commit_conversion() {
                *converting = false;
                // Same as KEY_RETURN: keep the preedit visible when a
                // Space-path pending buffer ("m") is still live so IBus
                // doesn't leave it as an invisible ghost (bug [27]).
                let preedit = engine.preedit();
                return KeyOutcome {
                    consumed: true,
                    commit: Some(text),
                    display: if preedit.is_empty() {
                        DisplayUpdate::Cleared
                    } else {
                        DisplayUpdate::Preedit(preedit)
                    },
                    ..KeyOutcome::default()
                };
            }
            *converting = false;
            return KeyOutcome::cleared();
        }
        let preedit = engine.preedit();
        if preedit.is_empty() {
            return KeyOutcome::default();
        }
        engine.commit(&preedit);
        return KeyOutcome::commit(preedit);
    }

    // F6 → hiragana form (in-conversion swap, or start-then-swap).
    if keyval == KEY_F6 {
        return kana_form_key(engine, converting, 0, KanaForm::Hiragana);
    }
    // F7/F8/F9/F10 → other kana / romaji forms.
    if let Some((form_idx, form)) = fkey_to_form(keyval) {
        return kana_form_key(engine, converting, form_idx, form);
    }

    // Conversion-mode key handling. Recognised navigation / commit keys
    // return `Some(outcome)`; unrecognised keys fall through to the
    // "commit conversion, then process the new char" path below.
    if *converting {
        if let Some(outcome) = conversion_key(engine, converting, keyval, has_shift) {
            return outcome;
        }
        if let Some(ch) = printable_char(keyval) {
            // Commit the pending conversion, then feed the new char.
            let commit = engine.commit_conversion();
            *converting = false;
            let display = process_char(engine, ch);
            return KeyOutcome {
                consumed: true,
                commit,
                display,
                ..KeyOutcome::default()
            };
        }
        // Non-printable key we don't recognise while converting: silently
        // consume so a stray Home / End / F11 doesn't leak to the app.
        return KeyOutcome::consumed_noop();
    }

    // Space → start conversion (and schedule a rerank-refresh poll so the
    // background LLM pass surfaces without another keystroke).
    if keyval == KEY_SPACE {
        if engine.start_conversion().is_some() {
            *converting = true;
            return KeyOutcome {
                consumed: true,
                display: DisplayUpdate::Conversion,
                schedule_rerank_refresh: true,
                ..KeyOutcome::default()
            };
        }
        // start_conversion declined but a preedit is being built
        // (buffer holds a partial syllable like "k" / "ky"). Consume
        // Space silently so it does not leak to the app while the
        // user's next keystroke completes the kana. Only pass through
        // when truly nothing is pending.
        if !engine.preedit().is_empty() {
            return KeyOutcome::consumed_noop();
        }
        return KeyOutcome::default();
    }

    // Enter → commit preedit as-is (hiragana / raw preedit).
    if keyval == KEY_RETURN {
        let preedit = engine.preedit();
        if preedit.is_empty() {
            return KeyOutcome::default();
        }
        engine.commit(&preedit);
        return KeyOutcome::commit(preedit);
    }

    // Escape → cancel input, but only consume if there is something to
    // cancel. Otherwise vim's mode exit and dialog Escapes would be
    // swallowed while the IME is on but idle.
    if keyval == KEY_ESCAPE {
        if *converting || !engine.preedit().is_empty() {
            engine.reset();
            engine.clear_conversion();
            *converting = false;
            return KeyOutcome::cleared();
        }
        return KeyOutcome::default();
    }

    // Backspace → delete last preedit char; passthrough if nothing to
    // delete (so the app's own Backspace still fires).
    if keyval == KEY_BACKSPACE {
        if !engine.delete_last() {
            return KeyOutcome::default();
        }
        return KeyOutcome::preedit_or_cleared(engine.preedit());
    }

    // Arrow / Page keys → consume if a preedit is being built (otherwise
    // Shift+Arrow could inject stray chars during composition), pass
    // through if nothing is composing.
    if matches!(
        keyval,
        KEY_LEFT | KEY_RIGHT | KEY_UP | KEY_DOWN | KEY_PAGE_UP | KEY_PAGE_DOWN
    ) {
        return KeyOutcome {
            consumed: !engine.preedit().is_empty(),
            ..KeyOutcome::default()
        };
    }

    // Printable ASCII → fullwidth symbol or romaji buildup.
    if let Some(ch) = printable_char(keyval) {
        let display = process_char(engine, ch);
        if !matches!(display, DisplayUpdate::Unchanged) {
            return KeyOutcome {
                consumed: true,
                display,
                ..KeyOutcome::default()
            };
        }
    }

    // Last resort: consume any other key while a preedit is active so a
    // random F11 / PrintScreen doesn't corrupt the composition. Passes
    // through when nothing is composing.
    KeyOutcome {
        consumed: !engine.preedit().is_empty(),
        ..KeyOutcome::default()
    }
}

/// Map a keypad keysym to its main-keyboard equivalent so the dispatch
/// ladder — which only knows the main-keyboard values — treats KP_Enter
/// as Enter, KP_5 as '5', KP_Down as ↓, etc. Non-keypad values pass
/// through unchanged. Mozc / ATOK / Google IME all do this; without it,
/// preedit / conversion mode consumed every keypad press as a silent
/// no-op (KP_Enter couldn't commit, digits didn't fold into romaji).
fn normalise_keypad(keyval: u32) -> u32 {
    // Digits: KP_0..KP_9 → '0'..'9'.
    if (KEY_KP_0..=KEY_KP_9).contains(&keyval) {
        return b'0' as u32 + (keyval - KEY_KP_0);
    }
    match keyval {
        KEY_KP_ENTER => KEY_RETURN,
        KEY_KP_SPACE => KEY_SPACE,
        KEY_KP_TAB => KEY_TAB,
        KEY_KP_LEFT => KEY_LEFT,
        KEY_KP_RIGHT => KEY_RIGHT,
        KEY_KP_UP => KEY_UP,
        KEY_KP_DOWN => KEY_DOWN,
        KEY_KP_PAGE_UP => KEY_PAGE_UP,
        KEY_KP_PAGE_DOWN => KEY_PAGE_DOWN,
        KEY_KP_HOME => KEY_HOME,
        KEY_KP_END => KEY_END,
        KEY_KP_DELETE => KEY_DELETE,
        KEY_KP_MULTIPLY => b'*' as u32,
        KEY_KP_ADD => b'+' as u32,
        KEY_KP_SEPARATOR => b',' as u32,
        KEY_KP_SUBTRACT => b'-' as u32,
        KEY_KP_DECIMAL => b'.' as u32,
        KEY_KP_DIVIDE => b'/' as u32,
        KEY_KP_EQUAL => b'=' as u32,
        other => other,
    }
}

fn fkey_to_form(keyval: u32) -> Option<(usize, KanaForm)> {
    match keyval {
        KEY_F7 => Some((1, KanaForm::Katakana)),
        KEY_F8 => Some((2, KanaForm::HalfwidthKatakana)),
        KEY_F9 => Some((4, KanaForm::FullwidthRomaji)),
        KEY_F10 => Some((3, KanaForm::Romaji)),
        _ => None,
    }
}

/// Handle F6-F10 either as an in-conversion form swap or as a start-
/// then-swap when no conversion is active yet. Empty-preedit F-keys are
/// still consumed (never leak `\e[18~` etc. to the terminal).
fn kana_form_key(
    engine: &mut ConversionEngine,
    converting: &mut bool,
    form_idx: usize,
    form: KanaForm,
) -> KeyOutcome {
    if *converting {
        if engine.convert_focused_to(form).is_some() {
            return KeyOutcome::conversion();
        }
        return KeyOutcome::consumed_noop();
    }
    if engine.start_kana_conversion(form_idx).is_some() {
        *converting = true;
        return KeyOutcome::conversion();
    }
    KeyOutcome::consumed_noop()
}

/// Handle a key while in conversion mode. Returns `Some(outcome)` for
/// recognised navigation / commit keys; `None` for anything else, so the
/// caller can decide (printable → commit-then-process, non-printable →
/// silent consume).
fn conversion_key(
    engine: &mut ConversionEngine,
    converting: &mut bool,
    keyval: u32,
    has_shift: bool,
) -> Option<KeyOutcome> {
    match keyval {
        KEY_SPACE | KEY_DOWN => {
            engine.cycle_candidate(1);
            Some(KeyOutcome::conversion())
        }
        KEY_UP => {
            engine.cycle_candidate(-1);
            Some(KeyOutcome::conversion())
        }
        KEY_RIGHT => {
            if has_shift {
                engine.resize_segment(1);
            } else {
                engine.move_focus(1);
            }
            Some(KeyOutcome {
                consumed: true,
                display: DisplayUpdate::Conversion,
                schedule_rerank_refresh: has_shift,
                ..KeyOutcome::default()
            })
        }
        KEY_LEFT => {
            if has_shift {
                engine.resize_segment(-1);
            } else {
                engine.move_focus(-1);
            }
            Some(KeyOutcome {
                consumed: true,
                display: DisplayUpdate::Conversion,
                schedule_rerank_refresh: has_shift,
                ..KeyOutcome::default()
            })
        }
        KEY_RETURN => {
            let text = engine.commit_conversion();
            *converting = false;
            // commit_conversion restores the Space-path pending buffer
            // (e.g. "m" from "vim → Space → Enter") back into the
            // romaji converter. Reflect that in the display so IBus's
            // preedit stays in sync with `engine.preedit()`; otherwise
            // the "m" lingers invisibly and the next Enter commits it
            // as a phantom keystroke (bug [27]). apply_outcome handles
            // commit+preedit combined. F-key path's kana already
            // included the pending buffer, so preedit will be empty.
            Some(KeyOutcome {
                consumed: true,
                commit: text,
                display: if engine.preedit().is_empty() {
                    DisplayUpdate::Cleared
                } else {
                    DisplayUpdate::Preedit(engine.preedit())
                },
                ..KeyOutcome::default()
            })
        }
        // Escape or Backspace → cancel back to preedit (Mozc parity).
        // `cancel_conversion` drops the segmented state and, unlike the
        // global `clear_conversion`, restores any F-key-captured pending
        // consonant (e.g. the "m" in "vim → F7 → Esc") so the resulting
        // preedit reads "ゔぃm", not "ゔぃ". Fable 5 flagged the missing
        // IBus Backspace arm as [6] and the pending-loss as [15]; this
        // shared path covers both.
        KEY_ESCAPE | KEY_BACKSPACE => {
            engine.cancel_conversion();
            *converting = false;
            Some(KeyOutcome::preedit_or_cleared(engine.preedit()))
        }
        _ => None,
    }
}

fn printable_char(keyval: u32) -> Option<char> {
    if (0x0020..=0x007E).contains(&keyval) {
        char::from_u32(keyval)
    } else {
        None
    }
}

/// Route a printable char into the engine: fullwidth-symbol (append_raw)
/// or romaji buildup (process_key). Returns the display update to emit,
/// or `Unchanged` if the char neither mapped as a fullwidth symbol nor
/// as a romaji-input char (rare — space is one such case).
fn process_char(engine: &mut ConversionEngine, ch: char) -> DisplayUpdate {
    // Alphabetic / apostrophe / hyphen go through the romaji converter,
    // not through the fullwidth table (which would otherwise map "a" to
    // "ａ"). Space is neither romaji nor a fullwidth-symbol input.
    let use_fullwidth = !ch.is_ascii_alphabetic() && !matches!(ch, '\'' | '-' | ' ');
    if use_fullwidth {
        if let Some(fw) = crate::core::romaji::to_fullwidth_char(ch) {
            engine.append_raw(&fw.to_string());
            return DisplayUpdate::Preedit(engine.preedit());
        }
        return DisplayUpdate::Unchanged;
    }
    if ch.is_ascii_alphabetic() || ch == '-' || ch == '\'' {
        // Pass the original case through — RomajiConverter case-folds
        // internally for the kana table lookup and preserves the raw
        // case in its `raw_input` history so F9/F10 can round-trip
        // "VIM" as "ＶＩＭ"/"VIM" instead of "ｖｉｍ"/"vim".
        engine.process_key(ch);
        return DisplayUpdate::Preedit(engine.preedit());
    }
    DisplayUpdate::Unchanged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(keyval: u32) -> KeyEvent {
        KeyEvent { keyval, state: 0 }
    }
    fn ev_shift(keyval: u32) -> KeyEvent {
        KeyEvent {
            keyval,
            state: SHIFT_MASK,
        }
    }

    fn setup() -> (ConversionEngine, bool) {
        (ConversionEngine::new(), false)
    }

    /// Typing "a" builds up preedit; Escape clears it and returns Cleared.
    #[test]
    fn romaji_then_escape() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        assert!(out.consumed);
        assert!(matches!(out.display, DisplayUpdate::Preedit(ref p) if p == "あ"));
        let out = dispatch_key(&mut e, &mut c, ev(KEY_ESCAPE));
        assert!(out.consumed);
        assert!(matches!(out.display, DisplayUpdate::Cleared));
        assert_eq!(e.preedit(), "");
    }

    /// Escape with nothing composing → passthrough (so vim / dialog Esc
    /// still works while the IME is on but idle).
    #[test]
    fn escape_idle_passes_through() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev(KEY_ESCAPE));
        assert!(!out.consumed);
        assert!(out.commit.is_none());
    }

    /// Tab with nothing composing → passthrough.
    #[test]
    fn tab_idle_passes_through() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev(KEY_TAB));
        assert!(!out.consumed);
    }

    /// Tab while preedit exists → commit that preedit.
    #[test]
    fn tab_commits_preedit() {
        let (mut e, mut c) = setup();
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        let out = dispatch_key(&mut e, &mut c, ev(KEY_TAB));
        assert!(out.consumed);
        assert_eq!(out.commit.as_deref(), Some("あ"));
    }

    /// Enter with empty preedit → passthrough (do not consume newlines
    /// in editors when the IME is idle).
    #[test]
    fn enter_idle_passes_through() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev(KEY_RETURN));
        assert!(!out.consumed);
    }

    /// Backspace with no preedit → passthrough.
    #[test]
    fn backspace_idle_passes_through() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev(KEY_BACKSPACE));
        assert!(!out.consumed);
    }

    /// Backspace with preedit → deletes one char, returns updated preedit
    /// or Cleared when the last char goes.
    #[test]
    fn backspace_deletes_preedit() {
        let (mut e, mut c) = setup();
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        dispatch_key(&mut e, &mut c, ev(b'i' as u32));
        let out = dispatch_key(&mut e, &mut c, ev(KEY_BACKSPACE));
        assert!(out.consumed);
        assert!(matches!(out.display, DisplayUpdate::Preedit(ref p) if p == "あ"));
        let out = dispatch_key(&mut e, &mut c, ev(KEY_BACKSPACE));
        assert!(out.consumed);
        assert!(matches!(out.display, DisplayUpdate::Cleared));
    }

    /// Ctrl/Alt combos pass through so app shortcuts still fire.
    #[test]
    fn ctrl_alt_pass_through() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(
            &mut e,
            &mut c,
            KeyEvent {
                keyval: b'c' as u32,
                state: CONTROL_MASK,
            },
        );
        assert!(!out.consumed);
    }

    /// Arrow keys with no preedit → passthrough (app navigation works
    /// while IME is on but idle).
    #[test]
    fn arrow_idle_passes_through() {
        let (mut e, mut c) = setup();
        for k in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN] {
            let out = dispatch_key(&mut e, &mut c, ev(k));
            assert!(!out.consumed, "arrow {k:x} should passthrough when idle");
        }
    }

    /// Arrow keys during preedit → consumed (no repaint).
    #[test]
    fn arrow_during_preedit_consumed() {
        let (mut e, mut c) = setup();
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        let out = dispatch_key(&mut e, &mut c, ev(KEY_LEFT));
        assert!(out.consumed);
        assert_eq!(out.display, DisplayUpdate::Unchanged);
    }

    /// F6 with empty preedit → consumed noop (never leak `\e[17~` to the
    /// terminal, regression from before [5]).
    #[test]
    fn f_key_empty_preedit_consumed() {
        let (mut e, mut c) = setup();
        for k in [KEY_F6, KEY_F7, KEY_F8, KEY_F9, KEY_F10] {
            let out = dispatch_key(&mut e, &mut c, ev(k));
            assert!(out.consumed, "F key {k:x} should be consumed even with empty preedit");
            assert!(!c);
        }
    }

    /// Ctrl combo passes through even if a preedit is up.
    #[test]
    fn ctrl_passthrough_during_preedit() {
        let (mut e, mut c) = setup();
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        let out = dispatch_key(
            &mut e,
            &mut c,
            KeyEvent {
                keyval: b'a' as u32,
                state: CONTROL_MASK,
            },
        );
        assert!(!out.consumed);
    }

    /// Full "kya" + Space + Enter cycle → commits the conversion.
    #[test]
    fn conversion_cycle_commits() {
        let (mut e, mut c) = setup();
        for k in [b'k', b'y', b'a'] {
            dispatch_key(&mut e, &mut c, ev(k as u32));
        }
        assert_eq!(e.preedit(), "きゃ");
        let out = dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
        assert!(out.consumed);
        assert!(c);
        assert_eq!(out.display, DisplayUpdate::Conversion);
        assert!(out.schedule_rerank_refresh);
        let out = dispatch_key(&mut e, &mut c, ev(KEY_RETURN));
        assert!(out.consumed);
        assert!(!c);
        assert!(out.commit.is_some());
        assert_eq!(out.display, DisplayUpdate::Cleared);
    }

    /// Backspace during conversion cancels back to preedit (Mozc parity —
    /// this is the [6] fix, now uniform across both frontends).
    #[test]
    fn backspace_during_conversion_returns_to_preedit() {
        let (mut e, mut c) = setup();
        for k in [b'k', b'a'] {
            dispatch_key(&mut e, &mut c, ev(k as u32));
        }
        dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
        assert!(c);
        let out = dispatch_key(&mut e, &mut c, ev(KEY_BACKSPACE));
        assert!(out.consumed);
        assert!(!c);
        assert!(matches!(out.display, DisplayUpdate::Preedit(ref p) if p == "か"));
    }

    /// Shift+Right during conversion resizes and schedules a rerank.
    #[test]
    fn shift_right_during_conversion_resizes() {
        let (mut e, mut c) = setup();
        for k in [b'k', b'a'] {
            dispatch_key(&mut e, &mut c, ev(k as u32));
        }
        dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
        let out = dispatch_key(&mut e, &mut c, ev_shift(KEY_RIGHT));
        assert!(out.consumed);
        assert_eq!(out.display, DisplayUpdate::Conversion);
        assert!(out.schedule_rerank_refresh);
    }

    /// Fullwidth punctuation goes into preedit as one wide char.
    #[test]
    fn fullwidth_punctuation() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev(b'.' as u32));
        assert!(out.consumed);
        assert_eq!(e.preedit(), "。");
    }

    /// Shift+Space at an empty preedit inserts a full-width `　`
    /// directly — it does NOT start a conversion (unlike bare Space,
    /// which passes through when the preedit is empty). Mozc / Google
    /// IME convention.
    #[test]
    fn shift_space_inserts_fullwidth_space() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev_shift(KEY_SPACE));
        assert!(out.consumed);
        assert!(!c);
        assert!(out.commit.is_none());
        assert_eq!(e.preedit(), "\u{3000}");
        assert!(matches!(out.display, DisplayUpdate::Preedit(ref p) if p == "\u{3000}"));
    }

    /// Shift+Space appends to an existing preedit rather than replacing
    /// or triggering conversion.
    #[test]
    fn shift_space_appends_to_preedit() {
        let (mut e, mut c) = setup();
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        let out = dispatch_key(&mut e, &mut c, ev_shift(KEY_SPACE));
        assert!(out.consumed);
        assert!(!c);
        assert_eq!(e.preedit(), "あ\u{3000}");
    }

    /// Shift+Space during conversion commits the current conversion
    /// first (matching the "printable during conversion" flow), then
    /// starts a fresh preedit with the full-width space.
    #[test]
    fn shift_space_during_conversion_commits_and_appends() {
        let (mut e, mut c) = setup();
        for k in [b'k', b'a'] {
            dispatch_key(&mut e, &mut c, ev(k as u32));
        }
        dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
        assert!(c);
        let out = dispatch_key(&mut e, &mut c, ev_shift(KEY_SPACE));
        assert!(out.consumed);
        assert!(!c, "conversion should end");
        assert!(out.commit.is_some(), "current conversion should commit");
        assert!(matches!(out.display, DisplayUpdate::Preedit(ref p) if p.ends_with('\u{3000}')));
    }

    /// Bare Space with no preedit still passes through (unchanged) —
    /// only Shift+Space triggers the fullwidth insert.
    #[test]
    fn bare_space_no_preedit_still_passes_through() {
        let (mut e, mut c) = setup();
        let out = dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
        assert!(!out.consumed);
        assert_eq!(e.preedit(), "");
    }

    /// Regression [10]: KP_Enter (0xFF8D) must commit an in-flight
    /// conversion the same way main-keyboard Enter (0xFF0D) does.
    /// Without keypad normalisation, KP_Enter matched neither KEY_RETURN
    /// nor `printable_char` and got silently consumed by the last-resort
    /// `consumed_noop` arm — users on external keyboards / laptop
    /// number pads couldn't confirm any candidate.
    #[test]
    fn kp_enter_commits_conversion() {
        let (mut e, mut c) = setup();
        for k in [b'k', b'a'] {
            dispatch_key(&mut e, &mut c, ev(k as u32));
        }
        dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
        assert!(c, "Space should enter conversion");
        let out = dispatch_key(&mut e, &mut c, ev(0xFF8D)); // KP_Enter
        assert!(out.consumed);
        assert!(!c, "KP_Enter should end conversion");
        assert!(
            out.commit.is_some(),
            "KP_Enter must commit the conversion (Mozc parity)",
        );
    }

    /// Regression [10]: KP digits (0xFFB0..0xFFB9) must fold into the
    /// romaji buffer as their ASCII counterparts. Without normalisation
    /// they were silently consumed once a preedit was up.
    #[test]
    fn kp_digit_feeds_romaji_when_preedit_active() {
        let (mut e, mut c) = setup();
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        let out = dispatch_key(&mut e, &mut c, ev(0xFFB2)); // KP_2
        assert!(out.consumed);
        // KP_2 should behave as '2' — engine turns it into ２ (fullwidth) via
        // the same path as a top-row 2 press, appending to the preedit.
        assert!(
            e.preedit().contains('２') || e.preedit().ends_with('2'),
            "preedit did not receive KP_2: {:?}",
            e.preedit(),
        );
    }

    /// Regression [16]: Space-triggered conversion of a mid-syllable
    /// input like "vim" must not silently drop the trailing "m". The
    /// pending buffer is snapshotted at start_conversion; a follow-up
    /// commit (Enter) re-injects it as buffered input, and a cancel
    /// (Esc / Backspace) restores it — mirroring the F-key path from
    /// [15] and matching Mozc's continuation behaviour.
    #[test]
    fn space_conversion_preserves_pending_consonant_on_commit() {
        let (mut e, mut c) = setup();
        for k in b"vim" {
            dispatch_key(&mut e, &mut c, ev(*k as u32));
        }
        assert_eq!(e.preedit(), "ゔぃm");
        dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
        assert!(c, "Space should enter conversion");
        let out = dispatch_key(&mut e, &mut c, ev(KEY_RETURN));
        assert!(out.consumed);
        assert!(!c);
        assert!(out.commit.is_some(), "Enter should commit");
        // After commit the pending "m" returns to the buffer so the
        // next keystroke continues the syllable ("m" + "a" → "ま").
        assert_eq!(
            e.preedit(),
            "m",
            "commit did not preserve trailing 'm' as pending",
        );
        // Bug [27]: display must reflect the restored preedit so IBus
        // paints the "m" instead of clearing it into an invisible ghost
        // that Enter later "commits" as a phantom keystroke.
        assert!(
            matches!(&out.display, DisplayUpdate::Preedit(t) if t == "m"),
            "commit-with-pending must emit Preedit(\"m\"), got {:?}",
            out.display,
        );
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        assert!(
            e.preedit().ends_with('ま'),
            "'m' + 'a' should continue as 'ま', got {:?}",
            e.preedit(),
        );
    }

    /// Regression [25]: F-key commit's pending buffer is already part of
    /// the committed text (start_kana_conversion appended it to kana), so
    /// commit must NOT restore it to the romaji buffer. Otherwise
    /// "vim → F7 → Enter → a" produces "ヴィムま" — the "m" gets
    /// committed inside "ヴィム" AND lingers as buffered input.
    #[test]
    fn f_key_conversion_commit_does_not_double_count_pending() {
        for commit_key in [KEY_RETURN, KEY_TAB] {
            let (mut e, mut c) = setup();
            for k in b"vim" {
                dispatch_key(&mut e, &mut c, ev(*k as u32));
            }
            assert_eq!(e.preedit(), "ゔぃm");
            dispatch_key(&mut e, &mut c, ev(KEY_F7));
            assert!(c, "F7 should enter conversion");
            let out = dispatch_key(&mut e, &mut c, ev(commit_key));
            assert!(out.consumed);
            assert!(!c);
            assert!(out.commit.is_some(), "commit key 0x{commit_key:04X} should commit");
            assert_eq!(
                e.preedit(),
                "",
                "F-key commit via 0x{commit_key:04X} left preedit \"{}\" — pending 'm' double-counted",
                e.preedit(),
            );
            assert!(
                matches!(out.display, DisplayUpdate::Cleared),
                "F-key commit display should be Cleared (no leftover preedit), got {:?}",
                out.display,
            );
            dispatch_key(&mut e, &mut c, ev(b'a' as u32));
            assert_eq!(
                e.preedit(),
                "あ",
                "next keystroke after F-key commit should build fresh 'あ', got {:?}",
                e.preedit(),
            );
        }
    }

    /// Regression [16] symmetric: cancel from a Space-triggered
    /// conversion must restore the pending consonant too, just like
    /// the F-key cancel path from [15].
    #[test]
    fn space_conversion_cancel_restores_pending_consonant() {
        for cancel_key in [KEY_ESCAPE, KEY_BACKSPACE] {
            let (mut e, mut c) = setup();
            for k in b"vim" {
                dispatch_key(&mut e, &mut c, ev(*k as u32));
            }
            dispatch_key(&mut e, &mut c, ev(KEY_SPACE));
            assert!(c);
            let out = dispatch_key(&mut e, &mut c, ev(cancel_key));
            assert!(out.consumed);
            assert!(!c);
            assert_eq!(
                e.preedit(),
                "ゔぃm",
                "cancel via 0x{cancel_key:04X} lost pending 'm' from Space path",
            );
        }
    }

    /// Regression [15]: cancelling an F-key conversion (Escape /
    /// Backspace) must restore the trailing romaji buffer that
    /// start_kana_conversion snapshotted into the segment state.
    /// Without cancel_conversion writing the pending back to the
    /// converter, "vim → F7 → Esc" left preedit as "ゔぃ" — the "m"
    /// was silently lost, breaking Mozc / ATOK parity.
    #[test]
    fn f_key_cancel_restores_pending_consonant() {
        for cancel_key in [KEY_ESCAPE, KEY_BACKSPACE] {
            let (mut e, mut c) = setup();
            for k in b"vim" {
                dispatch_key(&mut e, &mut c, ev(*k as u32));
            }
            assert_eq!(e.preedit(), "ゔぃm", "pre-F7 preedit should show pending m");
            // F7 → katakana form; enters conversion.
            dispatch_key(&mut e, &mut c, ev(KEY_F7));
            assert!(c, "F7 should enter conversion");
            // Cancel back to preedit.
            let out = dispatch_key(&mut e, &mut c, ev(cancel_key));
            assert!(out.consumed);
            assert!(!c, "cancel should end conversion");
            assert_eq!(
                e.preedit(),
                "ゔぃm",
                "cancel via 0x{cancel_key:04X} lost pending 'm'",
            );
        }
    }

    /// Regression [12]: Super/Hyper/Meta+letter must pass through, not
    /// romaji-ify into preedit. Compositors usually grab Super+e, but
    /// X11 apps with their own bindings receive it and had it silently
    /// consumed as "え".
    #[test]
    fn super_plus_letter_passes_through() {
        for mask in [MOD4_MASK, HYPER_MASK, META_MASK] {
            let (mut e, mut c) = setup();
            let out = dispatch_key(
                &mut e,
                &mut c,
                KeyEvent {
                    keyval: b'e' as u32,
                    state: mask,
                },
            );
            assert!(!out.consumed, "modifier 0x{mask:08X}+e should passthrough");
            assert_eq!(e.preedit(), "", "modifier 0x{mask:08X}+e leaked into preedit");
        }
    }

    /// Release events are the frontend's job — the dispatcher must not
    /// touch state when one slips through.
    #[test]
    fn release_is_defensively_ignored() {
        let (mut e, mut c) = setup();
        dispatch_key(&mut e, &mut c, ev(b'a' as u32));
        let before = e.preedit();
        let out = dispatch_key(
            &mut e,
            &mut c,
            KeyEvent {
                keyval: b'b' as u32,
                state: RELEASE_MASK,
            },
        );
        assert!(!out.consumed);
        assert_eq!(e.preedit(), before);
    }
}
