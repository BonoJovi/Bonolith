//! C FFI layer for Bonolith engine.
//!
//! Provides a C-compatible API so that Fcitx5 (or other C/C++ frameworks)
//! can use Bonolith's conversion engine without Rust-specific dependencies.
//!
//! The FFI handles all key dispatch logic (Space→convert, Enter→commit, etc.)
//! so the C++ side only needs to forward key events and read back UI state.

use std::ffi::CString;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::core::dictionary::{DictionaryEntry, PartOfSpeech};
use crate::engine::{ConversionEngine, SharedCore};

/// Acquire a write lock, unpoisoning on the fly. Poisoning happens when a
/// previous holder panicked; the shared engine data is our own and cannot
/// leave the process in a state that we cannot recover from — returning
/// `into_inner()` gives us the guard back so the extern "C" caller keeps
/// working instead of the whole IME dying on the next FFI call.
fn write_lock_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

fn read_lock_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn mutex_lock_recover<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|e| e.into_inner())
}

/// Wrap an extern "C" function body so a panic can never unwind past the
/// FFI boundary. Any panic (which would otherwise be undefined behavior
/// when it escapes into C/C++) is caught, logged with a source-location
/// hint, and turned into the caller-provided default return value.
fn ffi_boundary<R, F>(default: R, body: F) -> R
where
    F: FnOnce() -> R,
{
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(v) => v,
        Err(_) => {
            log::error!("bonolith FFI: panic caught at boundary; returning default");
            default
        }
    }
}

// X11 keysym values (shared by IBus and Fcitx5)
const KEY_SPACE: u32 = 0x0020;
const KEY_TAB: u32 = 0xFF09;
const KEY_RETURN: u32 = 0xFF0D;
const KEY_ESCAPE: u32 = 0xFF1B;
const KEY_BACKSPACE: u32 = 0xFF08;
const KEY_UP: u32 = 0xFF52;
const KEY_DOWN: u32 = 0xFF54;
const KEY_LEFT: u32 = 0xFF51;
const KEY_RIGHT: u32 = 0xFF53;
const KEY_PAGE_UP: u32 = 0xFF55;
const KEY_PAGE_DOWN: u32 = 0xFF56;
const KEY_F6: u32 = 0xFFC3;
const KEY_F7: u32 = 0xFFC4;
const KEY_F8: u32 = 0xFFC5;
const KEY_F9: u32 = 0xFFC6;
const KEY_F10: u32 = 0xFFC7;

const SHIFT_MASK: u32 = 1 << 0;
const CONTROL_MASK: u32 = 1 << 2;
const MOD1_MASK: u32 = 1 << 3; // Alt
const RELEASE_MASK: u32 = 1 << 30;

/// Maximum number of segments in a conversion.
const MAX_SEGMENTS: usize = 32;
/// Maximum number of candidates per segment.
const MAX_CANDIDATES: usize = 64;

/// Segment info for batch UI state.
#[repr(C)]
pub struct BonolithSegmentInfo {
    /// Character start position in composed text.
    pub start_chars: i32,
    /// Character length in composed text.
    pub char_len: i32,
}

/// Batch UI state returned by bonolith_get_ui_state().
/// All string pointers are valid until the next call to bonolith_handle_key() or bonolith_get_ui_state().
#[repr(C)]
pub struct BonolithUiState {
    /// Committed text (null if none).
    pub committed: *const c_char,
    /// Whether the engine is in conversion mode.
    pub converting: bool,
    /// Whether there is preedit text (only meaningful when not converting).
    pub has_preedit: bool,
    /// Preedit string (when not converting) or composed text (when converting).
    /// Null if empty.
    pub preedit: *const c_char,
    /// Number of segments (0 when not converting).
    pub segment_count: i32,
    /// Focused segment index.
    pub focus_index: i32,
    /// Segment info array (up to MAX_SEGMENTS).
    pub segments: [BonolithSegmentInfo; MAX_SEGMENTS],
    /// Number of candidates for the focused segment.
    pub candidate_count: i32,
    /// Selected candidate index.
    pub selected_index: i32,
    /// Candidate text pointers (up to MAX_CANDIDATES).
    pub candidates: [*const c_char; MAX_CANDIDATES],
}

/// Opaque handle to the Bonolith engine context.
pub struct BonolithContext {
    engine: ConversionEngine,
    converting: bool,
    /// Pending committed text, polled by the framework after handle_key.
    pending_commit: Option<String>,
    /// Cached strings for FFI return values (kept alive between calls).
    cache_preedit: CString,
    cache_commit: CString,
    cache_composed: CString,
    cache_candidate: CString,
    /// Cached candidate strings for batch API.
    cache_candidates: Vec<CString>,
}

/// Full-width punctuation/digit mapping. Delegates to the shared table in
/// `core::romaji::to_fullwidth_char` so the Fcitx5, IBus, and F9 paths all
/// agree on which characters convert (previously three drifting copies).
///
/// Excludes characters the romaji converter owns (a-z, A-Z, `'` for the
/// n'-disambiguation trick, `-` for long-vowel input, ` ` as a bunsetsu
/// delimiter) — those go through `ctx.engine.process_key(ch)` instead of
/// landing in the preedit as full-width symbols.
fn to_fullwidth(ch: char) -> Option<String> {
    if ch.is_ascii_alphabetic() || matches!(ch, '\'' | '-' | ' ') {
        return None;
    }
    crate::core::romaji::to_fullwidth_char(ch).map(|c| c.to_string())
}

fn is_printable_ascii(keyval: u32) -> bool {
    (0x0020..=0x007E).contains(&keyval)
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_context_new() -> *mut BonolithContext {
    let ctx = Box::new(BonolithContext {
        engine: ConversionEngine::new(),
        converting: false,
        pending_commit: None,
        cache_preedit: CString::default(),
        cache_commit: CString::default(),
        cache_composed: CString::default(),
        cache_candidate: CString::default(),
        cache_candidates: Vec::new(),
    });
    Box::into_raw(ctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_context_free(ctx: *mut BonolithContext) {
    if !ctx.is_null() {
        unsafe { drop(Box::from_raw(ctx)) }
    }
}

// ── Key handling ─────────────────────────────────────────────────────────────

/// Process a key event. Returns true if the key was consumed.
///
/// After calling this, use bonolith_poll_commit() to check for committed text,
/// and bonolith_get_preedit() / bonolith_is_converting() / bonolith_*() for UI state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_handle_key(
    ctx: *mut BonolithContext,
    keyval: u32,
    state: u32,
) -> bool {
    let ctx = unsafe { &mut *ctx };

    // Consume releases for function keys (F6–F10) when preedit is active,
    // to prevent GTK apps from opening menus on F10 release.
    if state & RELEASE_MASK != 0 {
        if (keyval == KEY_F6
            || keyval == KEY_F7
            || keyval == KEY_F8
            || keyval == KEY_F9
            || keyval == KEY_F10)
            && (!ctx.engine.preedit().is_empty() || ctx.converting)
        {
            return true;
        }
        return false;
    }

    // Pass through Ctrl/Alt combos
    if state & (CONTROL_MASK | MOD1_MASK) != 0 {
        return false;
    }

    let has_shift = state & SHIFT_MASK != 0;

    // Tab while a preedit/conversion is active → commit current text and
    // consume the key (return true). Focus does NOT move: this matches the
    // standard Japanese IME convention (Mozc / Google IME / ATOK), where Tab
    // during composition is an IME key, not a focus-navigation key. When
    // nothing is composing we return false so Tab works normally. Both engines
    // consume Tab here, so IBus and Fcitx5 stay consistent instead of diverging
    // on framework preedit defaults.
    if keyval == KEY_TAB {
        if ctx.converting {
            if let Some(text) = ctx.engine.commit_conversion() {
                ctx.pending_commit = Some(text);
            }
            ctx.converting = false;
            return true;
        }
        let preedit = ctx.engine.preedit();
        if !preedit.is_empty() {
            ctx.engine.commit(&preedit);
            ctx.pending_commit = Some(preedit);
            return true;
        }
        return false;
    }

    // F6 → hiragana
    //
    // F6-F10 are always consumed while Bonolith is the active IM
    // (bonolith_handle_key is only entered in that case). Returning false
    // for an empty-preedit F-key leaked the raw keysym to the host, where
    // a terminal turned e.g. F7 into `\e[18~` and the trailing tilde
    // appeared as input. Only the `converting=true` latch is gated —
    // that flag must not go on when there is no reading to convert,
    // otherwise a subsequent Space is silently swallowed by
    // cycle_candidate's no-op.
    if keyval == KEY_F6 {
        if ctx.converting {
            ctx.engine.convert_focused_to_hiragana();
        } else if ctx.engine.start_kana_conversion(0).is_some() {
            ctx.converting = true;
        }
        return true;
    }

    // F7 → katakana, F8 → half-width katakana, F9 → full-width romaji, F10 → half-width romaji
    if keyval == KEY_F7 || keyval == KEY_F8 || keyval == KEY_F9 || keyval == KEY_F10 {
        let form = match keyval {
            KEY_F8 => 2,
            KEY_F9 => 4,
            KEY_F10 => 3,
            _ => 1,
        };
        if ctx.converting {
            match keyval {
                KEY_F8 => { ctx.engine.convert_focused_to_halfwidth_katakana(); }
                KEY_F9 => { ctx.engine.convert_focused_to_fullwidth_romaji(); }
                KEY_F10 => { ctx.engine.convert_focused_to_romaji(); }
                _ => { ctx.engine.convert_focused_to_katakana(); }
            }
        } else if ctx.engine.start_kana_conversion(form).is_some() {
            ctx.converting = true;
        }
        return true;
    }

    // Conversion mode key handling
    if ctx.converting {
        return handle_conversion_key(ctx, keyval, has_shift);
    }

    // Space → start conversion
    if keyval == KEY_SPACE {
        if ctx.engine.start_conversion().is_some() {
            ctx.converting = true;
            return true;
        }
        return false;
    }

    // Enter → commit preedit as-is
    if keyval == KEY_RETURN {
        let preedit = ctx.engine.preedit();
        if preedit.is_empty() {
            return false;
        }
        ctx.engine.commit(&preedit);
        ctx.pending_commit = Some(preedit);
        return true;
    }

    // Escape → cancel input, but only consume if there was actually something
    // to cancel. Swallowing Esc unconditionally would eat vim's mode exit and
    // dialog-close Esc while Bonolith is active (IBus side already checks
    // similarly on Tab/Enter).
    if keyval == KEY_ESCAPE {
        if ctx.converting || !ctx.engine.preedit().is_empty() {
            ctx.engine.reset();
            ctx.engine.clear_conversion();
            ctx.converting = false;
            return true;
        }
        return false;
    }

    // Backspace — consume if there was anything to delete
    if keyval == KEY_BACKSPACE {
        return ctx.engine.delete_last();
    }

    // Arrow/navigation keys → consume if preedit active, pass through otherwise
    if matches!(keyval, KEY_LEFT | KEY_RIGHT | KEY_UP | KEY_DOWN | KEY_PAGE_UP | KEY_PAGE_DOWN) {
        return !ctx.engine.preedit().is_empty();
    }

    // Symbol/punctuation → full-width
    if is_printable_ascii(keyval) {
        if let Some(ch) = char::from_u32(keyval) {
            if let Some(fw) = to_fullwidth(ch) {
                ctx.engine.append_raw(&fw);
                return true;
            }
            // Alphabetic → romaji input
            if ch.is_ascii_alphabetic() || ch == '-' || ch == '\'' {
                ctx.engine.process_key(ch.to_ascii_lowercase());
                return true;
            }
        }
    }

    false
}

/// Handle keys during conversion mode. Returns true if consumed.
fn handle_conversion_key(ctx: &mut BonolithContext, keyval: u32, has_shift: bool) -> bool {
    match keyval {
        KEY_SPACE | KEY_DOWN => {
            ctx.engine.cycle_candidate(1);
            true
        }
        KEY_UP => {
            ctx.engine.cycle_candidate(-1);
            true
        }
        KEY_RIGHT => {
            if has_shift {
                ctx.engine.resize_segment(1);
            } else {
                ctx.engine.move_focus(1);
            }
            true
        }
        KEY_LEFT => {
            if has_shift {
                ctx.engine.resize_segment(-1);
            } else {
                ctx.engine.move_focus(-1);
            }
            true
        }
        KEY_RETURN => {
            if let Some(text) = ctx.engine.commit_conversion() {
                ctx.pending_commit = Some(text);
            }
            ctx.converting = false;
            true
        }
        KEY_ESCAPE => {
            ctx.engine.clear_conversion();
            ctx.converting = false;
            true
        }
        // Backspace during conversion → cancel back to preedit (Mozc parity).
        // Without this arm the key falls through `_ => false`, so the host
        // app receives the Backspace and deletes a character of already-
        // committed text. The IBus side already consumes it.
        KEY_BACKSPACE => {
            ctx.engine.clear_conversion();
            ctx.converting = false;
            true
        }
        _ if is_printable_ascii(keyval) => {
            // Commit conversion first, then process the new character
            if let Some(text) = ctx.engine.commit_conversion() {
                ctx.pending_commit = Some(text);
            }
            ctx.converting = false;
            // Process the incoming character (punctuation, letter, etc.)
            if let Some(ch) = char::from_u32(keyval) {
                if let Some(fw) = to_fullwidth(ch) {
                    ctx.engine.append_raw(&fw);
                } else if ch.is_ascii_alphabetic() || ch == '-' || ch == '\'' {
                    ctx.engine.process_key(ch.to_ascii_lowercase());
                }
            }
            true
        }
        _ => false,
    }
}

// ── State queries ────────────────────────────────────────────────────────────

/// Get the current preedit string. Returns empty string if no preedit.
/// The returned pointer is valid until the next call to any bonolith_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_get_preedit(ctx: *mut BonolithContext) -> *const c_char {
    let ctx = unsafe { &mut *ctx };
    let preedit = ctx.engine.preedit();
    ctx.cache_preedit = CString::new(preedit).unwrap_or_default();
    ctx.cache_preedit.as_ptr()
}

/// Poll for committed text. Returns null if nothing to commit.
/// Clears the pending commit after returning.
/// The returned pointer is valid until the next call to any bonolith_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_poll_commit(ctx: *mut BonolithContext) -> *const c_char {
    let ctx = unsafe { &mut *ctx };
    match ctx.pending_commit.take() {
        Some(text) => {
            ctx.cache_commit = CString::new(text).unwrap_or_default();
            ctx.cache_commit.as_ptr()
        }
        None => ptr::null(),
    }
}

/// Returns true if the engine is in conversion mode.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_is_converting(ctx: *mut BonolithContext) -> bool {
    let ctx = unsafe { &*ctx };
    ctx.converting
}

/// Returns true while a background LLM rerank pass is outstanding (triggered by
/// the last conversion start or resize but not yet applied). The frontend uses
/// this to decide whether to poll `bonolith_poll_apply_rerank`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_rerank_pending(ctx: *mut BonolithContext) -> bool {
    let ctx = unsafe { &*ctx };
    ctx.engine.rerank_inflight()
}

/// Apply the background LLM rerank result if it is ready. Returns true if the
/// conversion candidates changed (the caller should refresh its UI). Non-
/// blocking: returns false when no result is ready yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_poll_apply_rerank(ctx: *mut BonolithContext) -> bool {
    let ctx = unsafe { &mut *ctx };
    ctx.engine.apply_llm_rerank()
}

/// Returns true if there is preedit text.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_has_preedit(ctx: *mut BonolithContext) -> bool {
    let ctx = unsafe { &*ctx };
    !ctx.engine.preedit().is_empty()
}

/// Get the composed text during conversion mode.
/// The returned pointer is valid until the next call to any bonolith_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_composed_text(ctx: *mut BonolithContext) -> *const c_char {
    let ctx = unsafe { &mut *ctx };
    let text = ctx.engine.conversion_state()
        .map(|s| s.composed_text())
        .unwrap_or_default();
    ctx.cache_composed = CString::new(text).unwrap_or_default();
    ctx.cache_composed.as_ptr()
}

/// Get the number of segments in the current conversion.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_segment_count(ctx: *mut BonolithContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.segments.len() as i32)
        .unwrap_or(0)
}

/// Get the currently focused segment index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_focus_index(ctx: *mut BonolithContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.focus as i32)
        .unwrap_or(0)
}

/// Get the character start position of a segment in the composed text.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_segment_start_chars(ctx: *mut BonolithContext, seg: i32) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .and_then(|s| {
            let ranges = s.segment_char_ranges();
            ranges.get(seg as usize).map(|(start, _)| *start as i32)
        })
        .unwrap_or(0)
}

/// Get the character length of a segment in the composed text.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_segment_char_len(ctx: *mut BonolithContext, seg: i32) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .and_then(|s| {
            let ranges = s.segment_char_ranges();
            ranges.get(seg as usize).map(|(start, end)| (end - start) as i32)
        })
        .unwrap_or(0)
}

/// Get the number of candidates for the focused segment.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_candidate_count(ctx: *mut BonolithContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.segments[s.focus].candidates.len() as i32)
        .unwrap_or(0)
}

/// Get a candidate text by index (for the focused segment).
/// The returned pointer is valid until the next call to any bonolith_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_candidate_text(ctx: *mut BonolithContext, index: i32) -> *const c_char {
    let ctx = unsafe { &mut *ctx };
    let text = ctx.engine.conversion_state()
        .and_then(|s| {
            let seg = &s.segments[s.focus];
            seg.candidates.get(index as usize).map(|c| c.as_str())
        })
        .unwrap_or("");
    ctx.cache_candidate = CString::new(text).unwrap_or_default();
    ctx.cache_candidate.as_ptr()
}

/// Get the selected candidate index for the focused segment.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_selected_index(ctx: *mut BonolithContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.segments[s.focus].selected as i32)
        .unwrap_or(0)
}

/// Reset the engine state (called on focus change, deactivation, etc.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_reset(ctx: *mut BonolithContext) {
    let ctx = unsafe { &mut *ctx };
    ctx.engine.reset();
    ctx.engine.clear_conversion();
    ctx.converting = false;
    ctx.pending_commit = None;
}

/// Commit any in-progress composition (conversion candidate or raw preedit)
/// into pending_commit, then clear composing state. Called on focus loss so
/// typed text is preserved instead of discarded — matches standard JP IMEs.
/// The committed text is delivered via the next bonolith_get_ui_state() /
/// bonolith_poll_commit(). No-op when nothing is composing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_commit_input(ctx: *mut BonolithContext) {
    let ctx = unsafe { &mut *ctx };
    if ctx.converting {
        if let Some(text) = ctx.engine.commit_conversion() {
            ctx.pending_commit = Some(text);
        }
        ctx.converting = false;
    } else {
        let preedit = ctx.engine.preedit();
        if !preedit.is_empty() {
            ctx.engine.commit(&preedit);
            ctx.pending_commit = Some(preedit);
        }
    }
}

// ── Batch UI state query ────────────────────────────────────────────────────

/// Get the complete UI state in a single FFI call.
/// The returned struct's string pointers are valid until the next call to
/// bonolith_handle_key() or bonolith_get_ui_state().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_get_ui_state(ctx: *mut BonolithContext, out: *mut BonolithUiState) {
    let ctx = unsafe { &mut *ctx };
    let out = unsafe { &mut *out };

    // Zero-init segments and candidates
    for i in 0..MAX_SEGMENTS {
        out.segments[i] = BonolithSegmentInfo { start_chars: 0, char_len: 0 };
    }
    for i in 0..MAX_CANDIDATES {
        out.candidates[i] = ptr::null();
    }

    // 1) Committed text
    match ctx.pending_commit.take() {
        Some(text) => {
            ctx.cache_commit = CString::new(text).unwrap_or_default();
            out.committed = ctx.cache_commit.as_ptr();
        }
        None => {
            out.committed = ptr::null();
        }
    }

    // 2) Conversion state
    out.converting = ctx.converting;

    if ctx.converting {
        if let Some(state) = ctx.engine.conversion_state() {
            let composed = state.composed_text();
            let ranges = state.segment_char_ranges();
            let seg_count = state.segments.len().min(MAX_SEGMENTS);
            let focus = state.focus;

            out.segment_count = seg_count as i32;
            out.focus_index = focus as i32;

            for i in 0..seg_count {
                let (start, end) = ranges[i];
                out.segments[i] = BonolithSegmentInfo {
                    start_chars: start as i32,
                    char_len: (end - start) as i32,
                };
            }

            // Candidates for focused segment
            let seg = &state.segments[focus];
            let cand_count = seg.candidates.len().min(MAX_CANDIDATES);
            out.candidate_count = cand_count as i32;
            out.selected_index = seg.selected as i32;

            ctx.cache_candidates.clear();
            for j in 0..cand_count {
                ctx.cache_candidates.push(
                    CString::new(seg.candidates[j].as_str()).unwrap_or_default()
                );
            }
            for (j, cs) in ctx.cache_candidates.iter().enumerate() {
                out.candidates[j] = cs.as_ptr();
            }

            ctx.cache_composed = CString::new(composed).unwrap_or_default();
            out.preedit = ctx.cache_composed.as_ptr();
            out.has_preedit = true;
        } else {
            out.preedit = ptr::null();
            out.has_preedit = false;
            out.segment_count = 0;
            out.focus_index = 0;
            out.candidate_count = 0;
            out.selected_index = 0;
        }
    } else {
        out.segment_count = 0;
        out.focus_index = 0;
        out.candidate_count = 0;
        out.selected_index = 0;

        let preedit = ctx.engine.preedit();
        out.has_preedit = !preedit.is_empty();
        if out.has_preedit {
            ctx.cache_preedit = CString::new(preedit).unwrap_or_default();
            out.preedit = ctx.cache_preedit.as_ptr();
        } else {
            out.preedit = ptr::null();
        }
    }
}

// ── Dictionary operations (global, not per-context) ─────────────────────────

/// Entry info returned by bonolith_dict_get_user_entries().
#[repr(C)]
pub struct BonolithDictEntry {
    pub reading: *const c_char,
    pub surface: *const c_char,
}

/// Result of bonolith_dict_get_user_entries().
/// Caller must free with bonolith_dict_free_entries().
#[repr(C)]
pub struct BonolithDictEntries {
    pub entries: *mut BonolithDictEntry,
    pub count: i32,
}

/// Add a word to the user dictionary and save to disk.
/// Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_dict_add_entry(
    reading: *const c_char,
    surface: *const c_char,
) -> bool {
    ffi_boundary(false, || {
        let reading = match unsafe { std::ffi::CStr::from_ptr(reading) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        };
        let surface = match unsafe { std::ffi::CStr::from_ptr(surface) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        };
        if reading.is_empty() || surface.is_empty() {
            return false;
        }

        let entry = DictionaryEntry {
            reading,
            surface,
            pos: PartOfSpeech::Noun,
            frequency: 8000,
        };

        let shared = SharedCore::global();
        let mut dict = write_lock_recover(&shared.dictionary);
        dict.add_entry(entry);
        dict.sync_user_entries_to_store().is_ok()
    })
}

/// Delete a user dictionary entry by index. Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_dict_delete_entry(index: i32) -> bool {
    ffi_boundary(false, || {
        let shared = SharedCore::global();
        let mut dict = write_lock_recover(&shared.dictionary);
        let mut entries = dict.user_entries().to_vec();
        let idx = index as usize;
        if idx >= entries.len() {
            return false;
        }
        entries.remove(idx);
        dict.replace_user_entries(entries);
        dict.sync_user_entries_to_store().is_ok()
    })
}

/// Update a user dictionary entry by index. Empty strings mean "no change".
/// Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_dict_update_entry(
    index: i32,
    new_reading: *const c_char,
    new_surface: *const c_char,
) -> bool {
    ffi_boundary(false, || {
        let new_reading = match unsafe { std::ffi::CStr::from_ptr(new_reading) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        };
        let new_surface = match unsafe { std::ffi::CStr::from_ptr(new_surface) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        };

        let shared = SharedCore::global();
        let mut dict = write_lock_recover(&shared.dictionary);
        let mut entries = dict.user_entries().to_vec();
        let idx = index as usize;
        if idx >= entries.len() {
            return false;
        }
        if !new_reading.is_empty() {
            entries[idx].reading = new_reading;
        }
        if !new_surface.is_empty() {
            entries[idx].surface = new_surface;
        }
        dict.replace_user_entries(entries);
        dict.sync_user_entries_to_store().is_ok()
    })
}

/// Get all user dictionary entries. Caller must free with bonolith_dict_free_entries().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_dict_get_user_entries() -> BonolithDictEntries {
    let empty = BonolithDictEntries {
        entries: ptr::null_mut(),
        count: 0,
    };
    ffi_boundary(empty, || {
        let shared = SharedCore::global();
        let dict = read_lock_recover(&shared.dictionary);
        let user = dict.user_entries();

        if user.is_empty() {
            return BonolithDictEntries {
                entries: ptr::null_mut(),
                count: 0,
            };
        }

        let count = user.len();
        let layout = match std::alloc::Layout::array::<BonolithDictEntry>(count) {
            Ok(l) => l,
            Err(e) => {
                log::error!("bonolith_dict_get_user_entries: layout overflow count={count}: {e}");
                return BonolithDictEntries {
                    entries: ptr::null_mut(),
                    count: 0,
                };
            }
        };
        let entries = unsafe { std::alloc::alloc(layout) as *mut BonolithDictEntry };
        if entries.is_null() {
            log::error!(
                "bonolith_dict_get_user_entries: allocation failed count={count}"
            );
            return BonolithDictEntries {
                entries: ptr::null_mut(),
                count: 0,
            };
        }

        for (i, e) in user.iter().enumerate() {
            let reading = CString::new(e.reading.as_str()).unwrap_or_default();
            let surface = CString::new(e.surface.as_str()).unwrap_or_default();
            unsafe {
                (*entries.add(i)).reading = reading.into_raw();
                (*entries.add(i)).surface = surface.into_raw();
            }
        }

        BonolithDictEntries {
            entries,
            count: count as i32,
        }
    })
}

/// Free entries returned by bonolith_dict_get_user_entries().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_dict_free_entries(result: BonolithDictEntries) {
    ffi_boundary((), || {
        if result.entries.is_null() {
            return;
        }
        for i in 0..result.count as usize {
            unsafe {
                let entry = &*result.entries.add(i);
                if !entry.reading.is_null() {
                    drop(CString::from_raw(entry.reading as *mut c_char));
                }
                if !entry.surface.is_null() {
                    drop(CString::from_raw(entry.surface as *mut c_char));
                }
            }
        }
        // Layout must match the one used to allocate. If array::<T>(count)
        // failed at allocation time, the caller received a null pointer
        // and returned above — so if we got this far, count is valid.
        match std::alloc::Layout::array::<BonolithDictEntry>(result.count as usize) {
            Ok(layout) => unsafe {
                std::alloc::dealloc(result.entries as *mut u8, layout);
            },
            Err(e) => log::error!(
                "bonolith_dict_free_entries: layout mismatch count={} err={e}",
                result.count,
            ),
        }
    })
}

/// Export dictionary to a file path. Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_dict_export(path: *const c_char) -> bool {
    ffi_boundary(false, || {
        let path = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
            Ok(s) => std::path::PathBuf::from(s),
            Err(_) => return false,
        };
        let shared = SharedCore::global();
        let dict = read_lock_recover(&shared.dictionary);
        dict.export(&path).is_ok()
    })
}

/// Import dictionary from a file path. Returns number of entries imported, or -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_dict_import(path: *const c_char) -> i32 {
    ffi_boundary(-1, || {
        let path = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
            Ok(s) => std::path::PathBuf::from(s),
            Err(_) => return -1,
        };
        let shared = SharedCore::global();
        let mut dict = write_lock_recover(&shared.dictionary);
        match dict.import(&path) {
            Ok(count) => {
                let _ = dict.sync_user_entries_to_store();
                count as i32
            }
            Err(_) => -1,
        }
    })
}

/// Clear all user learning history (in-memory counts and persistent store).
/// Returns the number of rows deleted, or -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonolith_clear_learning() -> i32 {
    ffi_boundary(-1, || {
        let shared = SharedCore::global();
        let mut user_scorer = mutex_lock_recover(&shared.user_scorer);
        match user_scorer.clear_scores() {
            Ok(n) => n as i32,
            Err(_) => -1,
        }
    })
}
