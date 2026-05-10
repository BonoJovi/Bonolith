//! C FFI layer for JaIM engine.
//!
//! Provides a C-compatible API so that Fcitx5 (or other C/C++ frameworks)
//! can use JaIM's conversion engine without Rust-specific dependencies.
//!
//! The FFI handles all key dispatch logic (Space→convert, Enter→commit, etc.)
//! so the C++ side only needs to forward key events and read back UI state.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use crate::core::dictionary::{DictionaryEntry, PartOfSpeech};
use crate::engine::{ConversionEngine, SharedCore};

// X11 keysym values (shared by IBus and Fcitx5)
const KEY_SPACE: u32 = 0x0020;
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
pub struct JaimSegmentInfo {
    /// Character start position in composed text.
    pub start_chars: i32,
    /// Character length in composed text.
    pub char_len: i32,
}

/// Batch UI state returned by jaim_get_ui_state().
/// All string pointers are valid until the next call to jaim_handle_key() or jaim_get_ui_state().
#[repr(C)]
pub struct JaimUiState {
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
    pub segments: [JaimSegmentInfo; MAX_SEGMENTS],
    /// Number of candidates for the focused segment.
    pub candidate_count: i32,
    /// Selected candidate index.
    pub selected_index: i32,
    /// Candidate text pointers (up to MAX_CANDIDATES).
    pub candidates: [*const c_char; MAX_CANDIDATES],
}

/// Opaque handle to the JaIM engine context.
pub struct JaimContext {
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

/// Full-width punctuation mapping.
fn to_fullwidth(ch: char) -> Option<&'static str> {
    match ch {
        ',' => Some("、"),
        '.' => Some("。"),
        '!' => Some("！"),
        '?' => Some("？"),
        '(' => Some("（"),
        ')' => Some("）"),
        '[' => Some("［"),
        ']' => Some("］"),
        '{' => Some("｛"),
        '}' => Some("｝"),
        '+' => Some("＋"),
        '=' => Some("＝"),
        '*' => Some("＊"),
        '/' => Some("／"),
        '\\' => Some("＼"),
        '&' => Some("＆"),
        '@' => Some("＠"),
        '#' => Some("＃"),
        '$' => Some("＄"),
        '%' => Some("％"),
        '^' => Some("＾"),
        '|' => Some("｜"),
        '~' => Some("～"),
        '<' => Some("＜"),
        '>' => Some("＞"),
        ':' => Some("："),
        ';' => Some("；"),
        '_' => Some("＿"),
        '"' => Some("＂"),
        '`' => Some("｀"),
        '0' => Some("０"),
        '1' => Some("１"),
        '2' => Some("２"),
        '3' => Some("３"),
        '4' => Some("４"),
        '5' => Some("５"),
        '6' => Some("６"),
        '7' => Some("７"),
        '8' => Some("８"),
        '9' => Some("９"),
        _ => None,
    }
}

fn is_printable_ascii(keyval: u32) -> bool {
    (0x0020..=0x007E).contains(&keyval)
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_context_new() -> *mut JaimContext {
    let ctx = Box::new(JaimContext {
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
pub unsafe extern "C" fn jaim_context_free(ctx: *mut JaimContext) {
    if !ctx.is_null() {
        unsafe { drop(Box::from_raw(ctx)) }
    }
}

// ── Key handling ─────────────────────────────────────────────────────────────

/// Process a key event. Returns true if the key was consumed.
///
/// After calling this, use jaim_poll_commit() to check for committed text,
/// and jaim_get_preedit() / jaim_is_converting() / jaim_*() for UI state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_handle_key(
    ctx: *mut JaimContext,
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

    // F6 → hiragana
    if keyval == KEY_F6 {
        if ctx.converting {
            ctx.engine.convert_focused_to_hiragana();
        } else {
            ctx.engine.start_kana_conversion(0);
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
        } else {
            ctx.engine.start_kana_conversion(form);
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

    // Escape → cancel input
    if keyval == KEY_ESCAPE {
        ctx.engine.reset();
        ctx.engine.clear_conversion();
        ctx.converting = false;
        return true;
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
                ctx.engine.append_raw(fw);
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
fn handle_conversion_key(ctx: &mut JaimContext, keyval: u32, has_shift: bool) -> bool {
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
        _ if is_printable_ascii(keyval) => {
            // Commit conversion first, then process the new character
            if let Some(text) = ctx.engine.commit_conversion() {
                ctx.pending_commit = Some(text);
            }
            ctx.converting = false;
            // Process the incoming character (punctuation, letter, etc.)
            if let Some(ch) = char::from_u32(keyval) {
                if let Some(fw) = to_fullwidth(ch) {
                    ctx.engine.append_raw(fw);
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
/// The returned pointer is valid until the next call to any jaim_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_get_preedit(ctx: *mut JaimContext) -> *const c_char {
    let ctx = unsafe { &mut *ctx };
    let preedit = ctx.engine.preedit();
    ctx.cache_preedit = CString::new(preedit).unwrap_or_default();
    ctx.cache_preedit.as_ptr()
}

/// Poll for committed text. Returns null if nothing to commit.
/// Clears the pending commit after returning.
/// The returned pointer is valid until the next call to any jaim_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_poll_commit(ctx: *mut JaimContext) -> *const c_char {
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
pub unsafe extern "C" fn jaim_is_converting(ctx: *mut JaimContext) -> bool {
    let ctx = unsafe { &*ctx };
    ctx.converting
}

/// Returns true if there is preedit text.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_has_preedit(ctx: *mut JaimContext) -> bool {
    let ctx = unsafe { &*ctx };
    !ctx.engine.preedit().is_empty()
}

/// Get the composed text during conversion mode.
/// The returned pointer is valid until the next call to any jaim_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_composed_text(ctx: *mut JaimContext) -> *const c_char {
    let ctx = unsafe { &mut *ctx };
    let text = ctx.engine.conversion_state()
        .map(|s| s.composed_text())
        .unwrap_or_default();
    ctx.cache_composed = CString::new(text).unwrap_or_default();
    ctx.cache_composed.as_ptr()
}

/// Get the number of segments in the current conversion.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_segment_count(ctx: *mut JaimContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.segments.len() as i32)
        .unwrap_or(0)
}

/// Get the currently focused segment index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_focus_index(ctx: *mut JaimContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.focus as i32)
        .unwrap_or(0)
}

/// Get the character start position of a segment in the composed text.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_segment_start_chars(ctx: *mut JaimContext, seg: i32) -> i32 {
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
pub unsafe extern "C" fn jaim_segment_char_len(ctx: *mut JaimContext, seg: i32) -> i32 {
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
pub unsafe extern "C" fn jaim_candidate_count(ctx: *mut JaimContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.segments[s.focus].candidates.len() as i32)
        .unwrap_or(0)
}

/// Get a candidate text by index (for the focused segment).
/// The returned pointer is valid until the next call to any jaim_* function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_candidate_text(ctx: *mut JaimContext, index: i32) -> *const c_char {
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
pub unsafe extern "C" fn jaim_selected_index(ctx: *mut JaimContext) -> i32 {
    let ctx = unsafe { &*ctx };
    ctx.engine.conversion_state()
        .map(|s| s.segments[s.focus].selected as i32)
        .unwrap_or(0)
}

/// Reset the engine state (called on focus change, deactivation, etc.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_reset(ctx: *mut JaimContext) {
    let ctx = unsafe { &mut *ctx };
    ctx.engine.reset();
    ctx.engine.clear_conversion();
    ctx.converting = false;
    ctx.pending_commit = None;
}

// ── Batch UI state query ────────────────────────────────────────────────────

/// Get the complete UI state in a single FFI call.
/// The returned struct's string pointers are valid until the next call to
/// jaim_handle_key() or jaim_get_ui_state().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_get_ui_state(ctx: *mut JaimContext, out: *mut JaimUiState) {
    let ctx = unsafe { &mut *ctx };
    let out = unsafe { &mut *out };

    // Zero-init segments and candidates
    for i in 0..MAX_SEGMENTS {
        out.segments[i] = JaimSegmentInfo { start_chars: 0, char_len: 0 };
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
                out.segments[i] = JaimSegmentInfo {
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

/// Entry info returned by jaim_dict_get_user_entries().
#[repr(C)]
pub struct JaimDictEntry {
    pub reading: *const c_char,
    pub surface: *const c_char,
}

/// Result of jaim_dict_get_user_entries().
/// Caller must free with jaim_dict_free_entries().
#[repr(C)]
pub struct JaimDictEntries {
    pub entries: *mut JaimDictEntry,
    pub count: i32,
}

/// Add a word to the user dictionary and save to disk.
/// Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_dict_add_entry(
    reading: *const c_char,
    surface: *const c_char,
) -> bool {
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
    let mut dict = shared.dictionary.write().unwrap();
    dict.add_entry(entry);
    dict.sync_user_entries_to_store().is_ok()
}

/// Delete a user dictionary entry by index. Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_dict_delete_entry(index: i32) -> bool {
    let shared = SharedCore::global();
    let mut dict = shared.dictionary.write().unwrap();
    let mut entries = dict.user_entries().to_vec();
    let idx = index as usize;
    if idx >= entries.len() {
        return false;
    }
    entries.remove(idx);
    dict.replace_user_entries(entries);
    dict.sync_user_entries_to_store().is_ok()
}

/// Update a user dictionary entry by index. Empty strings mean "no change".
/// Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_dict_update_entry(
    index: i32,
    new_reading: *const c_char,
    new_surface: *const c_char,
) -> bool {
    let new_reading = match unsafe { std::ffi::CStr::from_ptr(new_reading) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return false,
    };
    let new_surface = match unsafe { std::ffi::CStr::from_ptr(new_surface) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return false,
    };

    let shared = SharedCore::global();
    let mut dict = shared.dictionary.write().unwrap();
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
}

/// Get all user dictionary entries. Caller must free with jaim_dict_free_entries().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_dict_get_user_entries() -> JaimDictEntries {
    let shared = SharedCore::global();
    let dict = shared.dictionary.read().unwrap();
    let user = dict.user_entries();

    if user.is_empty() {
        return JaimDictEntries {
            entries: ptr::null_mut(),
            count: 0,
        };
    }

    let count = user.len();
    let layout = std::alloc::Layout::array::<JaimDictEntry>(count).unwrap();
    let entries = unsafe { std::alloc::alloc(layout) as *mut JaimDictEntry };

    for (i, e) in user.iter().enumerate() {
        let reading = CString::new(e.reading.as_str()).unwrap_or_default();
        let surface = CString::new(e.surface.as_str()).unwrap_or_default();
        unsafe {
            (*entries.add(i)).reading = reading.into_raw();
            (*entries.add(i)).surface = surface.into_raw();
        }
    }

    JaimDictEntries {
        entries,
        count: count as i32,
    }
}

/// Free entries returned by jaim_dict_get_user_entries().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_dict_free_entries(result: JaimDictEntries) {
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
    let layout = std::alloc::Layout::array::<JaimDictEntry>(result.count as usize).unwrap();
    unsafe { std::alloc::dealloc(result.entries as *mut u8, layout) };
}

/// Export dictionary to a file path. Returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_dict_export(path: *const c_char) -> bool {
    let path = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
        Ok(s) => std::path::PathBuf::from(s),
        Err(_) => return false,
    };
    let shared = SharedCore::global();
    let dict = shared.dictionary.read().unwrap();
    dict.export(&path).is_ok()
}

/// Import dictionary from a file path. Returns number of entries imported, or -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jaim_dict_import(path: *const c_char) -> i32 {
    let path = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
        Ok(s) => std::path::PathBuf::from(s),
        Err(_) => return -1,
    };
    let shared = SharedCore::global();
    let mut dict = shared.dictionary.write().unwrap();
    match dict.import(&path) {
        Ok(count) => {
            let _ = dict.sync_user_entries_to_store();
            count as i32
        }
        Err(_) => -1,
    }
}
