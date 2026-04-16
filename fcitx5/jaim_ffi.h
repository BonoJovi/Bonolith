/// JaIM C FFI header — generated from src/ffi.rs
/// Used by the Fcitx5 C++ addon to interface with the Rust engine.

#ifndef JAIM_FFI_H
#define JAIM_FFI_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a JaIM engine context.
typedef struct JaimContext JaimContext;

#define JAIM_MAX_SEGMENTS 32
#define JAIM_MAX_CANDIDATES 64

/// Segment info for batch UI state.
typedef struct {
    int32_t start_chars;
    int32_t char_len;
} JaimSegmentInfo;

/// Batch UI state returned by jaim_get_ui_state().
typedef struct {
    const char *committed;
    bool converting;
    bool has_preedit;
    const char *preedit;
    int32_t segment_count;
    int32_t focus_index;
    JaimSegmentInfo segments[JAIM_MAX_SEGMENTS];
    int32_t candidate_count;
    int32_t selected_index;
    const char *candidates[JAIM_MAX_CANDIDATES];
} JaimUiState;

// ── Lifecycle ────────────────────────────────────────────────────────────

JaimContext *jaim_context_new(void);
void jaim_context_free(JaimContext *ctx);

// ── Key handling ─────────────────────────────────────────────────────────

/// Process a key event. Returns true if the key was consumed.
bool jaim_handle_key(JaimContext *ctx, uint32_t keyval, uint32_t state);

// ── Batch state query ────────────────────────────────────────────────────

/// Get the complete UI state in a single call.
void jaim_get_ui_state(JaimContext *ctx, JaimUiState *out);

// ── Individual state queries (legacy) ────────────────────────────────────

const char *jaim_get_preedit(JaimContext *ctx);
const char *jaim_poll_commit(JaimContext *ctx);
bool jaim_is_converting(JaimContext *ctx);
bool jaim_has_preedit(JaimContext *ctx);
const char *jaim_composed_text(JaimContext *ctx);
int32_t jaim_segment_count(JaimContext *ctx);
int32_t jaim_focus_index(JaimContext *ctx);
int32_t jaim_segment_start_chars(JaimContext *ctx, int32_t seg);
int32_t jaim_segment_char_len(JaimContext *ctx, int32_t seg);
int32_t jaim_candidate_count(JaimContext *ctx);
const char *jaim_candidate_text(JaimContext *ctx, int32_t index);
int32_t jaim_selected_index(JaimContext *ctx);

void jaim_reset(JaimContext *ctx);

// ── Dictionary operations (global, not per-context) ─────────────────────

typedef struct {
    const char *reading;
    const char *surface;
} JaimDictEntry;

typedef struct {
    JaimDictEntry *entries;
    int32_t count;
} JaimDictEntries;

/// Add a word to the user dictionary and save. Returns true on success.
bool jaim_dict_add_entry(const char *reading, const char *surface);

/// Delete a user dictionary entry by index. Returns true on success.
bool jaim_dict_delete_entry(int32_t index);

/// Update a user dictionary entry by index. Empty strings mean "no change".
bool jaim_dict_update_entry(int32_t index, const char *new_reading,
                            const char *new_surface);

/// Get all user dictionary entries. Caller must free with jaim_dict_free_entries().
JaimDictEntries jaim_dict_get_user_entries(void);

/// Free entries returned by jaim_dict_get_user_entries().
void jaim_dict_free_entries(JaimDictEntries result);

/// Export dictionary to a file path. Returns true on success.
bool jaim_dict_export(const char *path);

/// Import dictionary from a file path. Returns count imported, or -1 on error.
int32_t jaim_dict_import(const char *path);

#ifdef __cplusplus
}
#endif

#endif // JAIM_FFI_H
