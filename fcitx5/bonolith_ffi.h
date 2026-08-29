/// Bonolith C FFI header — generated from src/ffi.rs
/// Used by the Fcitx5 C++ addon to interface with the Rust engine.
///
/// NULL safety: every entry point below tolerates a NULL BonolithContext*
/// (and NULL `out` for bonolith_get_ui_state, NULL string pointers for
/// dict entry / import / export operations). Passing NULL returns the
/// documented default sentinel for that function's return type — false /
/// -1 / NULL / 0 — and mutates nothing.  This lets the caller check
/// `bonolith_context_new()` for NULL once and delay allocation failure
/// handling instead of guarding every subsequent call, though the safe
/// pattern is still to check.

#ifndef BONOLITH_FFI_H
#define BONOLITH_FFI_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a Bonolith engine context.
typedef struct BonolithContext BonolithContext;

#define BONOLITH_MAX_SEGMENTS 32
#define BONOLITH_MAX_CANDIDATES 64

/// Segment info for batch UI state.
typedef struct {
    int32_t start_chars;
    int32_t char_len;
} BonolithSegmentInfo;

/// Batch UI state returned by bonolith_get_ui_state().
typedef struct {
    const char *committed;
    bool converting;
    bool has_preedit;
    const char *preedit;
    int32_t segment_count;
    int32_t focus_index;
    BonolithSegmentInfo segments[BONOLITH_MAX_SEGMENTS];
    int32_t candidate_count;
    int32_t selected_index;
    const char *candidates[BONOLITH_MAX_CANDIDATES];
} BonolithUiState;

// ── Lifecycle ────────────────────────────────────────────────────────────

BonolithContext *bonolith_context_new(void);
void bonolith_context_free(BonolithContext *ctx);

// ── Key handling ─────────────────────────────────────────────────────────

/// Process a key event. Returns true if the key was consumed.
bool bonolith_handle_key(BonolithContext *ctx, uint32_t keyval, uint32_t state);

// ── Batch state query ────────────────────────────────────────────────────

/// Get the complete UI state in a single call.
void bonolith_get_ui_state(BonolithContext *ctx, BonolithUiState *out);

// ── Individual state queries (legacy) ────────────────────────────────────

const char *bonolith_get_preedit(BonolithContext *ctx);
const char *bonolith_poll_commit(BonolithContext *ctx);
bool bonolith_is_converting(BonolithContext *ctx);
bool bonolith_has_preedit(BonolithContext *ctx);

/// True while a background LLM rerank pass is outstanding (triggered by the last
/// conversion start or resize but not yet applied). Poll target gate.
bool bonolith_rerank_pending(BonolithContext *ctx);

/// Apply the background LLM rerank result if ready. Returns true if candidates
/// changed (caller should refresh UI). Non-blocking; false when not ready yet.
bool bonolith_poll_apply_rerank(BonolithContext *ctx);
const char *bonolith_composed_text(BonolithContext *ctx);
int32_t bonolith_segment_count(BonolithContext *ctx);
int32_t bonolith_focus_index(BonolithContext *ctx);
int32_t bonolith_segment_start_chars(BonolithContext *ctx, int32_t seg);
int32_t bonolith_segment_char_len(BonolithContext *ctx, int32_t seg);
int32_t bonolith_candidate_count(BonolithContext *ctx);
const char *bonolith_candidate_text(BonolithContext *ctx, int32_t index);
int32_t bonolith_selected_index(BonolithContext *ctx);

void bonolith_reset(BonolithContext *ctx);

/// Commit any in-progress composition (conversion candidate or raw preedit),
/// then clear composing state. Committed text is delivered via the next
/// bonolith_get_ui_state()/bonolith_poll_commit(). No-op when nothing composes.
void bonolith_commit_input(BonolithContext *ctx);

// ── Dictionary operations (global, not per-context) ─────────────────────

typedef struct {
    const char *reading;
    const char *surface;
} BonolithDictEntry;

typedef struct {
    BonolithDictEntry *entries;
    int32_t count;
} BonolithDictEntries;

/// Add a word to the user dictionary and save. Returns true on success.
bool bonolith_dict_add_entry(const char *reading, const char *surface);

/// Delete a user dictionary entry by (reading, surface) identity.
/// Returns true if a matching row was found and removed. Callers
/// (the manage-dict dialog) capture the pair from the row the user
/// picks so a concurrent register between "show list" and "confirm
/// delete" doesn't clobber the newly-added row.
bool bonolith_dict_delete_entry_by_identity(const char *reading,
                                            const char *surface);

/// Update a user dictionary entry identified by (old_reading, old_surface)
/// to a new (reading, surface) pair. Returns true on success. Same
/// snapshot-safety story as bonolith_dict_delete_entry_by_identity.
bool bonolith_dict_update_entry_by_identity(const char *old_reading,
                                            const char *old_surface,
                                            const char *new_reading,
                                            const char *new_surface);

/// Get all user dictionary entries. Caller must free with bonolith_dict_free_entries().
BonolithDictEntries bonolith_dict_get_user_entries(void);

/// Free entries returned by bonolith_dict_get_user_entries().
void bonolith_dict_free_entries(BonolithDictEntries result);

/// Export dictionary to a file path. Returns true on success.
bool bonolith_dict_export(const char *path);

/// Import dictionary from a file path. Returns count imported, or -1 on error.
int32_t bonolith_dict_import(const char *path);

// ── User learning history ───────────────────────────────────────────────

/// Clear all user learning history. Returns number of rows deleted, or -1 on error.
int32_t bonolith_clear_learning(void);

#ifdef __cplusplus
}
#endif

#endif // BONOLITH_FFI_H
