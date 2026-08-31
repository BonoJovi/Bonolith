/// IBus Engine D-Bus interface implementation.
///
/// Implements org.freedesktop.IBus.Engine via zbus #[interface].
/// Bridges IBus key events to Bonolith's ConversionEngine and sends
/// preedit/commit/candidates back via D-Bus signals.
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use log::{debug, info, warn};
use zbus::object_server::SignalEmitter;
use zbus::{interface, zvariant, Connection};

use bonolith::core::dictionary::{Dictionary, DictionaryEntry, PartOfSpeech};
use bonolith::engine::dispatch::{dispatch_key, DisplayUpdate, KeyEvent, KeyOutcome};
use bonolith::engine::{ConversionEngine, ConversionState, SharedCore};

use super::config::{CompiledToggleKey, BonolithConfig};
use super::factory::ForceEnable;
use super::keymap::*;

/// Fold ASCII uppercase (A-Z) to lowercase (a-z) for keyval matching.
/// Mirrors `CompiledToggleKey::matches`'s Shift+letter normalisation so
/// pending_release records/lookups survive Shift-first release order:
/// press "Shift+A" → OS delivers 0x41; release Shift before A → the A
/// release arrives as 0x61 (Shift no longer set at the release event).
/// Without folding both sides, the release misses the recorded 0x41
/// and leaks as a phantom character in XIM-derived clients — the same
/// symptom bug [13] tackled for non-shifted keys. Bug [28].
fn unshift_letter(keyval: u32) -> u32 {
    if (0x41..=0x5A).contains(&keyval) {
        keyval + 0x20
    } else {
        keyval
    }
}

/// IBus Engine state
pub struct BonolithEngine {
    /// Arc so a detached background task (LLM rerank refresh) can hold a clone
    /// and mutate the conversion state after the key handler has returned.
    engine: Arc<Mutex<ConversionEngine>>,
    /// Whether the engine is active (enabled by IBus)
    enabled: Mutex<bool>,
    /// Whether we are in conversion mode (showing candidates). Arc for the same
    /// reason as `engine`: the rerank-refresh task checks it before emitting.
    converting: Arc<Mutex<bool>>,
    /// Compiled toggle key bindings (immutable after creation)
    toggle_keys: Vec<CompiledToggleKey>,
    /// Shared force-on window set by the word-register dialog (see
    /// `factory::ForceEnable`).
    force: ForceEnable,
    /// Object path this engine is registered at — used by `destroy` to remove
    /// itself from the connection's `ObjectServer` so old engines don't leak.
    object_path: String,
    /// Keyvals whose press we consumed and are still awaiting a release for.
    /// The release-consume gate checks membership here BEFORE `enabled` so a
    /// toggle-off (which flips enabled=false on the press) or Muhenkan-off
    /// still consumes its own release — otherwise the orphan release lets
    /// some XIM/IBus clients synthesise a phantom press+release pair,
    /// injecting the raw char (grave `, Enter, …) into the app. Bounded
    /// implicitly by the number of physical keys the user can hold.
    pending_release: Mutex<HashSet<u32>>,
    /// Wall-clock timestamp of the most recent transition from enabled=true
    /// to enabled=false (Muhenkan-off, toggle-off, disable()). Any release
    /// arriving within `DISABLE_GRACE_MS` after this timestamp is consumed
    /// even if its keyval was not pre-registered in `pending_release`.
    ///
    /// Fable-5 D-group #13 residue: if the user is holding a non-toggle
    /// key (say Space) when they hit Muhenkan, the press-of-Space was
    /// consumed while enabled=true and `pending_release` only tracks the
    /// Muhenkan keyval itself — the trailing release-of-Space then falls
    /// through to `enabled == false` and leaks to XIM as an orphan release,
    /// producing a phantom press for that key. The grace window catches
    /// exactly this race without needing to enumerate every pressed key.
    disabled_at: Mutex<Option<std::time::Instant>>,
}

impl BonolithEngine {
    pub fn new(config: &BonolithConfig, force: ForceEnable, object_path: String) -> Self {
        let toggle_keys = config.compile_toggle_keys();
        info!(
            "Bonolith: Engine created at {} with {} toggle key binding(s)",
            object_path,
            toggle_keys.len()
        );
        Self {
            engine: Arc::new(Mutex::new(ConversionEngine::new())),
            enabled: Mutex::new(false),
            converting: Arc::new(Mutex::new(false)),
            toggle_keys,
            force,
            object_path,
            pending_release: Mutex::new(HashSet::new()),
            disabled_at: Mutex::new(None),
        }
    }

    /// Mark the moment we transitioned to enabled=false so any release
    /// arriving in the next `DISABLE_GRACE_MS` window is consumed even if
    /// its press-side keyval was not pre-registered (Fable-5 D-group #13
    /// residue). Callers: Muhenkan handling, toggle-off, `disable()`.
    fn mark_disabled(&self) {
        *self.disabled_at.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(std::time::Instant::now());
    }

    /// If the word-register dialog has opened a force-on window, flip this
    /// engine 日本語ON. Idempotent; safe to call from focus/enable/key paths.
    /// Returns true if it (or a prior call) left the engine enabled.
    fn apply_force_enable(&self) -> bool {
        let active = matches!(
            *self.force.lock().unwrap_or_else(|e| e.into_inner()),
            Some(deadline) if std::time::Instant::now() < deadline
        );
        if active {
            let mut enabled = self.enabled.lock().unwrap_or_else(|e| e.into_inner());
            if !*enabled {
                *enabled = true;
                info!("Bonolith: force-enable window → enabled=true");
            }
        }
        active
    }
}

/// Empty attachments dict reused by all IBus serializable builders.
fn ibus_attachments() -> std::collections::HashMap<String, zvariant::Value<'static>> {
    std::collections::HashMap::new()
}

/// Build an IBusAttrList: ("IBusAttrList", {attachments}, av[])
fn ibus_attr_list() -> zvariant::Value<'static> {
    zvariant::Value::new(zvariant::Structure::from((
        "IBusAttrList",
        ibus_attachments(),
        Vec::<zvariant::Value>::new(),
    )))
}

/// Build an IBusText: ("IBusText", {attachments}, text, v(IBusAttrList))
fn ibus_text(text: &str) -> zvariant::Value<'static> {
    zvariant::Value::new(zvariant::Structure::from((
        "IBusText",
        ibus_attachments(),
        text.to_string(),
        ibus_attr_list(),
    )))
}

/// Build an IBusText with custom attributes.
fn ibus_text_with_attrs(text: &str, attrs: Vec<zvariant::Value<'static>>) -> zvariant::Value<'static> {
    zvariant::Value::new(zvariant::Structure::from((
        "IBusText",
        ibus_attachments(),
        text.to_string(),
        ibus_attr_list_with(attrs),
    )))
}

/// Build an IBusPropList: ("IBusPropList", {attachments}, av[properties])
fn ibus_prop_list(props: Vec<zvariant::Value<'static>>) -> zvariant::Value<'static> {
    zvariant::Value::new(zvariant::Structure::from((
        "IBusPropList",
        ibus_attachments(),
        props,
    )))
}

/// Build an IBusProperty:
/// ("IBusProperty", {attachments}, key, type, v(label), icon, v(tooltip),
///  sensitive, visible, state, v(sub_props))
fn ibus_property(
    key: &str,
    prop_type: u32,
    label: &str,
    icon: &str,
    tooltip: &str,
) -> zvariant::Value<'static> {
    zvariant::Value::new(zvariant::Structure::from((
        "IBusProperty",
        ibus_attachments(),
        key.to_string(),           // key (s)
        prop_type,                 // type (u)
        ibus_text(label),          // label (v → IBusText)
        icon.to_string(),          // icon (s)
        ibus_text(tooltip),        // tooltip (v → IBusText)
        true,                      // sensitive (b)
        true,                      // visible (b)
        0u32,                      // state (u)
        ibus_prop_list(vec![]),    // sub_props (v → IBusPropList)
        ibus_text(""),             // symbol (v → IBusText)
    )))
}

/// Build an IBusAttribute: ("IBusAttribute", {attachments}, type, value, start, end)
/// type: 1=underline, 2=foreground, 3=background
/// underline values: 0=none, 1=single, 2=double, 3=low
fn ibus_attribute(attr_type: u32, value: u32, start: u32, end: u32) -> zvariant::Value<'static> {
    zvariant::Value::new(zvariant::Structure::from((
        "IBusAttribute",
        ibus_attachments(),
        attr_type,
        value,
        start,
        end,
    )))
}

/// Build an IBusAttrList with the given attributes.
fn ibus_attr_list_with(attrs: Vec<zvariant::Value<'static>>) -> zvariant::Value<'static> {
    zvariant::Value::new(zvariant::Structure::from((
        "IBusAttrList",
        ibus_attachments(),
        attrs,
    )))
}

/// Build an IBusText with segment highlighting.
/// All text gets single underline; the focused segment gets double underline.
fn ibus_text_with_segments(text: &str, ranges: &[(usize, usize)], focus: usize) -> zvariant::Value<'static> {
    let total_chars = text.chars().count() as u32;
    let mut attrs = Vec::new();

    // Single underline for entire text
    attrs.push(ibus_attribute(1, 1, 0, total_chars));

    // Double underline for focused segment
    if let Some(&(start, end)) = ranges.get(focus) {
        attrs.push(ibus_attribute(1, 2, start as u32, end as u32));
    }

    zvariant::Value::new(zvariant::Structure::from((
        "IBusText",
        ibus_attachments(),
        text.to_string(),
        ibus_attr_list_with(attrs),
    )))
}

/// Build an IBusLookupTable:
/// ("IBusLookupTable", {attachments}, page_size, cursor_pos, cursor_visible,
///  round, orientation, candidates[], labels[])
fn ibus_lookup_table(candidates: &[String], selected: usize) -> zvariant::Value<'static> {
    let candidate_values: Vec<zvariant::Value> = candidates
        .iter()
        .map(|c| ibus_text(c))
        .collect();
    let labels: Vec<zvariant::Value> = (0..candidates.len())
        .map(|i| ibus_text(&format!("{}.", i + 1)))
        .collect();
    zvariant::Value::new(zvariant::Structure::from((
        "IBusLookupTable",
        ibus_attachments(),
        9u32,                    // page_size
        selected as u32,         // cursor_pos
        true,                    // cursor_visible
        true,                    // round
        1i32,                    // orientation: 0=horizontal, 1=vertical
        candidate_values,        // candidates
        labels,                  // labels
    )))
}

#[interface(name = "org.freedesktop.IBus.Engine")]
impl BonolithEngine {
    /// Process a key event. Returns true if handled.
    async fn process_key_event(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        keyval: u32,
        _keycode: u32,
        state: u32,
    ) -> zbus::fdo::Result<bool> {
        // Consume all key releases while IME is active. Leaking a release for
        // a key whose press we consumed (Space, arrows, Enter, F6–F10) lets
        // some XIM/IBus clients reconstruct a phantom press+release pair from
        // the orphan release. Text-carrying apps derive input from press
        // events, so consuming releases is safe.
        //
        // Reading `enabled` here alone is wrong for the toggle-off / Muhenkan-
        // off case: the *press* consumed those and flipped enabled=false, and
        // then this arm would let the matching release through — the client
        // synthesises a phantom press+release for the toggle key (grave `,
        // Enter, …) and injects the raw char. So consult `pending_release`
        // first: any key whose press we consumed must have its release
        // consumed too, regardless of the current enabled state.
        if is_release(state) {
            // Fold A-Z → a-z on lookup so a Shift+letter binding that
            // recorded the uppercase keysym on press still matches when
            // the user releases Shift before the letter (bug [28]).
            let lookup_key = unshift_letter(keyval);
            let owed = self
                .pending_release
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&lookup_key);
            if owed {
                return Ok(true);
            }
            let enabled = *self.enabled.lock().unwrap_or_else(|e| e.into_inner());
            if enabled {
                return Ok(true);
            }
            // Post-disable grace window: any key whose press we consumed
            // while enabled=true can have its release arrive AFTER a
            // Muhenkan / toggle-off flipped enabled=false. Consuming it
            // here prevents XIM clients from synthesising a phantom press
            // for the still-held key (Fable-5 D-group #13 residue).
            //
            // Kept short (80 ms) so a fast typist hitting Muhenkan and
            // then a letter within the same fingering doesn't have the
            // fresh letter's release swallowed — 500 ms was wide enough
            // to eat 2-3 keystrokes at 5 chars/sec, producing phantom
            // held-key state in XIM/state-tracking clients (games,
            // browser keyup handlers). Real held-key releases arrive
            // within one input cycle (< 20 ms), so 80 ms leaves a
            // comfortable margin on slow systems without exposing the
            // over-consumption tail (Fable-5 bug_001).
            const DISABLE_GRACE_MS: u128 = 80;
            let recently_disabled = self
                .disabled_at
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .map_or(false, |t| t.elapsed().as_millis() < DISABLE_GRACE_MS);
            if recently_disabled {
                return Ok(true);
            }
            return Ok(false);
        }

        // If the word-register dialog opened a force-on window, adopt it now —
        // before the `if !enabled` gate below — so the very first keystroke in
        // the dialog is already Japanese, even when the focus_in callback fired
        // before the dialog set the flag.
        self.apply_force_enable();

        debug!(
            "Bonolith: KeyEvent keyval=0x{:04X} keycode={} state=0x{:08X}",
            keyval, _keycode, state
        );

        // Absolute IME on/off via the Japanese-keyboard 変換 / 無変換 keys.
        // Unlike the toggle key, these set a *known* state regardless of the
        // current one — handy on a physical JP keyboard. (The word-register
        // dialog no longer relies on this; it uses the org.bonolith.Control
        // session-bus surface instead.) Skipped if the user explicitly bound
        // these as toggle keys.
        if !has_modifier(state) && !self.is_toggle_key(keyval, state) {
            if keyval == IBUS_KEY_HENKAN_MODE {
                *self.enabled.lock().unwrap_or_else(|e| e.into_inner()) = true;
                info!("Bonolith: Henkan → enabled=true (absolute ON)");
                return Ok(true);
            }
            if keyval == IBUS_KEY_MUHENKAN {
                let _ = self.cancel_input(&emitter).await;
                *self.enabled.lock().unwrap_or_else(|e| e.into_inner()) = false;
                self.mark_disabled();
                info!("Bonolith: Muhenkan → enabled=false (absolute OFF)");
                // Record so the matching release is consumed even though
                // enabled just flipped to false.
                self.pending_release
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(unshift_letter(keyval));
                return Ok(true);
            }
        }

        // Toggle key check — must come before modifier pass-through
        if self.is_toggle_key(keyval, state) {
            let was_enabled = *self.enabled.lock().unwrap_or_else(|e| e.into_inner());
            if was_enabled {
                let _ = self.cancel_input(&emitter).await;
                *self.enabled.lock().unwrap_or_else(|e| e.into_inner()) = false;
                self.mark_disabled();
            } else {
                *self.enabled.lock().unwrap_or_else(|e| e.into_inner()) = true;
            }
            let now = *self.enabled.lock().unwrap_or_else(|e| e.into_inner());
            info!("Bonolith: Toggle → enabled={}", now);
            // The toggle-off press leaves us with enabled=false; without
            // this the matching release would slip through the release-
            // consume gate and let XIM clients synthesise a phantom press
            // for the toggle key (e.g. a raw ` on grave-based bindings).
            // Unshift-fold so Shift+letter toggles (bug [11]) survive
            // Shift-first release order — bug [28].
            self.pending_release
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(unshift_letter(keyval));
            return Ok(true);
        }

        // Pass through modifier combos (Ctrl+C, Alt+Tab, etc.) — checked
        // both here and inside `dispatch_key`, but returning early avoids
        // the engine lock and stays symmetric with the older code path.
        if has_modifier(state) {
            return Ok(false);
        }

        let enabled = *self.enabled.lock().unwrap_or_else(|e| e.into_inner());
        if !enabled {
            return Ok(false);
        }

        // Everything past this point (Tab, F6-F10, Space, arrows, Enter,
        // Escape, Backspace, romaji buildup, fullwidth symbols) is shared
        // with the Fcitx5 FFI via `dispatch_key`. See [10c] in
        // `work/bug_fix_progress_*.md` for the unification rationale.
        let event = KeyEvent { keyval, state };
        let (outcome, conversion_snapshot) = {
            let mut engine = self.engine.lock().unwrap_or_else(|e| e.into_inner());
            let mut converting_flag = *self.converting.lock().unwrap_or_else(|e| e.into_inner());
            let outcome = dispatch_key(&mut engine, &mut converting_flag, event);
            // Snapshot the conversion state under the same lock so the D-Bus
            // emission below sees the version the dispatcher just mutated.
            let snapshot = if matches!(outcome.display, DisplayUpdate::Conversion) {
                engine.conversion_state().cloned()
            } else {
                None
            };
            drop(engine);
            *self.converting.lock().unwrap_or_else(|e| e.into_inner()) = converting_flag;
            (outcome, snapshot)
        };
        self.apply_outcome(&emitter, outcome, conversion_snapshot).await
    }

    /// Called when the engine gains focus.
    async fn focus_in(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) {
        // Adopt an open force-on window so the dialog's field shows 日本語ON the
        // moment it gains focus (no keystroke needed).
        self.apply_force_enable();
        info!("Bonolith: FocusIn (enabled={})", *self.enabled.lock().unwrap_or_else(|e| e.into_inner()));
        // Reset the client's IM state on focus so the XIM proxy doesn't carry
        // a stale buffer from the previous window (Mozc parity).
        let _ = Self::hide_preedit_text(&emitter).await;
        let _ = Self::hide_lookup_table(&emitter).await;
        let _ = Self::hide_auxiliary_text(&emitter).await;
        if let Err(e) = self.register_menu(&emitter).await {
            warn!("Bonolith: Failed to register properties: {}", e);
        }
    }

    /// Called when a menu item is activated.
    async fn property_activate(&self, prop_name: &str, _state: u32) {
        info!("Bonolith: PropertyActivate({})", prop_name);
        match prop_name {
            "bonolith-export" => {
                std::thread::spawn(|| {
                    Self::run_dict_export();
                });
            }
            "bonolith-import" => {
                std::thread::spawn(|| {
                    Self::run_dict_import();
                });
            }
            "bonolith-register-word" => {
                std::thread::spawn(|| {
                    Self::run_word_register();
                });
            }
            "bonolith-manage-dict" => {
                std::thread::spawn(|| {
                    Self::run_manage_dict();
                });
            }
            "bonolith-clear-learning" => {
                let shared = self.engine.lock().unwrap_or_else(|e| e.into_inner()).shared_core();
                std::thread::spawn(move || {
                    Self::run_clear_learning(shared);
                });
            }
            _ => {}
        }
    }

    /// Called when the engine loses focus.
    async fn focus_out(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) {
        // Preedit is set with mode=1 (PREEDIT_COMMIT), so hiding it here
        // lets IBus auto-commit the current text into the focused-away
        // client. That is the same "click-away keeps text" contract as
        // before, but delivered by the framework instead of by our own
        // commit_text — the old explicit commit raced with IBus's own
        // focus-loss dispatch and ended up dropped by some clients
        // (Ghostty / Gnome Terminal), silently losing the composition.
        //
        // Symmetric with the Fcitx5 addon, which relies on the Wayland
        // compositor to auto-finalize its client preedit on the same
        // event. User-scorer learning is skipped for focus-loss commits
        // (record() only runs on explicit commit_conversion in the key
        // handler); rare edge case, Enter/Space commits still learn.
        let _ = Self::hide_preedit_text(&emitter).await;
        let _ = Self::hide_lookup_table(&emitter).await;
        let _ = Self::hide_auxiliary_text(&emitter).await;
        {
            let mut engine = self.engine.lock().unwrap_or_else(|e| e.into_inner());
            engine.reset();
            engine.clear_conversion();
        }
        *self.converting.lock().unwrap_or_else(|e| e.into_inner()) = false;
        // Drop any pending release records — the matching release may
        // never reach us if the client swallowed it during the focus
        // change, and a stale entry would silently eat the NEXT
        // press of the same key on a fresh input context (Devin PR
        // #3 #8).
        self.pending_release
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        // Deliberately keep `enabled` — standard IMEs hold 日本語ON/OFF per
        // input context across focus changes. Clearing it here made every
        // focus-out drop the mode, and neither focus_in nor IBus' enable
        // callback restore it, so the next keystroke fell through as raw
        // Latin. Preedit/converting state is cleaned above; the mode bit
        // belongs to the context, not to the transient composition.
    }

    /// Reset engine state.
    async fn reset(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) {
        debug!("Bonolith: Reset");
        let _ = self.cancel_input(&emitter).await;
    }

    /// Enable the engine.
    async fn enable(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) {
        self.apply_force_enable();
        info!("Bonolith: Enable (enabled={})", *self.enabled.lock().unwrap_or_else(|e| e.into_inner()));
        // Reset the client's IM state on enable (Mozc parity).
        let _ = Self::hide_preedit_text(&emitter).await;
        let _ = Self::hide_lookup_table(&emitter).await;
        let _ = Self::hide_auxiliary_text(&emitter).await;
    }

    /// Disable the engine.
    async fn disable(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) {
        info!("Bonolith: Disable");
        let _ = self.cancel_input(&emitter).await;
        *self.enabled.lock().unwrap_or_else(|e| e.into_inner()) = false;
        self.mark_disabled();
        // Same rationale as focus_out — a release we're owed may
        // never arrive across a disable, and letting stale entries
        // sit would eat the first matching press after re-enable
        // (Devin PR #3 #8).
        self.pending_release
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Set cursor location (unused but required by interface).
    async fn set_cursor_location(&self, _x: i32, _y: i32, _w: i32, _h: i32) {}

    /// Set capabilities (unused but required by interface).
    async fn set_capabilities(&self, _cap: u32) {}

    /// Called by IBus daemon when this engine is no longer needed (typically
    /// when the input context that owns it goes away). Removes the object
    /// from the `ObjectServer` so we don't leak engines — without this, every
    /// context we ever served stays live for the lifetime of the process.
    async fn destroy(
        &self,
        #[zbus(connection)] connection: &Connection,
    ) -> zbus::fdo::Result<()> {
        info!("Bonolith: Destroy engine at {}", self.object_path);
        let path = self.object_path.clone();
        match connection
            .object_server()
            .remove::<Self, _>(path.as_str())
            .await
        {
            Ok(true) => {}
            Ok(false) => warn!(
                "Bonolith: Destroy called on {} but object was not registered",
                path
            ),
            Err(e) => warn!("Bonolith: Destroy remove({}) failed: {}", path, e),
        }
        Ok(())
    }

    // ---- IBus Signals ----

    /// Commit composed text to the application.
    #[zbus(signal)]
    async fn commit_text(emitter: &SignalEmitter<'_>, text: zvariant::Value<'_>)
        -> zbus::Result<()>;

    /// Update preedit text displayed in the input area.
    /// mode: 0 = IBUS_ENGINE_PREEDIT_CLEAR, 1 = IBUS_ENGINE_PREEDIT_COMMIT
    #[zbus(signal)]
    async fn update_preedit_text(
        emitter: &SignalEmitter<'_>,
        text: zvariant::Value<'_>,
        cursor_pos: u32,
        visible: bool,
        mode: u32,
    ) -> zbus::Result<()>;

    /// Hide preedit text.
    #[zbus(signal)]
    async fn hide_preedit_text(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;

    /// Update the lookup table (candidate list).
    #[zbus(signal)]
    async fn update_lookup_table(
        emitter: &SignalEmitter<'_>,
        table: zvariant::Value<'_>,
        visible: bool,
    ) -> zbus::Result<()>;

    /// Hide the lookup table.
    #[zbus(signal)]
    async fn hide_lookup_table(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;

    /// Update the auxiliary text (romaji buffer / status area).
    /// Emitted alongside preedit/lookup updates for parity with Mozc so the
    /// XIM proxy's internal buffer never accumulates stale content.
    #[zbus(signal)]
    async fn update_auxiliary_text(
        emitter: &SignalEmitter<'_>,
        text: zvariant::Value<'_>,
        visible: bool,
    ) -> zbus::Result<()>;

    /// Hide the auxiliary text.
    #[zbus(signal)]
    async fn hide_auxiliary_text(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;

    /// Register the property list (menu items).
    #[zbus(signal)]
    async fn register_properties(
        emitter: &SignalEmitter<'_>,
        properties: zvariant::Value<'_>,
    ) -> zbus::Result<()>;
}

// Private helper methods (not exposed via D-Bus)
impl BonolithEngine {
    /// Check if the given key event matches any configured toggle binding.
    /// The Shift+letter normalisation lives on `CompiledToggleKey::matches`
    /// so it can be unit-tested without spinning up a whole engine.
    fn is_toggle_key(&self, keyval: u32, state: u32) -> bool {
        let relevant_mask = IBUS_CONTROL_MASK | IBUS_MOD1_MASK | IBUS_SHIFT_MASK;
        let active_modifiers = state & relevant_mask;
        self.toggle_keys
            .iter()
            .any(|tk| tk.matches(keyval, active_modifiers))
    }

    /// Translate a [`KeyOutcome`] from the shared dispatcher into IBus
    /// D-Bus signals. Best-effort emission: engine state has already been
    /// mutated by the dispatcher, so a broken signal here (D-Bus dying,
    /// XIM proxy hiccup) is logged rather than propagated — a `?` used to
    /// leave `converting=true` latched and jam further input.
    async fn apply_outcome(
        &self,
        emitter: &SignalEmitter<'_>,
        outcome: KeyOutcome,
        conversion: Option<ConversionState>,
    ) -> zbus::fdo::Result<bool> {
        // Preedit is set with mode=1 (PREEDIT_COMMIT), so a subsequent
        // hide would auto-commit the last-shown preedit text ("きょう")
        // even when we're about to commit our own converted text
        // ("京都") explicitly — that would double-insert as "京都きょう".
        // Neutralise by pushing an empty preedit with mode=0 first;
        // then commit_text + hide_preedit_text combine cleanly. Needed
        // whenever we commit or clear the display; safe (no-op-ish) at
        // other times.
        let had_commit = outcome.commit.is_some();
        let needs_neutralise = had_commit
            || matches!(outcome.display, DisplayUpdate::Cleared);
        if needs_neutralise {
            if let Err(e) = Self::update_preedit_text(
                emitter,
                ibus_text(""),
                0,
                false,
                0,
            ).await {
                warn!("Bonolith: preedit neutralise failed: {}", e);
            }
        }
        if let Some(text) = outcome.commit {
            if let Err(e) = Self::commit_text(emitter, ibus_text(&text)).await {
                warn!("Bonolith: commit_text emission failed: {}", e);
            }
        }
        match outcome.display {
            DisplayUpdate::Unchanged => {}
            DisplayUpdate::Cleared => {
                let _ = Self::hide_preedit_text(emitter).await;
                let _ = Self::hide_lookup_table(emitter).await;
                let _ = Self::hide_auxiliary_text(emitter).await;
            }
            DisplayUpdate::Preedit(text) => {
                // If we just committed (a conversion finished with a
                // Space-path pending buffer still live), the lookup
                // table and aux text from that conversion are still
                // visible. Hide them before showing the fresh preedit
                // so the leftover candidate window doesn't linger over
                // the "m" the user is about to keep typing (bug [27]).
                if had_commit {
                    let _ = Self::hide_lookup_table(emitter).await;
                    let _ = Self::hide_auxiliary_text(emitter).await;
                }
                // Best-effort: engine state has already been mutated by the
                // dispatcher (buffer advanced, romaji flushed, etc.). If
                // the D-Bus emit fails we still need to return
                // outcome.consumed — propagating the error via `?` used to
                // make ProcessKeyEvent return Err, which ibus-daemon
                // treated as "unprocessed" and forwarded the raw keysym
                // to the app while the preedit already held the char.
                // Same warn+continue policy as the commit arm above.
                if let Err(e) = self.send_preedit(emitter, &text).await {
                    warn!("Bonolith: send_preedit emission failed: {}", e);
                }
            }
            DisplayUpdate::Conversion => {
                if let Some(state) = conversion {
                    if let Err(e) = self.show_conversion_state(emitter, &state).await {
                        warn!("Bonolith: show_conversion_state emission failed: {}", e);
                    }
                }
            }
        }
        if outcome.schedule_rerank_refresh {
            self.spawn_rerank_refresh(emitter);
        }
        Ok(outcome.consumed)
    }

    /// Show the conversion state: segmented preedit + lookup table for focused segment.
    async fn show_conversion_state(
        &self,
        emitter: &SignalEmitter<'_>,
        state: &ConversionState,
    ) -> zbus::fdo::Result<()> {
        Self::emit_conversion_state(emitter, state).await
    }

    /// Emit the preedit + lookup-table signals for `state`. Standalone (no
    /// `&self`) so the background rerank-refresh task can call it with an owned
    /// signal emitter after the key handler has returned.
    async fn emit_conversion_state(
        emitter: &SignalEmitter<'_>,
        state: &ConversionState,
    ) -> zbus::fdo::Result<()> {
        let text = state.composed_text();
        let ranges = state.segment_char_ranges();
        let focus = state.focus;

        // Preedit with segment highlighting.
        // mode=1 (PREEDIT_COMMIT): IBus auto-commits the current preedit
        // text (the composed candidate text) into the client on hide /
        // focus loss. Escape / Enter / Space paths neutralise this in
        // apply_outcome by pre-clearing the preedit with mode=0.
        let cursor = text.chars().count() as u32;
        Self::update_preedit_text(
            emitter,
            ibus_text_with_segments(&text, &ranges, focus),
            cursor,
            true,
            1,
        ).await.map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;

        // Lookup table for the focused segment's candidates.
        // Devin PR #3 review #7: defensive `.get` so a stale focus
        // that outran resize doesn't panic — the zbus handler would
        // otherwise take the input context down with it (Fcitx5's
        // ffi_boundary catches, IBus's dispatcher does not).
        if let Some(seg) = state.segments.get(focus) {
            Self::update_lookup_table(
                emitter,
                ibus_lookup_table(&seg.candidates, seg.selected),
                true,
            ).await.map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        }

        // Keep auxiliary text explicitly empty so the XIM proxy doesn't
        // accumulate stale content (Mozc parity).
        Self::update_auxiliary_text(
            emitter,
            ibus_text(""),
            false,
        ).await.map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;

        Ok(())
    }

    /// Spawn a detached task that waits for the background LLM rerank to finish,
    /// applies it, and re-emits the conversion display — so context reranking
    /// surfaces without the user pressing another key. Cheap no-op if no result
    /// ever arrives. Called after a conversion is (re)built (start / resize).
    fn spawn_rerank_refresh(&self, emitter: &SignalEmitter<'_>) {
        let engine = Arc::clone(&self.engine);
        let converting = Arc::clone(&self.converting);
        let emitter = emitter.to_owned();

        tokio::spawn(async move {
            // Poll for the background result with a bounded budget. The rerank
            // runs on a llama-server round-trip per segment, so allow ~2s.
            const POLL_INTERVAL_MS: u64 = 60;
            const MAX_POLLS: u32 = 34; // ~2.0s total
            for _ in 0..MAX_POLLS {
                if engine.lock().unwrap_or_else(|e| e.into_inner()).has_llm_rerank_result() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;
            }

            // Apply and snapshot the refreshed state under one lock. Skip if the
            // user already left conversion mode (committed / cancelled). Also
            // capture the rerank generation so we can bail before the emit if
            // a commit / cancel bumps it between locks.
            let refreshed = {
                if !*converting.lock().unwrap_or_else(|e| e.into_inner()) {
                    return;
                }
                let mut engine = engine.lock().unwrap_or_else(|e| e.into_inner());
                if !engine.apply_llm_rerank() {
                    return;
                }
                let state = engine.conversion_state().cloned();
                let epoch = engine.rerank_generation();
                state.map(|s| (s, epoch))
            };

            if let Some((state, emit_gen)) = refreshed {
                // Between the lock drop above and the D-Bus emit below, the
                // key handler may commit or cancel the conversion. Emitting
                // afterwards leaves a mode=1 ghost preedit that IBus auto-
                // commits on focus loss (duplicate insertion). Re-check
                // converting AND the rerank generation as a single guard —
                // any invalidation bumps the generation, so a mismatch means
                // the snapshot above is already stale.
                {
                    let engine = engine.lock().unwrap_or_else(|e| e.into_inner());
                    if !*converting.lock().unwrap_or_else(|e| e.into_inner())
                        || engine.rerank_generation() != emit_gen
                    {
                        return;
                    }
                }
                if let Err(e) = Self::emit_conversion_state(&emitter, &state).await {
                    debug!("Bonolith: rerank refresh emit failed: {e}");
                }
            }
        });
    }

    async fn cancel_input(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        {
            let mut engine = self.engine.lock().unwrap_or_else(|e| e.into_inner());
            engine.reset();
            engine.clear_conversion();
        }
        *self.converting.lock().unwrap_or_else(|e| e.into_inner()) = false;
        // Every preedit we display is mode=1 (PREEDIT_COMMIT), so calling
        // hide_preedit_text directly triggers the client's auto-commit of
        // whatever preedit text was last shown — the exact opposite of
        // "cancel". Neutralise with an empty mode=0 preedit first, mirroring
        // apply_outcome's DisplayUpdate::Cleared / commit branch. Without
        // this, Muhenkan / toggle-off / reset / disable while a preedit
        // ("きょう") or conversion ("京都") is on screen silently inserts
        // it into the client.
        if let Err(e) = Self::update_preedit_text(
            emitter,
            ibus_text(""),
            0,
            false,
            0,
        ).await {
            warn!("Bonolith: cancel_input preedit neutralise failed: {}", e);
        }
        Self::hide_preedit_text(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_lookup_table(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_auxiliary_text(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Ok(true)
    }

    async fn send_preedit(
        &self,
        emitter: &SignalEmitter<'_>,
        text: &str,
    ) -> zbus::fdo::Result<()> {
        let cursor = text.chars().count() as u32;
        let visible = !text.is_empty();
        // Preedit text must have an underline attribute for terminals to render it
        let preedit_text = if visible {
            let attrs = vec![ibus_attribute(1, 1, 0, cursor)]; // single underline
            ibus_text_with_attrs(text, attrs)
        } else {
            ibus_text(text)
        };
        // mode=1 (PREEDIT_COMMIT): IBus auto-commits the current preedit
        // text on hide / focus loss (Ghostty / Gnome Terminal used to
        // silently lose focus-loss text with mode=0). Escape / Enter
        // paths pre-clear with mode=0 in apply_outcome to prevent
        // double-commit against our own commit_text.
        Self::update_preedit_text(emitter, preedit_text, cursor, visible, 1).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        // Keep lookup and auxiliary explicitly cleared during romaji buildup
        // so no stale state accumulates in the client (Mozc parity).
        Self::update_lookup_table(
            emitter,
            ibus_lookup_table(&[], 0),
            false,
        ).await.map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::update_auxiliary_text(
            emitter,
            ibus_text(""),
            false,
        ).await.map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Ok(())
    }

    /// Register menu items (Export/Import) in the IBus property panel.
    async fn register_menu(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        // prop_type: 0=normal, 1=toggle, 2=radio, 3=separator, 4=menu
        let export_prop = ibus_property(
            "bonolith-export", 0,
            "辞書エクスポート...", "",
            "辞書をJSONファイルにエクスポート",
        );
        let import_prop = ibus_property(
            "bonolith-import", 0,
            "辞書インポート...", "",
            "JSONファイルから辞書をインポート",
        );
        let register_prop = ibus_property(
            "bonolith-register-word", 0,
            "単語登録...", "",
            "ユーザー辞書に新しい単語を登録",
        );
        let manage_prop = ibus_property(
            "bonolith-manage-dict", 0,
            "辞書管理...", "",
            "ユーザー辞書の編集・削除",
        );

        let clear_prop = ibus_property(
            "bonolith-clear-learning", 0,
            "学習履歴をクリア...", "",
            "変換の学習履歴をすべて消去する",
        );

        let prop_list = ibus_prop_list(vec![register_prop, manage_prop, export_prop, import_prop, clear_prop]);

        Self::register_properties(emitter, prop_list).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Ok(())
    }

    /// Run dictionary export via zenity file dialog.
    fn run_dict_export() {
        let output = std::process::Command::new("zenity")
            .args(["--file-selection", "--save", "--confirm-overwrite",
                   "--title=Bonolith: 辞書エクスポート",
                   "--file-filter=JSON files (*.json) | *.json",
                   "--filename=bonolith_dict.json"])
            .output();

        match output {
            Ok(out) if out.status.success() => {
                let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if path.is_empty() {
                    return;
                }
                let dict = match Dictionary::with_default_store() {
                    Ok(d) => d,
                    Err(e) => {
                        warn!("Bonolith: could not open dict store for export: {}", e);
                        return;
                    }
                };
                match dict.export(std::path::Path::new(&path)) {
                    Ok(()) => {
                        info!("Bonolith: Dictionary exported to {}", path);
                        let _ = std::process::Command::new("zenity")
                            .args(["--info", "--title=Bonolith",
                                   &format!("--text=辞書をエクスポートしました: {}", path)])
                            .spawn();
                    }
                    Err(e) => {
                        warn!("Bonolith: Export failed: {}", e);
                        let _ = std::process::Command::new("zenity")
                            .args(["--error", "--title=Bonolith",
                                   &format!("--text=エクスポートに失敗しました: {}", e)])
                            .spawn();
                    }
                }
            }
            _ => { /* user cancelled or zenity not available */ }
        }
    }

    /// Run dictionary import via zenity file dialog.
    fn run_dict_import() {
        let output = std::process::Command::new("zenity")
            .args(["--file-selection",
                   "--title=Bonolith: 辞書インポート",
                   "--file-filter=JSON files (*.json) | *.json"])
            .output();

        match output {
            Ok(out) if out.status.success() => {
                let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if path.is_empty() {
                    return;
                }
                // Import into the SHARED live dictionary (not a fresh
                // Dictionary::with_default_store). The fresh instance would
                // persist the imported entries to sqlite fine, but the
                // running engine's in-memory dict would not pick them up
                // until the daemon restarts — the user reported this as
                // "count is shown but entries don't appear".
                let shared = SharedCore::global();
                let mut dict = shared.dictionary.write().unwrap_or_else(|e| e.into_inner());
                // Per-entry upsert (Devin PR #3 #2) — the prior
                // `import` + whole-table sync overwrote rows the other
                // frontend added since this process started.
                match dict.import_and_persist(std::path::Path::new(&path)) {
                    Ok(added) => {
                        info!("Bonolith: Imported {} entries from {}", added, path);
                        let _ = std::process::Command::new("zenity")
                            .args(["--info", "--title=Bonolith",
                                   &format!("--text={}件の単語をインポートしました ({})", added, path)])
                            .spawn();
                    }
                    Err(e) => {
                        warn!("Bonolith: Import failed: {}", e);
                        let _ = std::process::Command::new("zenity")
                            .args(["--error", "--title=Bonolith",
                                   &format!("--text=インポートに失敗しました: {}", e)])
                            .spawn();
                    }
                }
            }
            _ => { /* user cancelled or zenity not available */ }
        }
    }

    /// Register a new word to user dictionary via the GTK dialog.
    /// The dialog runs as /usr/share/bonolith/scripts/bonolith_word_register.py and
    /// re-activates IBus on every entry focus-in, so 単語 stays 日本語ON
    /// even after Tab. Output format: "<reading>|<surface>" on OK.
    fn run_word_register() {
        // No GDK_BACKEND override: the dialog turns the IME on via the
        // org.bonolith.Control session-bus call (engine-side, backend-agnostic),
        // so it runs as a native client and works on Wayland and X11 alike.
        let output = std::process::Command::new("/usr/bin/python3")
            .args([
                "/usr/share/bonolith/scripts/bonolith_word_register.py",
                "ibus",
            ])
            .output();

        match output {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                // Split on the FIRST '|' only so a surface carrying a
                // literal '|' (e.g. "a|b") is preserved verbatim. The
                // prior `split('|').collect()[1]` dropped everything
                // past the second pipe, silently truncating "a|b" to
                // "a" on the IBus side while Fcitx5 (which does
                // `find('|') + substr`) kept it — the two frontends
                // then diverged on the same user input. Bug [5] in
                // Fable-5 review 2026-08-31.
                let Some(sep) = text.find('|') else { return; };
                let reading = text[..sep].trim();
                let surface = text[sep + 1..].trim();
                if reading.is_empty() || surface.is_empty() {
                    let _ = std::process::Command::new("zenity")
                        .args(["--error", "--title=Bonolith",
                               "--text=よみと単語の両方を入力してください"])
                        .spawn();
                    return;
                }

                let entry = DictionaryEntry {
                    reading: reading.to_string(),
                    surface: surface.to_string(),
                    pos: PartOfSpeech::Noun,
                    frequency: 8000,
                };

                // Add to live dictionary via SharedCore. Per-row upsert
                // (Devin PR #3 #2) — the prior add_entry + whole-table
                // sync wiped rows the other frontend added since this
                // process started.
                let shared = SharedCore::global();
                {
                    let mut dict = shared.dictionary.write().unwrap_or_else(|e| e.into_inner());
                    if let Err(e) = dict.add_user_entry_and_persist(entry) {
                        warn!("Bonolith: Failed to save user dict: {}", e);
                        let _ = std::process::Command::new("zenity")
                            .args(["--error", "--title=Bonolith",
                                   &format!("--text=保存に失敗しました: {}", e)])
                            .spawn();
                        return;
                    }
                }

                info!("Bonolith: Registered word: {} → {}", reading, surface);
                let _ = std::process::Command::new("zenity")
                    .args(["--info", "--title=Bonolith",
                           &format!("--text=登録しました: {} → {}", reading, surface)])
                    .spawn();
            }
            _ => { /* user cancelled or zenity not available */ }
        }
    }

    /// Clear all user learning history after zenity confirmation.
    fn run_clear_learning(shared: std::sync::Arc<SharedCore>) {
        let confirmed = std::process::Command::new("zenity")
            .args([
                "--question",
                "--title=Bonolith 学習履歴クリア",
                "--text=変換の学習履歴をすべて消去します。\nこの操作は元に戻せません。よろしいですか？",
                "--ok-label=クリア",
                "--cancel-label=キャンセル",
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if !confirmed {
            return;
        }

        let mut user_scorer = shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
        match user_scorer.clear_scores() {
            Ok(n) => {
                let _ = std::process::Command::new("zenity")
                    .args([
                        "--info",
                        "--title=Bonolith",
                        &format!("--text=学習履歴を消去しました（{} 件）。\n次回起動時から反映されます。", n),
                    ])
                    .status();
            }
            Err(e) => {
                let _ = std::process::Command::new("zenity")
                    .args([
                        "--error",
                        "--title=Bonolith",
                        &format!("--text=エラー: {}", e),
                    ])
                    .status();
            }
        }
    }

    /// Manage user dictionary: list entries, then choose edit or delete.
    fn run_manage_dict() {
        let shared = SharedCore::global();
        let user_entries: Vec<DictionaryEntry> = {
            let dict = shared.dictionary.read().unwrap_or_else(|e| e.into_inner());
            dict.user_entries().to_vec()
        };

        if user_entries.is_empty() {
            let _ = std::process::Command::new("zenity")
                .args(["--info", "--title=Bonolith",
                       "--text=ユーザー辞書にエントリがありません"])
                .spawn();
            return;
        }

        const MAX_DISPLAY: usize = 500;
        if user_entries.len() > MAX_DISPLAY {
            let _ = std::process::Command::new("zenity")
                .args(["--warning", "--title=Bonolith",
                       &format!("--text=ユーザー辞書のエントリが{}件を超えています ({} 件)。\n先頭 {} 件のみ表示します。\nエクスポートして内容を確認してください。",
                           MAX_DISPLAY, user_entries.len(), MAX_DISPLAY)])
                .output();
        }

        // Step 1: Show list and let user select an entry
        let display_entries = &user_entries[..user_entries.len().min(MAX_DISPLAY)];
        let mut args = vec![
            "--list".to_string(),
            "--title=Bonolith: 辞書管理".to_string(),
            "--text=エントリを選択してOKを押してください".to_string(),
            "--column=#".to_string(),
            "--column=よみ".to_string(),
            "--column=単語".to_string(),
            "--print-column=1".to_string(),
            "--width=500".to_string(),
            "--height=400".to_string(),
        ];
        for (i, entry) in display_entries.iter().enumerate() {
            args.push(format!("{}", i));
            args.push(entry.reading.clone());
            args.push(entry.surface.clone());
        }

        let output = std::process::Command::new("zenity")
            .args(&args)
            .output();

        let idx: usize = match output {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                // zenity --list emits the selected value twice, pipe-joined
                // ("3|3"), when the user double-clicks a row — the most
                // natural gesture for picking from a list. Taking the
                // first pipe-delimited field keeps the single-click
                // ("3") case working while surviving the double-click
                // one (bug [7] in Fable-5 review 2026-08-31).
                let first = stdout.split('|').next().unwrap_or("");
                match first.parse() {
                    Ok(i) if i < user_entries.len() => i,
                    _ => return,
                }
            }
            _ => return,
        };

        // Step 2: Ask what to do with the selected entry
        let selected = &user_entries[idx];
        let action = std::process::Command::new("zenity")
            .args([
                "--list", "--radiolist",
                "--title=Bonolith: 操作を選択",
                &format!("--text=選択中: {} → {}", selected.reading, selected.surface),
                "--column=選択", "--column=操作",
                "TRUE", "編集",
                "FALSE", "削除",
            ])
            .output();

        // Capture the selected entry's identity so the apply path below
        // can re-fetch the LIVE user entries (a concurrent register /
        // delete may have shifted indices between showing the list and
        // confirming here — bug [17]).
        let sel_reading = user_entries[idx].reading.clone();
        let sel_surface = user_entries[idx].surface.clone();
        drop(user_entries);

        match action {
            Ok(out) if out.status.success() => {
                let choice = String::from_utf8_lossy(&out.stdout).trim().to_string();
                match choice.as_str() {
                    "編集" => Self::edit_user_entry(sel_reading, sel_surface),
                    "削除" => Self::delete_user_entry(sel_reading, sel_surface),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Delete a user dictionary entry identified by (reading, surface).
    /// Re-fetches the live user_entries under the write lock so any
    /// concurrent addition survives (bug [17]).
    fn delete_user_entry(reading: String, surface: String) {
        let confirm = std::process::Command::new("zenity")
            .args([
                "--question", "--title=Bonolith: 削除の確認",
                &format!("--text=「{}」→「{}」を削除しますか？", reading, surface),
            ])
            .status();

        if !matches!(confirm, Ok(s) if s.success()) {
            return;
        }

        let shared = SharedCore::global();
        let mut dict = shared.dictionary.write().unwrap_or_else(|e| e.into_inner());
        // Per-row DELETE (Devin PR #3 #2).
        match dict.remove_user_entry_and_persist(&reading, &surface) {
            Ok(true) => Self::notify_dict_updated(),
            Ok(false) => {} // row already gone (raced)
            Err(e) => {
                warn!("Bonolith: Failed to delete user entry: {}", e);
                let _ = std::process::Command::new("zenity")
                    .args(["--error", "--title=Bonolith",
                           &format!("--text=削除に失敗しました: {}", e)])
                    .spawn();
            }
        }
    }

    /// Edit a user dictionary entry identified by (old_reading, old_surface).
    /// Reuses the GTK register dialog in edit mode, prefilled with the
    /// current reading and surface so 単語 stays 日本語ON after Tab.
    /// Applies the change to LIVE entries by identity — see [17].
    fn edit_user_entry(old_reading: String, old_surface: String) {
        let output = std::process::Command::new("/usr/bin/python3")
            .args([
                "/usr/share/bonolith/scripts/bonolith_word_register.py",
                "ibus",
                "--mode", "edit",
                "--reading", &old_reading,
                "--surface", &old_surface,
            ])
            .output();

        let out = match output {
            Ok(o) if o.status.success() => o,
            _ => return,
        };
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        // First '|' only — see the register-path comment above for
        // rationale (bug [5]).
        let Some(sep) = text.find('|') else { return; };
        let new_reading = text[..sep].trim();
        let new_surface = text[sep + 1..].trim();
        if new_reading.is_empty() || new_surface.is_empty() {
            return;
        }
        if new_reading == old_reading && new_surface == old_surface {
            return; // no change
        }

        let shared = SharedCore::global();
        let mut dict = shared.dictionary.write().unwrap_or_else(|e| e.into_inner());
        // Per-row DELETE-then-UPSERT (Devin PR #3 #2). POS/frequency
        // preserved from the old row.
        match dict.update_user_entry_and_persist(
            &old_reading, &old_surface, new_reading, new_surface,
        ) {
            Ok(true) => Self::notify_dict_updated(),
            Ok(false) => {} // concurrently deleted
            Err(e) => {
                warn!("Bonolith: Failed to update user entry: {}", e);
                let _ = std::process::Command::new("zenity")
                    .args(["--error", "--title=Bonolith",
                           &format!("--text=保存に失敗しました: {}", e)])
                    .spawn();
            }
        }
    }

    /// Notify the user that the dictionary was successfully updated —
    /// used by delete/edit paths that previously routed through
    /// `apply_user_entries_locked` (removed with Devin PR #3 #2 in
    /// favour of row-level persist methods on Dictionary).
    fn notify_dict_updated() {
        let _ = std::process::Command::new("zenity")
            .args(["--info", "--title=Bonolith",
                   "--text=辞書を更新しました"])
            .spawn();
    }

}

#[cfg(test)]
mod tests {
    use super::unshift_letter;

    /// Bug [28] regression: pending_release records and lookups both go
    /// through `unshift_letter`, so a Shift+A press (delivered as 0x41)
    /// still matches its release when Shift is released first and the
    /// release keysym arrives as 0x61.
    #[test]
    fn unshift_letter_folds_uppercase_ascii() {
        // A..Z → a..z
        for (upper, lower) in [(0x41, 0x61), (0x4D, 0x6D), (0x5A, 0x7A)] {
            assert_eq!(unshift_letter(upper), lower);
            // Idempotent on lowercase.
            assert_eq!(unshift_letter(lower), lower);
        }
        // Non-letters (digits, punctuation, space, F-keys, IBus keysyms)
        // pass through unchanged — they're layout-dependent under Shift
        // and users must configure the shifted keyval explicitly.
        for k in [0x20u32, 0x30, 0x39, 0x40, 0x5B, 0x60, 0x7B, 0xFF0D, 0xFF1B] {
            assert_eq!(unshift_letter(k), k, "keyval 0x{k:04X} should not be folded");
        }
    }
}
