/// IBus Engine D-Bus interface implementation.
///
/// Implements org.freedesktop.IBus.Engine via zbus #[interface].
/// Bridges IBus key events to Bonolith's ConversionEngine and sends
/// preedit/commit/candidates back via D-Bus signals.
use std::sync::{Arc, Mutex};

use log::{debug, info, warn};
use zbus::object_server::SignalEmitter;
use zbus::{interface, zvariant, Connection};

use bonolith::core::dictionary::{Dictionary, DictionaryEntry, PartOfSpeech};
use bonolith::engine::{ConversionEngine, ConversionState, SharedCore};

use super::config::{CompiledToggleKey, BonolithConfig};
use super::factory::ForceEnable;
use super::keymap::*;

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
        }
    }

    /// If the word-register dialog has opened a force-on window, flip this
    /// engine 日本語ON. Idempotent; safe to call from focus/enable/key paths.
    /// Returns true if it (or a prior call) left the engine enabled.
    fn apply_force_enable(&self) -> bool {
        let active = matches!(
            *self.force.lock().unwrap(),
            Some(deadline) if std::time::Instant::now() < deadline
        );
        if active {
            let mut enabled = self.enabled.lock().unwrap();
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
        if is_release(state) {
            let enabled = *self.enabled.lock().unwrap();
            if enabled {
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
                *self.enabled.lock().unwrap() = true;
                info!("Bonolith: Henkan → enabled=true (absolute ON)");
                return Ok(true);
            }
            if keyval == IBUS_KEY_MUHENKAN {
                let _ = self.cancel_input(&emitter).await;
                *self.enabled.lock().unwrap() = false;
                info!("Bonolith: Muhenkan → enabled=false (absolute OFF)");
                return Ok(true);
            }
        }

        // Toggle key check — must come before modifier pass-through
        if self.is_toggle_key(keyval, state) {
            let was_enabled = *self.enabled.lock().unwrap();
            if was_enabled {
                let _ = self.cancel_input(&emitter).await;
                *self.enabled.lock().unwrap() = false;
            } else {
                *self.enabled.lock().unwrap() = true;
            }
            let now = *self.enabled.lock().unwrap();
            info!("Bonolith: Toggle → enabled={}", now);
            return Ok(true);
        }

        // Pass through modifier combos (Ctrl+C, Alt+Tab, etc.)
        if has_modifier(state) {
            return Ok(false);
        }

        let enabled = *self.enabled.lock().unwrap();
        if !enabled {
            return Ok(false);
        }

        let converting = *self.converting.lock().unwrap();

        // Tab while a preedit/conversion is active → commit current text and
        // consume the key (Ok(true)). Focus does NOT move: this matches the
        // standard Japanese IME convention (Mozc / Google IME / ATOK), where
        // Tab during composition is an IME key, not a focus-navigation key.
        // When nothing is composing we fall through to Ok(false) so Tab works
        // normally. Both engines consume Tab here, so IBus and Fcitx5 stay
        // consistent instead of diverging on framework preedit defaults.
        if keyval == IBUS_KEY_TAB {
            let text = {
                let mut engine = self.engine.lock().unwrap();
                if converting {
                    engine.commit_conversion()
                } else {
                    let p = engine.preedit().to_string();
                    if p.is_empty() {
                        None
                    } else {
                        engine.commit(&p);
                        Some(p)
                    }
                }
            };
            if let Some(text) = text {
                // Best-effort emit. Engine state has already been mutated
                // (commit_conversion / engine.commit take internal state),
                // so a `?` here used to leave `converting=true` latched when
                // a hide-signal failed, jamming further input. Log and
                // finish the state cleanup instead — a broken D-Bus means
                // the daemon is already gone and text loss is unavoidable,
                // but the residual `converting=true` was.
                if let Err(e) = Self::commit_text(&emitter, ibus_text(&text)).await {
                    warn!("Bonolith: commit_text emission failed: {}", e);
                }
                let _ = Self::hide_preedit_text(&emitter).await;
                let _ = Self::hide_lookup_table(&emitter).await;
                let _ = Self::hide_auxiliary_text(&emitter).await;
                *self.converting.lock().unwrap() = false;
                return Ok(true);
            }
            return Ok(false);
        }

        // F6-F10 → kana form selection.
        //
        // Bonolith always owns F6-F10 while the engine is enabled (we get
        // here only after the `!enabled` gate above). start_kana_conversion
        // returns Ok(false) for an empty preedit — propagating that let
        // IBus pass the raw keysym to the app, where a terminal expanded
        // F7 into `\e[18~` and printed the trailing tilde. Discard the
        // "did we actually enter conversion mode" bool and always report
        // the key as consumed; the empty-preedit case is a no-op instead
        // of a passthrough.
        if keyval == IBUS_KEY_F6 {
            if converting {
                let conv = {
                    let mut engine = self.engine.lock().unwrap();
                    engine.convert_focused_to_hiragana().cloned()
                };
                if let Some(conv) = conv {
                    self.show_conversion_state(&emitter, &conv).await?;
                }
            } else {
                let _ = self.start_kana_conversion(&emitter, 0).await?;
            }
            return Ok(true);
        }

        // F7 → full-width katakana, F8 → half-width katakana, F9 → full-width romaji, F10 → half-width romaji
        if keyval == IBUS_KEY_F7 || keyval == IBUS_KEY_F8 || keyval == IBUS_KEY_F9 || keyval == IBUS_KEY_F10 {
            info!("Bonolith: F-key 0x{:04X} converting={}", keyval, converting);
            let form = match keyval {
                IBUS_KEY_F8 => 2,
                IBUS_KEY_F9 => 4,
                IBUS_KEY_F10 => 3,
                _ => 1,
            };
            if converting {
                match keyval {
                    IBUS_KEY_F9 => return self.convert_focused_to_fullwidth_romaji(&emitter).await,
                    IBUS_KEY_F10 => return self.convert_focused_to_romaji(&emitter).await,
                    _ => {
                        let half = keyval == IBUS_KEY_F8;
                        return self.convert_focused_to_kana(&emitter, half).await;
                    }
                }
            } else {
                let _ = self.start_kana_conversion(&emitter, form).await?;
                return Ok(true);
            }
        }

        // Handle keys during conversion mode
        if converting {
            let result = self.handle_conversion_key(&emitter, keyval, state).await?;
            if result {
                return Ok(true);
            }
            // Non-printable keys (modifiers, function keys, etc.) — consume without committing
            if keyval_to_char(keyval).is_none() {
                return Ok(true);
            }
            // Printable key not handled by conversion — commit conversion first,
            // then fall through to process the key as new input
            let text = {
                let mut engine = self.engine.lock().unwrap();
                engine.commit_conversion()
            };
            if let Some(text) = text {
                // Same best-effort emission pattern — see the Tab arm above.
                if let Err(e) = Self::commit_text(&emitter, ibus_text(&text)).await {
                    warn!("Bonolith: commit_text emission failed: {}", e);
                }
                let _ = Self::hide_preedit_text(&emitter).await;
                let _ = Self::hide_lookup_table(&emitter).await;
                let _ = Self::hide_auxiliary_text(&emitter).await;
                *self.converting.lock().unwrap() = false;
            }
            // Fall through to process the key as new input
        }

        // Space → trigger conversion
        if keyval == IBUS_KEY_SPACE {
            return self.start_conversion(&emitter).await;
        }

        // Enter → commit current preedit as-is (hiragana)
        if keyval == IBUS_KEY_RETURN {
            return self.commit_preedit(&emitter).await;
        }

        // Escape → cancel input
        if keyval == IBUS_KEY_ESCAPE {
            return self.cancel_input(&emitter).await;
        }

        // Backspace → delete last character from buffer
        if keyval == IBUS_KEY_BACKSPACE {
            return self.handle_backspace(&emitter).await;
        }

        // Arrow keys / navigation keys → consume if preedit is active to prevent
        // interference (e.g. Shift+Arrow inserting stray characters), pass through otherwise
        if matches!(keyval, IBUS_KEY_LEFT | IBUS_KEY_RIGHT | IBUS_KEY_UP | IBUS_KEY_DOWN
                          | IBUS_KEY_PAGE_UP | IBUS_KEY_PAGE_DOWN) {
            let has_preedit = !self.engine.lock().unwrap().preedit().is_empty();
            return Ok(has_preedit);
        }

        // Symbol/punctuation/digit → full-width equivalent in preedit (F8 for half-width)
        if let Some(ch) = keyval_to_char(keyval) {
            let fw = match ch {
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
            };
            if let Some(sym) = fw {
                let preedit = {
                    let mut engine = self.engine.lock().unwrap();
                    engine.append_raw(sym);
                    engine.preedit().to_string()
                };
                self.send_preedit(&emitter, &preedit).await?;
                return Ok(true);
            }
        }

        // Printable ASCII → feed to romaji converter
        if let Some(ch) = keyval_to_char(keyval) {
            if ch.is_ascii_alphabetic() || ch == '-' || ch == '\'' {
                let preedit = {
                    let mut engine = self.engine.lock().unwrap();
                    engine.process_key(ch.to_ascii_lowercase());
                    engine.preedit().to_string()
                };

                self.send_preedit(&emitter, &preedit).await?;
                return Ok(true);
            }
        }

        // Consume unhandled keys while preedit is active to prevent stray characters
        let has_preedit = !self.engine.lock().unwrap().preedit().is_empty();
        if has_preedit {
            return Ok(true);
        }

        // Unhandled key
        Ok(false)
    }

    /// Called when the engine gains focus.
    async fn focus_in(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) {
        // Adopt an open force-on window so the dialog's field shows 日本語ON the
        // moment it gains focus (no keystroke needed).
        self.apply_force_enable();
        info!("Bonolith: FocusIn (enabled={})", *self.enabled.lock().unwrap());
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
                let shared = self.engine.lock().unwrap().shared_core();
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
        // With mode=0 (PREEDIT_CLEAR) IBus drops the preedit on focus loss, so
        // we must commit manually to preserve the "click-away keeps typed text"
        // contract (Mozc/Google IME parity).
        let commit_str = {
            let mut engine = self.engine.lock().unwrap();
            let converting = *self.converting.lock().unwrap();
            let preedit = engine.preedit().to_string();
            if converting {
                engine.commit_conversion().unwrap_or_default()
            } else if !preedit.is_empty() {
                engine.commit(&preedit);
                preedit
            } else {
                String::new()
            }
        };

        if !commit_str.is_empty() {
            if let Err(e) = Self::commit_text(&emitter, ibus_text(&commit_str)).await {
                warn!("Bonolith: focus_out commit_text failed: {e}");
            }
        }
        // Hide any stale UI, clear internal state.
        let _ = Self::hide_preedit_text(&emitter).await;
        let _ = Self::hide_lookup_table(&emitter).await;
        let _ = Self::hide_auxiliary_text(&emitter).await;
        {
            let mut engine = self.engine.lock().unwrap();
            engine.reset();
            engine.clear_conversion();
        }
        *self.converting.lock().unwrap() = false;
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
        info!("Bonolith: Enable (enabled={})", *self.enabled.lock().unwrap());
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
        *self.enabled.lock().unwrap() = false;
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
    fn is_toggle_key(&self, keyval: u32, state: u32) -> bool {
        let relevant_mask = IBUS_CONTROL_MASK | IBUS_MOD1_MASK | IBUS_SHIFT_MASK;
        let active_modifiers = state & relevant_mask;
        self.toggle_keys
            .iter()
            .any(|tk| keyval == tk.keyval && active_modifiers == tk.modifier_mask)
    }

    /// Start a kana-form conversion (F6/F7/F8 outside conversion mode).
    /// form: 0 = hiragana, 1 = katakana, 2 = half-width katakana
    async fn start_kana_conversion(
        &self,
        emitter: &SignalEmitter<'_>,
        form: usize,
    ) -> zbus::fdo::Result<bool> {
        let state = {
            let mut engine = self.engine.lock().unwrap();
            engine.start_kana_conversion(form).cloned()
        };
        let Some(state) = state else {
            return Ok(false);
        };
        self.show_conversion_state(emitter, &state).await?;
        *self.converting.lock().unwrap() = true;
        Ok(true)
    }

    async fn start_conversion(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let state = {
            let mut engine = self.engine.lock().unwrap();
            engine.start_conversion().cloned()
        };

        let Some(state) = state else {
            return Ok(false);
        };

        self.show_conversion_state(emitter, &state).await?;
        *self.converting.lock().unwrap() = true;
        // Surface LLM context reranking once the background pass finishes.
        self.spawn_rerank_refresh(emitter);

        Ok(true)
    }

    async fn handle_conversion_key(
        &self,
        emitter: &SignalEmitter<'_>,
        keyval: u32,
        state: u32,
    ) -> zbus::fdo::Result<bool> {
        let has_shift = state & IBUS_SHIFT_MASK != 0;

        match keyval {
            // Space / Down → next candidate for focused segment
            IBUS_KEY_SPACE | IBUS_KEY_DOWN => {
                let conv = {
                    let mut engine = self.engine.lock().unwrap();
                    engine.cycle_candidate(1).cloned()
                };
                if let Some(conv) = conv {
                    self.show_conversion_state(emitter, &conv).await?;
                }
                Ok(true)
            }
            // Up → previous candidate for focused segment
            IBUS_KEY_UP => {
                let conv = {
                    let mut engine = self.engine.lock().unwrap();
                    engine.cycle_candidate(-1).cloned()
                };
                if let Some(conv) = conv {
                    self.show_conversion_state(emitter, &conv).await?;
                }
                Ok(true)
            }
            // Right → move focus to next segment (or Shift+Right → extend segment)
            IBUS_KEY_RIGHT => {
                let conv = {
                    let mut engine = self.engine.lock().unwrap();
                    if has_shift {
                        engine.resize_segment(1).cloned()
                    } else {
                        engine.move_focus(1).cloned()
                    }
                };
                if let Some(conv) = conv {
                    self.show_conversion_state(emitter, &conv).await?;
                }
                // A resize re-triggers the background rerank; refresh when ready.
                if has_shift {
                    self.spawn_rerank_refresh(emitter);
                }
                Ok(true)
            }
            // Left → move focus to previous segment (or Shift+Left → shrink segment)
            IBUS_KEY_LEFT => {
                let conv = {
                    let mut engine = self.engine.lock().unwrap();
                    if has_shift {
                        engine.resize_segment(-1).cloned()
                    } else {
                        engine.move_focus(-1).cloned()
                    }
                };
                if let Some(conv) = conv {
                    self.show_conversion_state(emitter, &conv).await?;
                }
                // A resize re-triggers the background rerank; refresh when ready.
                if has_shift {
                    self.spawn_rerank_refresh(emitter);
                }
                Ok(true)
            }
            // Enter → commit composed text (with learning)
            IBUS_KEY_RETURN => {
                let text = {
                    let mut engine = self.engine.lock().unwrap();
                    engine.commit_conversion()
                };
                if let Some(text) = text {
                    // Same best-effort emission pattern — see the Tab arm.
                    if let Err(e) = Self::commit_text(emitter, ibus_text(&text)).await {
                        warn!("Bonolith: commit_text emission failed: {}", e);
                    }
                    let _ = Self::hide_preedit_text(emitter).await;
                    let _ = Self::hide_lookup_table(emitter).await;
                    let _ = Self::hide_auxiliary_text(emitter).await;
                    *self.converting.lock().unwrap() = false;
                }
                Ok(true)
            }
            // Escape → cancel conversion, return to preedit
            IBUS_KEY_ESCAPE => {
                self.cancel_conversion(emitter).await?;
                Ok(true)
            }
            _ => Ok(false),
        }
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
        // mode=0 (PREEDIT_CLEAR): drop preedit when hidden; focus_out commits
        // manually so the click-away-keeps-text contract still holds.
        let cursor = text.chars().count() as u32;
        Self::update_preedit_text(
            emitter,
            ibus_text_with_segments(&text, &ranges, focus),
            cursor,
            true,
            0,
        ).await.map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;

        // Lookup table for the focused segment's candidates
        let seg = &state.segments[focus];
        Self::update_lookup_table(
            emitter,
            ibus_lookup_table(&seg.candidates, seg.selected),
            true,
        ).await.map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;

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
                if engine.lock().unwrap().has_llm_rerank_result() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;
            }

            // Apply and snapshot the refreshed state under one lock. Skip if the
            // user already left conversion mode (committed / cancelled).
            let refreshed = {
                if !*converting.lock().unwrap() {
                    return;
                }
                let mut engine = engine.lock().unwrap();
                if !engine.apply_llm_rerank() {
                    return;
                }
                engine.conversion_state().cloned()
            };

            if let Some(state) = refreshed {
                // Re-check converting after the await-free section; harmless race
                // at worst repaints a still-valid panel.
                if *converting.lock().unwrap() {
                    if let Err(e) = Self::emit_conversion_state(&emitter, &state).await {
                        debug!("Bonolith: rerank refresh emit failed: {e}");
                    }
                }
            }
        });
    }

    /// Convert the focused segment to katakana (F7/F8 during conversion mode).
    async fn convert_focused_to_kana(
        &self,
        emitter: &SignalEmitter<'_>,
        half: bool,
    ) -> zbus::fdo::Result<bool> {
        let conv = {
            let mut engine = self.engine.lock().unwrap();
            if half {
                engine.convert_focused_to_halfwidth_katakana().cloned()
            } else {
                engine.convert_focused_to_katakana().cloned()
            }
        };
        if let Some(conv) = conv {
            self.show_conversion_state(emitter, &conv).await?;
        }
        Ok(true)
    }

    /// Convert the focused segment to romaji (F9 during conversion mode).
    async fn convert_focused_to_romaji(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let conv = {
            let mut engine = self.engine.lock().unwrap();
            engine.convert_focused_to_romaji().cloned()
        };
        if let Some(conv) = conv {
            self.show_conversion_state(emitter, &conv).await?;
        }
        Ok(true)
    }

    /// Convert the focused segment to full-width romaji (F10 during conversion mode).
    async fn convert_focused_to_fullwidth_romaji(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let conv = {
            let mut engine = self.engine.lock().unwrap();
            engine.convert_focused_to_fullwidth_romaji().cloned()
        };
        if let Some(conv) = conv {
            self.show_conversion_state(emitter, &conv).await?;
        }
        Ok(true)
    }

    async fn commit_preedit(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let preedit = {
            let mut engine = self.engine.lock().unwrap();
            let p = engine.preedit().to_string();
            if p.is_empty() {
                return Ok(false);
            }
            engine.commit(&p);
            p
        };

        Self::commit_text(emitter, ibus_text(&preedit)).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_preedit_text(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_lookup_table(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_auxiliary_text(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;

        Ok(true)
    }

    async fn cancel_input(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        {
            let mut engine = self.engine.lock().unwrap();
            engine.reset();
            engine.clear_conversion();
        }
        *self.converting.lock().unwrap() = false;
        Self::hide_preedit_text(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_lookup_table(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_auxiliary_text(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Ok(true)
    }

    async fn cancel_conversion(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        {
            let mut engine = self.engine.lock().unwrap();
            engine.clear_conversion();
        }
        *self.converting.lock().unwrap() = false;
        let preedit = {
            let engine = self.engine.lock().unwrap();
            engine.preedit().to_string()
        };
        self.send_preedit(emitter, &preedit).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_lookup_table(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Self::hide_auxiliary_text(emitter).await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Ok(())
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
        // mode=0 (PREEDIT_CLEAR): drop preedit when hidden; focus_out commits
        // manually to preserve the click-away-keeps-text contract.
        Self::update_preedit_text(emitter, preedit_text, cursor, visible, 0).await
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
                let mut dict = match Dictionary::with_default_store() {
                    Ok(d) => d,
                    Err(e) => {
                        warn!("Bonolith: could not open dict store for import: {}", e);
                        return;
                    }
                };
                match dict.import(std::path::Path::new(&path)) {
                    Ok(added) => {
                        if let Err(e) = dict.sync_user_entries_to_store() {
                            warn!("Bonolith: Failed to save after import: {}", e);
                            let _ = std::process::Command::new("zenity")
                                .args(["--error", "--title=Bonolith",
                                       &format!("--text=保存に失敗しました: {}", e)])
                                .spawn();
                            return;
                        }
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
                let parts: Vec<&str> = text.split('|').collect();
                if parts.len() < 2 {
                    return;
                }
                let reading = parts[0].trim();
                let surface = parts[1].trim();
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

                // Add to live dictionary via SharedCore
                let shared = SharedCore::global();
                {
                    let mut dict = shared.dictionary.write().unwrap();
                    dict.add_entry(entry);
                    if let Err(e) = dict.sync_user_entries_to_store() {
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

        let mut user_scorer = shared.user_scorer.lock().unwrap();
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
            let dict = shared.dictionary.read().unwrap();
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
                match stdout.parse() {
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

        match action {
            Ok(out) if out.status.success() => {
                let choice = String::from_utf8_lossy(&out.stdout).trim().to_string();
                match choice.as_str() {
                    "編集" => Self::edit_user_entry(user_entries, idx),
                    "削除" => Self::delete_user_entry(user_entries, idx),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Delete a user dictionary entry by index.
    fn delete_user_entry(mut entries: Vec<DictionaryEntry>, idx: usize) {
        let entry = &entries[idx];
        let confirm = std::process::Command::new("zenity")
            .args([
                "--question", "--title=Bonolith: 削除の確認",
                &format!("--text=「{}」→「{}」を削除しますか？", entry.reading, entry.surface),
            ])
            .status();

        match confirm {
            Ok(s) if s.success() => {
                entries.remove(idx);
                Self::save_and_apply_user_entries(entries);
            }
            _ => {}
        }
    }

    /// Edit a user dictionary entry by index.
    /// Reuses the GTK register dialog in edit mode, prefilled with the
    /// current reading and surface so 単語 stays 日本語ON after Tab.
    fn edit_user_entry(mut entries: Vec<DictionaryEntry>, idx: usize) {
        let old_reading = entries[idx].reading.clone();
        let old_surface = entries[idx].surface.clone();

        let output = std::process::Command::new("/usr/bin/python3")
            .args([
                "/usr/share/bonolith/scripts/bonolith_word_register.py",
                "ibus",
                "--mode", "edit",
                "--reading", &old_reading,
                "--surface", &old_surface,
            ])
            .output();

        match output {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let parts: Vec<&str> = text.split('|').collect();
                if parts.len() < 2 {
                    return;
                }
                let new_reading = parts[0].trim();
                let new_surface = parts[1].trim();
                if new_reading.is_empty() || new_surface.is_empty() {
                    return;
                }
                if new_reading == old_reading && new_surface == old_surface {
                    return; // no change
                }
                entries[idx].reading = new_reading.to_string();
                entries[idx].surface = new_surface.to_string();
                Self::save_and_apply_user_entries(entries);
            }
            _ => {}
        }
    }

    /// Save modified user entries to file and apply to live dictionary.
    fn save_and_apply_user_entries(entries: Vec<DictionaryEntry>) {
        let shared = SharedCore::global();
        let mut dict = shared.dictionary.write().unwrap();
        dict.replace_user_entries(entries);
        if let Err(e) = dict.sync_user_entries_to_store() {
            warn!("Bonolith: Failed to save user dict: {}", e);
            let _ = std::process::Command::new("zenity")
                .args(["--error", "--title=Bonolith",
                       &format!("--text=保存に失敗しました: {}", e)])
                .spawn();
            return;
        }
        let _ = std::process::Command::new("zenity")
            .args(["--info", "--title=Bonolith",
                   "--text=辞書を更新しました"])
            .spawn();
    }

    async fn handle_backspace(
        &self,
        emitter: &SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        let new_preedit = {
            let mut engine = self.engine.lock().unwrap();
            if !engine.delete_last() {
                return Ok(false);
            }
            engine.preedit().to_string()
        };

        if new_preedit.is_empty() {
            Self::hide_preedit_text(emitter).await
                .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
            Self::hide_lookup_table(emitter).await
                .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
            Self::hide_auxiliary_text(emitter).await
                .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        } else {
            self.send_preedit(emitter, &new_preedit).await?;
        }
        Ok(true)
    }
}
