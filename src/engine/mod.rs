/// Bonolith Conversion Engine
///
/// Orchestrates the 3-stage conversion pipeline:
/// 1. Dictionary lookup + segmentation (fast, < 1ms)
/// 2. Grammar scoring (fast, < 1ms)
/// 3. LLM reranking (20-40ms, background pre-computation)
///
/// Flow: keystroke → romaji → kana → dictionary segment → grammar score
///       → LLM rerank → candidate list → user selects → commit

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread;

use crate::core::{
    dictionary::{connection_cost, Dictionary, DictionaryEntry, PartOfSpeech, Segment},
    grammar::{GrammarEngine, GrammarToken},
    llm::LlmEngine,
    romaji::RomajiConverter,
    store::DictStore,
    user_scorer::UserScorer,
};

/// Frontend-agnostic key dispatcher shared by the Fcitx5 FFI and IBus
/// engine. See the module docs for the KeyEvent → KeyOutcome contract.
pub mod dispatch;

/// Per-segment state during conversion
#[derive(Debug, Clone)]
pub struct SegmentState {
    /// Hiragana reading for this segment
    pub reading: String,
    /// Start position in kana (char offset)
    pub start: usize,
    /// Candidate surfaces (sorted by score)
    pub candidates: Vec<String>,
    /// Currently selected candidate index
    pub selected: usize,
    /// Whether the user explicitly changed the candidate for this segment
    pub user_selected: bool,
}

/// Active conversion state (set after Space is pressed)
#[derive(Debug, Clone)]
pub struct ConversionState {
    /// The original kana string
    pub kana: String,
    /// Per-segment conversion state
    pub segments: Vec<SegmentState>,
    /// Currently focused segment index
    pub focus: usize,
}

/// The five kana display forms Bonolith cycles through with F6-F10.
/// Used by [`ConversionEngine::convert_focused_to`] so the shared
/// "swap the focused segment's surface to X" code path has one place
/// to live instead of five near-identical wrappers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KanaForm {
    Hiragana,          // F6  — reading as-is
    Katakana,          // F7  — full-width カタカナ
    HalfwidthKatakana, // F8  — ｶﾀｶﾅ
    FullwidthRomaji,   // F9  — ａｂｃ (Mozc / Google IME / ATOK convention)
    Romaji,            // F10 — abc (half-width ASCII)
}

impl KanaForm {
    /// Transform the given hiragana reading into this form. Hiragana is
    /// the identity mapping; the others delegate to `core::romaji`.
    pub fn apply(self, reading: &str) -> String {
        match self {
            KanaForm::Hiragana => reading.to_string(),
            KanaForm::Katakana => crate::core::romaji::hiragana_to_katakana(reading),
            KanaForm::HalfwidthKatakana => {
                crate::core::romaji::hiragana_to_halfwidth_katakana(reading)
            }
            KanaForm::Romaji => crate::core::romaji::hiragana_to_romaji(reading),
            KanaForm::FullwidthRomaji => {
                crate::core::romaji::hiragana_to_fullwidth_romaji(reading)
            }
        }
    }
}

impl ConversionState {
    /// Get the composed text from all segments' selected candidates.
    pub fn composed_text(&self) -> String {
        self.segments
            .iter()
            .map(|seg| seg.candidates[seg.selected].as_str())
            .collect()
    }

    /// Get segment boundary info: Vec of (start_char, end_char) in composed text.
    pub fn segment_char_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut pos = 0;
        for seg in &self.segments {
            let text = &seg.candidates[seg.selected];
            let len = text.chars().count();
            ranges.push((pos, pos + len));
            pos += len;
        }
        ranges
    }
}

/// Opt-in conversion logger for evaluation dataset curation.
///
/// When `BONOLITH_LOG_CONVERSIONS=1` is set in the environment, every committed
/// conversion is appended as a JSONL record to
/// `$HOME/.local/share/bonolith/conversions.jsonl`. Records contain the reading,
/// composed output, per-segment alternatives, and which segments the user
/// explicitly re-selected (a strong signal that the system's first choice was
/// wrong). See `scripts/curate_cases.py` for the curation workflow.
///
/// Privacy: this logs raw committed text. Enable only for short collection
/// sessions and delete the log file when done. Errors are silently swallowed
/// so the IME never breaks because of logging.
fn log_conversion_for_eval(state: &ConversionState) {
    if std::env::var("BONOLITH_LOG_CONVERSIONS").ok().as_deref() != Some("1") {
        return;
    }
    let Some(home) = std::env::var_os("HOME") else { return };
    let dir = std::path::Path::new(&home).join(".local/share/bonolith");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join("conversions.jsonl");

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let segments: Vec<serde_json::Value> = state
        .segments
        .iter()
        .map(|seg| {
            serde_json::json!({
                "reading": seg.reading,
                "selected": seg.candidates.get(seg.selected).cloned().unwrap_or_default(),
                "user_selected": seg.user_selected,
                "alternatives": seg.candidates.iter().take(5).cloned().collect::<Vec<_>>(),
            })
        })
        .collect();

    let record = serde_json::json!({
        "ts": ts,
        "kana": state.kana,
        "composed": state.composed_text(),
        "segments": segments,
        "version": env!("CARGO_PKG_VERSION"),
    });

    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let _ = writeln!(file, "{}", record);
}

/// Shared heavy resources (dictionary, grammar, LLM, user scorer).
/// Initialized once per process and shared across all InputContexts.
pub struct SharedCore {
    pub dictionary: RwLock<Dictionary>,
    pub grammar: GrammarEngine,
    pub llm: Mutex<LlmEngine>,
    pub user_scorer: Mutex<UserScorer>,
    /// Persistent SQLite store for user entries and scores. None when
    /// the database could not be opened (in which case the engine still
    /// runs but mutations are not persisted).
    pub store: Option<Arc<DictStore>>,
}

/// Global shared core, initialized on first use.
static SHARED_CORE: OnceLock<Arc<SharedCore>> = OnceLock::new();

impl SharedCore {
    /// Get or initialize the global shared core.
    pub fn global() -> Arc<SharedCore> {
        SHARED_CORE
            .get_or_init(|| {
                let store = match DictStore::open_default_with_migration() {
                    Ok(s) => Some(Arc::new(s)),
                    Err(e) => {
                        log::warn!(
                            "failed to open user dictionary store: {}; \
                             persistence disabled",
                            e
                        );
                        None
                    }
                };

                let user_scorer = match &store {
                    Some(s) => UserScorer::from_store(s.clone()).unwrap_or_else(|e| {
                        log::warn!("failed to load user scores from store: {}", e);
                        UserScorer::new()
                    }),
                    None => UserScorer::new(),
                };

                let mut dictionary = Dictionary::new();
                if let Some(s) = &store {
                    dictionary.attach_store(s.clone());
                    match dictionary.load_from_store() {
                        Ok(n) if n > 0 => {
                            log::info!("loaded {} user dictionary entries from store", n)
                        }
                        Err(e) => log::warn!("failed to load user dictionary: {}", e),
                        _ => {}
                    }
                }

                // Unit tests must be hermetic: ConversionEngine::new() reaches
                // this global core, and binding it to a live llama-server makes
                // ranking-order tests depend on the server's model and the
                // background rerank's timing (flaky under parallel load). Use
                // the deterministic MockScorer in test builds; the #[ignore]
                // integration tests construct their own HttpLlamaScorer.
                let llm = if cfg!(test) {
                    LlmEngine::with_scorer(Box::new(crate::core::llm::MockScorer))
                } else {
                    LlmEngine::new()
                };

                Arc::new(SharedCore {
                    dictionary: RwLock::new(dictionary),
                    grammar: GrammarEngine::new(),
                    llm: Mutex::new(llm),
                    user_scorer: Mutex::new(user_scorer),
                    store,
                })
            })
            .clone()
    }

    /// Build a fresh, isolated core for evaluation harnesses, wired to a
    /// specific LLM scorer.
    ///
    /// Unlike [`global`], this does not touch the process-wide singleton or the
    /// user's SQLite store: it pairs the embedded dictionary/grammar with an
    /// empty [`UserScorer`] (no learned history), while still exercising the
    /// real production pipeline (`start_conversion` → `trigger_llm_rerank` →
    /// `apply_llm_rerank`). The scorer is the only moving part, so the hermetic
    /// and live conversion-quality layers differ only in what they pass here.
    pub fn new_eval(scorer: Box<dyn crate::core::llm::LlmScorer>) -> Arc<SharedCore> {
        Arc::new(SharedCore {
            dictionary: RwLock::new(Dictionary::new()),
            grammar: GrammarEngine::new(),
            llm: Mutex::new(LlmEngine::with_scorer(scorer)),
            user_scorer: Mutex::new(UserScorer::new()),
            store: None,
        })
    }

    /// Hermetic evaluation core: [`new_eval`] wired to the deterministic
    /// [`MockScorer`]. No llama-server, reproducible across machines and CI.
    /// The `#[ignore]` live-quality tests call [`new_eval`] with an
    /// `HttpLlamaScorer` instead.
    pub fn new_hermetic() -> Arc<SharedCore> {
        Self::new_eval(Box::new(crate::core::llm::MockScorer))
    }
}

/// Result of background LLM reranking: reranked candidate lists per segment.
/// One reranked segment: the reading it was computed for, paired with its
/// reordered candidate list. The reading lets `apply_llm_rerank` reject a
/// result whose segmentation no longer matches the live conversion (a pass
/// that started before a resize / boundary change moved the bunsetsu).
type LlmRerankResult = Vec<(String, Vec<String>)>;

pub struct ConversionEngine {
    romaji: RomajiConverter,
    shared: Arc<SharedCore>,
    /// Active conversion state (None when not converting)
    conversion: Option<ConversionState>,
    /// Background LLM reranking result (populated asynchronously)
    llm_rerank_result: Arc<Mutex<Option<LlmRerankResult>>>,
    /// True from when a background rerank is triggered until its result is
    /// applied. Frontends poll this to know whether to wait for a refresh.
    rerank_inflight: Arc<AtomicBool>,
}

impl ConversionEngine {
    pub fn new() -> Self {
        Self::with_shared(SharedCore::global())
    }

    /// Construct an engine bound to a specific [`SharedCore`].
    ///
    /// Production uses [`new`] (the global core). Evaluation harnesses pass a
    /// [`SharedCore::new_hermetic`] core to run the full conversion pipeline
    /// deterministically, without the global singleton, user store, or a live
    /// LLM server.
    pub fn with_shared(shared: Arc<SharedCore>) -> Self {
        Self {
            romaji: RomajiConverter::new(),
            shared,
            conversion: None,
            llm_rerank_result: Arc::new(Mutex::new(None)),
            rerank_inflight: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Process a key event from the IME framework.
    /// Returns the appropriate action for the UI layer.
    pub fn process_key(&mut self, key: char) -> EngineAction {
        if let Some(_kana) = self.romaji.process_key(key) {
            EngineAction::UpdatePreedit(self.preedit())
        } else {
            EngineAction::Buffering(self.preedit())
        }
    }

    /// Append a raw string directly to the preedit (e.g., punctuation).
    pub fn append_raw(&mut self, s: &str) {
        self.romaji.flush();
        self.romaji.append_raw(s);
    }

    /// Get the current preedit string (kana output + pending romaji buffer).
    pub fn preedit(&self) -> String {
        let mut preedit = self.romaji.output().to_string();
        let buf = self.romaji.buffer();
        if !buf.is_empty() {
            preedit.push_str(buf);
        }
        preedit
    }

    /// Start segment-based conversion (space key pressed).
    /// Returns the conversion state if successful.
    pub fn start_conversion(&mut self) -> Option<&ConversionState> {
        self.romaji.flush();
        let kana = self.romaji.output().to_string();
        if kana.is_empty() {
            return None;
        }

        let dict = self.shared.dictionary.read().unwrap_or_else(|e| e.into_inner());
        let segments = {
            let user_scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
            dict.segment_with_boost(&kana, |reading, entries| {
                // Don't boost single-char segments — learned single-kana scores (の, が, い...)
                // are very high and distort segmentation by encouraging excessive splitting.
                if reading.chars().count() <= 1 {
                    return 0.0;
                }
                entries
                    .iter()
                    .map(|e| user_scorer.score(reading, &e.surface))
                    .fold(0.0_f64, f64::max)
                    * 10.0 // Scale boost to be significant vs segment cost
            })
            // user_scorer lock released here
        };
        if segments.is_empty() {
            return None;
        }

        // Apply AI segmentation filter: try alternative segmentations and pick the best
        let segments = self.filter_segmentation(segments, &kana, &dict);
        drop(dict);

        let segment_states = self.build_segment_states(&segments);
        self.conversion = Some(ConversionState {
            kana,
            segments: segment_states,
            focus: 0,
        });

        // Trigger LLM reranking in background — results applied on next interaction
        self.trigger_llm_rerank();

        self.conversion.as_ref()
    }

    /// Get the current conversion state.
    pub fn conversion_state(&self) -> Option<&ConversionState> {
        self.conversion.as_ref()
    }

    /// Move focus to the next/previous segment. delta: +1 = right, -1 = left.
    pub fn move_focus(&mut self, delta: i32) -> Option<&ConversionState> {
        let state = self.conversion.as_mut()?;
        let len = state.segments.len();
        if len == 0 {
            return self.conversion.as_ref();
        }
        state.focus = if delta > 0 {
            (state.focus + 1) % len
        } else if state.focus == 0 {
            len - 1
        } else {
            state.focus - 1
        };
        self.conversion.as_ref()
    }

    /// Cycle the candidate for the focused segment. delta: +1 = next, -1 = previous.
    pub fn cycle_candidate(&mut self, delta: i32) -> Option<&ConversionState> {
        let state = self.conversion.as_mut()?;
        let seg = &mut state.segments[state.focus];
        let len = seg.candidates.len();
        if len == 0 {
            return self.conversion.as_ref();
        }
        seg.selected = if delta > 0 {
            (seg.selected + 1) % len
        } else if seg.selected == 0 {
            len - 1
        } else {
            seg.selected - 1
        };
        seg.user_selected = true;
        self.conversion.as_ref()
    }

    /// Resize the focused segment. delta: +1 = extend right, -1 = shrink right.
    /// Re-segments the affected regions and re-looks up candidates.
    pub fn resize_segment(&mut self, delta: i32) -> Option<&ConversionState> {
        let state = self.conversion.as_mut()?;
        let focus = state.focus;
        let seg_count = state.segments.len();

        if delta > 0 {
            // Extend: take one char from the next segment
            if focus + 1 >= seg_count {
                return self.conversion.as_ref();
            }
            let next_reading: Vec<char> = state.segments[focus + 1].reading.chars().collect();
            if next_reading.is_empty() {
                return self.conversion.as_ref();
            }
            // Move first char of next segment to current segment
            let ch = next_reading[0];
            state.segments[focus].reading.push(ch);
            let new_next: String = next_reading[1..].iter().collect();
            if new_next.is_empty() {
                state.segments.remove(focus + 1);
            } else {
                state.segments[focus + 1].reading = new_next;
                state.segments[focus + 1].start += 1;
            }
        } else {
            // Shrink: move last char of current segment to next segment
            let cur_reading: Vec<char> = state.segments[focus].reading.chars().collect();
            if cur_reading.len() <= 1 {
                return self.conversion.as_ref();
            }
            // Guarded above: cur_reading.len() > 1 → last() is Some.
            let Some(&last_ch) = cur_reading.last() else {
                return self.conversion.as_ref();
            };
            let new_cur: String = cur_reading[..cur_reading.len() - 1].iter().collect();
            state.segments[focus].reading = new_cur;

            if focus + 1 < state.segments.len() {
                let next = &mut state.segments[focus + 1];
                next.reading.insert(0, last_ch);
                next.start -= 1;
            } else {
                // Create a new segment after current
                let start = state.segments[focus].start
                    + state.segments[focus].reading.chars().count();
                state.segments.push(SegmentState {
                    reading: last_ch.to_string(),
                    start,
                    candidates: vec![last_ch.to_string()],
                    selected: 0,
                    user_selected: false,
                });
            }
        }

        // Re-lookup candidates for the affected segments. On a shrink, the
        // following segment passively received the pushed char and may now span
        // a particle + word (e.g. "がふる"); split it back into separate bunsetsu
        // so each piece stays independently selectable instead of collapsing
        // into one inseparable chunk. On an extend the focused segment grew by a
        // deliberate merge, so it is kept whole.
        self.relookup_segment(focus);
        let seg_len = self
            .conversion
            .as_ref()
            .map(|s| s.segments.len())
            .unwrap_or(0);
        if focus + 1 < seg_len {
            if delta < 0 {
                self.relookup_or_split_segment(focus + 1);
            } else {
                self.relookup_segment(focus + 1);
            }
        }

        // A manual boundary change is *not* a surface choice: `user_selected`
        // is reserved for segments where the user explicitly picked a candidate
        // (cycle / kana / hiragana), which `apply_llm_rerank` must not override.
        // Leaving the resized bunsetsu rerank-eligible lets the LLM order their
        // candidates by context. Re-run the background rerank so it scores the
        // new bunsetsu instead of the stale pre-resize segmentation.
        self.trigger_llm_rerank();

        self.conversion.as_ref()
    }

    /// Set the focused segment's selected candidate to its hiragana reading (F6).
    pub fn convert_focused_to_hiragana(&mut self) -> Option<&ConversionState> {
        self.convert_focused_to(KanaForm::Hiragana)
    }

    /// Reset the focused segment to one of the five kana forms — hiragana
    /// (F6), full/half-width katakana (F7/F8), full/half-width romaji
    /// (F9/F10). Adds the transformed text as a new candidate when the
    /// segment doesn't already contain it, and marks the segment as
    /// user-selected so later LLM rerank passes leave it alone. Replaces
    /// five near-identical convert_focused_to_* helpers.
    pub fn convert_focused_to(&mut self, form: KanaForm) -> Option<&ConversionState> {
        let state = self.conversion.as_mut()?;
        let seg = &mut state.segments[state.focus];
        let text = form.apply(&seg.reading);
        seg.selected = match seg.candidates.iter().position(|c| c == &text) {
            Some(p) => p,
            None => {
                seg.candidates.push(text);
                seg.candidates.len() - 1
            }
        };
        seg.user_selected = true;
        self.conversion.as_ref()
    }

    /// Start a kana-form conversion (F6/F7/F8/F9/F10 outside conversion mode).
    /// Creates a single-segment conversion with hiragana, katakana, half-width katakana,
    /// half-width romaji, and full-width romaji as candidates, and selects the one matching the requested form.
    /// form: 0 = hiragana, 1 = katakana, 2 = half-width katakana, 3 = half-width romaji, 4 = full-width romaji
    pub fn start_kana_conversion(&mut self, form: usize) -> Option<&ConversionState> {
        self.romaji.flush();
        let kana = self.romaji.output().to_string();
        if kana.is_empty() {
            return None;
        }

        let katakana = crate::core::romaji::hiragana_to_katakana(&kana);
        let half_katakana = crate::core::romaji::hiragana_to_halfwidth_katakana(&kana);
        let romaji = crate::core::romaji::hiragana_to_romaji(&kana);
        let fw_romaji = crate::core::romaji::hiragana_to_fullwidth_romaji(&kana);

        // Look up the requested form BEFORE deduping. The old code deduped
        // the vec first and then indexed by `form`, so on strings whose
        // renderings collide — e.g. "ー" where kana==katakana == "ー" —
        // F8 (form=2, half-width katakana) mapped to index 2 of the deduped
        // list, which was the ASCII "-", not "ｰ".
        let forms = [kana.clone(), katakana, half_katakana, romaji, fw_romaji];
        let selected_text = forms
            .get(form)
            .cloned()
            .unwrap_or_else(|| forms[0].clone());

        let mut candidates: Vec<String> = Vec::with_capacity(forms.len());
        let mut seen = std::collections::HashSet::new();
        for f in forms.iter() {
            if seen.insert(f.clone()) {
                candidates.push(f.clone());
            }
        }

        let selected = candidates
            .iter()
            .position(|c| c == &selected_text)
            .unwrap_or(0);

        self.conversion = Some(ConversionState {
            kana: kana.clone(),
            segments: vec![SegmentState {
                reading: kana,
                start: 0,
                candidates,
                selected,
                user_selected: form != 0,
            }],
            focus: 0,
        });
        self.conversion.as_ref()
    }

    /// Convert the focused segment's reading to katakana and set it as the selected candidate.
    pub fn convert_focused_to_katakana(&mut self) -> Option<&ConversionState> {
        self.convert_focused_to(KanaForm::Katakana)
    }

    /// Clear conversion state (on commit or cancel).
    pub fn clear_conversion(&mut self) {
        self.conversion = None;
    }

    /// Trigger conversion (legacy interface for tests).
    /// Runs the full 3-stage pipeline: dictionary → grammar → LLM.
    pub fn convert(&mut self) -> Vec<ConversionCandidate> {
        self.romaji.flush();
        let kana = self.romaji.output().to_string();
        if kana.is_empty() {
            return Vec::new();
        }

        let segments = self.shared.dictionary.read().unwrap_or_else(|e| e.into_inner()).segment(&kana);
        if segments.is_empty() {
            return Vec::new();
        }

        let candidates = self.build_candidates(&segments);

        let llm = self.shared.llm.lock().unwrap_or_else(|e| e.into_inner());
        let mut scored: Vec<ConversionCandidate> = candidates
            .into_iter()
            .map(|text| {
                let grammar_tokens = self.tokens_for_grammar(&text, &segments);
                let grammar_result = self.shared.grammar.score(&grammar_tokens);
                let llm_score = llm.score_candidate(&text);
                let combined = grammar_result.score * 0.4 + llm_score * 0.6;
                ConversionCandidate {
                    text,
                    grammar_score: grammar_result.score,
                    llm_score,
                    score: combined,
                }
            })
            .collect();

        drop(llm);
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Convert the focused segment's reading to half-width romaji (F10 during conversion mode).
    pub fn convert_focused_to_romaji(&mut self) -> Option<&ConversionState> {
        self.convert_focused_to(KanaForm::Romaji)
    }

    /// Convert the focused segment's reading to full-width romaji (F9 during conversion mode).
    pub fn convert_focused_to_fullwidth_romaji(&mut self) -> Option<&ConversionState> {
        self.convert_focused_to(KanaForm::FullwidthRomaji)
    }

    /// Convert the focused segment's reading to half-width katakana.
    pub fn convert_focused_to_halfwidth_katakana(&mut self) -> Option<&ConversionState> {
        self.convert_focused_to(KanaForm::HalfwidthKatakana)
    }

    /// Commit the selected candidate and update context.
    pub fn commit(&mut self, candidate: &str) -> String {
        match self.shared.llm.try_lock() {
            Ok(mut llm) => llm.update_context(candidate),
            Err(_) => log::debug!("LLM lock busy during commit, skipping context update"),
        }
        self.romaji.reset();
        candidate.to_string()
    }

    /// Commit the current conversion, recording user selections for learning.
    /// Returns the composed text if there was an active conversion.
    pub fn commit_conversion(&mut self) -> Option<String> {
        let state = self.conversion.take()?;
        let text = state.composed_text();

        // Opt-in: dump committed conversion to JSONL for (c) eval dataset curation.
        // Gated by BONOLITH_LOG_CONVERSIONS=1 — privacy-sensitive (logs raw text).
        log_conversion_for_eval(&state);

        // Record only segments where the user explicitly chose a candidate.
        // record() persists immediately when the scorer is store-attached,
        // so no separate save step is needed.
        {
            let mut user_scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
            for seg in &state.segments {
                if seg.user_selected {
                    let surface = &seg.candidates[seg.selected];
                    user_scorer.record(&seg.reading, surface);
                }
            }
        }

        // Use try_lock to avoid blocking if LLM background thread holds the lock.
        // If we can't acquire the lock now, update context on next available opportunity.
        match self.shared.llm.try_lock() {
            Ok(mut llm) => llm.update_context(&text),
            Err(_) => log::debug!("LLM lock busy during commit, skipping context update"),
        }
        self.romaji.reset();
        Some(text)
    }

    /// Clear all user learning history (scores) from memory and the database.
    /// Returns the number of rows deleted, or an error string.
    pub fn clear_learning_history(&self) -> Result<usize, String> {
        let mut user_scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
        user_scorer.clear_scores().map_err(|e| e.to_string())
    }

    /// Return a reference to the shared core, for use in background threads.
    pub fn shared_core(&self) -> Arc<SharedCore> {
        self.shared.clone()
    }

    /// Delete the last character from the preedit (backspace).
    /// Returns true if something was deleted.
    pub fn delete_last(&mut self) -> bool {
        self.romaji.delete_last()
    }

    /// Reset the engine state (e.g., on focus change).
    pub fn reset(&mut self) {
        self.romaji.reset();
    }

    /// Build SegmentState list from dictionary Segments.
    /// Candidates are ordered by effective score (dictionary frequency + user learning).
    /// LLM reranking is triggered separately in the background.
    fn build_segment_states(&self, segments: &[Segment]) -> Vec<SegmentState> {
        let user_scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
        segments
            .iter()
            .map(|seg| {
                let mut entries: Vec<&DictionaryEntry> = seg.candidates.iter().collect();
                entries.sort_by(|a, b| {
                    let score_a = Self::effective_score_with(&user_scorer, &seg.reading, a);
                    let score_b = Self::effective_score_with(&user_scorer, &seg.reading, b);
                    score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut candidates: Vec<String> =
                    entries.iter().map(|e| e.surface.clone()).collect();
                // Always include the raw reading (kana) as a candidate.
                // Insert at a position based on user learning score so that
                // frequently selected kana forms can outrank kanji entries.
                if candidates.is_empty() || !candidates.contains(&seg.reading) {
                    let kana_user = user_scorer.score(&seg.reading, &seg.reading);
                    let kana_score = kana_user * 2.0 + 0.1; // small base + user learning
                    let insert_pos = entries
                        .iter()
                        .position(|e| {
                            Self::effective_score_with(&user_scorer, &seg.reading, e) < kana_score
                        })
                        .unwrap_or(candidates.len());
                    candidates.insert(insert_pos, seg.reading.clone());
                }
                SegmentState {
                    reading: seg.reading.clone(),
                    start: seg.start,
                    candidates,
                    selected: 0,
                    user_selected: false,
                }
            })
            .collect()
    }

    /// Trigger background LLM reranking for the current conversion state.
    /// Results are stored in `llm_rerank_result` and can be applied later.
    fn trigger_llm_rerank(&self) {
        let state = match self.conversion.as_ref() {
            Some(s) => s,
            None => return,
        };

        // Collect segment info needed for background scoring. Snapshot each
        // candidate's user-learning score here (we're on the engine thread and
        // can lock the scorer); the rerank itself runs on a background thread
        // and uses the magnitude directly, so repeated selections keep adding
        // weight instead of saturating once the surface reaches rank 0.
        let seg_info: Vec<(String, Vec<(String, f64)>)> = {
            let user_scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
            state
                .segments
                .iter()
                .map(|seg| {
                    let cands = seg
                        .candidates
                        .iter()
                        .map(|s| (s.clone(), user_scorer.score(&seg.reading, s)))
                        .collect();
                    (seg.reading.clone(), cands)
                })
                .collect()
        };

        let shared = self.shared.clone();
        let result_slot = self.llm_rerank_result.clone();
        // Clone the inflight flag into the worker so a panic path can clear
        // it — otherwise `rerank_inflight` stays latched and the frontend
        // wastes ~2 s polling for a result that will never arrive.
        let inflight = self.rerank_inflight.clone();

        // Clear previous result and mark a pass in flight (cleared when applied).
        *result_slot.lock().unwrap_or_else(|e| e.into_inner()) = None;
        inflight.store(true, Ordering::Relaxed);

        thread::spawn(move || {
            // Catch any panics to prevent crashing the IBus process
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let llm = shared.llm.lock().unwrap_or_else(|e| e.into_inner());
                let committed_context = llm.context().to_string();
                // running_context accumulates chosen candidates as we go through segments.
                // Start empty — committed_context is passed separately to score_with_context.
                let mut preceding_text = String::new();

                const LLM_RERANK_TOP_N: usize = 5;
                // Weight on the user-learning magnitude. The user score
                // saturates at 1.0 (~20 selections; ~0.23 at one), so a fully
                // learned surface adds up to this much — chosen so it can
                // overcome the LLM's max swing (0.6 * 0.6 = 0.36) when the user
                // has deliberately reinforced a reading, while a single
                // selection only nudges near-ties.
                const USER_LEARNING_WEIGHT: f64 = 0.5;
                let mut reranked_segments: Vec<(String, Vec<String>)> = Vec::new();

                for (reading, candidates) in &seg_info {
                    if candidates.len() > 1 && reading.chars().count() >= 2 {
                        let rerank_count = candidates.len().min(LLM_RERANK_TOP_N);
                        // Context = committed text + preceding segments' chosen candidates
                        let context = format!("{}{}", committed_context, preceding_text);
                        let mut top_with_scores: Vec<(usize, f64)> = (0..rerank_count)
                            .map(|i| {
                                let (surface, user) = &candidates[i];
                                let llm_score = Self::rerank_llm_score(reading, surface, || {
                                    llm.score_with_context(&context, surface)
                                });
                                let rank_base =
                                    1.0 - (i as f64 / rerank_count as f64) * 0.3;
                                let combined = rank_base * 0.4
                                    + llm_score * 0.6
                                    + user * USER_LEARNING_WEIGHT;
                                (i, combined)
                            })
                            .collect();
                        top_with_scores.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });

                        let mut reranked: Vec<String> = top_with_scores
                            .iter()
                            .map(|(idx, _)| candidates[*idx].0.clone())
                            .collect();
                        reranked
                            .extend(candidates[rerank_count..].iter().map(|(s, _)| s.clone()));

                        log::debug!(
                            "LLM rerank segment '{}': top='{}'",
                            reading,
                            reranked[0],
                        );

                        preceding_text.push_str(&reranked[0]);
                        reranked_segments.push((reading.clone(), reranked));
                    } else {
                        preceding_text.push_str(&candidates[0].0);
                        reranked_segments.push((
                            reading.clone(),
                            candidates.iter().map(|(s, _)| s.clone()).collect(),
                        ));
                    }
                }

                reranked_segments
            }));

            match result {
                Ok(reranked) => {
                    *result_slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(reranked);
                    log::info!("LLM background reranking complete");
                }
                Err(_) => {
                    log::warn!("LLM background reranking panicked, discarding results");
                    // Without this, `rerank_inflight` stays true until the
                    // next conversion overwrites it, so the current pass's
                    // poll-refresh loop burns ~2 s waiting for a result the
                    // panicked thread will never produce.
                    inflight.store(false, Ordering::Relaxed);
                }
            }
        });
    }

    /// Apply LLM reranking results if available.
    /// Returns true if candidates were updated.
    pub fn apply_llm_rerank(&mut self) -> bool {
        let reranked = {
            let mut slot = self.llm_rerank_result.lock().unwrap_or_else(|e| e.into_inner());
            slot.take()
        };

        let reranked = match reranked {
            // No result yet — leave `rerank_inflight` set so the frontend keeps
            // polling for the pass that is still running.
            Some(r) => r,
            None => return false,
        };

        let state = match self.conversion.as_mut() {
            Some(s) => s,
            None => {
                // No active conversion to apply to; the pass is moot.
                self.rerank_inflight.store(false, Ordering::Relaxed);
                return false;
            }
        };

        // Reject a stale result whose segmentation no longer matches the live
        // conversion. Background passes from before a resize / commit can finish
        // late and be grabbed by a still-polling refresh task; applying their
        // candidate lists positionally onto a changed bunsetsu layout corrupts
        // the display (dropped or duplicated segments). Require a segment-for-
        // segment reading match; on mismatch, drop it and leave the pass marked
        // in flight so the frontend keeps waiting for the matching result.
        let aligned = reranked.len() == state.segments.len()
            && reranked
                .iter()
                .zip(state.segments.iter())
                .all(|((reading, _), seg)| *reading == seg.reading);
        if !aligned {
            log::debug!("Discarding stale LLM rerank result (segmentation changed)");
            return false;
        }

        // We consumed a matching background result; the pass is no longer in flight.
        self.rerank_inflight.store(false, Ordering::Relaxed);

        let mut updated = false;
        for (seg, (_, new_candidates)) in state.segments.iter_mut().zip(reranked.into_iter()) {
            // Don't override if user already manually selected a candidate
            if seg.user_selected {
                continue;
            }
            if seg.candidates != new_candidates {
                seg.candidates = new_candidates;
                seg.selected = 0;
                updated = true;
            }
        }
        if updated {
            log::info!("LLM reranking applied to conversion state");
        }
        updated
    }

    /// Check if LLM reranking results are ready (non-blocking).
    pub fn has_llm_rerank_result(&self) -> bool {
        self.llm_rerank_result.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// True while a background rerank pass is outstanding (triggered but its
    /// result not yet applied). Frontends poll this to decide whether to keep
    /// waiting for a display refresh.
    pub fn rerank_inflight(&self) -> bool {
        self.rerank_inflight.load(Ordering::Relaxed)
    }

    /// LLM score to use for a candidate during reranking.
    ///
    /// Plain-kana candidates (surface == reading) are the explicit fallback,
    /// not an LLM-preferred conversion. The model systematically over-rates a
    /// bare-kana continuation (it's always a plausible next token), which would
    /// let きょう/はし beat 今日/橋. Neutralize the LLM term for the kana form so
    /// it can only win on frequency/position or deliberate user learning (the
    /// user-magnitude term in the combine is untouched).
    fn rerank_llm_score(reading: &str, surface: &str, score: impl FnOnce() -> f64) -> f64 {
        if surface == reading {
            crate::core::llm::NEUTRAL_SCORE
        } else {
            score()
        }
    }

    /// Compute effective score combining dictionary frequency, user learning,
    /// and surface-form adjustments.
    fn effective_score_with(user_scorer: &UserScorer, reading: &str, entry: &DictionaryEntry) -> f64 {
        let freq_norm = (entry.frequency as f64) / 10000.0;
        let user = user_scorer.score(reading, &entry.surface);

        // Surface-form adjustments to correct IPADIC frequency biases
        let surface_adj = Self::surface_adjustment(reading, &entry.surface, entry.pos);

        freq_norm + user * 2.0 + surface_adj
    }

    /// Adjustment based on surface form characteristics.
    /// Returns a bonus (positive) or penalty (negative) added to the effective score.
    fn surface_adjustment(reading: &str, surface: &str, pos: PartOfSpeech) -> f64 {
        // Katakana-only surfaces are rarely the desired conversion in normal text.
        // e.g. イイ(いい), テキ(てき), タイ(たい) — demote significantly.
        let all_katakana = !surface.is_empty()
            && surface.chars().all(|c| {
                ('\u{30A1}'..='\u{30F6}').contains(&c) || c == 'ー'
            });
        if all_katakana && surface != reading {
            return -0.3;
        }

        // If surface exactly matches reading (kana-only), give a small boost
        // for functional words — they are often the correct choice.
        // e.g. これ, それ, いる, する, いい
        if surface == reading {
            return match pos {
                PartOfSpeech::Particle | PartOfSpeech::Auxiliary => 0.2,
                PartOfSpeech::Verb | PartOfSpeech::Adjective | PartOfSpeech::Adverb => 0.15,
                PartOfSpeech::Noun => 0.1,  // pronouns are Noun
                _ => 0.0,
            };
        }

        0.0
    }

    /// AI segmentation filter: generate alternative segmentations and pick the best.
    ///
    /// Uses a two-stage scoring approach:
    /// 1. Heuristic score based on dictionary frequency and segment quality (always available)
    /// 2. LLM score for naturalness check (when available, used as tiebreaker)
    fn filter_segmentation(
        &self,
        base_segments: Vec<Segment>,
        kana: &str,
        dict: &Dictionary,
    ) -> Vec<Segment> {
        // Skip if too few segments to have meaningful alternatives
        if base_segments.len() <= 1 {
            return base_segments;
        }

        let alternatives = self.generate_alternative_segmentations(&base_segments, kana, dict);
        if alternatives.len() <= 1 {
            return base_segments;
        }

        // Score each alternative with heuristic
        let mut scored: Vec<(usize, f64)> = alternatives
            .iter()
            .enumerate()
            .map(|(i, alt)| (i, Self::score_segmentation_heuristic(alt)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // If the top heuristic candidate differs from base, try LLM as tiebreaker
        let heuristic_best = scored[0].0;
        if heuristic_best != 0 {
            // Try LLM scoring on the top 2 candidates for final decision
            if let Ok(llm) = self.shared.llm.try_lock() {
                let base_llm = Self::score_segmentation_llm(&alternatives[0], &llm);
                let best_llm = Self::score_segmentation_llm(&alternatives[heuristic_best], &llm);
                log::debug!(
                    "Segmentation filter LLM: base={:.3} '{}' vs best={:.3} '{}'",
                    base_llm,
                    Self::compose_top_candidates(&alternatives[0]),
                    best_llm,
                    Self::compose_top_candidates(&alternatives[heuristic_best]),
                );
                // Use LLM result only if it strongly disagrees (base scores much higher)
                if base_llm > best_llm + 0.15 {
                    log::info!("Segmentation filter: LLM overrode heuristic, keeping base");
                    // Invariant: alternatives always includes the base
                    // segmentation at index 0 (seeded above).
                    return alternatives
                        .into_iter()
                        .next()
                        .expect("alternatives always contains the base segmentation");
                }
            }
            log::info!(
                "Segmentation filter: changed from '{}' to '{}'",
                Self::compose_top_candidates(&alternatives[0]),
                Self::compose_top_candidates(&alternatives[heuristic_best]),
            );
        }

        for (i, score) in &scored {
            log::debug!(
                "Segmentation filter: alt[{}] h_score={:.3} text='{}'",
                i,
                score,
                Self::compose_top_candidates(&alternatives[*i]),
            );
        }

        // scored[0].0 is an index into alternatives (built above from the
        // same alternatives.iter().enumerate()), so nth() is always Some.
        alternatives
            .into_iter()
            .nth(scored[0].0)
            .expect("scored indices are always valid alternatives positions")
    }

    /// Generate alternative segmentations by merging adjacent segment pairs.
    /// Returns a list where the first entry is always the original segmentation.
    fn generate_alternative_segmentations(
        &self,
        base: &[Segment],
        _kana: &str,
        dict: &Dictionary,
    ) -> Vec<Vec<Segment>> {
        let mut alternatives: Vec<Vec<Segment>> = vec![base.to_vec()];

        const MAX_ALTERNATIVES: usize = 8;

        // Find all valid merge positions (where merging produces a dictionary word)
        let mut valid_merges: Vec<usize> = Vec::new();
        for i in 0..base.len().saturating_sub(1) {
            let merged_reading = format!("{}{}", base[i].reading, base[i + 1].reading);
            if !dict.lookup(&merged_reading).is_empty() {
                valid_merges.push(i);
            }
        }

        // Try each single merge
        for &i in &valid_merges {
            if alternatives.len() >= MAX_ALTERNATIVES {
                break;
            }
            if let Some(alt) = Self::build_merged_segmentation(base, &[i], dict) {
                alternatives.push(alt);
            }
        }

        // Try greedy multi-merge: apply all non-overlapping merges left-to-right.
        // Two merges at positions i and j overlap if j == i+1 (both consume segment i+1).
        if valid_merges.len() >= 2 {
            let mut multi_merges: Vec<usize> = Vec::new();
            for &i in &valid_merges {
                // Skip if this position overlaps with the previous merge
                if let Some(&last) = multi_merges.last() {
                    if i <= last + 1 {
                        continue;
                    }
                }
                multi_merges.push(i);
            }
            if multi_merges.len() >= 2 && alternatives.len() < MAX_ALTERNATIVES {
                if let Some(alt) = Self::build_merged_segmentation(base, &multi_merges, dict) {
                    alternatives.push(alt);
                }
            }
        }

        // Try splitting segments that are 4+ chars (may contain two words)
        for i in 0..base.len() {
            if alternatives.len() >= MAX_ALTERNATIVES {
                break;
            }
            let reading_chars: Vec<char> = base[i].reading.chars().collect();
            if reading_chars.len() < 4 {
                continue;
            }

            // Try splitting at each internal position (prefer middle splits)
            let mid = reading_chars.len() / 2;
            let mut split_positions: Vec<usize> = (2..reading_chars.len() - 1).collect();
            split_positions.sort_by_key(|&p| (p as i32 - mid as i32).unsigned_abs());

            for p in split_positions {
                if alternatives.len() >= MAX_ALTERNATIVES {
                    break;
                }
                let left_reading: String = reading_chars[..p].iter().collect();
                let right_reading: String = reading_chars[p..].iter().collect();

                let left_entries = dict.lookup(&left_reading);
                let right_entries = dict.lookup(&right_reading);
                if left_entries.is_empty() || right_entries.is_empty() {
                    continue;
                }

                let mut alt: Vec<Segment> = Vec::with_capacity(base.len() + 1);
                alt.extend_from_slice(&base[..i]);
                alt.push(Segment {
                    reading: left_reading,
                    start: base[i].start,
                    len: p,
                    candidates: left_entries.into_iter().cloned().collect(),
                });
                alt.push(Segment {
                    reading: right_reading,
                    start: base[i].start + p,
                    len: reading_chars.len() - p,
                    candidates: right_entries.into_iter().cloned().collect(),
                });
                if i + 1 < base.len() {
                    alt.extend_from_slice(&base[i + 1..]);
                }
                alternatives.push(alt);
            }
        }

        alternatives
    }

    /// Build a segmentation by applying merges at the given positions.
    /// Positions must be sorted and non-overlapping.
    fn build_merged_segmentation(
        base: &[Segment],
        merge_positions: &[usize],
        dict: &Dictionary,
    ) -> Option<Vec<Segment>> {
        let merge_set: std::collections::HashSet<usize> = merge_positions.iter().copied().collect();
        let mut alt: Vec<Segment> = Vec::new();
        let mut i = 0;
        while i < base.len() {
            if merge_set.contains(&i) && i + 1 < base.len() {
                let merged_reading = format!("{}{}", base[i].reading, base[i + 1].reading);
                let entries = dict.lookup(&merged_reading);
                if entries.is_empty() {
                    return None;
                }
                alt.push(Segment {
                    reading: merged_reading,
                    start: base[i].start,
                    len: base[i].len + base[i + 1].len,
                    candidates: entries.into_iter().cloned().collect(),
                });
                i += 2; // skip merged pair
            } else {
                alt.push(base[i].clone());
                i += 1;
            }
        }
        Some(alt)
    }

    /// Heuristic score for segmentation quality.
    /// Prefers: fewer segments, higher word frequencies, longer matched words,
    /// and natural POS bigram transitions (via the dictionary's CONN table —
    /// same one the segmentation DP uses, so the filter and the DP agree on
    /// what "natural" means).
    fn score_segmentation_heuristic(segments: &[Segment]) -> f64 {
        if segments.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let total_chars: usize = segments.iter().map(|s| s.len).sum();
        let mut prev_pos: Option<PartOfSpeech> = None;

        for seg in segments {
            let top = seg.candidates.first();
            let top_freq = top.map(|c| c.frequency as f64).unwrap_or(100.0);

            // Weighted frequency: longer segments contribute more (reward compound recognition)
            let weight = seg.len as f64 / total_chars as f64;
            score += weight * top_freq.ln();

            // Penalty for single-char non-particle segments (likely fragmented)
            if seg.len == 1 {
                let is_particle = top
                    .map(|c| matches!(c.pos, PartOfSpeech::Particle | PartOfSpeech::Auxiliary))
                    .unwrap_or(false);
                if !is_particle {
                    score -= 1.0;
                }
            }

            // POS-bigram penalty using the dictionary CONN table. Scaled by 0.3
            // so a worst-case transition (~8.0) costs ~2.4 — comparable to a
            // weighted-frequency term but never dominates it.
            if let Some(top) = top {
                let conn = connection_cost(prev_pos, top.pos);
                score -= conn * 0.3;
                prev_pos = Some(top.pos);
            }
        }

        // Bonus for fewer segments (prefer cohesive segmentation)
        score -= segments.len() as f64 * 0.5;

        score
    }

    /// Score a segmentation using LLM (naturalness check).
    fn score_segmentation_llm(segments: &[Segment], llm: &LlmEngine) -> f64 {
        let text = Self::compose_top_candidates(segments);
        llm.score_with_context(llm.context(), &text)
    }

    /// Compose text from the top candidate of each segment.
    fn compose_top_candidates(segments: &[Segment]) -> String {
        segments
            .iter()
            .map(|seg| {
                seg.candidates
                    .first()
                    .map(|c| c.surface.as_str())
                    .unwrap_or(&seg.reading)
            })
            .collect()
    }

    /// Re-lookup candidates for a segment after its reading changed.
    fn relookup_segment(&mut self, idx: usize) {
        let reading = match self.conversion.as_ref() {
            Some(state) => state.segments[idx].reading.clone(),
            None => return,
        };
        let dict = self.shared.dictionary.read().unwrap_or_else(|e| e.into_inner());
        // Use candidates_for_unit (not a flat lookup) so a manually resized
        // boundary that glues a particle onto an adjacent word still yields real
        // word candidates instead of collapsing to bare kana.
        let mut entries = dict.candidates_for_unit(&reading);
        let user_scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
        entries.sort_by(|a, b| {
            let score_a = Self::effective_score_with(&user_scorer, &reading, a);
            let score_b = Self::effective_score_with(&user_scorer, &reading, b);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut candidates: Vec<String> = entries.iter().map(|e| e.surface.clone()).collect();
        if candidates.is_empty() || !candidates.contains(&reading) {
            let kana_user = user_scorer.score(&reading, &reading);
            let kana_score = kana_user * 2.0 + 0.1;
            let insert_pos = entries
                .iter()
                .position(|e| {
                    Self::effective_score_with(&user_scorer, &reading, e) < kana_score
                })
                .unwrap_or(candidates.len());
            candidates.insert(insert_pos, reading);
        }
        drop(user_scorer);
        if let Some(state) = self.conversion.as_mut() {
            state.segments[idx].candidates = candidates;
            state.segments[idx].selected = 0;
        }
    }

    /// Re-lookup the segment at `idx`, splitting it into separate bunsetsu when
    /// a resize has glued a particle onto an adjacent word so its reading now
    /// spans more than one bunsetsu (e.g. "がふる" = が + 降る). Splitting keeps
    /// each piece independently selectable instead of fusing them into one
    /// inseparable chunk. Falls back to a plain in-place relookup when the
    /// reading is a single bunsetsu (incl. genuine Noun+Particle units like
    /// "はしを", which the segmenter keeps merged).
    fn relookup_or_split_segment(&mut self, idx: usize) {
        let reading = match self.conversion.as_ref() {
            Some(state) => state.segments[idx].reading.clone(),
            None => return,
        };
        let subs = {
            let dict = self.shared.dictionary.read().unwrap_or_else(|e| e.into_inner());
            dict.segment(&reading)
        };
        if subs.len() < 2 {
            self.relookup_segment(idx);
            return;
        }
        let base_start = match self.conversion.as_ref() {
            Some(state) => state.segments[idx].start,
            None => return,
        };
        // Build a SegmentState per sub-bunsetsu (mirrors the auto path) and
        // shift each to the glued segment's absolute kana offset. Leave
        // user_selected = false (build_segment_states default) so the freshly
        // split bunsetsu remain eligible for LLM context reranking.
        let mut new_states = self.build_segment_states(&subs);
        for st in &mut new_states {
            st.start += base_start;
        }
        if let Some(state) = self.conversion.as_mut() {
            state.segments.splice(idx..=idx, new_states);
        }
    }

    /// Build candidate sentences from segmented words.
    /// For each segment, pick the top candidates and combine.
    fn build_candidates(&self, segments: &[Segment]) -> Vec<String> {
        // Start with the top candidate for each segment (best conversion)
        let mut results = Vec::new();

        // Best candidate: top surface for each segment
        let best: String = segments
            .iter()
            .map(|seg| {
                seg.candidates
                    .first()
                    .map(|c| c.surface.as_str())
                    .unwrap_or(&seg.reading)
            })
            .collect();
        results.push(best);

        // Generate alternatives by swapping one segment at a time
        for (i, seg) in segments.iter().enumerate() {
            for candidate in seg.candidates.iter().skip(1).take(3) {
                let alt: String = segments
                    .iter()
                    .enumerate()
                    .map(|(j, s)| {
                        if j == i {
                            candidate.surface.as_str()
                        } else {
                            s.candidates
                                .first()
                                .map(|c| c.surface.as_str())
                                .unwrap_or(&s.reading)
                        }
                    })
                    .collect();
                if !results.contains(&alt) {
                    results.push(alt);
                }
            }
        }

        // Also include the raw kana as a candidate
        let raw_kana: String = segments.iter().map(|s| s.reading.as_str()).collect();
        if !results.contains(&raw_kana) {
            results.push(raw_kana);
        }

        results
    }

    /// Create grammar tokens from a candidate text and its segments.
    fn tokens_for_grammar(&self, _text: &str, segments: &[Segment]) -> Vec<GrammarToken> {
        segments
            .iter()
            .map(|seg| {
                let pos = seg
                    .candidates
                    .first()
                    .map(|c| c.pos)
                    .unwrap_or(crate::core::dictionary::PartOfSpeech::Other);
                GrammarToken {
                    surface: seg
                        .candidates
                        .first()
                        .map(|c| c.surface.clone())
                        .unwrap_or_else(|| seg.reading.clone()),
                    pos,
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub enum EngineAction {
    /// Key was buffered (incomplete romaji). Contains current preedit.
    Buffering(String),
    /// Preedit text was updated (kana produced). Contains current preedit.
    UpdatePreedit(String),
    /// Candidates are ready to display.
    ShowCandidates(Vec<ConversionCandidate>),
    /// Text was committed.
    Commit(String),
}

#[derive(Debug, Clone)]
pub struct ConversionCandidate {
    /// Converted text (kanji/mixed)
    pub text: String,
    /// Grammar score (0.0–1.0)
    pub grammar_score: f64,
    /// LLM score (0.0–1.0)
    pub llm_score: f64,
    /// Combined score (grammar * 0.4 + LLM * 0.6)
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (hermetic): resizing a segment boundary so a particle is
    /// pushed onto an adjacent word must split the pair back into separate
    /// bunsetsu, each independently selectable — not fuse them into one
    /// inseparable chunk. Earlier, `relookup_segment` did a flat `dict.lookup`
    /// on the multi-token reading (e.g. "がふる"), which collapsed to bare kana
    /// (fixed in bf15e50 by composing a "が降る" candidate); but that left the
    /// particle glued onto the verb as a single segment. Now the receiving
    /// segment is re-segmented into bunsetsu (あめ | が | 降る). Drives the real
    /// engine through the resize → relookup → split path.
    #[test]
    fn resize_splits_particle_off_adjacent_word() {
        fn has_kanji(s: &str) -> bool {
            s.chars().any(|c| {
                ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c)
            })
        }

        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        // Default bunsetsu: seg0="あめが", seg1="ふる". Shrink seg0 so its last
        // char (the particle が) is pushed onto the following verb. The glued
        // reading "がふる" must be split into が | ふる, not kept as one chunk.
        engine.resize_segment(-1);
        let segs = engine.conversion_state().unwrap().segments.clone();

        // No segment should carry the fused multi-token reading.
        assert!(
            segs.iter().all(|s| s.reading != "がふる"),
            "particle stayed glued onto the verb as one chunk: {:?}",
            segs.iter().map(|s| &s.reading).collect::<Vec<_>>(),
        );
        let readings: Vec<&str> = segs.iter().map(|s| s.reading.as_str()).collect();
        assert_eq!(
            readings,
            vec!["あめ", "が", "ふる"],
            "resize should split into independent bunsetsu",
        );
        // The verb bunsetsu still offers its kanji candidate (降る), and the
        // particle is its own selectable segment.
        let verb = segs.last().unwrap();
        assert!(
            verb.candidates.iter().any(|c| has_kanji(c)),
            "verb segment '{}' offered no word candidate: {:?}",
            verb.reading,
            verb.candidates,
        );
    }

    /// Regression (hermetic): the bare verb する must default to hiragana, not the
    /// slangy katakana スる. スる (ス + hiragana る) is mixed-kana so it dodged the
    /// all-katakana demotion in surface_adjustment; PRIORITY_OVERRIDES demotes スる
    /// and lifts する so build_segment_states ranks する first.
    #[test]
    fn suru_defaults_to_hiragana_not_katakana() {
        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "suru".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        let seg = &engine.conversion_state().unwrap().segments[0];
        assert_eq!(seg.reading, "する");
        assert_eq!(
            seg.candidates.first().map(String::as_str),
            Some("する"),
            "する should default to hiragana, got {:?}",
            seg.candidates.iter().take(4).collect::<Vec<_>>(),
        );
        assert_ne!(seg.candidates.first().map(String::as_str), Some("スる"));
    }

    /// Regression (hermetic): a manual resize must re-trigger the background LLM
    /// rerank and must NOT lock the touched bunsetsu as `user_selected` — a
    /// boundary change is not a surface choice, so the LLM may still reorder
    /// their candidates by context. (Previously resize set user_selected=true,
    /// which `apply_llm_rerank` skips, so context reranking never reached a
    /// resized segment.)
    #[test]
    fn resize_retriggers_rerank_and_leaves_segments_rerankable() {
        use std::time::{Duration, Instant};

        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        assert!(engine.rerank_inflight(), "start_conversion should trigger a rerank");

        // Drain the initial pass so we can observe the resize re-trigger cleanly.
        let deadline = Instant::now() + Duration::from_secs(2);
        while !engine.has_llm_rerank_result() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        engine.apply_llm_rerank();
        assert!(!engine.rerank_inflight(), "applying the result clears the inflight flag");

        engine.resize_segment(-1);
        assert!(engine.rerank_inflight(), "resize should re-trigger the rerank");
        let segs = &engine.conversion_state().unwrap().segments;
        assert!(
            segs.iter().all(|s| !s.user_selected),
            "resized/split bunsetsu must stay rerank-eligible (user_selected=false)",
        );
    }

    /// Regression: a background rerank from a superseded segmentation (e.g. one
    /// triggered before a resize / left-commit changed the bunsetsu boundaries)
    /// can finish late and be grabbed by a still-polling refresh task. Applying
    /// its candidate lists positionally onto the *current* layout corrupted the
    /// display (dropped / duplicated segments — 再現性→再性, …のですがですが).
    /// `apply_llm_rerank` must reject any result whose readings don't line up
    /// segment-for-segment, leave the conversion untouched, and keep the pass
    /// marked in flight so the matching result is still awaited.
    #[test]
    fn apply_rerank_rejects_stale_segmentation() {
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};

        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();

        // Drain the initial pass so the slot is empty and state is settled.
        let deadline = Instant::now() + Duration::from_secs(2);
        while !engine.has_llm_rerank_result() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        engine.apply_llm_rerank();

        let before: Vec<Vec<String>> = engine
            .conversion_state()
            .unwrap()
            .segments
            .iter()
            .map(|s| s.candidates.clone())
            .collect();

        // Inject a stale result whose readings don't match the live layout.
        *engine.llm_rerank_result.lock().unwrap_or_else(|e| e.into_inner()) = Some(vec![(
            "ZZ-bogus-reading".to_string(),
            vec!["☃".to_string(), "☔".to_string()],
        )]);
        engine.rerank_inflight.store(true, Ordering::Relaxed);

        assert!(
            !engine.apply_llm_rerank(),
            "a result from a different segmentation must not be applied",
        );
        assert!(
            engine.rerank_inflight(),
            "a discarded stale result leaves the pass in flight for the matching one",
        );

        let after: Vec<Vec<String>> = engine
            .conversion_state()
            .unwrap()
            .segments
            .iter()
            .map(|s| s.candidates.clone())
            .collect();
        assert_eq!(
            before, after,
            "candidates must be untouched by a stale rerank result",
        );
    }

    /// Learning regression (hermetic): recording user selections must be able
    /// to promote a non-default but valid surface to top-1 through the *full*
    /// pipeline (build_segment_states ordering + background rerank), and must
    /// never demote it once learned. Deterministic under MockScorer, so it
    /// guards the user-learning weight in the rerank combine without a server.
    /// Replaces the old `learning_curve` probe, which hand-rolled the combine
    /// (a partial pipeline) instead of driving the real engine.
    #[test]
    fn learning_promotes_surface_to_top1() {
        // (reading, target) where `target` is a valid homophone that is *not*
        // the cold-start default (e.g. 飴 sits below 雨) — so any flip to it can
        // only come from the recorded user selections.
        let cases = [("はし", "橋"), ("きしゃ", "汽車"), ("あめ", "飴")];
        const MAX_N: u32 = 20;

        for (reading, target) in cases {
            let top1_after = |n: u32| -> String {
                let shared = SharedCore::new_hermetic();
                {
                    let mut user = shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
                    for _ in 0..n {
                        user.record(reading, target);
                    }
                }
                let mut engine = ConversionEngine::with_shared(shared);
                engine.append_raw(reading);
                if engine.start_conversion().is_none() {
                    return String::new();
                }
                // Drain the deterministic background rerank.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while !engine.has_llm_rerank_result() && std::time::Instant::now() < deadline {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                engine.apply_llm_rerank();
                engine
                    .conversion_state()
                    .map(|s| s.composed_text())
                    .unwrap_or_default()
            };

            let flip = (0..=MAX_N).find(|&n| top1_after(n) == target);
            assert!(
                flip.is_some(),
                "{reading} -> {target}: learning never promotes it to top-1 within N={MAX_N} (got {:?})",
                top1_after(MAX_N),
            );
            // Once learned, it must stay on top at higher N (no demotion).
            assert_eq!(
                top1_after(MAX_N),
                target,
                "{reading} -> {target}: not stable at N={MAX_N}",
            );
            eprintln!(
                "learning: {reading} -> {target} reaches top-1 at N={}",
                flip.unwrap(),
            );
        }
    }

    /// Hermetic guard for ひらがな固着: the plain-kana form (surface == reading)
    /// must get the neutral LLM score so a model that over-rates bare kana can't
    /// promote it, while kanji forms keep the model's real score. Deliberate
    /// user learning is applied separately in the combine and stays unaffected.
    #[test]
    fn rerank_neutralizes_plain_kana_llm_score() {
        // A scorer that loves bare kana (the real failure mode).
        let kana_loving = |_surface: &str| 0.9_f64;

        // Plain kana → neutral, regardless of what the model would say.
        let kana = ConversionEngine::rerank_llm_score("はし", "はし", || kana_loving("はし"));
        assert_eq!(kana, crate::core::llm::NEUTRAL_SCORE);

        // Kanji form → the model's real score is used.
        let kanji = ConversionEngine::rerank_llm_score("はし", "橋", || kana_loving("橋"));
        assert_eq!(kanji, 0.9);
    }

    #[test]
    fn process_key_buffering() {
        let mut engine = ConversionEngine::new();
        match engine.process_key('k') {
            EngineAction::Buffering(preedit) => assert_eq!(preedit, "k"),
            _ => panic!("expected Buffering"),
        }
    }

    #[test]
    fn process_key_produces_kana() {
        let mut engine = ConversionEngine::new();
        engine.process_key('k');
        match engine.process_key('a') {
            EngineAction::UpdatePreedit(preedit) => assert_eq!(preedit, "か"),
            _ => panic!("expected UpdatePreedit"),
        }
    }

    #[test]
    fn preedit_shows_buffer() {
        let mut engine = ConversionEngine::new();
        engine.process_key('k');
        assert_eq!(engine.preedit(), "k");
        engine.process_key('a');
        assert_eq!(engine.preedit(), "か");
        engine.process_key('n');
        assert_eq!(engine.preedit(), "かn");
    }

    #[test]
    fn convert_basic() {
        let mut engine = ConversionEngine::new();
        // Type "kyou" → きょう
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let candidates = engine.convert();
        assert!(!candidates.is_empty());
        // Top candidate should be 今日 (highest frequency + kanji bonus)
        assert_eq!(candidates[0].text, "今日");
    }

    #[test]
    fn convert_sentence() {
        let mut engine = ConversionEngine::new();
        // Type "kyouhaiitenki" → きょうはいいてんき
        for ch in "kyouhaiitenki".chars() {
            engine.process_key(ch);
        }
        let candidates = engine.convert();
        assert!(!candidates.is_empty());
        // Some candidate should contain kanji conversion
        let any_has_kanji = candidates.iter().any(|c| {
            c.text.chars().any(|ch| ('\u{4E00}'..='\u{9FFF}').contains(&ch))
        });
        assert!(any_has_kanji, "Expected kanji in candidates: {:?}", candidates.iter().map(|c| &c.text).collect::<Vec<_>>());
    }

    #[test]
    fn convert_empty() {
        let mut engine = ConversionEngine::new();
        let candidates = engine.convert();
        assert!(candidates.is_empty());
    }

    #[test]
    fn commit_resets_state() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let candidates = engine.convert();
        let committed = engine.commit(&candidates[0].text);
        assert_eq!(committed, candidates[0].text);
        assert_eq!(engine.preedit(), "");
    }

    #[test]
    fn candidates_have_scores() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let candidates = engine.convert();
        for c in &candidates {
            assert!(c.score >= 0.0);
            assert!(c.score <= 1.0);
            assert!(c.grammar_score >= 0.0);
            assert!(c.llm_score >= 0.0);
        }
    }

    #[test]
    fn kanji_ranked_above_kana() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let candidates = engine.convert();
        // Raw kana きょう should be ranked below kanji candidates
        let kana_pos = candidates.iter().position(|c| c.text == "きょう");
        let kanji_pos = candidates.iter().position(|c| c.text == "今日");
        if let (Some(kp), Some(kap)) = (kana_pos, kanji_pos) {
            assert!(kap < kp, "kanji should rank above kana");
        }
    }

    #[test]
    fn segment_conversion_basic() {
        let mut engine = ConversionEngine::new();
        for ch in "kyouhaiitenki".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_conversion().unwrap();
        assert!(state.segments.len() >= 3);
        let text = state.composed_text();
        assert!(!text.is_empty());
    }

    #[test]
    fn segment_move_focus() {
        let mut engine = ConversionEngine::new();
        for ch in "kyouhaiitenki".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        let state = engine.move_focus(1).unwrap();
        assert_eq!(state.focus, 1);
        let state = engine.move_focus(-1).unwrap();
        assert_eq!(state.focus, 0);
    }

    #[test]
    fn segment_cycle_candidate() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        let state = engine.conversion_state().unwrap();
        let first = state.segments[0].candidates[0].clone();
        let state = engine.cycle_candidate(1).unwrap();
        let second = state.segments[0].candidates[state.segments[0].selected].clone();
        assert_ne!(first, second);
    }

    #[test]
    fn segment_resize() {
        let mut engine = ConversionEngine::new();
        for ch in "kyouhaiitenki".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        let orig_reading = engine.conversion_state().unwrap().segments[0].reading.clone();
        engine.resize_segment(1); // extend first segment by one char
        let new_reading = engine.conversion_state().unwrap().segments[0].reading.clone();
        assert_eq!(new_reading.chars().count(), orig_reading.chars().count() + 1);
    }

    #[test]
    fn segment_composed_text_and_ranges() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        let state = engine.conversion_state().unwrap();
        let ranges = state.segment_char_ranges();
        assert_eq!(ranges[0].0, 0);
        assert!(ranges.last().unwrap().1 > 0);
    }

    #[test]
    fn segmentation_filter_ryoukai_shimashita() {
        let mut engine = ConversionEngine::new();
        // Type "ryoukaisimashita" → りょうかいしました
        for ch in "ryoukaisimashita".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_conversion().unwrap();
        let readings: Vec<&str> = state.segments.iter().map(|s| s.reading.as_str()).collect();
        // The filter (even with MockScorer) should keep りょうかい as one segment
        assert!(
            readings.contains(&"りょうかい"),
            "Expected 'りょうかい' as a segment, got: {:?}",
            readings,
        );
    }

    #[test]
    fn segmentation_filter_multi_merge() {
        // Verify that the filter generates multi-merge alternatives
        // (merging non-overlapping segment pairs simultaneously)
        let engine = ConversionEngine::new();
        let dict = engine.shared.dictionary.read().unwrap_or_else(|e| e.into_inner());

        let base = vec![
            Segment {
                reading: "りょ".to_string(),
                start: 0,
                len: 2,
                candidates: dict.lookup("りょ").into_iter().cloned().collect(),
            },
            Segment {
                reading: "うかい".to_string(),
                start: 2,
                len: 3,
                candidates: dict.lookup("うかい").into_iter().cloned().collect(),
            },
            Segment {
                reading: "しま".to_string(),
                start: 5,
                len: 2,
                candidates: dict.lookup("しま").into_iter().cloned().collect(),
            },
            Segment {
                reading: "した".to_string(),
                start: 7,
                len: 2,
                candidates: dict.lookup("した").into_iter().cloned().collect(),
            },
        ];

        let alts = engine.generate_alternative_segmentations(&base, "りょうかいしました", &dict);
        let has_correct = alts.iter().any(|alt| {
            let readings: Vec<&str> = alt.iter().map(|s| s.reading.as_str()).collect();
            readings == vec!["りょうかい", "しました"]
        });
        assert!(has_correct, "Expected multi-merge alternative りょうかい+しました");
    }

    #[test]
    fn segmentation_filter_single_segment_passthrough() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_conversion().unwrap();
        // Single word — filter should pass through without change
        assert_eq!(state.segments.len(), 1);
        assert_eq!(state.segments[0].reading, "きょう");
    }

    #[test]
    fn f9_kana_conversion_fullwidth_romaji() {
        let mut engine = ConversionEngine::new();
        for ch in "tesuto".chars() {
            engine.process_key(ch);
        }
        assert_eq!(engine.preedit(), "てすと");

        // F9 → start_kana_conversion(4) = full-width romaji
        let state = engine.start_kana_conversion(4);
        assert!(state.is_some(), "start_kana_conversion(4) returned None");
        let state = state.unwrap();
        let composed = state.composed_text();
        assert_eq!(composed, "ｔｅｓｕｔｏ", "F9 should produce full-width romaji, got: {}", composed);
    }

    #[test]
    fn f10_kana_conversion_halfwidth_romaji() {
        let mut engine = ConversionEngine::new();
        for ch in "tesuto".chars() {
            engine.process_key(ch);
        }
        assert_eq!(engine.preedit(), "てすと");

        // F10 → start_kana_conversion(3) = half-width romaji
        let state = engine.start_kana_conversion(3);
        assert!(state.is_some(), "start_kana_conversion(3) returned None");
        let state = state.unwrap();
        let composed = state.composed_text();
        assert_eq!(composed, "tesuto", "F10 should produce half-width romaji, got: {}", composed);
    }
}
