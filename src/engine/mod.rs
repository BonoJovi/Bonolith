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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::core::{
    dictionary::{connection_cost, Dictionary, DictionaryEntry, PartOfSpeech, Segment},
    grammar::GrammarEngine,
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
    /// Case+spelling-preserved raw input snapshot for the whole
    /// conversion. Set only for single-segment conversions (F-key
    /// origin, or Space that yielded a single segment) so a follow-up
    /// F9/F10 form swap can round-trip "VIM"→"ＶＩＭ" and "shi"→"shi"
    /// via `convert_focused_to` instead of deriving from kana (which
    /// would flatten case and normalise spelling to "si"). `None` for
    /// multi-segment conversions — we don't track per-segment raw→kana
    /// boundaries, so `convert_focused_to` falls back to `KanaForm::apply`.
    pub raw_input: Option<String>,
    /// Segment boundaries at the moment start_conversion (or
    /// start_kana_conversion) built this state, expressed as segment
    /// start positions in char offsets and excluding 0. On commit,
    /// `commit_conversion` compares the final boundaries against this
    /// snapshot — if the user resized (`resize_segment`) into a
    /// different layout, the new segmentation is recorded via
    /// `UserScorer::record_segmentation` so the next time the same
    /// kana is typed the learned layout is applied up front.
    pub initial_boundaries: Vec<usize>,
}

/// Extract segment start positions (excluding 0) from a segment list.
/// Mirrors [`DictStore::record_segmentation`]'s wire format.
fn boundaries_of(segments: &[SegmentState]) -> Vec<usize> {
    segments.iter().skip(1).map(|s| s.start).collect()
}

/// Rebuild a segment list from a learned boundary layout. Returns
/// `None` if the boundaries don't line up with the kana char count
/// (defensive against corrupt DB rows). The reading for each slice is
/// looked up via [`Dictionary::candidates_for_unit`] so kanji
/// candidates still surface for the learned bunsetsu.
fn segments_from_boundaries(
    kana: &str,
    boundaries: &[usize],
    dict: &Dictionary,
) -> Option<Vec<Segment>> {
    let chars: Vec<char> = kana.chars().collect();
    let mut cuts = Vec::with_capacity(boundaries.len() + 2);
    cuts.push(0);
    cuts.extend_from_slice(boundaries);
    cuts.push(chars.len());

    let mut segments = Vec::with_capacity(cuts.len().saturating_sub(1));
    for w in cuts.windows(2) {
        let start = w[0];
        let end = w[1];
        if start >= end {
            return None;
        }
        let reading: String = chars[start..end].iter().collect();
        let candidates = dict.candidates_for_unit(&reading);
        segments.push(Segment {
            reading,
            start,
            len: end - start,
            candidates,
        });
    }
    Some(segments)
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
    /// Background LLM reranking result (populated asynchronously), tagged
    /// with the rerank generation of the pass that produced it. `apply_llm_rerank`
    /// drops the result if the tag doesn't match the current generation
    /// (a newer trigger / commit / cancel has invalidated it).
    llm_rerank_result: Arc<Mutex<Option<(u64, LlmRerankResult)>>>,
    /// Monotonically increasing "rerank epoch". Bumped on every event that
    /// invalidates an in-flight rerank pass (trigger, commit, cancel/clear).
    /// Workers capture the generation at spawn and gate their result-store /
    /// panic-recovery on it still matching, so a stale worker cannot poison
    /// the slot or clear the inflight flag for a newer pass. Frontends can
    /// snapshot it around the emit path to skip a repaint whose data no
    /// longer represents the live conversion (see the ibus refresh task).
    rerank_generation: Arc<AtomicU64>,
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
            rerank_generation: Arc::new(AtomicU64::new(0)),
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
        // Non-destructive precheck: only flush if there is already
        // committed output OR the pending buffer is a lone "n" (the
        // sole case where flush produces kana). Without this guard,
        // a Space on a mid-syllable buffer like "k" / "ky" silently
        // dropped the buffer, returned None, and let dispatch pass
        // Space through to the app — while the client's preedit
        // still displayed "k" until the next redraw, so the next
        // 'a' produced "あ" instead of "か". Leaving the buffer
        // intact lets the caller consume Space as a no-op and the
        // user keeps building the syllable.
        let flush_would_produce_kana = self.romaji.buffer() == "n";
        if self.romaji.output().is_empty() && !flush_would_produce_kana {
            return None;
        }
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

        let mut segment_states = self.build_segment_states(&segments);

        // If the user has previously re-segmented this exact kana into a
        // different layout, override the DP output with their learned
        // preference. Guarded by exact-match on the whole kana (see
        // A案 in the design discussion) so a mistake on "ものがきになる"
        // doesn't leak into unrelated inputs like "ものがきになった".
        // Boundaries beyond kana length are validated defensively;
        // corrupt DB rows fall back to the DP result.
        let kana_char_len = kana.chars().count();
        let learned = {
            let scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
            scorer.lookup_segmentation(&kana).map(|v| v.to_vec())
        };
        if let Some(learned) = learned {
            let valid = learned
                .iter()
                .all(|&b| b > 0 && b < kana_char_len)
                && learned.windows(2).all(|w| w[0] < w[1]);
            if valid && boundaries_of(&segment_states) != learned {
                if let Some(rebuilt) = segments_from_boundaries(&kana, &learned, &dict) {
                    segment_states = self.build_segment_states(&rebuilt);
                }
            }
        }
        drop(dict);

        let initial_boundaries = boundaries_of(&segment_states);
        // Only snapshot raw_input for a single-segment conversion — that's
        // the one case where the whole raw input maps unambiguously to
        // the segment's reading. Multi-segment conversions would need
        // per-segment raw slices we don't track.
        let raw_input = if segment_states.len() == 1 {
            self.romaji.raw_input().map(str::to_string)
        } else {
            None
        };
        self.conversion = Some(ConversionState {
            kana,
            segments: segment_states,
            focus: 0,
            raw_input,
            initial_boundaries,
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

        // raw_input snapshots the romaji spelling of a single-segment
        // conversion so F9/F10 can restore the exact keystrokes (`shi`
        // rather than a re-derived `si`). Resize always breaks that
        // invariant — the one segment that mapped to raw_input has just
        // been split into two, and neither piece owns the whole spelling
        // any more. If we left raw_input in place, F9/F10 on either
        // piece would paste the entire pre-resize spelling into that
        // single focused segment (kyou → shift-Left → F10 yields
        // "kyouう" or "きょkyou"). Invalidate here so subsequent F9/F10
        // fall back to per-segment kana derivation.
        if let Some(state) = self.conversion.as_mut() {
            state.raw_input = None;
        }

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
        // For F9/F10 (romaji forms), prefer the case+spelling-preserved
        // raw_input snapshot when it is available — otherwise the second
        // F9/F10 press would re-derive "し" as "ｓｉ"/"si" and add a
        // spurious candidate (the first press already put "shi" from
        // start_kana_conversion into the list). Only meaningful for
        // single-segment conversions where raw_input maps unambiguously
        // to the whole reading — start_conversion sets raw_input=None
        // on multi-segment entry and resize_segment nulls it out on any
        // subsequent split, so the invariant "raw_input is Some ⇒
        // segments.len() == 1" holds without an explicit len guard.
        let text = match (form, state.raw_input.as_deref()) {
            (KanaForm::Romaji, Some(raw)) if !raw.is_empty() => raw.to_string(),
            (KanaForm::FullwidthRomaji, Some(raw)) if !raw.is_empty() => raw
                .chars()
                .map(|c| crate::core::romaji::to_fullwidth_char(c).unwrap_or(c))
                .collect(),
            _ => {
                let seg = &state.segments[state.focus];
                form.apply(&seg.reading)
            }
        };
        let seg = &mut state.segments[state.focus];
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
        // Capture pending romaji buffer (e.g. lone "m" from "vim") BEFORE
        // flush drops it. `preedit()` already shows kana+buffer ("ゔぃm"),
        // so the user expects the buffer chars to survive into every
        // F-key form — most obviously F9/F10 where "vim" should round-
        // trip to "ｖｉｍ"/"vim" instead of "ｖｉ"/"vi". Each converter
        // (hiragana_to_katakana/_halfwidth_katakana/_romaji/_fullwidth_romaji)
        // already passes non-kana chars through unchanged, so appending
        // the buffer to the source string flows through cleanly.
        let pending = self.romaji.buffer().to_string();
        // Snapshot the case-preserved raw input before flush for F9/F10.
        // When available, F9/F10 uses it directly so "VIM" round-trips
        // as "ＶＩＭ"/"VIM" and "shi" as "shi" (not "si") — deriving
        // from kana would lose both case and spelling.
        let raw_input = self.romaji.raw_input().map(str::to_string);
        self.romaji.flush();
        let mut kana = self.romaji.output().to_string();
        kana.push_str(&pending);
        if kana.is_empty() {
            return None;
        }

        let katakana = crate::core::romaji::hiragana_to_katakana(&kana);
        let half_katakana = crate::core::romaji::hiragana_to_halfwidth_katakana(&kana);
        let (romaji, fw_romaji) = match raw_input.as_deref() {
            Some(raw) if !raw.is_empty() => {
                let fw: String = raw
                    .chars()
                    .map(|c| crate::core::romaji::to_fullwidth_char(c).unwrap_or(c))
                    .collect();
                (raw.to_string(), fw)
            }
            _ => (
                crate::core::romaji::hiragana_to_romaji(&kana),
                crate::core::romaji::hiragana_to_fullwidth_romaji(&kana),
            ),
        };

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
            // Snapshot the raw input so a subsequent F9/F10 form swap
            // via convert_focused_to picks up the case-preserved spelling
            // instead of deriving from kana (which would flatten "shi"→"si").
            raw_input,
            // Single-segment F-key conversion has no boundaries to record.
            initial_boundaries: Vec::new(),
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
        // Any in-flight rerank now belongs to a conversation the user has
        // walked away from. Bump the epoch so its result / panic path can't
        // touch the slot or clear inflight for a *later* pass, and clear the
        // slot + inflight so a follow-up refresh task exits its poll loop.
        self.invalidate_rerank_state();
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
        // so no separate save step is needed. When the final segmentation
        // differs from what the DP segmenter produced (user resized via
        // Shift+←/→), also record the layout so the same kana next time
        // starts pre-split the way the user wants — see
        // [`UserScorer::record_segmentation`].
        {
            let mut user_scorer = self.shared.user_scorer.lock().unwrap_or_else(|e| e.into_inner());
            for seg in &state.segments {
                if seg.user_selected {
                    let surface = &seg.candidates[seg.selected];
                    user_scorer.record(&seg.reading, surface);
                }
            }
            let final_boundaries = boundaries_of(&state.segments);
            if final_boundaries != state.initial_boundaries {
                user_scorer.record_segmentation(&state.kana, final_boundaries);
            }
        }

        // Use try_lock to avoid blocking if LLM background thread holds the lock.
        // If we can't acquire the lock now, update context on next available opportunity.
        match self.shared.llm.try_lock() {
            Ok(mut llm) => llm.update_context(&text),
            Err(_) => log::debug!("LLM lock busy during commit, skipping context update"),
        }
        self.romaji.reset();
        // Retire any in-flight rerank pass so a late-arriving worker cannot
        // repaint over the just-committed (hidden) preedit — a mode=1
        // ghost that would auto-commit on focus loss = duplicate insertion.
        self.invalidate_rerank_state();
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
        let generation = self.rerank_generation.clone();

        // Bump the rerank epoch: this is a new pass, and any earlier pass's
        // late-arriving result / panic must not touch our slot or inflight
        // flag. Capture the fresh generation into the worker so it can gate
        // its slot-store and panic path on the pass still being current.
        let my_gen = generation.fetch_add(1, Ordering::AcqRel) + 1;

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
                // Wall-clock cap on the whole rerank pass. Per-request
                // timeouts already bound each HTTP call at 1.5 s, but a
                // multi-segment conversion (say 5 segments × 5 candidates)
                // would amplify that to 37.5 s of a bg thread holding
                // the LLM lock — long enough that the *next* keystroke's
                // rerank waits behind it. Match the frontend's own poll
                // budget (~2 s in IBus spawn_rerank_refresh / Fcitx5
                // scheduleRerankRefresh) so we never linger past what
                // the UI is willing to wait for. When the deadline
                // passes we keep remaining segments in original
                // candidate order (partial rerank) rather than dropping
                // them — dictionary-derived candidates are still valid.
                const RERANK_TOTAL_BUDGET: Duration = Duration::from_millis(1800);
                let deadline = Instant::now() + RERANK_TOTAL_BUDGET;
                let mut reranked_segments: Vec<(String, Vec<String>)> = Vec::new();

                for (reading, candidates) in &seg_info {
                    if Instant::now() >= deadline {
                        log::debug!(
                            "LLM rerank budget spent ({}ms) — segment '{}' and remaining kept in original order",
                            RERANK_TOTAL_BUDGET.as_millis(),
                            reading,
                        );
                        preceding_text.push_str(&candidates[0].0);
                        reranked_segments.push((
                            reading.clone(),
                            candidates.iter().map(|(s, _)| s.clone()).collect(),
                        ));
                        continue;
                    }
                    if candidates.len() > 1 && reading.chars().count() >= 2 {
                        let rerank_count = candidates.len().min(LLM_RERANK_TOP_N);
                        // Context = committed text + preceding segments' chosen candidates
                        let context = format!("{}{}", committed_context, preceding_text);
                        // Score each candidate imperatively so we can break on
                        // the wall-clock deadline. Without an inner check, a
                        // hung llama-server let one segment run all 5
                        // top-N HTTP calls (each up to 1.5 s per-request
                        // timeout) — ~7.5 s while holding shared.llm.
                        // The top-of-loop deadline only fires at segment
                        // boundaries, so it was useless for a single-segment
                        // conversion. Check between candidates so the pass
                        // exits at most one 1.5 s HTTP hang past the deadline.
                        let mut top_with_scores: Vec<(usize, f64)> =
                            Vec::with_capacity(rerank_count);
                        let mut budget_hit = false;
                        for i in 0..rerank_count {
                            if Instant::now() >= deadline {
                                budget_hit = true;
                                break;
                            }
                            let (surface, user) = &candidates[i];
                            let llm_score = Self::rerank_llm_score(reading, surface, || {
                                llm.score_with_context(&context, surface)
                            });
                            let rank_base = 1.0 - (i as f64 / rerank_count as f64) * 0.3;
                            let combined =
                                rank_base * 0.4 + llm_score * 0.6 + user * USER_LEARNING_WEIGHT;
                            top_with_scores.push((i, combined));
                        }
                        if budget_hit {
                            // Partial scores would order this segment
                            // against candidates that were never scored.
                            // Keep the original dictionary order and let
                            // subsequent segments fall through the
                            // top-of-loop deadline check.
                            log::debug!(
                                "LLM rerank budget spent mid-segment '{}' — keeping original order",
                                reading,
                            );
                            preceding_text.push_str(&candidates[0].0);
                            reranked_segments.push((
                                reading.clone(),
                                candidates.iter().map(|(s, _)| s.clone()).collect(),
                            ));
                            continue;
                        }
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

            // Gate slot-store and panic-recovery on the pass still being
            // current. Without this check, an old worker that outlived its
            // trigger could either (a) overwrite a newer pass's result in
            // the slot (later discarded by the alignment guard, leaving
            // `rerank_inflight` latched at true), or (b) clear
            // `rerank_inflight` on panic while a newer pass is still
            // running — the frontend then stops polling and misses the
            // valid result.
            let current_gen = generation.load(Ordering::Acquire);
            if current_gen != my_gen {
                log::debug!(
                    "LLM rerank pass {} superseded by {} — dropping result",
                    my_gen,
                    current_gen,
                );
                return;
            }
            match result {
                Ok(reranked) => {
                    let mut slot = result_slot.lock().unwrap_or_else(|e| e.into_inner());
                    // Re-check under the slot lock: a trigger could have fired
                    // between the load above and here. Tagging the result with
                    // `my_gen` lets `apply_llm_rerank` drop it silently if it
                    // has been superseded before the frontend picks it up.
                    if generation.load(Ordering::Acquire) == my_gen {
                        *slot = Some((my_gen, reranked));
                        log::info!("LLM background reranking complete");
                    } else {
                        log::debug!(
                            "LLM rerank pass {} superseded before store — dropping result",
                            my_gen,
                        );
                    }
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

        let (result_gen, reranked) = match reranked {
            // No result yet — leave `rerank_inflight` set so the frontend keeps
            // polling for the pass that is still running.
            Some(pair) => pair,
            None => return false,
        };

        // Drop a result whose pass has been superseded by a newer trigger /
        // commit / cancel. `rerank_inflight` here belongs to whichever pass is
        // current — leave it alone so the frontend keeps polling for that one
        // (or the invalidation already cleared it).
        let current_gen = self.rerank_generation.load(Ordering::Acquire);
        if result_gen != current_gen {
            log::debug!(
                "Discarding superseded LLM rerank result (gen {} vs current {})",
                result_gen,
                current_gen,
            );
            return false;
        }

        let state = match self.conversion.as_mut() {
            Some(s) => s,
            None => {
                // No active conversion to apply to; the pass is moot.
                self.rerank_inflight.store(false, Ordering::Relaxed);
                return false;
            }
        };

        // Defensive: with the generation gate above, a mismatched segmentation
        // shouldn't reach here (any state change that alters bunsetsu
        // boundaries bumps the generation). Keep the alignment check as a
        // safety net; a mismatch now means an assumption above is wrong, not
        // that the pass merely predates a resize.
        let aligned = reranked.len() == state.segments.len()
            && reranked
                .iter()
                .zip(state.segments.iter())
                .all(|((reading, _), seg)| *reading == seg.reading);
        if !aligned {
            log::debug!("Discarding LLM rerank result with mismatched segmentation");
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

    /// Current rerank epoch. Frontends can snapshot this before scheduling a
    /// refresh emit and re-check it right before repainting — a bump between
    /// the two (from a commit / cancel that fired while the refresh was
    /// polling) means the conversion is gone and the pending emit would leave
    /// a mode=1 ghost preedit on screen.
    pub fn rerank_generation(&self) -> u64 {
        self.rerank_generation.load(Ordering::Acquire)
    }

    /// Bump the rerank epoch and clear the pending-result slot / inflight
    /// flag. Called from commit / cancel paths so any late-arriving worker
    /// result is dropped and the frontend stops polling for a pass that is
    /// no longer meaningful. `trigger_llm_rerank` does the equivalent inline
    /// (bump + clear slot) but leaves `inflight=true` for the new pass.
    fn invalidate_rerank_state(&self) {
        self.rerank_generation.fetch_add(1, Ordering::AcqRel);
        *self.llm_rerank_result.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.rerank_inflight.store(false, Ordering::Relaxed);
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

}

/// Return value from `ConversionEngine::process_key`. The dispatcher
/// discards it — `preedit()` provides the same information — but the
/// two variants stay for the `process_key_buffering`/`_produces_kana`
/// tests which pattern-match on them as a state-machine smoke check.
#[derive(Debug, Clone)]
pub enum EngineAction {
    /// Key was buffered (incomplete romaji). Contains current preedit.
    Buffering(String),
    /// Preedit text was updated (kana produced). Contains current preedit.
    UpdatePreedit(String),
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
        // Tag it with the current generation so the epoch gate lets it
        // through — the alignment guard is what's under test here.
        let cur_gen = engine.rerank_generation();
        *engine.llm_rerank_result.lock().unwrap_or_else(|e| e.into_inner()) = Some((
            cur_gen,
            vec![(
                "ZZ-bogus-reading".to_string(),
                vec!["☃".to_string(), "☔".to_string()],
            )],
        ));
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

    /// Regression: a worker whose pass has been superseded (by a newer trigger
    /// or by commit / cancel) must not have its result applied. Without the
    /// generation gate, a late worker could overwrite a newer pass's slot; the
    /// alignment guard would then discard it silently but leave `rerank_inflight`
    /// latched at true, and the frontend would poll for a result that will
    /// never arrive.
    #[test]
    fn apply_rerank_drops_result_from_superseded_pass() {
        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();

        // Snapshot the current generation, then simulate a follow-up trigger
        // that bumps it (as commit / cancel / resize would). A worker still
        // holding the earlier `stale_gen` must not be applied.
        let stale_gen = engine.rerank_generation();
        let live_segs: Vec<String> = engine
            .conversion_state()
            .unwrap()
            .segments
            .iter()
            .map(|s| s.reading.clone())
            .collect();
        engine.rerank_generation.fetch_add(1, Ordering::AcqRel);

        // Inject the stale worker's result: readings match the live layout so
        // only the generation gate can catch it.
        let fake_result: LlmRerankResult = live_segs
            .iter()
            .map(|r| (r.clone(), vec![r.clone()]))
            .collect();
        *engine.llm_rerank_result.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((stale_gen, fake_result));
        engine.rerank_inflight.store(true, Ordering::Relaxed);

        assert!(
            !engine.apply_llm_rerank(),
            "a result tagged with a superseded generation must be dropped",
        );
        assert!(
            engine.rerank_inflight(),
            "dropping a superseded result must not clear inflight for the newer pass",
        );
    }

    /// Regression: commit_conversion must invalidate any in-flight rerank so a
    /// late worker's result cannot be applied to the (now-hidden) conversion.
    /// Without this, the IBus rerank-refresh task would see a fresh result,
    /// call `apply_llm_rerank`, and re-emit a mode=1 preedit — a ghost that
    /// auto-commits on focus loss (duplicate insertion).
    #[test]
    fn commit_conversion_invalidates_inflight_rerank() {
        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        assert!(engine.rerank_inflight(), "start_conversion arms inflight");
        let pre_commit_gen = engine.rerank_generation();

        // Freeze a snapshot of the segment readings so we can inject a result
        // that would pass the alignment guard — only the generation gate (via
        // commit_conversion's invalidate) should keep it out.
        let live_segs: Vec<String> = engine
            .conversion_state()
            .unwrap()
            .segments
            .iter()
            .map(|s| s.reading.clone())
            .collect();

        engine.commit_conversion().expect("commit succeeds");

        assert!(
            !engine.rerank_inflight(),
            "commit must clear inflight so refresh tasks stop polling",
        );
        assert!(
            engine.rerank_generation() > pre_commit_gen,
            "commit must bump the rerank epoch",
        );

        // A worker for the pre-commit pass finishes late and drops its result
        // into the slot. Apply must not touch anything.
        let stale: LlmRerankResult = live_segs
            .iter()
            .map(|r| (r.clone(), vec![r.clone()]))
            .collect();
        *engine.llm_rerank_result.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((pre_commit_gen, stale));

        assert!(
            !engine.apply_llm_rerank(),
            "a late pre-commit result must not be applied post-commit",
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

    /// Regression (bug_list_fable_5_review_2026-08-28 #4): Space on a
    /// mid-syllable buffer like "k" must not destroy the buffer. Before
    /// the fix, start_conversion flushed unconditionally, dropped "k",
    /// returned None, and the next 'a' produced "あ" instead of "か".
    /// The fix makes start_conversion a no-op precheck when output is
    /// empty and buffer isn't a lone "n".
    #[test]
    fn start_conversion_preserves_partial_buffer() {
        let mut engine = ConversionEngine::new();
        engine.process_key('k');
        assert_eq!(engine.preedit(), "k");
        assert!(
            engine.start_conversion().is_none(),
            "start_conversion should decline on empty output + partial buffer",
        );
        assert_eq!(
            engine.preedit(),
            "k",
            "start_conversion must not destroy the pending buffer",
        );
        // Next keystroke should still complete the syllable.
        engine.process_key('a');
        assert_eq!(engine.preedit(), "か");
    }

    /// A lone "n" in the buffer is the one case where flush produces
    /// kana — start_conversion still runs its full pipeline.
    #[test]
    fn start_conversion_flushes_lone_n() {
        let mut engine = ConversionEngine::new();
        engine.process_key('n');
        assert_eq!(engine.preedit(), "n");
        let state = engine.start_conversion().expect("lone n should convert to ん");
        assert!(!state.segments.is_empty());
    }

    /// "kyou" (きょう) — the single segment's top candidate should be
    /// the kanji "今日" (highest dictionary frequency + kanji preference).
    #[test]
    fn convert_basic() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_conversion().expect("start_conversion returned None");
        let seg = &state.segments[0];
        assert!(!seg.candidates.is_empty());
        assert_eq!(seg.candidates[0], "今日");
    }

    /// A multi-segment sentence produces at least one segment whose top
    /// candidate contains a kanji — the dictionary layer must find kanji
    /// candidates for the common words in "きょうはいいてんき".
    #[test]
    fn convert_sentence() {
        let mut engine = ConversionEngine::new();
        for ch in "kyouhaiitenki".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_conversion().expect("start_conversion returned None");
        let has_kanji = state.segments.iter().any(|s| {
            s.candidates.iter().any(|c| {
                c.chars().any(|ch| ('\u{4E00}'..='\u{9FFF}').contains(&ch))
            })
        });
        assert!(
            has_kanji,
            "expected kanji in some segment's candidates: {:?}",
            state.segments.iter().map(|s| &s.candidates).collect::<Vec<_>>()
        );
    }

    /// start_conversion returns None when there is nothing to convert.
    #[test]
    fn convert_empty() {
        let mut engine = ConversionEngine::new();
        assert!(engine.start_conversion().is_none());
    }

    /// commit_conversion clears the composing state — preedit is empty
    /// afterwards.
    #[test]
    fn commit_resets_state() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let committed_text = {
            let state = engine.start_conversion().expect("start_conversion returned None");
            state.segments[0].candidates[state.segments[0].selected].clone()
        };
        let committed = engine.commit_conversion().expect("commit_conversion returned None");
        assert_eq!(committed, committed_text);
        assert_eq!(engine.preedit(), "");
    }

    /// The raw kana "きょう" must not outrank the kanji "今日" in the
    /// candidate list — the dictionary + rerank layers preserve the
    /// standard "kanji first" ordering.
    #[test]
    fn kanji_ranked_above_kana() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_conversion().expect("start_conversion returned None");
        let candidates = &state.segments[0].candidates;
        let kana_pos = candidates.iter().position(|c| c == "きょう");
        let kanji_pos = candidates.iter().position(|c| c == "今日");
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

    /// F9/F10 must preserve a lone trailing consonant sitting in the
    /// romaji buffer (e.g. the "m" in "vim") — flush drops it, so the
    /// F-key path used to lose it and turn "vim" (preedit "ゔぃm") into
    /// "ｖｉ"/"vi" instead of "ｖｉｍ"/"vim".
    #[test]
    fn f_key_preserves_pending_romaji_buffer() {
        for (form, expected) in [(3, "vim"), (4, "ｖｉｍ")] {
            let mut engine = ConversionEngine::new();
            for ch in "vim".chars() {
                engine.process_key(ch);
            }
            assert_eq!(engine.preedit(), "ゔぃm", "preedit should show kana+buffer");
            let state = engine.start_kana_conversion(form)
                .expect("start_kana_conversion returned None");
            assert_eq!(
                state.composed_text(),
                expected,
                "F-key form {form} lost pending 'm'",
            );
        }
    }

    /// Non-romaji forms (F6 hiragana, F7 katakana, F8 halfwidth katakana)
    /// also carry the pending buffer through as-is — the converters pass
    /// non-kana chars through unchanged, so "vim" → "ゔぃm"/"ヴィm"/"ｳﾞィm".
    #[test]
    fn f_key_pending_buffer_flows_through_kana_forms() {
        for (form, expected) in [(0, "ゔぃm"), (1, "ヴィm"), (2, "ｳﾞｨm")] {
            let mut engine = ConversionEngine::new();
            for ch in "vim".chars() {
                engine.process_key(ch);
            }
            let state = engine.start_kana_conversion(form)
                .expect("start_kana_conversion returned None");
            assert_eq!(
                state.composed_text(),
                expected,
                "F-key form {form} composition wrong",
            );
        }
    }

    /// F9/F10 round-trip preserves original case ("VIM" → "ＶＩＭ"/"VIM")
    /// via raw_input tracking. Without it the dispatcher-side lowercase
    /// or the kana→romaji derivation would flatten the case.
    #[test]
    fn f_key_preserves_uppercase() {
        for (form, expected) in [(3, "VIM"), (4, "ＶＩＭ")] {
            let mut engine = ConversionEngine::new();
            for ch in "VIM".chars() {
                engine.process_key(ch);
            }
            let state = engine.start_kana_conversion(form)
                .expect("start_kana_conversion returned None");
            assert_eq!(
                state.composed_text(),
                expected,
                "F-key form {form} did not preserve uppercase",
            );
        }
    }

    /// F9/F10 preserves the original spelling — "shi" stays "shi" rather
    /// than being normalised to "si" via the reverse kana→romaji table.
    #[test]
    fn f_key_preserves_spelling() {
        let mut engine = ConversionEngine::new();
        for ch in "shi".chars() {
            engine.process_key(ch);
        }
        assert_eq!(engine.preedit(), "し");
        let state = engine.start_kana_conversion(3)
            .expect("start_kana_conversion returned None");
        assert_eq!(
            state.composed_text(),
            "shi",
            "F10 lost original spelling (would have returned 'si' via kana derivation)",
        );
    }

    /// Mixed case ("Vim") preserves per-char case in F9/F10.
    #[test]
    fn f_key_preserves_mixed_case() {
        let mut engine = ConversionEngine::new();
        for ch in "Vim".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_kana_conversion(3)
            .expect("start_kana_conversion returned None");
        assert_eq!(state.composed_text(), "Vim");
    }

    /// Pressing F9/F10 a second time in a row (which routes through
    /// `convert_focused_to` instead of `start_kana_conversion`) must
    /// keep the raw-input-based romaji, not derive a fresh "si"/"ｓｉ"
    /// from the kana reading and append it as a new candidate.
    #[test]
    fn f_key_repeat_keeps_raw_input_romaji() {
        for form in [3, 4] {
            let mut engine = ConversionEngine::new();
            for ch in "shi".chars() {
                engine.process_key(ch);
            }
            // First press: start_kana_conversion path.
            let first = engine.start_kana_conversion(form)
                .expect("start_kana_conversion returned None")
                .clone();
            let first_text = first.composed_text();
            let first_candidates = first.segments[0].candidates.clone();
            // Second press: convert_focused_to path.
            let second = engine.convert_focused_to(match form {
                3 => KanaForm::Romaji,
                4 => KanaForm::FullwidthRomaji,
                _ => unreachable!(),
            }).expect("convert_focused_to returned None");
            assert_eq!(
                second.composed_text(), first_text,
                "second F{} press changed the composed text",
                if form == 3 { 10 } else { 9 },
            );
            assert_eq!(
                second.segments[0].candidates, first_candidates,
                "second F{} press appended a spurious kana-derived candidate",
                if form == 3 { 10 } else { 9 },
            );
        }
    }

    /// When raw_input is invalidated (e.g. Backspace popped from the
    /// committed output), F9/F10 falls back to the kana-derived romaji.
    /// Case/spelling can no longer be recovered — best-effort only.
    #[test]
    fn f_key_falls_back_when_raw_input_invalidated() {
        let mut engine = ConversionEngine::new();
        for ch in "VIM".chars() {
            engine.process_key(ch);
        }
        // Delete the pending "m", then also delete a kana from output —
        // this invalidates raw_input tracking.
        assert!(engine.delete_last(), "delete pending m");
        assert!(engine.delete_last(), "delete kana from output");
        // Retype 'i' — buffer has 'i', but raw_input stays None.
        engine.process_key('i');
        // Now F10 should fall back to kana-derived romaji (lowercase).
        let state = engine.start_kana_conversion(3)
            .expect("start_kana_conversion returned None");
        // The fallback path uses the kana + lowercase pending buffer.
        // Just verify it doesn't panic and returns something non-empty.
        assert!(!state.composed_text().is_empty());
    }

    /// Regression (bug_list_fable_5_review_2026-08-28 #2): a resize
    /// that turns a single-segment conversion into multiple segments
    /// must invalidate raw_input. Otherwise F9/F10 on any of the
    /// resulting segments would paste the whole pre-resize romaji
    /// spelling into that single focused segment (kyou → Shift+Left
    /// → F10 used to yield "kyouう" or "きょkyou").
    #[test]
    fn resize_invalidates_raw_input_for_f_key_fallback() {
        let mut engine = ConversionEngine::new();
        for ch in "kyou".chars() {
            engine.process_key(ch);
        }
        let state = engine.start_conversion().expect("start_conversion");
        assert_eq!(state.segments.len(), 1, "test setup: expected 1 segment");
        assert!(state.raw_input.is_some(), "raw_input should snapshot the single-segment romaji");
        engine.resize_segment(-1).expect("resize_segment shrink");
        let after = engine.conversion_state().expect("conversion cleared");
        assert!(
            after.segments.len() >= 2,
            "resize should have produced multiple segments, got {}",
            after.segments.len(),
        );
        assert!(
            after.raw_input.is_none(),
            "resize must null raw_input once it invalidates the single-segment invariant",
        );
        let focused = engine
            .convert_focused_to(KanaForm::Romaji)
            .expect("convert_focused_to Romaji");
        let focused_seg = &focused.segments[focused.focus];
        let selected = &focused_seg.candidates[focused_seg.selected];
        assert!(
            !selected.eq_ignore_ascii_case("kyou"),
            "F10 on a resized segment must derive romaji from its own reading, \
             not paste the whole pre-resize spelling; got {selected:?}",
        );
    }

    /// A user resize that changes segmentation is recorded, and the
    /// same kana next time comes back pre-split the way the user last
    /// left it. Uses the hermetic core so learning + start_conversion
    /// go through the full pipeline without a live LLM.
    #[test]
    fn resegmentation_is_learned_and_reapplied() {
        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        // First conversion: capture whatever DP gave us.
        let dp_boundaries = {
            let state = engine.start_conversion().expect("start_conversion returned None");
            boundaries_of(&state.segments)
        };
        // Force a resize so the final layout differs from the DP one.
        // Extending the focused (leftmost) segment by one char is enough
        // to shift every downstream boundary.
        engine.resize_segment(1).expect("resize_segment returned None");
        let resized_boundaries = boundaries_of(
            &engine.conversion_state().expect("conversion cleared").segments,
        );
        assert_ne!(
            dp_boundaries, resized_boundaries,
            "resize should have changed boundaries; test setup broken",
        );
        engine.commit_conversion().expect("commit_conversion returned None");

        // Re-type the same kana. The learned segmentation should now
        // come back before the user needs to resize again.
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        let replayed = {
            let state = engine.start_conversion().expect("start_conversion returned None");
            boundaries_of(&state.segments)
        };
        assert_eq!(
            replayed, resized_boundaries,
            "learned segmentation should be re-applied on the same kana",
        );
        // initial_boundaries snapshot is the learned layout, so a subsequent
        // commit without further resizing does NOT re-record (idempotent).
        let state = engine.conversion_state().unwrap();
        assert_eq!(state.initial_boundaries, replayed);
    }

    /// Learning is exact-match by design (A案): a slightly different
    /// kana ("amegafutta" vs the learned "amegafuru") must fall back to
    /// the DP segmenter, not inherit the learned boundaries.
    #[test]
    fn resegmentation_learning_is_exact_match_only() {
        let mut engine = ConversionEngine::with_shared(SharedCore::new_hermetic());
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        engine.start_conversion();
        engine.resize_segment(1);
        let resized = boundaries_of(&engine.conversion_state().unwrap().segments);
        engine.commit_conversion();

        // Different kana — the DP result stands.
        for ch in "amegafutta".chars() {
            engine.process_key(ch);
        }
        let other = engine.start_conversion().expect("start_conversion returned None");
        assert_ne!(
            boundaries_of(&other.segments),
            resized,
            "learning must not bleed across kanas — only exact matches",
        );
    }

    /// Background rerank must honour the wall-clock budget so a slow LLM
    /// server can't amplify per-request timeouts across N segments into
    /// a multi-second bg thread holding the LLM lock. With a mock scorer
    /// that sleeps 400 ms per call, a multi-segment reading of
    /// "amegafuru" would (5 candidates × 3 segments × 400 ms = 6 s)
    /// blow past the 1800 ms budget; the pass must finish by ~2 s and
    /// still deliver a usable (possibly partial) result.
    #[test]
    fn rerank_respects_wall_clock_budget() {
        use crate::core::llm::LlmScorer;
        use std::time::{Duration, Instant};

        struct SlowScorer;
        impl LlmScorer for SlowScorer {
            fn score(&self, _context: &str, _candidate: &str) -> f64 {
                std::thread::sleep(Duration::from_millis(400));
                0.5
            }
            fn warm_cache(&self, _context: &str) {}
        }

        let mut engine =
            ConversionEngine::with_shared(SharedCore::new_eval(Box::new(SlowScorer)));
        for ch in "amegafuru".chars() {
            engine.process_key(ch);
        }
        let start = Instant::now();
        engine.start_conversion();

        // Poll for the rerank to land. Budget is 1800 ms + one in-flight
        // segment's slop (up to ~5 * 400 ms = 2 s). Give the poll loop
        // 5 s of headroom before we call it a hang.
        let deadline = start + Duration::from_secs(5);
        while !engine.has_llm_rerank_result() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let elapsed = start.elapsed();
        assert!(
            engine.has_llm_rerank_result(),
            "rerank never completed within 5 s (elapsed {}ms)",
            elapsed.as_millis(),
        );
        // Wall-clock cap is 1800 ms + one segment slop; 4 s is a very
        // loose upper bound that still catches the "amplification with
        // no budget at all" regression (would be ~6 s here).
        assert!(
            elapsed < Duration::from_millis(4000),
            "rerank exceeded budget: {}ms",
            elapsed.as_millis(),
        );

        // Partial result is still fully-shaped (every segment present,
        // in the same order) so `apply_llm_rerank` will accept it.
        let live_segs: Vec<String> = engine
            .conversion_state()
            .unwrap()
            .segments
            .iter()
            .map(|s| s.reading.clone())
            .collect();
        let result = engine
            .llm_rerank_result
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let result = result.as_ref().expect("result should be populated");
        let segments = &result.1;
        assert_eq!(
            segments.len(),
            live_segs.len(),
            "partial result must still cover every segment",
        );
        for (i, (reading, _)) in segments.iter().enumerate() {
            assert_eq!(reading, &live_segs[i]);
        }
    }

    /// Regression [9]: the wall-clock budget must fire mid-segment, not only
    /// between segments. Without an in-candidate deadline check, a slow
    /// llama-server let a 1-segment (or first-segment) conversion burn its
    /// per-request timeout (up to 1.5 s) across all 5 top-N candidates —
    /// ~7.5 s while holding `shared.llm`, defeating the budget entirely for
    /// the pathological single-segment case.
    #[test]
    fn rerank_budget_fires_between_candidates() {
        use crate::core::llm::LlmScorer;
        use std::time::{Duration, Instant};

        struct VerySlowScorer;
        impl LlmScorer for VerySlowScorer {
            fn score(&self, _context: &str, _candidate: &str) -> f64 {
                std::thread::sleep(Duration::from_millis(700));
                0.5
            }
            fn warm_cache(&self, _context: &str) {}
        }

        let mut engine =
            ConversionEngine::with_shared(SharedCore::new_eval(Box::new(VerySlowScorer)));
        // A short reading likely to segment as a single bunsetsu with many
        // homophone candidates (橋 / 端 / 箸 …). Even if it splits, the
        // FIRST segment will still exercise the inner-loop deadline path
        // the same way — the bug is about not checking between candidates.
        for ch in "hashi".chars() {
            engine.process_key(ch);
        }
        let start = Instant::now();
        engine.start_conversion();

        let poll_deadline = start + Duration::from_secs(6);
        while !engine.has_llm_rerank_result() && Instant::now() < poll_deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let elapsed = start.elapsed();
        assert!(
            engine.has_llm_rerank_result(),
            "rerank never completed within 6 s (elapsed {}ms)",
            elapsed.as_millis(),
        );
        // Without the inner-loop check, 5 candidates × 700 ms = 3500 ms of
        // sequential score() calls per segment. With it, we break as soon
        // as the 1800 ms budget passes, so worst case is one in-flight
        // slop past the deadline (~2500 ms).
        assert!(
            elapsed < Duration::from_millis(3300),
            "single-segment rerank blew inner-loop budget: {}ms",
            elapsed.as_millis(),
        );
    }
}
