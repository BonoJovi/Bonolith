//! Conversion-quality evaluation harness against tests/conversion_cases/cases.jsonl.
//!
//! Primary metric: **bunsetsu refinement check**.
//! Each case carries `expected_readings` (hiragana per phrase); their cumulative
//! char counts define the human-expected phrase boundaries. Bonolith's segmenter
//! returns morpheme-level boundaries — finer than phrases. We declare a case
//! "boundary-correct" when every expected boundary appears in Bonolith's boundary
//! set (i.e. Bonolith is a *refinement* of the expected segmentation). This lines
//! up the two granularities instead of demanding equal segment counts.
//!
//! Secondary signals (informational, not gating):
//! - `boundary_recall` — fraction of expected boundaries present in Bonolith output
//! - `over_segments` — extra Bonolith splits inside expected phrases (lower = closer
//!   to phrase-level; high values still pass refinement)
//! - `segs_parity` — legacy segment-count parity, kept for historical comparison
//!
//! This is a v2.x harness for the IPADIC POS-connection-cost initiative.
//!
//! Run with:
//!   cargo test --test conversion_quality -- --nocapture

use bonolith::core::dictionary::Dictionary;
use bonolith::engine::{ConversionEngine, ConversionState, SharedCore};
use std::fs;
use std::time::{Duration, Instant};

const CASES_PATH: &str = "tests/conversion_cases/cases.jsonl";

#[derive(serde::Deserialize)]
struct Case {
    id: String,
    input_hiragana: String,
    expected: Vec<String>,
    expected_readings: Vec<String>,
    pos_solvable: String,
    #[serde(default)]
    category: String,
}

fn load_cases() -> Vec<Case> {
    let content = fs::read_to_string(CASES_PATH).expect("read cases.jsonl");
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse case"))
        .collect()
}

/// Cumulative char offsets after each chunk: returns `[len(c0), len(c0)+len(c1), …]`.
/// The terminal offset (== full length) is included; the leading 0 is not.
fn cumulative_char_offsets(chunks: &[String]) -> Vec<usize> {
    let mut acc = 0usize;
    chunks
        .iter()
        .map(|c| {
            acc += c.chars().count();
            acc
        })
        .collect()
}

#[derive(Default, Clone, Copy)]
struct BucketStats {
    total: u32,
    refinement_pass: u32,
    parity_pass: u32,
    boundary_recall_num: u32,
    boundary_recall_den: u32,
    over_segments_sum: u32,
}

#[test]
fn evaluate_conversion_cases() {
    let path = "tests/conversion_cases/cases.jsonl";
    let content = fs::read_to_string(path).expect("read cases.jsonl");
    let dict = Dictionary::new();

    let mut buckets: std::collections::BTreeMap<String, BucketStats> =
        std::collections::BTreeMap::new();
    let mut total = 0u32;
    let mut refinement_pass = 0u32;
    let mut parity_pass = 0u32;
    let mut recall_num_total = 0u32;
    let mut recall_den_total = 0u32;
    let mut over_segments_total = 0u32;

    eprintln!("\n=== Per-case results ===");
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(line).expect("parse case");

        // Sanity: expected_readings must reconstruct input_hiragana exactly.
        let joined: String = case.expected_readings.concat();
        assert_eq!(
            joined, case.input_hiragana,
            "{}: expected_readings concat mismatch (got {:?}, want {:?})",
            case.id, joined, case.input_hiragana
        );
        assert_eq!(
            case.expected.len(),
            case.expected_readings.len(),
            "{}: expected and expected_readings length mismatch",
            case.id
        );

        let segments = dict.segment(&case.input_hiragana);

        // Internal expected boundaries: cumulative reading offsets, dropping
        // the terminal one (== full length, trivially shared by any segmentation).
        let mut exp_offsets = cumulative_char_offsets(&case.expected_readings);
        exp_offsets.pop();
        let exp_boundaries: std::collections::BTreeSet<usize> = exp_offsets.into_iter().collect();

        // Bonolith internal boundaries: cumulative segment lengths minus terminal.
        let mut bonolith_offsets = Vec::with_capacity(segments.len());
        let mut acc = 0usize;
        for s in &segments {
            acc += s.reading.chars().count();
            bonolith_offsets.push(acc);
        }
        bonolith_offsets.pop();
        let bonolith_boundaries: std::collections::BTreeSet<usize> =
            bonolith_offsets.into_iter().collect();

        let exp_count = exp_boundaries.len() as u32;
        let matched = exp_boundaries.intersection(&bonolith_boundaries).count() as u32;
        let is_refinement = exp_boundaries.is_subset(&bonolith_boundaries);
        let over_segments = bonolith_boundaries.difference(&exp_boundaries).count();
        let parity_ok = segments.len() == case.expected.len();

        total += 1;
        if is_refinement {
            refinement_pass += 1;
        }
        if parity_ok {
            parity_pass += 1;
        }
        recall_num_total += matched;
        recall_den_total += exp_count;
        over_segments_total += over_segments as u32;

        let b = buckets.entry(case.pos_solvable.clone()).or_default();
        b.total += 1;
        if is_refinement {
            b.refinement_pass += 1;
        }
        if parity_ok {
            b.parity_pass += 1;
        }
        b.boundary_recall_num += matched;
        b.boundary_recall_den += exp_count;
        b.over_segments_sum += over_segments as u32;

        let readings: Vec<&str> = segments.iter().map(|s| s.reading.as_str()).collect();
        let refine_tag = if is_refinement { "REFINE" } else { "miss  " };
        let parity_tag = if parity_ok { "P" } else { "·" };
        let recall_str = if exp_count == 0 {
            "n/a".to_string()
        } else {
            format!("{}/{}", matched, exp_count)
        };
        eprintln!(
            "{:>10} [{:>7} / {:>14}] {} {} recall={:<5} over={}  input={:<30} got={:?}",
            case.id,
            case.pos_solvable,
            case.category,
            refine_tag,
            parity_tag,
            recall_str,
            over_segments,
            case.input_hiragana,
            readings,
        );
    }

    let pct = |num: u32, den: u32| -> f64 {
        if den == 0 {
            0.0
        } else {
            100.0 * num as f64 / den as f64
        }
    };

    eprintln!("\n=== Summary ===");
    eprintln!(
        "Refinement pass:   {}/{} ({:.1}%)   [Bonolith boundaries ⊇ expected]",
        refinement_pass,
        total,
        pct(refinement_pass, total),
    );
    eprintln!(
        "Boundary recall:   {}/{} ({:.1}%)   [internal expected boundaries hit]",
        recall_num_total,
        recall_den_total,
        pct(recall_num_total, recall_den_total),
    );
    eprintln!(
        "Segs-count parity: {}/{} ({:.1}%)   [legacy metric, kept for comparison]",
        parity_pass,
        total,
        pct(parity_pass, total),
    );
    let avg_over = if total > 0 {
        over_segments_total as f64 / total as f64
    } else {
        0.0
    };
    eprintln!(
        "Over-segments:     {} total, {:.2} avg/case   [Bonolith boundaries that fall inside an expected phrase]",
        over_segments_total, avg_over,
    );

    eprintln!("\n=== By pos_solvable ===");
    eprintln!("{:<8}  refinement     boundary-recall  over  parity", "bucket");
    for (bucket, s) in &buckets {
        let bucket_avg_over = if s.total > 0 {
            s.over_segments_sum as f64 / s.total as f64
        } else {
            0.0
        };
        eprintln!(
            "{:<8}  {:>3}/{:>3} ({:>5.1}%)  {:>3}/{:>3} ({:>5.1}%)  {:>4.2}  {:>3}/{:>3} ({:>5.1}%)",
            bucket,
            s.refinement_pass,
            s.total,
            pct(s.refinement_pass, s.total),
            s.boundary_recall_num,
            s.boundary_recall_den,
            pct(s.boundary_recall_num, s.boundary_recall_den),
            bucket_avg_over,
            s.parity_pass,
            s.total,
            pct(s.parity_pass, s.total),
        );
    }

    // PoC harness — visibility only. The CONN tuning cycle will gate on this
    // once the metric stabilizes, but for now regressions shouldn't break CI.
}

// ---------------------------------------------------------------------------
// Top-1 conversion accuracy suite (hermetic layer).
//
// Unlike `evaluate_conversion_cases` (which only checks bunsetsu *boundaries*),
// this drives the **full production conversion pipeline** — segmentation,
// effective-score ordering, and LLM rerank — through `ConversionEngine`, then
// checks the actual top-1 *surface* against each case's `expected` kanji.
//
// Hermetic by construction: `SharedCore::new_hermetic` wires the embedded
// dictionary to an empty `UserScorer` (no learned history) and the
// deterministic `MockScorer` (no llama-server). Results are reproducible on any
// machine and in CI. Semantic-only cases (`pos_solvable == "no"`, e.g. はし/
// きしゃ disambiguation) cannot be solved without a real LLM, so they are
// reported but excluded from the gate — the live layer (a future `#[ignore]`
// test wired to HttpLlamaScorer) is what measures those.
// ---------------------------------------------------------------------------

/// Run one reading through the full pipeline and return the top-1 surface.
///
/// A fresh engine per case (sharing the hermetic core) keeps romaji/conversion
/// state clean; the shared core's user-learning and LLM context stay empty
/// because we never commit. The background rerank uses the deterministic
/// MockScorer, so the bounded wait below always resolves quickly.
fn convert_top1(shared: &std::sync::Arc<SharedCore>, kana: &str) -> String {
    let mut engine = ConversionEngine::with_shared(shared.clone());
    engine.append_raw(kana);
    if engine.start_conversion().is_none() {
        return String::new();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while !engine.has_llm_rerank_result() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    engine.apply_llm_rerank();
    engine
        .conversion_state()
        .map(|s| s.composed_text())
        .unwrap_or_default()
}

#[derive(Default, Clone, Copy)]
struct Top1Stats {
    total: u32,
    exact: u32,
    phrase_hit: u32,
    phrase_total: u32,
}

impl Top1Stats {
    fn record(&mut self, exact: bool, phrase_hit: u32, phrase_total: u32) {
        self.total += 1;
        if exact {
            self.exact += 1;
        }
        self.phrase_hit += phrase_hit;
        self.phrase_total += phrase_total;
    }
}

fn pct(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

/// Aggregated scorecard for one suite run.
struct Scorecard {
    overall: Top1Stats,
    by_pos: std::collections::BTreeMap<String, Top1Stats>,
    by_cat: std::collections::BTreeMap<String, Top1Stats>,
    /// `(id, want, got)` for every case that missed top-1 exact.
    misses: Vec<(String, String, String)>,
}

impl Scorecard {
    fn bucket(&self, by_pos: &str) -> Top1Stats {
        self.by_pos.get(by_pos).copied().unwrap_or_default()
    }
}

/// Drive the full conversion pipeline over every case against `shared`,
/// printing a per-case line and bucketed scorecard. The only difference
/// between the hermetic and live layers is the scorer wired into `shared`.
fn run_top1_suite(shared: &std::sync::Arc<SharedCore>, label: &str) -> Scorecard {
    let cases = load_cases();
    let mut card = Scorecard {
        overall: Top1Stats::default(),
        by_pos: std::collections::BTreeMap::new(),
        by_cat: std::collections::BTreeMap::new(),
        misses: Vec::new(),
    };

    eprintln!("\n=== Top-1 conversion accuracy ({label}) ===");
    for case in &cases {
        let want: String = case.expected.concat();
        let got = convert_top1(shared, &case.input_hiragana);
        let exact = got == want;

        // Phrase recall: expected surfaces present (in order) as the output is
        // scanned left-to-right. Granularity-independent partial credit.
        let mut cursor = 0usize;
        let mut phrase_hit = 0u32;
        for phrase in &case.expected {
            if let Some(rel) = got[cursor..].find(phrase.as_str()) {
                phrase_hit += 1;
                cursor += rel + phrase.len();
            }
        }
        let phrase_total = case.expected.len() as u32;

        card.overall.record(exact, phrase_hit, phrase_total);
        card.by_pos
            .entry(case.pos_solvable.clone())
            .or_default()
            .record(exact, phrase_hit, phrase_total);
        card.by_cat
            .entry(case.category.clone())
            .or_default()
            .record(exact, phrase_hit, phrase_total);
        if !exact {
            card.misses
                .push((case.id.clone(), want.clone(), got.clone()));
        }

        let tag = if exact { "OK  " } else { "MISS" };
        eprintln!(
            "{:>10} [{:>7} / {:>12}] {} phrases={}/{}  want={:<18} got={}",
            case.id, case.pos_solvable, case.category, tag, phrase_hit, phrase_total, want, got,
        );
    }

    eprintln!("\n=== Summary ===");
    eprintln!(
        "Top-1 exact:    {}/{} ({:.1}%)   [full-sentence surface == expected]",
        card.overall.exact,
        card.overall.total,
        pct(card.overall.exact, card.overall.total),
    );
    eprintln!(
        "Phrase recall:  {}/{} ({:.1}%)   [expected phrases present in order]",
        card.overall.phrase_hit,
        card.overall.phrase_total,
        pct(card.overall.phrase_hit, card.overall.phrase_total),
    );

    let print_bucket = |title: &str, m: &std::collections::BTreeMap<String, Top1Stats>| {
        eprintln!("\n=== By {} ===", title);
        eprintln!("{:<14}  exact            phrase-recall", "bucket");
        for (k, s) in m {
            eprintln!(
                "{:<14}  {:>3}/{:>3} ({:>5.1}%)  {:>3}/{:>3} ({:>5.1}%)",
                k,
                s.exact,
                s.total,
                pct(s.exact, s.total),
                s.phrase_hit,
                s.phrase_total,
                pct(s.phrase_hit, s.phrase_total),
            );
        }
    };
    print_bucket("pos_solvable", &card.by_pos);
    print_bucket("category", &card.by_cat);

    card
}

/// Hermetic layer (CI gate): the deterministic MockScorer measures
/// dictionary + POS quality with no llama-server and no learned history.
#[test]
fn evaluate_top1_accuracy() {
    let shared = SharedCore::new_hermetic();
    let card = run_top1_suite(&shared, "hermetic: MockScorer, no learning");

    // Gate: the `yes` bucket is the POS/dictionary-solvable set — it must not
    // regress. `partial`/`no` lean on the real LLM and are measured in the live
    // layer, so they are informational here.
    let yes = card.bucket("yes");
    // Baseline 2026-06-21: yes=7/13 (53.8%). Floor just below it, so any
    // single-case regression in the solvable bucket (→ 6/13 = 46.2%) trips CI.
    // Raise as dictionary/POS quality improves.
    const YES_GATE_PCT: f64 = 53.0;
    let yes_pct = pct(yes.exact, yes.total);
    eprintln!(
        "\nGate: pos_solvable=yes top-1 = {:.1}% (floor {:.1}%)",
        yes_pct, YES_GATE_PCT,
    );
    assert!(
        yes_pct >= YES_GATE_PCT,
        "pos_solvable=yes top-1 accuracy {:.1}% fell below floor {:.1}%",
        yes_pct,
        YES_GATE_PCT,
    );
}

/// Live layer (informational, needs a running llama-server): the real
/// HttpLlamaScorer measures semantic disambiguation — the `partial`/`no`
/// buckets (はし/きしゃ/公開 …) that MockScorer cannot solve. Not a CI gate:
/// the score depends on the served model, so it reports rather than asserts.
///
/// Run with a server up:
///   cargo test --test conversion_quality -- --ignored --nocapture live
#[test]
#[ignore]
fn evaluate_top1_accuracy_live() {
    use bonolith::core::llm::HttpLlamaScorer;

    let scorer = match HttpLlamaScorer::from_default_endpoint() {
        Some(s) => s,
        None => {
            eprintln!("no llama-server reachable; skipping live conversion-quality suite");
            return;
        }
    };
    let shared = SharedCore::new_eval(Box::new(scorer));
    let card = run_top1_suite(&shared, "live: HttpLlamaScorer");

    // The live layer earns its keep on the semantic buckets the hermetic layer
    // must skip. Surface those numbers explicitly so a model/endpoint change is
    // easy to spot, then list every miss for inspection.
    let partial = card.bucket("partial");
    let no = card.bucket("no");
    eprintln!(
        "\nSemantic buckets (LLM-dependent): partial {}/{} ({:.1}%)  no {}/{} ({:.1}%)",
        partial.exact,
        partial.total,
        pct(partial.exact, partial.total),
        no.exact,
        no.total,
        pct(no.exact, no.total),
    );
    if !card.misses.is_empty() {
        eprintln!("misses:");
        for (id, want, got) in &card.misses {
            eprintln!("  {} want={:?} got={:?}", id, want, got);
        }
    }
}

// ---------------------------------------------------------------------------
// Oracle-prefix ceiling (live, informational).
//
// Quantifies the upside of incremental re-ranking — re-evaluating the rest of
// the sentence each time the user confirms a word. We drive the REAL engine
// confirm-and-continue: at each expected bunsetsu we convert the *remaining*
// reading with the LLM context built from prior confirmations, then commit a
// confirmation surface so it becomes left context for the next round. The two
// modes differ only in that surface:
//   - oracle: the correct expected surface (a perfect user confirmation)
//   - self:   the system's own leading output (today's single-pass behaviour)
// The gap on downstream bunsetsu (j>=1, where left context exists) is the
// ceiling of the feature. MockScorer ignores context, so this is live-only and
// never gates — it informs whether the feature is worth building.
// ---------------------------------------------------------------------------

/// Surface produced for the first `target_chars` of reading, or None if a
/// segment boundary straddles that offset (then we can't isolate the bunsetsu).
fn leading_surface(state: &ConversionState, target_chars: usize) -> Option<String> {
    let mut acc = 0usize;
    let mut surf = String::new();
    for seg in &state.segments {
        acc += seg.reading.chars().count();
        surf.push_str(&seg.candidates[seg.selected]);
        if acc == target_chars {
            return Some(surf);
        }
        if acc > target_chars {
            return None;
        }
    }
    None
}

/// Confirm-and-continue over one case. Returns, per bunsetsu, `Some(hit)` when
/// the leading output is alignable (so measurable), else `None`.
fn confirm_and_continue(case: &Case, oracle: bool) -> Vec<Option<bool>> {
    let scorer = bonolith::core::llm::HttpLlamaScorer::from_default_endpoint()
        .expect("llama-server checked reachable by caller");
    let shared = SharedCore::new_eval(Box::new(scorer));
    let mut engine = ConversionEngine::with_shared(shared);

    let mut out = Vec::with_capacity(case.expected.len());
    for j in 0..case.expected.len() {
        let remaining: String = case.expected_readings[j..].concat();
        engine.append_raw(&remaining);
        if engine.start_conversion().is_none() {
            engine.reset();
            out.push(None);
            continue;
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        while !engine.has_llm_rerank_result() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        engine.apply_llm_rerank();

        let target = case.expected_readings[j].chars().count();
        let got = engine
            .conversion_state()
            .and_then(|s| leading_surface(s, target));
        out.push(got.as_ref().map(|g| g == &case.expected[j]));

        // Confirm a surface so it becomes left context: the correct one in
        // oracle mode, the system's own leading output otherwise (falling back
        // to correct when unalignable, so context stays sane either way).
        let confirm = if oracle {
            case.expected[j].clone()
        } else {
            got.unwrap_or_else(|| case.expected[j].clone())
        };
        engine.clear_conversion();
        engine.commit(&confirm);
    }
    out
}

#[derive(Default, Clone, Copy)]
struct CeilStats {
    aligned: u32,
    oracle_hit: u32,
    self_hit: u32,
}

#[test]
#[ignore]
fn measure_oracle_prefix_ceiling_live() {
    if bonolith::core::llm::HttpLlamaScorer::from_default_endpoint().is_none() {
        eprintln!("no llama-server reachable; skipping oracle-prefix ceiling");
        return;
    }
    let cases = load_cases();

    let mut overall = CeilStats::default();
    let mut by_pos: std::collections::BTreeMap<String, CeilStats> = std::collections::BTreeMap::new();
    let mut wins: Vec<String> = Vec::new();

    eprintln!("\n=== Oracle-prefix ceiling (downstream bunsetsu, j>=1) ===");
    for case in &cases {
        if case.expected.len() < 2 {
            continue; // no downstream position to measure
        }
        let oracle = confirm_and_continue(case, true);
        let zelf = confirm_and_continue(case, false);

        for j in 1..case.expected.len() {
            // Segmentation of the remaining reading is context-independent, so
            // alignment matches across modes; require both measurable.
            if let (Some(oh), Some(sh)) = (oracle[j], zelf[j]) {
                let b = by_pos.entry(case.pos_solvable.clone()).or_default();
                overall.aligned += 1;
                b.aligned += 1;
                if oh {
                    overall.oracle_hit += 1;
                    b.oracle_hit += 1;
                }
                if sh {
                    overall.self_hit += 1;
                    b.self_hit += 1;
                }
                if oh && !sh {
                    wins.push(format!(
                        "{} bunsetsu[{}] {:?} fixed by correct left context",
                        case.id, j, case.expected[j]
                    ));
                }
            }
        }
    }

    let pct = |n: u32, d: u32| if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 };
    eprintln!(
        "\nDownstream aligned positions: {}",
        overall.aligned
    );
    eprintln!(
        "  self  (current single-pass): {}/{} ({:.1}%)",
        overall.self_hit,
        overall.aligned,
        pct(overall.self_hit, overall.aligned),
    );
    eprintln!(
        "  oracle (perfect left context): {}/{} ({:.1}%)   <- ceiling",
        overall.oracle_hit,
        overall.aligned,
        pct(overall.oracle_hit, overall.aligned),
    );
    eprintln!(
        "  ceiling gain: +{} positions (+{:.1} pts)",
        overall.oracle_hit.saturating_sub(overall.self_hit),
        pct(overall.oracle_hit, overall.aligned) - pct(overall.self_hit, overall.aligned),
    );

    eprintln!("\nBy pos_solvable (self -> oracle):");
    for (k, s) in &by_pos {
        eprintln!(
            "  {:<8} {}/{} ({:.1}%) -> {}/{} ({:.1}%)",
            k,
            s.self_hit,
            s.aligned,
            pct(s.self_hit, s.aligned),
            s.oracle_hit,
            s.aligned,
            pct(s.oracle_hit, s.aligned),
        );
    }

    if wins.is_empty() {
        eprintln!("\nNo downstream position flips on correct left context — feature looks inert.");
    } else {
        eprintln!("\nPositions fixed purely by correct left context ({}):", wins.len());
        for w in &wins {
            eprintln!("  {}", w);
        }
    }
}
