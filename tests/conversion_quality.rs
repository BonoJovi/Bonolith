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
use bonolith::engine::{ConversionEngine, SharedCore};
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

#[test]
fn evaluate_top1_accuracy() {
    let cases = load_cases();
    let shared = SharedCore::new_hermetic();

    let pct = |num: u32, den: u32| -> f64 {
        if den == 0 {
            0.0
        } else {
            100.0 * num as f64 / den as f64
        }
    };

    let mut overall = Top1Stats::default();
    let mut by_pos: std::collections::BTreeMap<String, Top1Stats> = std::collections::BTreeMap::new();
    let mut by_cat: std::collections::BTreeMap<String, Top1Stats> = std::collections::BTreeMap::new();
    let mut yes_failures: Vec<String> = Vec::new();

    eprintln!("\n=== Top-1 conversion accuracy (hermetic: MockScorer, no learning) ===");
    for case in &cases {
        let want: String = case.expected.concat();
        let got = convert_top1(&shared, &case.input_hiragana);
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

        overall.record(exact, phrase_hit, phrase_total);
        by_pos
            .entry(case.pos_solvable.clone())
            .or_default()
            .record(exact, phrase_hit, phrase_total);
        by_cat
            .entry(case.category.clone())
            .or_default()
            .record(exact, phrase_hit, phrase_total);

        if !exact && case.pos_solvable == "yes" {
            yes_failures.push(format!("{} {:?} -> got {:?}", case.id, want, got));
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
        overall.exact,
        overall.total,
        pct(overall.exact, overall.total),
    );
    eprintln!(
        "Phrase recall:  {}/{} ({:.1}%)   [expected phrases present in order]",
        overall.phrase_hit,
        overall.phrase_total,
        pct(overall.phrase_hit, overall.phrase_total),
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
    print_bucket("pos_solvable", &by_pos);
    print_bucket("category", &by_cat);

    // Gate: the `yes` bucket is the POS/dictionary-solvable set — it must not
    // regress. `partial`/`no` lean on the real LLM and are tracked in the live
    // layer, so they are informational here. Threshold is calibrated to the
    // current hermetic baseline; tighten it as dictionary/POS quality improves.
    let yes = by_pos.get("yes").copied().unwrap_or_default();
    // Baseline 2026-06-21: yes=7/13 (53.8%). Floor just below it, so any
    // single-case regression in the solvable bucket (→ 6/13 = 46.2%) trips CI.
    // Raise as dictionary/POS quality improves.
    const YES_GATE_PCT: f64 = 53.0;
    let yes_pct = pct(yes.exact, yes.total);
    eprintln!(
        "\nGate: pos_solvable=yes top-1 = {:.1}% (floor {:.1}%)",
        yes_pct, YES_GATE_PCT,
    );
    if !yes_failures.is_empty() {
        eprintln!("yes-bucket misses:");
        for f in &yes_failures {
            eprintln!("  {}", f);
        }
    }
    assert!(
        yes_pct >= YES_GATE_PCT,
        "pos_solvable=yes top-1 accuracy {:.1}% fell below floor {:.1}%",
        yes_pct,
        YES_GATE_PCT,
    );
}
