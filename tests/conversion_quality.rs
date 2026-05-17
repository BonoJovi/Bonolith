//! Conversion-quality evaluation harness against tests/conversion_cases/cases.jsonl.
//!
//! Primary metric: **bunsetsu refinement check**.
//! Each case carries `expected_readings` (hiragana per phrase); their cumulative
//! char counts define the human-expected phrase boundaries. JaIM's segmenter
//! returns morpheme-level boundaries — finer than phrases. We declare a case
//! "boundary-correct" when every expected boundary appears in JaIM's boundary
//! set (i.e. JaIM is a *refinement* of the expected segmentation). This lines
//! up the two granularities instead of demanding equal segment counts.
//!
//! Secondary signals (informational, not gating):
//! - `boundary_recall` — fraction of expected boundaries present in JaIM output
//! - `over_segments` — extra JaIM splits inside expected phrases (lower = closer
//!   to phrase-level; high values still pass refinement)
//! - `segs_parity` — legacy segment-count parity, kept for historical comparison
//!
//! This is a v2.x harness for the IPADIC POS-connection-cost initiative.
//!
//! Run with:
//!   cargo test --test conversion_quality -- --nocapture

use jaim::core::dictionary::Dictionary;
use std::fs;

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

        // JaIM internal boundaries: cumulative segment lengths minus terminal.
        let mut jaim_offsets = Vec::with_capacity(segments.len());
        let mut acc = 0usize;
        for s in &segments {
            acc += s.reading.chars().count();
            jaim_offsets.push(acc);
        }
        jaim_offsets.pop();
        let jaim_boundaries: std::collections::BTreeSet<usize> =
            jaim_offsets.into_iter().collect();

        let exp_count = exp_boundaries.len() as u32;
        let matched = exp_boundaries.intersection(&jaim_boundaries).count() as u32;
        let is_refinement = exp_boundaries.is_subset(&jaim_boundaries);
        let over_segments = jaim_boundaries.difference(&exp_boundaries).count();
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
        "Refinement pass:   {}/{} ({:.1}%)   [JaIM boundaries ⊇ expected]",
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

    eprintln!("\n=== By pos_solvable ===");
    eprintln!("{:<8}  refinement     boundary-recall  parity", "bucket");
    for (bucket, s) in &buckets {
        eprintln!(
            "{:<8}  {:>3}/{:>3} ({:>5.1}%)  {:>3}/{:>3} ({:>5.1}%)  {:>3}/{:>3} ({:>5.1}%)",
            bucket,
            s.refinement_pass,
            s.total,
            pct(s.refinement_pass, s.total),
            s.boundary_recall_num,
            s.boundary_recall_den,
            pct(s.boundary_recall_num, s.boundary_recall_den),
            s.parity_pass,
            s.total,
            pct(s.parity_pass, s.total),
        );
    }

    // PoC harness — visibility only. The CONN tuning cycle will gate on this
    // once the metric stabilizes, but for now regressions shouldn't break CI.
}
