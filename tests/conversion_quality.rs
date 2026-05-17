//! Conversion-quality evaluation harness against tests/conversion_cases/cases.jsonl.
//!
//! Coarse metric: segment count parity (segments.len() == expected.len()).
//! This is a v2.x PoC harness for the IPADIC POS-connection-cost initiative —
//! it tracks whether segmentation boundaries roughly match human expectation
//! while we tune CONNECTION_COST in src/core/dictionary/mod.rs.
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
    pos_solvable: String,
    #[serde(default)]
    category: String,
}

#[test]
fn evaluate_conversion_cases() {
    let path = "tests/conversion_cases/cases.jsonl";
    let content = fs::read_to_string(path).expect("read cases.jsonl");
    let dict = Dictionary::new();

    let mut buckets: std::collections::BTreeMap<String, (u32, u32)> =
        std::collections::BTreeMap::new();
    let mut total = 0u32;
    let mut pass = 0u32;

    eprintln!("\n=== Per-case results ===");
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(line).expect("parse case");
        let segments = dict.segment(&case.input_hiragana);
        let segs_count = segments.len();
        let expected_count = case.expected.len();
        let ok = segs_count == expected_count;

        total += 1;
        if ok {
            pass += 1;
        }
        let entry = buckets.entry(case.pos_solvable.clone()).or_insert((0, 0));
        entry.0 += 1;
        if ok {
            entry.1 += 1;
        }

        let readings: Vec<&str> = segments.iter().map(|s| s.reading.as_str()).collect();
        eprintln!(
            "{:>10} [{:>7} / {:>14}] segs={}/{} {}  input={:<30} got={:?}  expected={:?}",
            case.id,
            case.pos_solvable,
            case.category,
            segs_count,
            expected_count,
            if ok { "PASS" } else { "FAIL" },
            case.input_hiragana,
            readings,
            case.expected,
        );
    }

    eprintln!("\n=== Summary (segment-count parity) ===");
    eprintln!(
        "Total: {}/{} ({:.1}%)",
        pass,
        total,
        100.0 * pass as f64 / total as f64
    );
    for (bucket, (b_total, b_pass)) in &buckets {
        eprintln!(
            "  pos_solvable={:<7}: {}/{} ({:.1}%)",
            bucket,
            b_pass,
            b_total,
            100.0 * *b_pass as f64 / *b_total as f64
        );
    }

    // Don't fail the suite on coarse-metric regressions during PoC tuning.
    // The harness exists for visibility, not gating.
}
