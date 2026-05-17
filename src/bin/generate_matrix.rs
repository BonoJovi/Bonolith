//! Generate JaIM CONNECTION_COST table from IPADIC matrix.def.
//!
//! Reads IPADIC left-id.def, right-id.def, and matrix.def from
//! /usr/share/mecab/dic/ipadic, maps each IPADIC POS string to one of
//! JaIM's 11 PartOfSpeech variants (same mapping as generate_dict.rs),
//! aggregates costs per (left_jaim_pos, right_jaim_pos) using the arithmetic
//! mean, normalizes to [0, 8] (to match the engine boost ×10 scale), and
//! emits a Rust source file.
//!
//! Usage:
//!   1. Install IPADIC: `sudo apt install mecab-ipadic-utf8` (or equivalent)
//!   2. Generate:        `cargo run --bin generate-matrix --release > src/core/dictionary/connection_cost.rs`
//!   3. Build/test:      `cargo build --release && cargo test --lib dictionary`

use std::fs;
use std::io::{self, Write};
use std::path::Path;

const IPADIC_DIR: &str = "/usr/share/mecab/dic/ipadic";

const POS_COUNT: usize = 11;

const POS_NAMES: [&str; POS_COUNT] = [
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Particle",
    "Auxiliary",
    "Conjunction",
    "Interjection",
    "Prefix",
    "Suffix",
    "Other",
];

fn pos_to_idx(name: &str) -> usize {
    POS_NAMES
        .iter()
        .position(|&n| n == name)
        .expect("unknown POS name")
}

/// Hand-tuned overrides applied **after** IPADIC normalization.
///
/// IPADIC's 1316×1316 matrix has fine-grained particle/adjective subtypes
/// (格助詞 vs 終助詞, 連体形 vs 終止形, …) whose costs span a wide range.
/// Averaging them into JaIM's 11×11 buckets washes out cells that are
/// universally cheap in modern written Japanese, producing pathological
/// asymmetries — most notably N→Part (1.667) vs Part→N (4.679), which
/// makes the DP prefer wrong merges like は|いい → はい|い.
///
/// Each override is a `(left_pos, right_pos, value, reason)` quad. Keep
/// `reason` short — it is emitted as a comment in the generated table and
/// also documents the design intent for future regenerations.
const OVERRIDES: &[(&str, &str, f64, &str)] = &[
    // Particle → {Noun, Adj, Verb, Adv}: the most common openings after a
    // case/topic particle. Aggregation artifact from 終助詞 contamination.
    //
    // Tuned upward from initial pass (1.5 / 2.0): values too low encouraged
    // fragmentation through high-frequency single-mora Particle entries
    // (e.g. たべたい → た|べ|たい, もも → も|も). The values below keep the
    // important fixes (case_0019, case_0020) while preserving compounds.
    ("Particle", "Noun",      3.500, "は|わたし; raised from 2.5 — single-mora Particles (も,に,は freq 8.5k-9.8k) still pulled もも→も|も apart"),
    ("Particle", "Adjective", 2.500, "は|いい etc — fixes case_0019, segmentation_basic"),
    ("Particle", "Verb",      3.000, "を|たべる; kept moderate to avoid Particle-glued false splits"),
    ("Particle", "Adverb",    3.500, "は|すごく etc — valid, less frequent"),
    // Adjective → Noun: default modification pattern (連体修飾).
    ("Adjective", "Noun",     2.000, "いい|てんき, あたらしい|ほん — default 連体修飾"),
    // Adverb → {Verb, Adjective}: standard modification of predicates.
    ("Adverb", "Verb",        2.500, "はやく|たべる — adverb modifies verb"),
    ("Adverb", "Adjective",   2.500, "とても|うつくしい — adverb modifies adjective"),
    // Noun → Suffix: IPADIC normalization gave 0.000 (universal compound-form
    // magnet). The Suffix bucket is heterogeneous in JaIM's 11-class scheme
    // (氏/様/的/化/etc.), so a universal 0.000 lets the DP take spurious
    // suffix splits like りょうかい|し(suf)|ました over りょうかい|しました.
    ("Noun", "Suffix",        2.500, "氏/様/的 chain is natural but not free; was 0.000 IPADIC artifact"),
];

/// Same mapping as generate_dict.rs::map_pos — keep in sync.
fn map_pos(major: &str, sub: &str) -> &'static str {
    match major {
        "名詞" => {
            if sub == "接尾" {
                "Suffix"
            } else {
                "Noun"
            }
        }
        "動詞" => "Verb",
        "形容詞" => "Adjective",
        "副詞" => "Adverb",
        "助詞" => "Particle",
        "助動詞" => "Auxiliary",
        "接続詞" => "Conjunction",
        "感動詞" => "Interjection",
        "接頭詞" => "Prefix",
        _ => "Other",
    }
}

fn read_eucjp(path: &Path) -> io::Result<String> {
    let raw = fs::read(path)?;
    let (utf8, _, had_errors) = encoding_rs::EUC_JP.decode(&raw);
    if had_errors {
        eprintln!("WARNING: encoding errors in {}", path.display());
    }
    Ok(utf8.into_owned())
}

/// Parse left-id.def or right-id.def into a Vec<jaim_pos_idx> indexed by IPADIC id.
/// Format per line: `<id> <pos_csv>` where pos_csv is e.g. `名詞,一般,*,*,*,*,*,*,*`
fn parse_id_def(path: &Path) -> io::Result<Vec<usize>> {
    let text = read_eucjp(path)?;
    let other_idx = pos_to_idx("Other");
    let mut by_id: Vec<usize> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut sp = trimmed.splitn(2, ' ');
        let id: usize = sp
            .next()
            .and_then(|s| s.parse().ok())
            .expect("id parse");
        let pos_csv = sp.next().unwrap_or("");
        let parts: Vec<&str> = pos_csv.split(',').collect();
        let major = parts.first().copied().unwrap_or("");
        let sub = parts.get(1).copied().unwrap_or("");
        let jaim_pos = map_pos(major, sub);
        let idx = pos_to_idx(jaim_pos);
        if by_id.len() <= id {
            by_id.resize(id + 1, other_idx);
        }
        by_id[id] = idx;
    }
    Ok(by_id)
}

fn main() -> io::Result<()> {
    let dir = Path::new(IPADIC_DIR);
    if !dir.exists() {
        eprintln!("ERROR: IPADIC not found at {}", IPADIC_DIR);
        eprintln!("Install with: sudo apt install mecab-ipadic-utf8");
        std::process::exit(1);
    }

    eprintln!("Reading left-id.def...");
    let left = parse_id_def(&dir.join("left-id.def"))?;
    eprintln!("  {} left ids", left.len());

    eprintln!("Reading right-id.def...");
    let right = parse_id_def(&dir.join("right-id.def"))?;
    eprintln!("  {} right ids", right.len());

    eprintln!("Reading matrix.def...");
    let matrix_text = read_eucjp(&dir.join("matrix.def"))?;
    let mut lines = matrix_text.lines();
    let header = lines.next().unwrap_or("0 0");
    eprintln!("  header: {}", header);

    let mut sum = [[0i64; POS_COUNT]; POS_COUNT];
    let mut cnt = [[0u64; POS_COUNT]; POS_COUNT];
    let other_idx = pos_to_idx("Other");

    let mut row_count = 0u64;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut sp = trimmed.split_whitespace();
        let lid: usize = sp.next().and_then(|s| s.parse().ok()).expect("lid");
        let rid: usize = sp.next().and_then(|s| s.parse().ok()).expect("rid");
        let cost: i64 = sp.next().and_then(|s| s.parse().ok()).expect("cost");
        let lpos = left.get(lid).copied().unwrap_or(other_idx);
        let rpos = right.get(rid).copied().unwrap_or(other_idx);
        sum[lpos][rpos] += cost;
        cnt[lpos][rpos] += 1;
        row_count += 1;
    }
    eprintln!("  {} matrix entries aggregated", row_count);

    // Mean per cell (NaN if no data)
    let mut mean = [[f64::NAN; POS_COUNT]; POS_COUNT];
    for i in 0..POS_COUNT {
        for j in 0..POS_COUNT {
            if cnt[i][j] > 0 {
                mean[i][j] = sum[i][j] as f64 / cnt[i][j] as f64;
            }
        }
    }

    // Fill empty cells with global mean (no-data → neutral)
    let valid: Vec<f64> = mean.iter().flatten().copied().filter(|v| !v.is_nan()).collect();
    let overall = valid.iter().sum::<f64>() / valid.len() as f64;
    for i in 0..POS_COUNT {
        for j in 0..POS_COUNT {
            if mean[i][j].is_nan() {
                mean[i][j] = overall;
            }
        }
    }

    // Min-max normalize to [0.0, 8.0]
    let mn = mean.iter().flatten().copied().fold(f64::INFINITY, f64::min);
    let mx = mean.iter().flatten().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (mx - mn).max(1.0);
    let mut normed = [[0.0_f64; POS_COUNT]; POS_COUNT];
    for i in 0..POS_COUNT {
        for j in 0..POS_COUNT {
            normed[i][j] = ((mean[i][j] - mn) / range * 8.0 * 1000.0).round() / 1000.0;
        }
    }
    eprintln!(
        "  raw mean range: [{:.1}, {:.1}] -> normalized to [0.000, 8.000]",
        mn, mx
    );

    // Apply hand-tuned overrides (post-normalization so they survive regen).
    let mut override_marks = [[false; POS_COUNT]; POS_COUNT];
    for (left, right, value, reason) in OVERRIDES {
        let li = pos_to_idx(left);
        let ri = pos_to_idx(right);
        let before = normed[li][ri];
        normed[li][ri] = *value;
        override_marks[li][ri] = true;
        eprintln!(
            "  override [{:>10} -> {:>10}] {:.3} -> {:.3}  ({})",
            left, right, before, value, reason
        );
    }

    // Emit Rust source to stdout
    let out = io::stdout();
    let mut out = out.lock();
    writeln!(
        out,
        "//! Bigram connection cost table between (prev_pos, cur_pos), indexed by"
    )?;
    writeln!(
        out,
        "//! `PartOfSpeech::idx()`. Lower = more natural; added to segmentation cost"
    )?;
    writeln!(
        out,
        "//! so higher = penalty. Normalized to [0, 8] to match engine boost ×10 scale."
    )?;
    writeln!(out, "//!")?;
    writeln!(
        out,
        "//! AUTO-GENERATED from IPADIC matrix.def by src/bin/generate_matrix.rs."
    )?;
    writeln!(out, "//! To regenerate:")?;
    writeln!(
        out,
        "//!   cargo run --bin generate-matrix --release > src/core/dictionary/connection_cost.rs"
    )?;
    writeln!(out)?;
    writeln!(out, "use super::PartOfSpeech;")?;
    writeln!(out)?;
    writeln!(
        out,
        "pub const CONNECTION_COST: [[f64; PartOfSpeech::COUNT]; PartOfSpeech::COUNT] = ["
    )?;
    let short = ["N", "V", "Adj", "Adv", "Part", "Aux", "Conj", "Intj", "Pref", "Suf", "Other"];
    write!(out, "    //  ")?;
    for s in &short {
        write!(out, "{:>6} ", s)?;
    }
    writeln!(out)?;
    for i in 0..POS_COUNT {
        write!(out, "    [ ")?;
        for j in 0..POS_COUNT {
            write!(out, "{:>5.3}", normed[i][j])?;
            if j < POS_COUNT - 1 {
                write!(out, ", ")?;
            } else {
                write!(out, "  ")?;
            }
        }
        // Mark overrides on the row so they are easy to audit at a glance.
        let overridden_cols: Vec<&str> = (0..POS_COUNT)
            .filter(|&j| override_marks[i][j])
            .map(|j| short[j])
            .collect();
        if overridden_cols.is_empty() {
            writeln!(out, "],  // {} ->", short[i])?;
        } else {
            writeln!(
                out,
                "],  // {} ->   override: {}",
                short[i],
                overridden_cols.join(", ")
            )?;
        }
    }
    writeln!(out, "];")?;

    Ok(())
}
