/// IPADIC dictionary generator for JaIM.
///
/// Reads IPADIC CSV files (EUC-JP), filters and transforms entries,
/// and outputs `builtin_dict.rs` with a static const array.
///
/// Usage: cargo run --bin generate-dict
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

const IPADIC_DIR: &str = "/usr/share/mecab/dic/ipadic";

/// Output path relative to project root.
const OUTPUT_PATH: &str = "src/core/dictionary/builtin_dict.rs";

/// IPADIC source file with its cost threshold (None = no threshold).
struct DictSource {
    filename: &'static str,
    max_cost: Option<i32>,
    /// Which conjugation forms to include.
    /// None = all forms, Some(list) = only matching forms.
    allowed_forms: Option<&'static [&'static str]>,
}

/// Common conjugation forms for daily Japanese input.
/// 基本形: base (食べる), 連用タ接続: ta-form (食べた), 連用形: te-form (食べて),
/// 未然形: negative stem (食べない), 仮定形: conditional (食べれば)
const PRACTICAL_FORMS: &[&str] = &["基本形", "連用タ接続", "連用形", "未然形", "仮定形"];

const SOURCES: &[DictSource] = &[
    // --- Nouns ---
    DictSource { filename: "Noun.csv", max_cost: Some(8000), allowed_forms: None },
    DictSource { filename: "Noun.adjv.csv", max_cost: Some(8000), allowed_forms: None },
    DictSource { filename: "Noun.adverbal.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Noun.verbal.csv", max_cost: Some(8000), allowed_forms: None },
    DictSource { filename: "Noun.demonst.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Noun.nai.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Noun.number.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Noun.others.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Noun.place.csv", max_cost: Some(5500), allowed_forms: None },
    DictSource { filename: "Noun.name.csv", max_cost: Some(5500), allowed_forms: None },
    DictSource { filename: "Noun.org.csv", max_cost: Some(5500), allowed_forms: None },
    DictSource { filename: "Noun.proper.csv", max_cost: Some(6000), allowed_forms: None },
    // --- Verbs (practical conjugation forms) ---
    DictSource { filename: "Verb.csv", max_cost: None, allowed_forms: Some(PRACTICAL_FORMS) },
    // --- Adjectives (practical conjugation forms) ---
    DictSource { filename: "Adj.csv", max_cost: None, allowed_forms: Some(PRACTICAL_FORMS) },
    // --- Small categories (no threshold) ---
    DictSource { filename: "Adverb.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Postp.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Postp-col.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Auxil.csv", max_cost: None, allowed_forms: Some(PRACTICAL_FORMS) },
    DictSource { filename: "Conjunction.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Interjection.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Prefix.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Suffix.csv", max_cost: None, allowed_forms: None },
    DictSource { filename: "Adnominal.csv", max_cost: None, allowed_forms: None },
];

#[derive(Clone)]
struct Entry {
    reading: String,
    surface: String,
    pos: &'static str,
    frequency: u32,
}

/// Intermediate entry that carries conjugation metadata for compound generation.
struct ParsedEntry {
    entry: Entry,
    conjugation_type: String,
    conjugation_form: String,
}

/// Conjugation types that take voiced auxiliaries (だ/で) instead of (た/て).
const VOICED_CONJUGATION_TYPES: &[&str] = &[
    "五段・ガ行",
    "五段・バ行",
    "五段・マ行",
    "五段・ナ行",
];

/// Conjugation types where 連用形 can take た/て directly (ichidan-style).
const ICHIDAN_TYPES: &[&str] = &[
    "一段",
    "一段・クレル",
    "カ変・来ル",
    "サ変・スル",
    "サ変・−スル",
];

fn main() {
    let mut entries: Vec<Entry> = Vec::new();
    let mut parsed_for_compounds: Vec<ParsedEntry> = Vec::new();

    for source in SOURCES {
        let path = Path::new(IPADIC_DIR).join(source.filename);
        if !path.exists() {
            eprintln!("WARNING: {} not found, skipping", path.display());
            continue;
        }

        let raw = fs::read(&path).expect("failed to read file");
        let (utf8, _, had_errors) = encoding_rs::EUC_JP.decode(&raw);
        if had_errors {
            eprintln!("WARNING: encoding errors in {}", source.filename);
        }

        let mut count = 0;
        for line in utf8.lines() {
            if let Some(parsed) = parse_line(line, source) {
                // Collect verb/adjective conjugation stems for compound generation.
                // Include kana-only forms (し, い, き, etc.) — they produce essential
                // compounds like します, います, きます.
                if matches!(parsed.entry.pos, "Verb" | "Adjective")
                    && matches!(parsed.conjugation_form.as_str(), "連用タ接続" | "連用形" | "未然形")
                {
                    parsed_for_compounds.push(ParsedEntry {
                        entry: parsed.entry.clone(),
                        conjugation_type: parsed.conjugation_type,
                        conjugation_form: parsed.conjugation_form,
                    });
                }
                entries.push(parsed.entry);
                count += 1;
            }
        }
        eprintln!("{}: {} entries", source.filename, count);
    }

    // Generate compound entries (verb stem + auxiliary)
    let compounds = generate_compounds(&parsed_for_compounds);
    eprintln!("Generated {} compound entries", compounds.len());
    entries.extend(compounds);

    // Manual extra entries (single kanji etc. missing from IPADIC)
    let extras = extra_entries();
    eprintln!("Added {} manual extra entries", extras.len());
    entries.extend(extras);

    eprintln!("Total before dedup: {}", entries.len());

    // Deduplicate: same reading+surface+pos → keep highest frequency
    let mut dedup: HashMap<(String, String, String), Entry> = HashMap::new();
    for entry in entries {
        let key = (entry.reading.clone(), entry.surface.clone(), entry.pos.to_string());
        dedup
            .entry(key)
            .and_modify(|existing| {
                if entry.frequency > existing.frequency {
                    *existing = entry.clone();
                }
            })
            .or_insert(entry);
    }

    let mut final_entries: Vec<Entry> = dedup.into_values().collect();
    // Sort by frequency descending, then reading, then surface for stable output
    final_entries.sort_by(|a, b| {
        b.frequency
            .cmp(&a.frequency)
            .then_with(|| a.reading.cmp(&b.reading))
            .then_with(|| a.surface.cmp(&b.surface))
    });

    eprintln!("Total after dedup: {}", final_entries.len());

    write_output(&final_entries);
    eprintln!("Written to {}", OUTPUT_PATH);
}

/// Parse a single IPADIC CSV line into a ParsedEntry, or None if filtered out.
fn parse_line(line: &str, source: &DictSource) -> Option<ParsedEntry> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 13 {
        return None;
    }

    let surface = fields[0];
    let cost: i32 = fields[3].parse().ok()?;
    let pos_major = fields[4]; // 品詞
    let pos_sub = fields[5]; // 品詞細分類1
    let conjugation_type = fields[8]; // 活用型
    let conjugation_form = fields[9]; // 活用形
    let reading_katakana = fields[11]; // 読み (katakana)

    // Filter by cost threshold (skip for single-kanji entries to ensure full coverage)
    let is_single_kanji = surface.chars().count() == 1
        && surface.chars().next().map_or(false, |c| ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c));
    if !is_single_kanji {
        if let Some(max) = source.max_cost {
            if cost >= max {
                return None;
            }
        }
    }

    // Filter by allowed conjugation forms (verbs/adjectives/auxiliaries)
    if let Some(forms) = source.allowed_forms {
        if !forms.iter().any(|&f| f == conjugation_form) {
            return None;
        }
    }

    // Skip entries with no reading
    if reading_katakana.is_empty() || reading_katakana == "*" {
        return None;
    }

    // Convert katakana reading to hiragana
    let reading = katakana_to_hiragana(reading_katakana);

    // Skip if surface == reading (pure kana words add no value for kana→kanji conversion)
    // Exception: keep functional words, verb/adjective conjugations, and pronouns
    // as they are needed for segmentation even without kanji conversion
    let is_pronoun = pos_major == "名詞" && pos_sub == "代名詞";
    let dominated_by_kana = surface == reading
        && !is_pronoun
        && !matches!(
            pos_major,
            "助詞" | "助動詞" | "接続詞" | "副詞" | "感動詞" | "連体詞" | "動詞" | "形容詞"
        );
    if dominated_by_kana {
        return None;
    }

    // Skip single-katakana surfaces with single-hiragana readings (e.g., を→ヲ, が→ガ)
    // These are rarely useful in normal Japanese input and pollute candidate lists.
    if reading.chars().count() == 1 && surface.chars().count() == 1 {
        let sc = surface.chars().next().unwrap();
        if ('\u{30A1}'..='\u{30F6}').contains(&sc) {
            return None;
        }
    }

    // Map POS
    let pos = map_pos(pos_major, pos_sub);

    // Convert cost to frequency (lower cost = more common), then adjust by POS
    let raw_frequency = cost_to_frequency(cost);
    let frequency = adjust_frequency(raw_frequency, surface, pos_major, pos_sub);

    Some(ParsedEntry {
        entry: Entry {
            reading,
            surface: surface.to_string(),
            pos,
            frequency,
        },
        conjugation_type: conjugation_type.to_string(),
        conjugation_form: conjugation_form.to_string(),
    })
}

/// Generate compound entries by combining verb/adjective stems with auxiliaries.
///
/// For each 連用タ接続 stem, generates:
///   stem + た/だ (past tense, e.g., 食べた, 読んだ)
///   stem + て/で (te-form, e.g., 食べて, 読んで)
///   stem + ている/でいる (progressive, e.g., 食べている, 読んでいる)
///   stem + ています/でいます (polite progressive, e.g., 食べています)
///
/// For each 連用形 stem, generates polite forms (all verb types):
///   stem + ます/ました/ません/ましょう (e.g., 書きます, 食べました, しません)
/// For ichidan-type verbs, also generates:
///   stem + た/て (past/te-form, e.g., 食べた, 食べて)
///   stem + ている/ています (progressive)
///
/// For each 未然形 stem, generates passive/potential forms:
///   stem + れる/れた/れて (godan passive, e.g., 書かれる)
///   stem + られる/られた/られて (ichidan passive, e.g., 食べられる)
fn generate_compounds(parsed: &[ParsedEntry]) -> Vec<Entry> {
    let mut compounds = Vec::new();

    for p in parsed {
        let ctype = p.conjugation_type.as_str();
        let cform = p.conjugation_form.as_str();

        match cform {
            "連用タ接続" => {
                // Voiced consonant stems (ガ/バ/マ/ナ行) take だ/で
                let is_voiced =
                    VOICED_CONJUGATION_TYPES.iter().any(|&v| ctype.starts_with(v));
                let (ta, te) = if is_voiced { ("だ", "で") } else { ("た", "て") };

                let boosted = boost_freq(p.entry.frequency);

                // Past and te-form
                for &suffix in &[ta, te] {
                    compounds.push(make_compound(&p.entry, suffix, suffix, boosted));
                }
                // Progressive: て+いる, て+いた, て+いない, て+います
                for &(r_suf, s_suf) in &[
                    (te, te),  // already added above, skip
                ] {
                    let _ = (r_suf, s_suf); // placeholder
                }
                let te_progressive: &[(&str, &str)] = &[
                    (&format!("{}いる", te), &format!("{}いる", te)),
                    (&format!("{}いた", te), &format!("{}いた", te)),
                    (&format!("{}いない", te), &format!("{}いない", te)),
                    (&format!("{}います", te), &format!("{}います", te)),
                ];
                for (r_suf, s_suf) in te_progressive {
                    compounds.push(Entry {
                        reading: format!("{}{}", p.entry.reading, r_suf),
                        surface: format!("{}{}", p.entry.surface, s_suf),
                        pos: p.entry.pos,
                        frequency: boosted,
                    });
                }
            }
            "連用形" => {
                let boosted = boost_freq(p.entry.frequency);

                // Polite forms: ALL verb types use 連用形 + ます/ました/ません/ましょう
                for &suffix in &["ます", "ました", "ません", "ましょう"] {
                    compounds.push(make_compound(&p.entry, suffix, suffix, boosted));
                }

                // Only ichidan-style verbs form past/te directly from 連用形
                if ICHIDAN_TYPES.iter().any(|&t| ctype.starts_with(t)) {
                    for &suffix in &["た", "て"] {
                        compounds.push(make_compound(&p.entry, suffix, suffix, boosted));
                    }
                    // Progressive forms
                    for &suffix in &["ている", "ていた", "ていない", "ています"] {
                        compounds.push(make_compound(&p.entry, suffix, suffix, boosted));
                    }
                }
            }
            "未然形" => {
                // Passive/potential forms
                let boosted = boost_freq(p.entry.frequency);
                if ICHIDAN_TYPES.iter().any(|&t| ctype.starts_with(t)) {
                    // Ichidan: stem + られる/られた/られて
                    for &suffix in &["られる", "られた", "られて", "られない"] {
                        compounds.push(make_compound(&p.entry, suffix, suffix, boosted));
                    }
                } else if ctype.starts_with("五段") {
                    // Godan: stem + れる/れた/れて (未然形 already ends with あ-row)
                    for &suffix in &["れる", "れた", "れて", "れない"] {
                        compounds.push(make_compound(&p.entry, suffix, suffix, boosted));
                    }
                }
            }
            _ => continue,
        }
    }

    compounds
}

fn boost_freq(frequency: u32) -> u32 {
    (frequency as f64 * 2.0).min(20000.0) as u32
}

fn make_compound(base: &Entry, reading_suffix: &str, surface_suffix: &str, freq: u32) -> Entry {
    Entry {
        reading: format!("{}{}", base.reading, reading_suffix),
        surface: format!("{}{}", base.surface, surface_suffix),
        pos: base.pos,
        frequency: freq,
    }
}

/// Convert katakana string to hiragana.
fn katakana_to_hiragana(s: &str) -> String {
    s.chars()
        .map(|c| {
            if ('\u{30A1}'..='\u{30F6}').contains(&c) {
                // Katakana → Hiragana (offset 0x60)
                char::from_u32(c as u32 - 0x60).unwrap_or(c)
            } else if c == '\u{30FC}' {
                // Katakana long vowel mark ー → keep as-is (used in loanwords)
                'ー'
            } else {
                c
            }
        })
        .collect()
}

/// Map IPADIC POS tags to JaIM PartOfSpeech variant name.
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
        _ => "Other", // 連体詞, フィラー, etc.
    }
}

/// Convert IPADIC cost to frequency score.
/// IPADIC: lower cost = more common. Range roughly -7000 to +16000.
/// JaIM: higher frequency = more common.
fn cost_to_frequency(cost: i32) -> u32 {
    let freq = 10000 - cost;
    freq.clamp(1, 20000) as u32
}

/// Adjust frequency based on POS and surface characteristics.
///
/// IPADIC cost values don't always reflect practical input frequency.
/// For example, particles (に, は, が) are the most frequent words in
/// Japanese text but have relatively high IPADIC cost.  Conversely,
/// katakana-only nouns (テキ, タイ) rarely appear in typical kana input.
fn adjust_frequency(frequency: u32, surface: &str, pos_major: &str, pos_sub: &str) -> u32 {
    let freq = frequency as f64;
    let adjusted = match pos_major {
        // Particles (に, は, が, を, で, etc.) — most frequent in running text
        "助詞" => freq * 1.6,
        // Auxiliaries (です, ます, た, etc.)
        "助動詞" => freq * 1.4,
        // Adverbs: boost kana-surface adverbs (うまく, たぶん, やはり) that lose
        // to single-kanji splits.  Skip kanji-surface adverbs (依然, 全然)
        // which are already high-freq and break segmentation when boosted.
        "副詞" => {
            let has_kanji = surface.chars().any(|c| {
                ('\u{4E00}'..='\u{9FFF}').contains(&c)
                    || ('\u{3400}'..='\u{4DBF}').contains(&c)
            });
            if has_kanji { freq } else { freq * 1.4 }
        }
        // Verbs: する/し/して etc. have very high IPADIC cost despite being
        // the most common verbs in Japanese.  Boost kana-only verb surfaces.
        "動詞" => {
            let all_kana = !surface.is_empty()
                && surface.chars().all(|c| {
                    ('\u{3040}'..='\u{309F}').contains(&c)  // hiragana
                    || ('\u{30A0}'..='\u{30FF}').contains(&c) // katakana
                });
            if all_kana { (freq * 1.5).min(8000.0) } else { freq }
        }
        // Suffixes (的, 性, 化, etc.) — very common in compounds
        "名詞" if pos_sub == "接尾" => freq * 1.4,
        // Pronouns (これ, それ, あれ, etc.) — very common, need segmentation presence
        "名詞" if pos_sub == "代名詞" => freq * 1.5,
        // サ変接続 nouns (換字, 自書, etc.) — IPADIC gives them lower cost than
        // common nouns (漢字, 辞書), which is backwards for typical input.
        "名詞" if pos_sub == "サ変接続" => freq * 0.8,
        // 副詞可能 nouns (後, 前, 時, 上, 間, etc.) — IPADIC gives them very
        // high cost (often > 10000 → freq=1), but they are common everyday words.
        // Use a floor + boost to keep them competitive with general nouns.
        "名詞" if pos_sub == "副詞可能" => {
            let floored = freq.max(4500.0);
            floored * 1.1
        }
        // Number kanji (二, 三, etc.) — rarely typed as kanji via kana
        "名詞" if pos_sub == "数" => {
            let is_kanji = surface.chars().all(|c| {
                ('\u{4E00}'..='\u{9FFF}').contains(&c)
                    || ('\u{3400}'..='\u{4DBF}').contains(&c)
            });
            if is_kanji { freq * 0.5 } else { freq }
        }
        // Katakana-only nouns (テキ, タイ, etc.) — demote
        "名詞" => {
            let all_katakana = !surface.is_empty()
                && surface.chars().all(|c| ('\u{30A1}'..='\u{30F6}').contains(&c) || c == 'ー');
            if all_katakana { freq * 0.5 } else { freq }
        }
        _ => freq,
    };
    (adjusted as u32).clamp(1, 20000)
}

/// Manual extra entries for kanji/words missing from IPADIC.
/// Add entries here as needed for single kanji or common words.
fn extra_entries() -> Vec<Entry> {
    let extras: &[(&str, &str, &str, u32)] = &[
        // (reading, surface, pos, frequency)
        ("ご", "誤", "Noun", 5000),
        // Common words missing from IPADIC
        ("なんかい", "何回", "Noun", 6000),
        ("ほしがる", "欲しがる", "Verb", 5500),
        ("ほしがり", "欲しがり", "Noun", 5000),
        ("ほしがって", "欲しがって", "Verb", 5500),
        ("ほしがった", "欲しがった", "Verb", 5500),
        ("ほしがらない", "欲しがらない", "Verb", 5500),
        // Frequency overrides: common words that lose to competing splits
        ("りょうかい", "了解", "Noun", 7000),
        ("かんりょう", "完了", "Noun", 6000),
        ("じゅんばん", "順番", "Noun", 6000),
        ("しょうがい", "障害", "Noun", 5500),
        // Common polite verb endings (base forms have unreasonably low IPADIC cost)
        ("しました", "しました", "Verb", 7000),
        ("します", "します", "Verb", 7000),
        ("しません", "しません", "Verb", 6500),
        ("しましょう", "しましょう", "Verb", 6500),
        ("います", "います", "Verb", 7000),
        ("いました", "いました", "Verb", 7000),
        ("いません", "いません", "Verb", 6500),
        ("ぷろんぷと", "プロンプト", "Noun", 5500),
        ("とーくん", "トークン", "Noun", 5500),
        ("あらーと", "アラート", "Noun", 5500),
        // Half-width digit → full-width digit conversion
        ("1", "１", "Noun", 7000),
        ("2", "２", "Noun", 7000),
        ("3", "３", "Noun", 7000),
        ("4", "４", "Noun", 7000),
        ("5", "５", "Noun", 7000),
        ("6", "６", "Noun", 7000),
        ("7", "７", "Noun", 7000),
        ("8", "８", "Noun", 7000),
        ("9", "９", "Noun", 7000),
        ("0", "０", "Noun", 7000),
    ];
    extras
        .iter()
        .map(|(reading, surface, pos, freq)| Entry {
            reading: reading.to_string(),
            surface: surface.to_string(),
            pos,
            frequency: *freq,
        })
        .collect()
}

/// Write the generated builtin_dict.rs file.
fn write_output(entries: &[Entry]) {
    let mut out = fs::File::create(OUTPUT_PATH).expect("failed to create output file");

    writeln!(out, "/// Auto-generated from IPADIC. Do not edit manually.").unwrap();
    writeln!(out, "/// Re-generate with: cargo run --bin generate-dict").unwrap();
    writeln!(
        out,
        "/// Source: /usr/share/mecab/dic/ipadic/ (mecab-ipadic package)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use super::PartOfSpeech;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "/// Built-in dictionary entries: (reading, surface, pos, frequency)"
    )
    .unwrap();
    writeln!(
        out,
        "pub const BUILTIN_ENTRIES: &[(&str, &str, PartOfSpeech, u32)] = &["
    )
    .unwrap();

    for entry in entries {
        // Escape any backslashes or quotes in strings
        let reading = entry.reading.replace('\\', "\\\\").replace('"', "\\\"");
        let surface = entry.surface.replace('\\', "\\\\").replace('"', "\\\"");
        writeln!(
            out,
            "    (\"{}\", \"{}\", PartOfSpeech::{}, {}),",
            reading, surface, entry.pos, entry.frequency
        )
        .unwrap();
    }

    writeln!(out, "];").unwrap();
}
