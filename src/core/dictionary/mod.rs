/// Dictionary-based Kana to Kanji conversion
///
/// Fast lookup (< 1ms) handling 70-80% of common conversions.
/// Uses trie-based data structure for efficient prefix matching.
/// Includes word segmentation via dynamic programming (minimum-cost path).

mod builtin_dict;
mod trie;

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use trie::Trie;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    /// Reading in hiragana
    pub reading: String,
    /// Surface form (kanji/mixed)
    pub surface: String,
    /// Part of speech
    pub pos: PartOfSpeech,
    /// Frequency score (higher = more common)
    pub frequency: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Particle,
    Auxiliary,
    Conjunction,
    Interjection,
    Prefix,
    Suffix,
    Other,
}

/// A segment produced by word segmentation
#[derive(Debug, Clone)]
pub struct Segment {
    /// Reading (kana substring)
    pub reading: String,
    /// Start position (char offset)
    pub start: usize,
    /// Length in characters
    pub len: usize,
    /// Candidate entries for this segment
    pub candidates: Vec<DictionaryEntry>,
}

pub struct Dictionary {
    entries: Vec<DictionaryEntry>,
    trie: Trie,
    /// Index of the first user-added entry (all entries before this are builtin)
    user_start: usize,
}

impl Dictionary {
    /// Create a new dictionary pre-loaded with the built-in word set.
    pub fn new() -> Self {
        let mut dict = Self {
            entries: Vec::new(),
            trie: Trie::new(),
            user_start: 0,
        };
        dict.load_builtin();
        dict.user_start = dict.entries.len();
        dict
    }

    /// Add a single entry.
    pub fn add_entry(&mut self, entry: DictionaryEntry) {
        let idx = self.entries.len();
        let reading = entry.reading.clone();
        let frequency = entry.frequency;
        self.entries.push(entry);
        self.trie.insert(&reading, idx, frequency);
    }

    /// Exact lookup: return all candidates for a reading, sorted by frequency (descending).
    pub fn lookup(&self, reading: &str) -> Vec<&DictionaryEntry> {
        let indices = self.trie.exact_lookup(reading);
        let mut entries: Vec<&DictionaryEntry> = indices.iter().map(|&i| &self.entries[i]).collect();
        entries.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        entries
    }

    /// Common prefix search: find all dictionary words that are prefixes of `input`.
    /// Returns Vec of (char_length, entries) sorted by prefix length.
    pub fn common_prefix_search(&self, input: &str) -> Vec<(usize, Vec<&DictionaryEntry>)> {
        self.trie
            .common_prefix_search(input)
            .into_iter()
            .map(|(len, indices)| {
                let mut entries: Vec<&DictionaryEntry> =
                    indices.iter().map(|&i| &self.entries[i]).collect();
                entries.sort_by(|a, b| b.frequency.cmp(&a.frequency));
                (len, entries)
            })
            .collect()
    }

    /// Prefix lookup: return candidates for all readings starting with `prefix`.
    pub fn prefix_lookup(&self, prefix: &str) -> Vec<&DictionaryEntry> {
        let indices = self.trie.prefix_lookup(prefix);
        let mut entries: Vec<&DictionaryEntry> = indices.iter().map(|&i| &self.entries[i]).collect();
        entries.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        entries
    }

    /// Segment a kana string into words using minimum-cost dynamic programming.
    /// Returns the best segmentation as a Vec of Segments.
    pub fn segment(&self, input: &str) -> Vec<Segment> {
        self.segment_with_boost(input, |_, _| 0.0)
    }

    /// Segment with an optional cost-reduction callback.
    /// `boost_fn(reading, entries)` returns a bonus (>= 0.0) that reduces segment cost.
    pub fn segment_with_boost<F>(&self, input: &str, boost_fn: F) -> Vec<Segment>
    where
        F: Fn(&str, &[&DictionaryEntry]) -> f64,
    {
        let chars: Vec<char> = input.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        // Pre-compute byte offsets for each char position to avoid String allocation in the loop
        let byte_offsets: Vec<usize> = input
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(input.len()))
            .collect();

        // DP: best_cost[i] = minimum cost to segment chars[0..i]
        const INF: f64 = 1e18;
        let mut best_cost = vec![INF; n + 1];
        let mut back: Vec<Option<usize>> = vec![None; n + 1]; // back[i] = start of last segment ending at i
        best_cost[0] = 0.0;

        for i in 0..n {
            if best_cost[i] >= INF {
                continue;
            }

            let remaining = &input[byte_offsets[i]..];
            let prefixes = self.trie.common_prefix_search(remaining);

            for (len, _indices) in &prefixes {
                // Cost: prefer longer matches and higher frequency
                let best_freq = _indices
                    .iter()
                    .map(|&idx| self.entries[idx].frequency)
                    .max()
                    .unwrap_or(1);
                let reading: String = chars[i..i + len].iter().collect();
                let entries: Vec<&DictionaryEntry> = _indices
                    .iter()
                    .map(|&idx| &self.entries[idx])
                    .collect();
                let boost = boost_fn(&reading, &entries);
                let cost = segment_cost(*len, best_freq) - boost;
                let total = best_cost[i] + cost;
                if total < best_cost[i + len] {
                    best_cost[i + len] = total;
                    back[i + len] = Some(i);
                }
            }

            // Fallback: single character as unknown word (high cost)
            let unknown_cost = best_cost[i] + 20.0;
            if unknown_cost < best_cost[i + 1] {
                best_cost[i + 1] = unknown_cost;
                back[i + 1] = Some(i);
            }
        }

        // Reconstruct path
        let mut boundaries = Vec::new();
        let mut pos = n;
        while pos > 0 {
            if let Some(start) = back[pos] {
                boundaries.push((start, pos));
                pos = start;
            } else {
                // Should not happen if fallback works, but handle gracefully
                boundaries.push((pos - 1, pos));
                pos -= 1;
            }
        }
        boundaries.reverse();

        // Build segments
        boundaries
            .into_iter()
            .map(|(start, end)| {
                let reading: String = chars[start..end].iter().collect();
                let candidates: Vec<DictionaryEntry> = self
                    .lookup(&reading)
                    .into_iter()
                    .cloned()
                    .collect();
                Segment {
                    reading,
                    start,
                    len: end - start,
                    candidates,
                }
            })
            .collect()
    }

    /// Return a slice of user-added entries.
    pub fn user_entries(&self) -> &[DictionaryEntry] {
        &self.entries[self.user_start..]
    }

    /// Replace all user entries. Removes old user entries from the trie
    /// and adds new ones.
    pub fn replace_user_entries(&mut self, new_entries: Vec<DictionaryEntry>) {
        // Remove old user entries from trie
        for idx in self.user_start..self.entries.len() {
            self.trie.remove(&self.entries[idx].reading, idx);
        }
        // Truncate to builtin only
        self.entries.truncate(self.user_start);
        // Add new entries
        for entry in new_entries {
            self.add_entry(entry);
        }
    }

    /// Total number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return the default path for the user dictionary file.
    pub fn default_user_dict_path() -> io::Result<PathBuf> {
        let data_dir = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                PathBuf::from(home).join(".local/share")
            })
            .join("jaim");
        Ok(data_dir.join("user_dict.json"))
    }

    /// Save user-added entries to a JSON file.
    pub fn save_user_entries(&self, path: &Path) -> io::Result<()> {
        let user_entries = &self.entries[self.user_start..];
        if user_entries.is_empty() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(user_entries)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load user entries from a JSON file and add them to the dictionary.
    pub fn load_user_entries(&mut self, path: &Path) -> io::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }
        let entries = read_and_parse_dict(path)?;
        let count = entries.len();
        for entry in entries {
            self.add_entry(entry);
        }
        Ok(count)
    }

    /// Export the entire dictionary (builtin + user) to a JSON file.
    pub fn export(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Import entries from a JSON file, adding them as user entries.
    /// Duplicate entries (same reading + surface) are skipped.
    pub fn import(&mut self, path: &Path) -> io::Result<usize> {
        let entries = read_and_parse_dict(path)?;
        let mut added = 0;
        for entry in entries {
            if !self.has_entry(&entry.reading, &entry.surface) {
                self.add_entry(entry);
                added += 1;
            }
        }
        Ok(added)
    }

    /// Check if an entry with the given reading and surface already exists.
    fn has_entry(&self, reading: &str, surface: &str) -> bool {
        self.lookup(reading).iter().any(|e| e.surface == surface)
    }

    fn load_builtin(&mut self) {
        for &(reading, surface, pos, frequency) in builtin_dict::BUILTIN_ENTRIES {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos,
                frequency,
            });
        }
        self.load_symbol_entries();
        self.load_emoji_entries();
    }

    /// Load emoji entries from the embedded TSV data.
    /// Emoji are added with low frequency so they appear after regular candidates.
    fn load_emoji_entries(&mut self) {
        const EMOJI_TSV: &str = include_str!("../../../data/emoji.tsv");
        const EMOJI_BASE_FREQ: u32 = 500;

        for line in EMOJI_TSV.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((reading, emojis)) = line.split_once('\t') else {
                continue;
            };
            for (i, emoji) in emojis.split_whitespace().enumerate() {
                self.add_entry(DictionaryEntry {
                    reading: reading.to_string(),
                    surface: emoji.to_string(),
                    pos: PartOfSpeech::Other,
                    frequency: EMOJI_BASE_FREQ.saturating_sub(i as u32 * 10),
                });
            }
        }
    }

    /// Load symbol/special character entries not found in IPADIC.
    fn load_symbol_entries(&mut self) {
        let symbols: &[(&str, &[&str])] = &[
            ("やじるし", &["→", "←", "↑", "↓", "⇒", "⇐", "⇑", "⇓", "↔", "↕"]),
            ("みぎ", &["→", "⇒"]),
            ("ひだり", &["←", "⇐"]),
            ("うえ", &["↑", "⇑"]),
            ("した", &["↓", "⇓"]),
            ("まる", &["○", "◎", "●", "◯"]),
            ("さんかく", &["△", "▲", "▽", "▼"]),
            ("しかく", &["□", "■", "◇", "◆"]),
            ("ほし", &["☆", "★"]),
            ("こめ", &["※"]),
            ("から", &["〜", "～"]),
            ("てん", &["・", "…", "‥", "、"]),
            ("まる", &["。", "○", "◎", "●"]),
            ("かっこ", &["「」", "「", "」", "『』", "『", "』", "【】", "【", "】", "（）", "（", "）", "〔〕", "［］", "｛｝", "〈〉", "《》"]),
            ("かぎかっこ", &["「」", "「", "」", "『』", "『", "』"]),
            ("すみかっこ", &["【】", "【", "】"]),
            ("まるかっこ", &["（）", "（", "）"]),
            ("ゆうびん", &["〒"]),
        ];
        for &(reading, surfaces) in symbols {
            for (i, &surface) in surfaces.iter().enumerate() {
                self.add_entry(DictionaryEntry {
                    reading: reading.to_string(),
                    surface: surface.to_string(),
                    pos: PartOfSpeech::Other,
                    // First candidate gets highest frequency
                    frequency: 8000 - (i as u32) * 100,
                });
            }
        }

        // Common auxiliary verb compound forms not in IPADIC
        let auxiliaries: &[(&str, &str)] = &[
            ("ましょう", "ましょう"),
            ("ません", "ません"),
            ("ました", "ました"),
            ("ませんでした", "ませんでした"),
            ("でしょう", "でしょう"),
            ("でした", "でした"),
            ("ですが", "ですが"),
            ("ですけど", "ですけど"),
            ("ですから", "ですから"),
            ("ですので", "ですので"),
            ("ですよね", "ですよね"),
            ("ですよ", "ですよ"),
            ("ですね", "ですね"),
            ("ですか", "ですか"),
            ("ますが", "ますが"),
            ("ますか", "ますか"),
            ("ますよ", "ますよ"),
            ("ますね", "ますね"),
            ("ください", "ください"),
            ("くださる", "くださる"),
            ("ております", "ております"),
            ("いたします", "いたします"),
        ];
        for &(reading, surface) in auxiliaries {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Auxiliary,
                frequency: 9000,
            });
        }
    }
}

/// Cost function for segmentation DP.
/// Lower cost = better.  The char_len multiplier makes longer words accumulate
/// more frequency benefit, while the +1.0 per-segment penalty discourages
/// excessive splitting.
fn segment_cost(char_len: usize, frequency: u32) -> f64 {
    (char_len as f64) * -(frequency as f64).ln() + 1.0
}

/// Read a dictionary JSON file and parse it into entries.
/// Returns a rich io::Error including file path, line/column, and a hint
/// when parsing fails (so users can fix manual edits without grepping logs).
fn read_and_parse_dict(path: &Path) -> io::Result<Vec<DictionaryEntry>> {
    let json = fs::read_to_string(path)?;
    parse_dict_json(&json).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to parse {} at line {} col {}: {} \
                 (hint: check for trailing commas, missing brackets, or unescaped quotes)",
                path.display(),
                e.line(),
                e.column(),
                e
            ),
        )
    })
}

/// Parse a dictionary JSON string. Tries strict parsing first; on failure,
/// strips trailing commas (a common manual-edit mistake) and retries.
/// Returns the original strict error if both attempts fail.
fn parse_dict_json(json: &str) -> serde_json::Result<Vec<DictionaryEntry>> {
    match serde_json::from_str::<Vec<DictionaryEntry>>(json) {
        Ok(entries) => Ok(entries),
        Err(strict_err) => {
            let cleaned = strip_trailing_commas(json);
            match serde_json::from_str::<Vec<DictionaryEntry>>(&cleaned) {
                Ok(entries) => {
                    log::warn!(
                        "user dictionary JSON had a syntax error (likely a trailing comma); \
                         auto-recovered. original error: {}",
                        strict_err
                    );
                    Ok(entries)
                }
                Err(_) => Err(strict_err),
            }
        }
    }
}

/// Remove commas that appear immediately before `}` or `]` (with optional
/// intervening whitespace), preserving any commas that occur inside JSON
/// string literals.
fn strip_trailing_commas(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escape = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if escape {
            out.push(c);
            escape = false;
            i += 1;
            continue;
        }
        if in_string {
            if c == '\\' {
                out.push(c);
                escape = true;
                i += 1;
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            out.push(c);
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == ',' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                i += 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_dictionary_loads() {
        let dict = Dictionary::new();
        assert!(dict.len() > 100);
    }

    #[test]
    fn lookup_common_words() {
        let dict = Dictionary::new();

        let results = dict.lookup("きょう");
        assert!(!results.is_empty());
        assert_eq!(results[0].surface, "今日"); // highest frequency
    }

    #[test]
    fn lookup_emoji() {
        let dict = Dictionary::new();

        let results = dict.lookup("えがお");
        let surfaces: Vec<&str> = results.iter().map(|e| e.surface.as_str()).collect();
        assert!(surfaces.contains(&"😊"), "Expected 😊 in results for えがお: {:?}", surfaces);

        let results = dict.lookup("ねこ");
        let surfaces: Vec<&str> = results.iter().map(|e| e.surface.as_str()).collect();
        assert!(surfaces.contains(&"🐱"), "Expected 🐱 in results for ねこ: {:?}", surfaces);

        // Emoji should have lower frequency than regular words
        let emoji = results.iter().find(|e| e.surface == "🐱").unwrap();
        assert!(emoji.frequency <= 500);
    }

    #[test]
    fn lookup_particles() {
        let dict = Dictionary::new();

        let results = dict.lookup("は");
        assert!(!results.is_empty());
        assert_eq!(results[0].pos, PartOfSpeech::Particle);
    }

    #[test]
    fn lookup_miss() {
        let dict = Dictionary::new();
        let results = dict.lookup("zzz");
        assert!(results.is_empty());
    }

    #[test]
    fn lookup_multiple_candidates() {
        let dict = Dictionary::new();

        // あめ should have multiple candidates including 雨
        let results = dict.lookup("あめ");
        assert!(results.len() >= 2);
        let surfaces: Vec<&str> = results.iter().map(|e| e.surface.as_str()).collect();
        assert!(surfaces.contains(&"雨"));
    }

    #[test]
    fn prefix_lookup_basic() {
        let dict = Dictionary::new();

        let results = dict.prefix_lookup("きょう");
        // Should include きょう (今日, 京) and きょうと (京都) and きょねん etc.
        let surfaces: Vec<&str> = results.iter().map(|e| e.surface.as_str()).collect();
        assert!(surfaces.contains(&"今日"));
        assert!(surfaces.contains(&"京都"));
    }

    #[test]
    fn common_prefix_search_basic() {
        let dict = Dictionary::new();

        let results = dict.common_prefix_search("きょうは");
        // Should find entries for き (木/気) and きょう (今日/京)
        assert!(results.len() >= 2);
    }

    #[test]
    fn segmentation_basic() {
        let dict = Dictionary::new();

        let segments = dict.segment("きょうはいいてんきです");
        let words: Vec<&str> = segments.iter().map(|s| s.reading.as_str()).collect();

        assert_eq!(words, vec!["きょう", "は", "いい", "てんき", "です"]);
    }

    #[test]
    fn segmentation_with_candidates() {
        let dict = Dictionary::new();

        let segments = dict.segment("きょうはいいてんきです");
        // きょう segment should have 今日 among its candidates
        let kyou_seg = segments.iter().find(|s| s.reading == "きょう").unwrap();
        let surfaces: Vec<&str> = kyou_seg.candidates.iter().map(|e| e.surface.as_str()).collect();
        assert!(surfaces.contains(&"今日"));
    }

    #[test]
    fn segmentation_unknown_word() {
        let dict = Dictionary::new();

        // ぱぴぷ is not in the dictionary — should be single-char segments
        let segments = dict.segment("ぱぴぷ");
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].reading, "ぱ");
    }

    #[test]
    fn add_entry_runtime() {
        let mut dict = Dictionary::new();
        let count_before = dict.len();

        dict.add_entry(DictionaryEntry {
            reading: "てすと".to_string(),
            surface: "テスト".to_string(),
            pos: PartOfSpeech::Noun,
            frequency: 8000,
        });

        assert_eq!(dict.len(), count_before + 1);
        let results = dict.lookup("てすと");
        assert_eq!(results[0].surface, "テスト");
    }

    #[test]
    fn save_and_load_user_entries() {
        let dir = std::env::temp_dir().join("jaim_test_save_load");
        let path = dir.join("user_dict.json");

        // Create dict and add a user entry
        let mut dict = Dictionary::new();
        let builtin_count = dict.len();
        dict.add_entry(DictionaryEntry {
            reading: "くろーど".to_string(),
            surface: "クロード".to_string(),
            pos: PartOfSpeech::Noun,
            frequency: 5000,
        });
        assert_eq!(dict.len(), builtin_count + 1);

        // Save user entries
        dict.save_user_entries(&path).unwrap();

        // Load into a fresh dictionary
        let mut dict2 = Dictionary::new();
        let loaded = dict2.load_user_entries(&path).unwrap();
        assert_eq!(loaded, 1);
        let results = dict2.lookup("くろーど");
        assert_eq!(results[0].surface, "クロード");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_and_import() {
        let dir = std::env::temp_dir().join("jaim_test_export_import");
        let path = dir.join("export.json");

        let mut dict = Dictionary::new();
        dict.add_entry(DictionaryEntry {
            reading: "らすと".to_string(),
            surface: "Rust".to_string(),
            pos: PartOfSpeech::Noun,
            frequency: 7000,
        });

        // Export all
        dict.export(&path).unwrap();

        // Import into a fresh dictionary — builtin entries should be skipped as duplicates
        let mut dict2 = Dictionary::new();
        let added = dict2.import(&path).unwrap();
        assert_eq!(added, 1); // only the user entry should be new
        let results = dict2.lookup("らすと");
        assert_eq!(results[0].surface, "Rust");

        // Import again — no duplicates
        let added2 = dict2.import(&path).unwrap();
        assert_eq!(added2, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_nonexistent_file_returns_zero() {
        let mut dict = Dictionary::new();
        let result = dict.load_user_entries(Path::new("/tmp/jaim_nonexistent_dict.json"));
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn strip_trailing_commas_basic_array_and_object() {
        let input = r#"[{"a":1,"b":2,},]"#;
        assert_eq!(strip_trailing_commas(input), r#"[{"a":1,"b":2}]"#);
    }

    #[test]
    fn strip_trailing_commas_with_whitespace() {
        let input = "[\n  1,\n  2,\n]\n";
        assert_eq!(strip_trailing_commas(input), "[\n  1,\n  2\n]\n");
    }

    #[test]
    fn strip_trailing_commas_preserves_string_content() {
        let input = r#"{"a":"b,]","c":1,}"#;
        assert_eq!(strip_trailing_commas(input), r#"{"a":"b,]","c":1}"#);
    }

    #[test]
    fn strip_trailing_commas_handles_escaped_quote_in_string() {
        let input = r#"{"a":"x\",y","b":2,}"#;
        assert_eq!(strip_trailing_commas(input), r#"{"a":"x\",y","b":2}"#);
    }

    #[test]
    fn load_user_entries_tolerates_trailing_comma() {
        let dir = std::env::temp_dir().join("jaim_test_trailing_comma");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("user_dict.json");
        let json = r#"[
  {"reading":"てすと","surface":"テスト","pos":"Noun","frequency":100},
  {"reading":"くろーど","surface":"クロード","pos":"Noun","frequency":200},
]"#;
        std::fs::write(&path, json).unwrap();

        let mut dict = Dictionary::new();
        let n = dict.load_user_entries(&path).unwrap();
        assert_eq!(n, 2);
        assert_eq!(dict.lookup("てすと")[0].surface, "テスト");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_user_entries_error_includes_path_and_hint() {
        let dir = std::env::temp_dir().join("jaim_test_bad_json");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("user_dict.json");
        // Genuinely broken: missing closing bracket — neither strict nor
        // lenient parsing can recover, so the error path runs.
        std::fs::write(&path, r#"[{"reading":"x","surface":"y","#).unwrap();

        let mut dict = Dictionary::new();
        let err = dict.load_user_entries(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(path.to_str().unwrap()), "msg should include path: {}", msg);
        assert!(msg.contains("hint:"), "msg should include hint: {}", msg);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn segmentation_common_words() {
        let dict = Dictionary::new();

        // Common compound words must not be split into short fragments,
        // both standalone and in-context (with particles/verbs following)
        let test_cases: &[(&str, &[&str])] = &[
            ("りょうかい", &["りょうかい"]),
            ("かんりょう", &["かんりょう"]),
            ("じゅんばん", &["じゅんばん"]),
            ("しょうがい", &["しょうがい"]),
            ("りょうかいしました", &["りょうかい", "しました"]),
            ("りょうかいです", &["りょうかい", "です"]),
            ("かんりょうした", &["かんりょう", "した"]),
            ("かんりょうです", &["かんりょう", "です"]),
            ("じゅんばんに", &["じゅんばん", "に"]),
            ("じゅんばんです", &["じゅんばん", "です"]),
            ("しょうがいがある", &["しょうがい", "が", "ある"]),
            ("しょうがいです", &["しょうがい", "です"]),
        ];
        for (input, expected) in test_cases {
            let segments = dict.segment(input);
            let words: Vec<&str> = segments.iter().map(|s| s.reading.as_str()).collect();
            assert_eq!(&words, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn segmentation_toukyou() {
        let dict = Dictionary::new();

        let segments = dict.segment("とうきょうにいく");
        let words: Vec<&str> = segments.iter().map(|s| s.reading.as_str()).collect();
        assert_eq!(words, vec!["とうきょう", "に", "いく"]);
    }
}
