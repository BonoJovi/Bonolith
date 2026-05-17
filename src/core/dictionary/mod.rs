/// Dictionary-based Kana to Kanji conversion
///
/// Fast lookup (< 1ms) handling 70-80% of common conversions.
/// Uses trie-based data structure for efficient prefix matching.
/// Includes word segmentation via dynamic programming (minimum-cost path).

mod builtin_dict;
mod connection_cost;
mod trie;

use connection_cost::CONNECTION_COST;

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use trie::Trie;

use crate::core::store::DictStore;

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

impl PartOfSpeech {
    pub const COUNT: usize = 11;

    pub const fn idx(self) -> usize {
        match self {
            PartOfSpeech::Noun => 0,
            PartOfSpeech::Verb => 1,
            PartOfSpeech::Adjective => 2,
            PartOfSpeech::Adverb => 3,
            PartOfSpeech::Particle => 4,
            PartOfSpeech::Auxiliary => 5,
            PartOfSpeech::Conjunction => 6,
            PartOfSpeech::Interjection => 7,
            PartOfSpeech::Prefix => 8,
            PartOfSpeech::Suffix => 9,
            PartOfSpeech::Other => 10,
        }
    }
}

/// Bigram connection cost lookup. Returns 0.0 at sentence start (prev=None).
///
/// Exposed publicly so downstream scorers (e.g. the engine's segmentation
/// filter heuristic) can score candidate segmentations on the same CONN
/// scale the DP optimizes against, keeping the two stages consistent.
pub fn connection_cost(prev: Option<PartOfSpeech>, cur: PartOfSpeech) -> f64 {
    match prev {
        None => 0.0,
        Some(p) => CONNECTION_COST[p.idx()][cur.idx()],
    }
}

/// Index -> PartOfSpeech reverse lookup, used in segmentation DP reconstruction.
const POS_BY_IDX: [PartOfSpeech; PartOfSpeech::COUNT] = [
    PartOfSpeech::Noun,
    PartOfSpeech::Verb,
    PartOfSpeech::Adjective,
    PartOfSpeech::Adverb,
    PartOfSpeech::Particle,
    PartOfSpeech::Auxiliary,
    PartOfSpeech::Conjunction,
    PartOfSpeech::Interjection,
    PartOfSpeech::Prefix,
    PartOfSpeech::Suffix,
    PartOfSpeech::Other,
];

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
    /// Optional persistent store. When attached, mutations to the user
    /// portion of the dictionary are written through to SQLite.
    store: Option<Arc<DictStore>>,
}

impl Dictionary {
    /// Create a new dictionary pre-loaded with the built-in word set.
    pub fn new() -> Self {
        let mut dict = Self {
            entries: Vec::new(),
            trie: Trie::new(),
            user_start: 0,
            store: None,
        };
        dict.load_builtin();
        dict.user_start = dict.entries.len();
        dict
    }

    /// Attach a persistent store. Subsequent calls to
    /// `sync_user_entries_to_store` write the user portion through to it.
    pub fn attach_store(&mut self, store: Arc<DictStore>) {
        self.store = Some(store);
    }

    /// Load all user entries from the attached store and add them to
    /// the in-memory dictionary. Returns the number of entries loaded.
    /// No-op when no store is attached.
    pub fn load_from_store(&mut self) -> io::Result<usize> {
        let store = match &self.store {
            Some(s) => s.clone(),
            None => return Ok(0),
        };
        let entries = store.load_user_entries()?;
        let count = entries.len();
        for entry in entries {
            self.add_entry(entry);
        }
        Ok(count)
    }

    /// Persist the current user portion of the dictionary to the store.
    /// Replaces all rows in user_entries in a single transaction.
    /// No-op when no store is attached.
    pub fn sync_user_entries_to_store(&self) -> io::Result<()> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(()),
        };
        store.replace_all_user_entries(&self.entries[self.user_start..])
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

        // 2D DP: best_cost[i][pos_idx] = minimum cost to reach char position i
        // with the last consumed segment having POS = pos_idx. This lets the
        // optimal path depend on (prev_pos, cur_pos) bigram transitions via
        // CONNECTION_COST.  back[i][pos_idx] = (start, prev_pos_idx at start).
        const INF: f64 = 1e18;
        const PC: usize = PartOfSpeech::COUNT;
        let mut best_cost = vec![[INF; PC]; n + 1];
        let mut back: Vec<[Option<(usize, usize)>; PC]> = vec![[None; PC]; n + 1];

        // BOS sentinel: seed position 0 in the Other slot; the i==0 check
        // below skips connection_cost lookup so the slot identity doesn't matter.
        let bos_slot = PartOfSpeech::Other.idx();
        best_cost[0][bos_slot] = 0.0;

        for i in 0..n {
            let is_bos = i == 0;
            let remaining = &input[byte_offsets[i]..];
            let prefixes = self.trie.common_prefix_search(remaining);

            for prev_p in 0..PC {
                let prev_cost = best_cost[i][prev_p];
                if prev_cost >= INF {
                    continue;
                }
                let prev_pos = if is_bos { None } else { Some(POS_BY_IDX[prev_p]) };

                for (len, indices) in &prefixes {
                    // Group candidates by POS; take the max-frequency entry per POS
                    // so each POS transition is scored on its strongest candidate.
                    let mut best_freq_by_pos = [0u32; PC];
                    for &idx in indices {
                        let e = &self.entries[idx];
                        let p = e.pos.idx();
                        if e.frequency > best_freq_by_pos[p] {
                            best_freq_by_pos[p] = e.frequency;
                        }
                    }

                    let reading: String = chars[i..i + len].iter().collect();
                    let entries: Vec<&DictionaryEntry> = indices
                        .iter()
                        .map(|&idx| &self.entries[idx])
                        .collect();
                    let boost = boost_fn(&reading, &entries);

                    for cur_p in 0..PC {
                        let best_freq = best_freq_by_pos[cur_p];
                        if best_freq == 0 {
                            continue;
                        }
                        let cur_pos = POS_BY_IDX[cur_p];
                        let conn = connection_cost(prev_pos, cur_pos);
                        let cost = segment_cost(*len, best_freq) + conn - boost;
                        let total = prev_cost + cost;
                        if total < best_cost[i + len][cur_p] {
                            best_cost[i + len][cur_p] = total;
                            back[i + len][cur_p] = Some((i, prev_p));
                        }
                    }
                }

                // Fallback: single unknown character → land in the Other slot.
                let unknown_conn = connection_cost(prev_pos, PartOfSpeech::Other);
                let unknown_cost = prev_cost + 20.0 + unknown_conn;
                let other_idx = PartOfSpeech::Other.idx();
                if unknown_cost < best_cost[i + 1][other_idx] {
                    best_cost[i + 1][other_idx] = unknown_cost;
                    back[i + 1][other_idx] = Some((i, prev_p));
                }
            }
        }

        // Pick the min-cost terminal slot at position n
        let mut final_p = 0;
        let mut final_cost = INF;
        for p in 0..PC {
            if best_cost[n][p] < final_cost {
                final_cost = best_cost[n][p];
                final_p = p;
            }
        }

        // Reconstruct boundary list by walking back-pointers
        let mut boundaries = Vec::new();
        let mut cur_idx = n;
        let mut cur_p = final_p;
        while cur_idx > 0 {
            if let Some((start, prev_p)) = back[cur_idx][cur_p] {
                boundaries.push((start, cur_idx));
                cur_idx = start;
                cur_p = prev_p;
            } else {
                boundaries.push((cur_idx - 1, cur_idx));
                cur_idx -= 1;
            }
        }
        boundaries.reverse();

        // Build segments
        let segments: Vec<Segment> = boundaries
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
            .collect();

        self.merge_affix_compounds(segments)
    }

    /// Return the dominant (highest-frequency) PartOfSpeech for a segment, if any.
    fn dominant_pos(seg: &Segment) -> Option<PartOfSpeech> {
        seg.candidates.iter().max_by_key(|e| e.frequency).map(|e| e.pos)
    }

    /// Merge adjacent compound pairs into single segments.
    ///
    /// Rules:
    /// - Noun + Suffix     → compound noun (技術+的, 委員+会)
    /// - Prefix + Noun     → compound noun (各+国, 全+体)
    /// - Prefix + Suffix   → compound noun (何+回, 何+人)
    /// - Noun + Particle   → bunsetsu unit (雨+が, 私+は, 感謝+の)
    /// - Verb + Aux        → verbal complex (食べ+たい, 見た+かった)
    /// - Noun + Conj       → clause head (天気+だから, 学生+だが)
    ///
    /// Merged POS reflects the right element so the result does not
    /// re-trigger the same rule.
    fn merge_affix_compounds(&self, mut segs: Vec<Segment>) -> Vec<Segment> {
        let mut i = 0;
        while i + 1 < segs.len() {
            let cur = Self::dominant_pos(&segs[i]);
            let nxt = Self::dominant_pos(&segs[i + 1]);
            let is_noun_particle = matches!(
                (cur, nxt),
                (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Particle))
            );
            let merge_right_pos = match (cur, nxt) {
                // Noun+Particle: use Particle so the result doesn't re-trigger N+P
                (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Particle)) => {
                    Some(PartOfSpeech::Particle)
                }
                // Verb+Aux: verbal complex (食べ+たい, 見た+かった)
                (Some(PartOfSpeech::Verb), Some(PartOfSpeech::Auxiliary)) => {
                    Some(PartOfSpeech::Auxiliary)
                }
                // Noun+Conj: clause head (天気+だから, 学生+だが)
                (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Conjunction)) => {
                    Some(PartOfSpeech::Conjunction)
                }
                // Affix compounds: always produce Noun
                (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Suffix))
                | (Some(PartOfSpeech::Prefix), Some(PartOfSpeech::Noun))
                | (Some(PartOfSpeech::Prefix), Some(PartOfSpeech::Suffix)) => {
                    Some(PartOfSpeech::Noun)
                }
                _ => None,
            };
            let should_merge = merge_right_pos.is_some();
            if should_merge {
                let right = segs.remove(i + 1);
                let left = &mut segs[i];
                let merged_reading = format!("{}{}", left.reading, right.reading);
                let synthetic_pos = merge_right_pos.unwrap();
                let merged_candidates = {
                    // For Noun+Particle, always use Cartesian product: the DP
                    // already chose N+P over any homophone dict entry (e.g.
                    // きょうは→教派), so preserving that intent is correct.
                    let skip_dict = is_noun_particle;
                    let from_dict: Vec<DictionaryEntry> = if skip_dict {
                        vec![]
                    } else {
                        self.lookup(&merged_reading).into_iter().cloned().collect()
                    };
                    if !from_dict.is_empty() {
                        from_dict
                    } else {
                        // Synthetic: Cartesian product of top-3 candidates from each side
                        let l_tops: Vec<_> = left.candidates.iter().take(3).collect();
                        let r_tops: Vec<_> = right.candidates.iter().take(3).collect();
                        let mr = merged_reading.clone();
                        l_tops
                            .iter()
                            .flat_map(|l| {
                                let mr = mr.clone();
                                r_tops.iter().map(move |r| DictionaryEntry {
                                    reading: mr.clone(),
                                    surface: format!("{}{}", l.surface, r.surface),
                                    pos: synthetic_pos,
                                    frequency: l.frequency.min(r.frequency),
                                })
                            })
                            .collect()
                    }
                };
                *left = Segment {
                    reading: merged_reading,
                    start: left.start,
                    len: left.len + right.len,
                    candidates: merged_candidates,
                };
                // Don't advance i — the new merged segment may itself be mergeable.
            } else {
                i += 1;
            }
        }
        segs
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

    /// Construct a dictionary attached to the default SQLite store at
    /// `~/.local/share/jaim/dict.sqlite`, with all user entries loaded
    /// in memory. Runs legacy JSON migration on first call. Used by the
    /// CLI and dialog flows that previously built a fresh Dictionary
    /// with JSON-backed persistence.
    pub fn with_default_store() -> io::Result<Self> {
        let store = Arc::new(DictStore::open_default_with_migration()?);
        let mut dict = Dictionary::new();
        dict.attach_store(store);
        dict.load_from_store()?;
        Ok(dict)
    }

    /// Export user dictionary entries to a JSON file.
    pub fn export(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.entries[self.user_start..])
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

        // Common サ変動詞 compounds. IPADIC inflects these as noun+する, so
        // high-frequency short pieces (Adv こう, Noun 一滴…) often beat the
        // compound. Adding the verb form directly prevents over-segmentation.
        let sahen_compounds: &[(&str, &[&str])] = &[
            ("こうかいする", &["公開する", "後悔する", "航海する"]),
            ("かいしする", &["開始する"]),
            ("かいしゃする", &["解釈する"]),
        ];
        for &(reading, surfaces) in sahen_compounds {
            for (i, &surface) in surfaces.iter().enumerate() {
                self.add_entry(DictionaryEntry {
                    reading: reading.to_string(),
                    surface: surface.to_string(),
                    pos: PartOfSpeech::Verb,
                    frequency: 9000u32.saturating_sub(i as u32 * 100),
                });
            }
        }

        // Formulaic phrases (挨拶・定型句) that IPADIC decomposes into morphemes.
        // いってき = 一滴 (Noun, 6812) pulls the DP away from いって+き+ます.
        let formulaic: &[(&str, &str)] = &[
            ("いってきます", "行ってきます"),
            ("いってらっしゃい", "行ってらっしゃい"),
            ("おかえりなさい", "お帰りなさい"),
            ("おやすみなさい", "おやすみなさい"),
            // Obligation negation chain. IPADIC splits this into し+なければ+なら+ない
            // (5 morphemes); a single entry prevents the DP from fragmenting it.
            ("しなければならない", "しなければならない"),
            // Desiderative verbal forms. たい dominant POS=Noun (対,freq=5260);
            // かった dominant POS=Verb (勝った,freq=8678). Without compound entries
            // the DP splits たべ+たい and みた+かった rather than treating them as
            // single verbal complexes.
            ("たべたい", "食べたい"),
            ("みたかった", "見たかった"),
        ];
        for &(reading, surface) in formulaic {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Auxiliary,
                frequency: 9000,
            });
        }

        // Compound particles (格助詞連結) absent from IPADIC as single entries.
        // High frequency keeps them competitive against adjacent single-particle splits.
        let compound_particles: &[(&str, &str)] = &[
            ("には", "には"),
            ("へは", "へは"),
            ("にも", "にも"),
            ("でも", "でも"),
            ("からも", "からも"),
            ("までも", "までも"),
        ];
        for &(reading, surface) in compound_particles {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Particle,
                frequency: 9200,
            });
        }

        // Compound nouns (慣用複合名詞) absent from IPADIC as single entries.
        let compound_nouns: &[(&str, &str)] = &[
            // 「目の前」: IPADIC splits as め(Noun)+の(Particle)+まえ(Noun).
            // Adding as a single Noun lets 「めのまえに」 = めのまえ+に (Particle).
            ("めのまえ", "目の前"),
            // 「桃のうち」: proverb phrase. IPADIC splits as もも+のうち (農地).
            // Single entry fixes the tail segment of すもももももももものうち.
            ("もものうち", "桃のうち"),
        ];
        for &(reading, surface) in compound_nouns {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 8000,
            });
        }

        // IT/tech katakana supplement — terms absent from IPADIC that cause
        // mis-segmentation or produce wrong kanji candidates.
        // Frequencies reflect typical input-method usage weight (8000 = common,
        // 7000 = moderately common, 6000 = less frequent but important).
        let katakana_it: &[(&str, &str, u32)] = &[
            // --- Infrastructure / DevOps ---
            ("どっかー",           "ドッカー",           8500), // Docker (ど+っ+かー without entry)
            ("くらうど",           "クラウド",           9000), // Cloud (くら=鞍, うど=独活 without entry)
            ("くーばーねてす",     "クーバーネテス",     7000), // Kubernetes
            ("くべるねてす",       "クベルネテス",       7000), // Kubernetes alt reading
            ("でぷろいめんと",     "デプロイメント",     7500), // Deployment
            ("くらすたー",         "クラスター",         8000), // Cluster
            ("みどるうぇあ",       "ミドルウェア",       7500), // Middleware
            // --- Development tools ---
            ("ふれーむわーく",     "フレームワーク",     8500), // Framework (フレーム+ワーク split)
            ("まいくろさーびす",   "マイクロサービス",   7500), // Microservice (マイクロ+サービス split)
            ("ぎっとはぶ",         "ギットハブ",         8000), // GitHub
            ("りびゅー",           "レビュー",           8500), // Review (りびゅー not in IPADIC)
            // --- AI / ML ---
            ("でぃーぷらーにんぐ", "ディープラーニング", 8000), // Deep Learning (broken without entry)
            // --- General IT ---
            ("くらいあんと",       "クライアント",       8500), // Client
            ("くえりー",           "クエリー",           8000), // Query
            ("えいぴーあい",       "API",                8000), // API
            ("すたーとあっぷ",     "スタートアップ",     7500), // Startup
            ("じゃば",             "ジャバ",             7500), // Java
        ];
        for &(reading, surface, freq) in katakana_it {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Noun,
                frequency: freq,
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

        // Noun+Particle merge: きょう+は → きょうは (bunsetsu unit)
        assert_eq!(words, vec!["きょうは", "いい", "てんき", "です"]);
    }

    #[test]
    fn segmentation_with_candidates() {
        let dict = Dictionary::new();

        let segments = dict.segment("きょうはいいてんきです");
        // Noun+Particle merge produces きょうは; candidates are Cartesian 今日×は
        let kyou_seg = segments.iter().find(|s| s.reading == "きょうは").unwrap();
        let surfaces: Vec<&str> = kyou_seg.candidates.iter().map(|e| e.surface.as_str()).collect();
        assert!(surfaces.contains(&"今日は"), "expected 今日は in {:?}", surfaces);
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
    fn sync_and_load_via_store() {
        let dir = std::env::temp_dir().join("jaim_test_dict_sync");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = Arc::new(DictStore::open(&dir.join("dict.sqlite")).unwrap());

        let mut dict = Dictionary::new();
        dict.attach_store(store.clone());
        let builtin_count = dict.len();
        dict.add_entry(DictionaryEntry {
            reading: "くろーど".to_string(),
            surface: "クロード".to_string(),
            pos: PartOfSpeech::Noun,
            frequency: 5000,
        });
        assert_eq!(dict.len(), builtin_count + 1);
        dict.sync_user_entries_to_store().unwrap();

        let mut dict2 = Dictionary::new();
        dict2.attach_store(store);
        let loaded = dict2.load_from_store().unwrap();
        assert_eq!(loaded, 1);
        let results = dict2.lookup("くろーど");
        assert_eq!(results[0].surface, "クロード");

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
    fn import_tolerates_trailing_comma() {
        let dir = std::env::temp_dir().join("jaim_test_trailing_comma");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("import.json");
        // Use clearly-unique surfaces so they're not deduped against builtin.
        let json = r#"[
  {"reading":"てすとよみ","surface":"𩸽𠮷𠮟","pos":"Noun","frequency":100},
  {"reading":"てすとよみ","surface":"𩸽𠮷𠮟2","pos":"Noun","frequency":200},
]"#;
        std::fs::write(&path, json).unwrap();

        let mut dict = Dictionary::new();
        let n = dict.import(&path).unwrap();
        assert_eq!(n, 2);
        assert_eq!(dict.lookup("てすとよみ").len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn import_error_includes_path_and_hint() {
        let dir = std::env::temp_dir().join("jaim_test_bad_json");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("import.json");
        std::fs::write(&path, r#"[{"reading":"x","surface":"y","#).unwrap();

        let mut dict = Dictionary::new();
        let err = dict.import(&path).unwrap_err();
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
            ("じゅんばんに", &["じゅんばんに"]),
            ("じゅんばんです", &["じゅんばん", "です"]),
            ("しょうがいがある", &["しょうがいが", "ある"]),
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
        // Noun+Particle merge: とうきょう+に → とうきょうに (bunsetsu unit)
        assert_eq!(words, vec!["とうきょうに", "いく"]);
    }

    #[test]
    fn segmentation_katakana_coverage() {
        let dict = Dictionary::new();
        let tests: &[(&str, &[&str])] = &[
            ("さーばー",         &["さーばー"]),
            ("でーたべーす",     &["でーたべーす"]),
            ("ふれーむわーく",   &["ふれーむわーく"]),
            ("らいぶらりー",     &["らいぶらりー"]),
            ("こんてなー",       &["こんてなー"]),
            ("どっかー",         &["どっかー"]),
            ("くらうど",         &["くらうど"]),
            ("えんどぽいんと",   &["えんどぽいんと"]),
            ("りくえすと",       &["りくえすと"]),
            ("れすぽんす",       &["れすぽんす"]),
            ("でぃーぷらーにんぐ", &["でぃーぷらーにんぐ"]),
            ("あるごりずむ",     &["あるごりずむ"]),
            ("きゃっしゅ",       &["きゃっしゅ"]),
            ("でぷろい",         &["でぷろい"]),
            ("まいくろさーびす", &["まいくろさーびす"]),
        ];
        let mut failures = 0;
        for &(input, expected) in tests {
            let segs: Vec<String> = dict.segment(input).iter().map(|s| s.reading.clone()).collect();
            let top: Vec<String> = dict.segment(input).iter().map(|s| {
                s.candidates.first().map(|c| c.surface.clone()).unwrap_or("?".into())
            }).collect();
            let pass = segs.iter().map(|s| s.as_str()).collect::<Vec<_>>() == expected;
            if !pass {
                failures += 1;
                eprintln!("FAIL {:20} → {:?}  (top={:?})", input, segs, top);
            } else {
                eprintln!("pass {:20} → {:?}", input, top);
            }
        }
        assert_eq!(failures, 0, "{failures} katakana cases failed");
    }
}
