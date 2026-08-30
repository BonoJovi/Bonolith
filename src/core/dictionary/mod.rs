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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Frequency overrides for the auto-generated IPADIC dictionary.
///
/// IPADIC's per-entry frequency does not reflect everyday *input* frequency:
/// many core everyday nouns are buried below rarer/abstract homophones (川 sat
/// at 1150, 海 at 1714; 道 < 未知, 目 < 眼, 髪 < 加味). For homophone groups with
/// a clear everyday default we raise that surface just above its group so it
/// wins the cold-start tie; the LLM rerank can still flip it from context.
///
/// Deliberately excluded: genuinely context-dependent pairs (雨/飴, 箸/橋,
/// 切る/着る, 石/医師/意思) — those are left to the LLM, never force-ordered here.
///
/// `(reading, surface, frequency)`. builtin_dict.rs is regenerated from IPADIC,
/// so corrections live here to survive regeneration.
const PRIORITY_OVERRIDES: &[(&str, &str, u32)] = &[
    ("みち", "道", 4100), // was 3786, below 未知 3957
    ("うみ", "海", 5700), // was 1714, below 生み 5576 / 膿 / 産み
    ("め", "目", 5500),   // was 3921, below 眼 5460 / 海布
    ("かわ", "川", 3900), // was 1150, below 皮 3755
    ("かみ", "髪", 4000), // was 3300, below 加味 3609
    ("かみ", "神", 3800), // was 3065, below 加味 3609 (髪 > 神 kept)
    ("そら", "空", 5000), // was 2561, below そら Interjection 4288 — sky/blank
    ("やま", "山", 4500), // was 938, below 止ま 2394 (a verb inflection) — mountain
    ("みず", "水", 4200), // was 2615, below 瑞 3948 (rare "auspicious") — water
    ("りょうり", "料理", 4400), // was 3968, below 良吏 4296 (rare "good official")
    ("かぎ", "鍵", 4200),       // was 4119, below 鈎 4124 (rare "hook/gaff")
    ("ひと", "人", 4100), // was 2882, below 匪徒/費途 3982 (rare "bandit"/"expense")
    ("こと", "事", 4700), // was 1188, buried under 古都 4603/糊塗/殊/琴 (ancient capital etc.)
    ("じしょ", "辞書", 4700), // was 4465, below 字書 4594 (rare/archaic "character dictionary")
    // つぎは: two stacked homophone problems. The 3-char Noun 継歯/継端 (4378, an
    // archaic dental-prosthesis term) wins as one word over つぎ+は, and even when
    // split, 継ぎ (3957) outranks 次 (3508). Raise 次 above 継ぎ AND demote the rare
    // 継歯/継端 so 「つぎは」→ 次(Noun)+は(Particle) (next/next time).
    ("つぎ", "次", 4400),    // was 3508, below 継ぎ 3957
    ("つぎは", "継歯", 1500), // was 4378, archaic; was beating the 次+は split as one word
    ("つぎは", "継端", 1500), // was 4378, archaic; same
    ("とうこう", "投稿", 4600), // was 4436, below 登校 4442 / 陶工 4440 (SNS投稿 is the everyday default)
    // おねがい was 3459, so おねがいします mis-segmented as 尾根(おね 4435)+害します
    // (がいします Verb 5772, a 5-char length bonus). Raising お願い fixes the whole
    // family (お願いします / お願いする / お願いいたします).
    ("おねがい", "お願い", 5800), // was 3459, below 尾根 4435 once します pulls in 害します
    // する: the everyday verb should default to hiragana, not the slangy カタカナ
    // スる. スる (ス + hiragana る) is mixed-kana, so it dodges the all-katakana
    // demotion in surface_adjustment and its IPADIC freq 6520 wins. Demote スる and
    // lift the plain する above the 摺る/擦る group.
    ("する", "スる", 800),  // was 6520, slangy katakana stylization
    ("する", "する", 2800), // was 1306, below 摺る/擦る 2632
    ("ところ", "所", 4200), // was 1685, below 野老 3755 (rare plant/surname) / 処 2978
    // 書き (連用/compound tail: 下書き, 落書き) is far more common as input than
    // 餓鬼; IPADIC buried it (Suffix 3635 < 餓鬼 4353), so がき defaulted to 餓鬼
    // and, worse, its Noun POS suppressed the Noun+Suffix compound merge.
    ("がき", "書き", 4400), // was 3635, below 餓鬼 4353
    // Verbs: the everyday default buried below rare/literary kanji variants.
    // Same exclusion rule — context-dependent verbs (着る/切る, 図る/測る,
    // 上る/登る, 治す/直す, 帰る/変える) are left to the LLM.
    ("みる", "見る", 4200),     // was 2999, below 海松/水松 4120 (seaweed nouns)
    // For the four families below, load_builtin propagates the freq to every
    // inflection whose reading starts with the stem and whose surface starts
    // with the kanji stem. Compound generation gives rare-kanji rivals like
    // 祷った / 游いだ / 綴じた / 捜します a ~5700 boosted freq, so the
    // elevation must sit above 5700 to keep the canonical kanji surface top.
    ("いのる", "祈る", 5800),   // was 2678; base competitor 祷る 2850, inflection 祷った 5700
    ("およぐ", "泳ぐ", 5800),   // was 2770; base competitor 游ぐ 2850, inflection 游いだ 5700
    ("とじる", "閉じる", 5800), // was 2845; base competitor 綴じる 2850, inflection 綴じた 5696
    // つくる family is the one PRIORITY_OVERRIDES verb whose hiragana inflections
    // are boosted well above the base (つくりました reaches 11444 in IPADIC), so
    // the elevation propagated to 作った / 作りました / 作られる needs a much
    // higher ceiling than the other families. Load-time propagation (see
    // load_builtin) applies this same freq to every inflection whose reading
    // starts with "つく" and whose surface starts with "作".
    ("つくる", "作る", 12000),  // was 2449 / bumped past hiragana つくりました 11444
    ("あう", "会う", 2850),     // was 2743, below 遭う/遇う 2768
    ("さがす", "探す", 5800),   // was 2817; base competitor 捜す 2850, inflection 捜します 5666
    // 再 Prefix (4213) is buried under 際 Noun (4950) as the dominant surface
    // for reading さい, so probe_2piece_alternatives puts 際+確認 above 再+
    // 確認 despite 際確認 being a nonsense compound. Lift 再 above 際 for
    // the Prefix path only — the Noun ranking (際 top when さい stands
    // alone) is left untouched.
    ("さい", "再", 5000),       // was 4213 (Prefix); bumps above 際 Noun 4950
    // します family — IPADIC ships these as Verb 7000, which is only just
    // above し(Noun 5002)+ます(Aux 6248) split cost. Once the user has
    // learned ます|ます (score 0.228 → boost 2.28) the split cost falls by
    // ~2, edging out します and turning お願いします into お願い+し+ます —
    // then the affix-compound merge picks up お願い+し as Noun+Suffix and
    // top surface becomes お願い市. Raise the whole family so the single
    // unit stays comfortably below the boosted split.
    ("します", "します", 9500),
    ("しました", "しました", 9500),
    ("しません", "しません", 9000),
    ("しましょう", "しましょう", 9000),
    // Stem+Particle homographs. IPADIC ships several 2/3-char Nouns whose
    // reading is exactly `<content stem> + <strong particle>` (きょうは→
    // 教派, かれは→枯れ葉, そらに→空似, …). The DP prefers those 1-word
    // entries over the Noun+Particle split by a hair (segment_cost -32.5
    // vs -31.8 for きょうは), so 「きょうは」 conversion resurfaces as
    // "教派" instead of "今日は". Demote the archaic / niche readings
    // below 3000 so the everyday N+P parse wins; the compound entry
    // remains reachable if the user actually types towards it. Same
    // shape as the existing つぎは→継歯 demotion (2026-05 fix).
    ("きょうは", "教派", 2500),       // religious sect; 今日+は is the default
    ("きょうが", "恭賀", 2500),       // formal congratulations; 今日+が default
    ("かれは", "枯れ葉", 2500),       // dead-leaves noun; 彼+は default
    ("かれは", "枯葉", 2500),         // alt surface for the same
    // These are 3-char readings; the DP's char_len multiplier keeps the
    // 1-seg entry ahead of そら+に / やま+が until freq < ~2260, so
    // demote further than the 2-char cases above.
    ("そらに", "空似", 1500),         // coincidental resemblance; 空+に default
    ("そらに", "そら似", 1500),       // alt surface
    ("そらで", "空手", 1500),         // からて reading (karate) is untouched
    ("やまが", "山家", 1500),         // mountain-dweller archaic; 山+が default
    // "これ" as 之 is a legitimate archaic reading, but Cartesian
    // product for compounds like これが / これは pulled 之が / 之は
    // into rank-2. Demoting 之 below the take_top cutoff keeps the
    // compound clean while the standalone 之 candidate (which the user
    // has to cycle to) stays intact through the Noun+Particle
    // compound-candidate limit.
    ("これ", "之", 2500),             // was 6273; archaic possessive form
];

pub struct Dictionary {
    entries: Vec<DictionaryEntry>,
    trie: Trie,
    /// Index of the first user-added entry (all entries before this are builtin)
    user_start: usize,
    /// Optional persistent store. When attached, mutations to the user
    /// portion of the dictionary are written through to SQLite.
    store: Option<Arc<DictStore>>,
    /// True once `finalize_sort` has been called. `add_entry` skips its
    /// per-node re-sort until this flips, so bulk load stays O(N) instead
    /// of the O(N²) it would be if every insertion re-sorted.
    sorted: bool,
}

impl Dictionary {
    /// Create a new dictionary pre-loaded with the built-in word set.
    pub fn new() -> Self {
        let mut dict = Self {
            entries: Vec::new(),
            trie: Trie::new(),
            user_start: 0,
            store: None,
            sorted: false,
        };
        dict.load_builtin();
        dict.user_start = dict.entries.len();
        dict.finalize_sort();
        dict
    }

    /// Sort every trie posting list by descending frequency so that
    /// `lookup` / `prefix_lookup` / `common_prefix_search` can hand back
    /// pre-ordered results directly. Runtime insertions call
    /// [`Trie::resort_node`] on just the affected reading.
    fn finalize_sort(&mut self) {
        let freqs: Vec<u32> = self.entries.iter().map(|e| e.frequency).collect();
        self.trie.sort_by_freq(&freqs);
        self.sorted = true;
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
    ///
    /// **Prefer the row-level `*_and_persist` methods for mutations
    /// (add / delete / update / import).** This whole-table replace is a
    /// multi-process data-loss vector: each process loads user_entries
    /// once at startup, and a subsequent replace_all here writes THIS
    /// process's stale snapshot back to the DB, wiping any row the
    /// other frontend (IBus vs Fcitx5) added since our startup. See
    /// Devin PR #3 #2. Kept for the few callers that legitimately want
    /// snapshot semantics (Import of a whole file with intent to
    /// overwrite, tests) — no current path should rely on it for
    /// individual edits.
    pub fn sync_user_entries_to_store(&self) -> io::Result<()> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(()),
        };
        store.replace_all_user_entries(&self.entries[self.user_start..])
    }

    /// Add a user entry to memory AND, if a store is attached, upsert
    /// the single row to the DB. Unlike `sync_user_entries_to_store`,
    /// this does NOT replace the whole `user_entries` table, so rows
    /// the other frontend added since our startup are preserved
    /// (Devin PR #3 #2).
    pub fn add_user_entry_and_persist(&mut self, entry: DictionaryEntry) -> io::Result<()> {
        if let Some(store) = self.store.as_ref() {
            store.upsert_user_entry(&entry)?;
        }
        self.add_entry(entry);
        Ok(())
    }

    /// Remove a user entry by (reading, surface) identity from memory
    /// AND, if a store is attached, DELETE the single row from the DB.
    /// Returns `Ok(true)` when a matching row was found and removed,
    /// `Ok(false)` when nothing matched, `Err` on DB failure.
    ///
    /// Uses `DictStore::remove_user_entry` so only the identified row
    /// is touched — other processes' concurrent additions survive.
    /// In-memory rebuild via `replace_user_entries` is O(n) in user
    /// portion, which is small.
    pub fn remove_user_entry_and_persist(
        &mut self,
        reading: &str,
        surface: &str,
    ) -> io::Result<bool> {
        let present = self.entries[self.user_start..]
            .iter()
            .any(|e| e.reading == reading && e.surface == surface);
        if !present {
            return Ok(false);
        }
        // DB first so a store failure aborts before we mutate memory.
        if let Some(store) = self.store.as_ref() {
            store.remove_user_entry(reading, surface)?;
        }
        let kept: Vec<DictionaryEntry> = self.entries[self.user_start..]
            .iter()
            .filter(|e| !(e.reading == reading && e.surface == surface))
            .cloned()
            .collect();
        self.replace_user_entries(kept);
        Ok(true)
    }

    /// Update a user entry identified by (old_reading, old_surface) to
    /// (new_reading, new_surface), preserving POS and frequency. Both
    /// memory and DB are updated by identity — no whole-table replace.
    /// Returns `Ok(true)` when the old row was found and updated,
    /// `Ok(false)` when nothing matched, `Err` on DB failure.
    pub fn update_user_entry_and_persist(
        &mut self,
        old_reading: &str,
        old_surface: &str,
        new_reading: &str,
        new_surface: &str,
    ) -> io::Result<bool> {
        let (pos, frequency) = match self
            .entries[self.user_start..]
            .iter()
            .find(|e| e.reading == old_reading && e.surface == old_surface)
        {
            Some(e) => (e.pos, e.frequency),
            None => return Ok(false),
        };
        let new_entry = DictionaryEntry {
            reading: new_reading.to_string(),
            surface: new_surface.to_string(),
            pos,
            frequency,
        };
        // DB: single-transaction DELETE-old + UPSERT-new (Devin PR #4
        // review #2). The prior pair of auto-committed statements
        // would leave the DB with neither row if the UPSERT failed
        // after a successful DELETE — the edited word vanished on
        // restart.
        if let Some(store) = self.store.as_ref() {
            store.update_user_entry_by_identity(old_reading, old_surface, &new_entry)?;
        }
        // Memory rebuild.
        let kept: Vec<DictionaryEntry> = self.entries[self.user_start..]
            .iter()
            .filter(|e| !(e.reading == old_reading && e.surface == old_surface))
            .cloned()
            .chain(std::iter::once(new_entry))
            .collect();
        self.replace_user_entries(kept);
        Ok(true)
    }

    /// Import from a JSON file. Each new (reading, surface) pair not
    /// already in memory is added to memory AND upserted to the DB
    /// row-by-row (no whole-table replace). Returns the count added.
    /// Duplicates in memory are skipped; the DB upsert overwrites an
    /// existing row's pos/frequency with the imported entry's values.
    pub fn import_and_persist(&mut self, path: &Path) -> io::Result<usize> {
        let entries = read_and_parse_dict(path)?;
        let mut added = 0;
        for entry in entries {
            if self.has_entry(&entry.reading, &entry.surface) {
                continue;
            }
            if let Some(store) = self.store.as_ref() {
                store.upsert_user_entry(&entry)?;
            }
            self.add_entry(entry);
            added += 1;
        }
        Ok(added)
    }

    /// Add a single entry. During bulk-load (before `finalize_sort`) this
    /// just appends — the trailing `finalize_sort` orders every posting
    /// list once. After finalization each add re-sorts only its own node
    /// so the pre-sort invariant relied on by `lookup` still holds.
    pub fn add_entry(&mut self, entry: DictionaryEntry) {
        let idx = self.entries.len();
        let reading = entry.reading.clone();
        self.entries.push(entry);
        self.trie.insert(&reading, idx);
        if self.sorted {
            let entries = &self.entries;
            self.trie.resort_node(&reading, |i| entries[i].frequency);
        }
    }

    /// Exact lookup: return all candidates for a reading, sorted by
    /// frequency (descending). Order comes from the pre-sorted trie
    /// posting list — no per-call sort.
    pub fn lookup(&self, reading: &str) -> Vec<&DictionaryEntry> {
        let indices = self.trie.exact_lookup(reading);
        indices.iter().map(|&i| &self.entries[i]).collect()
    }

    /// Common prefix search: find all dictionary words that are prefixes of `input`.
    /// Returns Vec of (char_length, entries) sorted by prefix length.
    pub fn common_prefix_search(&self, input: &str) -> Vec<(usize, Vec<&DictionaryEntry>)> {
        self.trie
            .common_prefix_search(input)
            .into_iter()
            .map(|(len, indices)| {
                let entries: Vec<&DictionaryEntry> =
                    indices.iter().map(|&i| &self.entries[i]).collect();
                (len, entries)
            })
            .collect()
    }

    /// Prefix lookup: return candidates for all readings starting with `prefix`.
    pub fn prefix_lookup(&self, prefix: &str) -> Vec<&DictionaryEntry> {
        let indices = self.trie.prefix_lookup(prefix);
        indices.iter().map(|&i| &self.entries[i]).collect()
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

        // Per-prefix data that depends on (i, len) only — hoisted out of
        // the prev_p loop so boost_fn and the accompanying allocations
        // run once per prefix instead of PC times.
        struct PrefixInfo {
            len: usize,
            best_freq_by_pos: [u32; PC],
            boost: f64,
        }

        for i in 0..n {
            let is_bos = i == 0;
            let remaining = &input[byte_offsets[i]..];
            let prefixes = self.trie.common_prefix_search(remaining);
            // Longest dictionary word starting here. A learning boost on a
            // segment shorter than this would let a frequently-typed short word
            // (e.g. けん→件) fragment a longer compound it is a prefix of
            // (けんさく→検索 becoming 件＋柵). Suppress the boost in that case so
            // user learning can reorder candidates but never break a longer
            // dictionary word apart.
            let max_prefix_len = prefixes.iter().map(|(l, _)| *l).max().unwrap_or(0);

            let prefix_infos: Vec<PrefixInfo> = prefixes
                .iter()
                .map(|&(len, indices)| {
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

                    let boost = if len < max_prefix_len {
                        0.0
                    } else {
                        let reading = &input[byte_offsets[i]..byte_offsets[i + len]];
                        let entries: Vec<&DictionaryEntry> = indices
                            .iter()
                            .map(|&idx| &self.entries[idx])
                            .collect();
                        boost_fn(reading, &entries)
                    };

                    PrefixInfo { len, best_freq_by_pos, boost }
                })
                .collect();

            for prev_p in 0..PC {
                let prev_cost = best_cost[i][prev_p];
                if prev_cost >= INF {
                    continue;
                }
                let prev_pos = if is_bos { None } else { Some(POS_BY_IDX[prev_p]) };

                for info in &prefix_infos {
                    for cur_p in 0..PC {
                        let best_freq = info.best_freq_by_pos[cur_p];
                        if best_freq == 0 {
                            continue;
                        }
                        let cur_pos = POS_BY_IDX[cur_p];
                        let conn = connection_cost(prev_pos, cur_pos);
                        let cost = segment_cost(info.len, best_freq) + conn - info.boost;
                        let total = prev_cost + cost;
                        if total < best_cost[i + info.len][cur_p] {
                            best_cost[i + info.len][cur_p] = total;
                            back[i + info.len][cur_p] = Some((i, prev_p));
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

        let segments = self.split_particle_head_segments(segments);
        self.merge_affix_compounds(segments)
    }

    /// Break up 2-char segments whose reading decomposes as (strong Particle
    /// single char) + (content word), when only the middle-of-a-sentence
    /// context makes the tail-as-Suffix reading (がき→書き) win in the DP.
    /// Turning がき into が+き lets the natural もの+が+き parse survive
    /// merge_affix_compounds — otherwise the Noun+Suffix reclass fuses
    /// もの+がき into 物書き and the whole bunsetsu collapses.
    ///
    /// Applied only to segments the DP judged as one word (via a single
    /// dict entry) — segments the DP already split remain untouched.
    ///
    /// The dominant-POS gate is critical. Without it, everyday 2-char
    /// content words that happen to start with a strong Particle kana
    /// (にく 肉 / はし 橋 / とき 時 / とち 土地 / のど 喉 / にわ 庭 /
    /// とし 年 …) get shredded into 助詞+単漢字 pairs — にく→に+苦,
    /// はし→は+市, とき→と+気, とち→と+血 — and the subsequent
    /// Noun+Particle merge with the next segment fuses the orphaned tail
    /// (く/し/き/ち) with the following particle so the original content
    /// word can no longer be recovered from ANY parse. The DP only keeps
    /// a 2-char reading whole via a tail-Suffix win when the segment's
    /// dominant candidate is a Suffix (がき: 書き Suffix 4400 tops 餓鬼
    /// Noun 4353 through PRIORITY_OVERRIDES); everyday Noun-dominant
    /// segments were kept whole for the right reason and must not be
    /// touched here.
    fn split_particle_head_segments(&self, segs: Vec<Segment>) -> Vec<Segment> {
        // A 2-char reading with a strong Noun/Verb/Adj runner-up alongside
        // its Suffix leader is an everyday content word wearing a Suffix
        // hat — 敵 (Noun 5474) alongside 的 (Suffix 7408) for てき,
        // 戦 (Noun 5x) alongside 戦 Suffix for せん, etc. Splitting drops
        // the content word from every reachable parse. がき's runner-up
        // is 餓鬼 (Noun 4353), so the threshold stays above that to keep
        // the がき→物書き Noun+Suffix rebuild firing.
        const CONTENT_RUNNER_UP_MIN: u32 = 4500;
        let mut out: Vec<Segment> = Vec::with_capacity(segs.len());
        for seg in segs {
            let chars: Vec<char> = seg.reading.chars().collect();
            if chars.len() != 2 {
                out.push(seg);
                continue;
            }
            if !matches!(Self::dominant_pos(&seg), Some(PartOfSpeech::Suffix)) {
                out.push(seg);
                continue;
            }
            // Suffix wins on freq, but is there a strong content-word
            // runner-up on the SAME 2-char reading? If so the split
            // would strand it — bug [1] 残: てき split loses 敵 entirely
            // because the ensuing き+を Noun+Particle merge fuses き's
            // 気 with を into 気を, so no parse ever reaches 敵.
            let has_strong_content = seg.candidates.iter().any(|e| {
                matches!(
                    e.pos,
                    PartOfSpeech::Noun | PartOfSpeech::Verb | PartOfSpeech::Adjective
                ) && e.frequency >= CONTENT_RUNNER_UP_MIN
            });
            if has_strong_content {
                out.push(seg);
                continue;
            }
            if !Self::right_reading_looks_like_particle_head(&seg, self) {
                out.push(seg);
                continue;
            }
            let head_r = chars[0].to_string();
            let tail_r = chars[1].to_string();
            let head_cands: Vec<DictionaryEntry> =
                self.lookup(&head_r).into_iter().cloned().collect();
            let tail_cands: Vec<DictionaryEntry> =
                self.lookup(&tail_r).into_iter().cloned().collect();
            if head_cands.is_empty() || tail_cands.is_empty() {
                out.push(seg);
                continue;
            }
            let head_start = seg.start;
            let head_seg = Segment {
                reading: head_r,
                start: head_start,
                len: 1,
                candidates: head_cands,
            };
            let tail_seg = Segment {
                reading: tail_r,
                start: head_start + 1,
                len: 1,
                candidates: tail_cands,
            };
            out.push(head_seg);
            out.push(tail_seg);
        }
        out
    }

    /// Return the dominant (highest-frequency) PartOfSpeech for a segment, if any.
    ///
    /// Tie-breaker: on equal frequency, prefer content POS (Noun / Verb /
    /// Adjective) over affix POS (Suffix / Prefix). PRIORITY_OVERRIDES
    /// keys on (reading, surface) and therefore hits every entry with the
    /// same kanji regardless of POS — 山 Noun and 山 Suffix both land at
    /// 4500 after the override, and `max_by_key`'s "last wins" default
    /// picked the Suffix and vetoed the Noun+Particle merge that would
    /// have turned やまは into 山は (bug: kana top1 for that reading).
    fn dominant_pos(seg: &Segment) -> Option<PartOfSpeech> {
        seg.candidates
            .iter()
            .max_by(|a, b| {
                a.frequency
                    .cmp(&b.frequency)
                    .then_with(|| Self::pos_tiebreaker(a.pos).cmp(&Self::pos_tiebreaker(b.pos)))
            })
            .map(|e| e.pos)
    }

    /// Higher = wins ties in `dominant_pos`. Content POS beats functional
    /// POS, which beats affixes.
    fn pos_tiebreaker(pos: PartOfSpeech) -> u8 {
        match pos {
            PartOfSpeech::Noun | PartOfSpeech::Verb | PartOfSpeech::Adjective => 3,
            PartOfSpeech::Adverb | PartOfSpeech::Interjection => 2,
            PartOfSpeech::Particle
            | PartOfSpeech::Auxiliary
            | PartOfSpeech::Conjunction => 1,
            PartOfSpeech::Prefix | PartOfSpeech::Suffix | PartOfSpeech::Other => 0,
        }
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
    /// Reclassify a segment's effective merge-time POS when it has a strong
    /// affix candidate that dominant_pos missed because a homograph wins on
    /// raw frequency. Common single-kana suffixes like か→化 (5054),
    /// かい→会 (3704), にん→人 (1859) lose the dominant race to か Particle
    /// (7424), かい Particle (4582), 忍 Noun (4206); without this the merge
    /// classifies せいじ+か as Noun+Particle and produces 政治か instead of
    /// 政治化, and skips なん+にん entirely so 何+忍 stays split.
    ///
    /// Only fires when the left-side POS makes an affix compound plausible
    /// (Noun or Prefix), and only when the Suffix candidate is common enough
    /// to be a real affix (freq ≥ 1500 — filters out と→都 800, は→派 649,
    /// が→画 366, で→出 1 for the everyday particle readings).
    ///
    /// The `dict` argument lets us peek inside the right segment: if its
    /// reading starts with a strong Particle (がき → が+き, 部+はい → は+
    /// い) and the remainder is a viable content word, the Noun+Suffix
    /// reclass would fuse across a natural Particle boundary. In that case
    /// we leave the dominant POS alone and let the merge fall through.
    fn effective_right_merge_pos(
        seg: &Segment,
        dominant: Option<PartOfSpeech>,
        left_pos: Option<PartOfSpeech>,
        left_reading: &str,
        dict: &Dictionary,
    ) -> Option<PartOfSpeech> {
        const SUFFIX_MERGE_THRESHOLD: u32 = 1500;
        if !matches!(
            left_pos,
            Some(PartOfSpeech::Noun) | Some(PartOfSpeech::Prefix)
        ) {
            return dominant;
        }
        // Long left → treat as a compound-noun head. The Particle-head veto
        // is aimed at 2-char stand-alone nouns (もの+がき, 部+はい) where the
        // right-hand kana genuinely reads as Particle+content; once the left
        // is 3+ chars it is almost always a compound term (いいん+かい →
        // 委員会, てんらん+かい → 展覧会, しんさいいん+かい → 審査委員会),
        // and the trailing kai/sa/ka/nin reads as a Suffix, not a bunsetsu
        // boundary.
        let left_long = left_reading.chars().count() >= 3;
        // Particle-head suppression runs even when dominant is already
        // Suffix — がき's 書き (Suffix) tops the group at 4400 thanks to a
        // PRIORITY_OVERRIDE, so returning Suffix here would still fuse
        // もの+がき into 物書き and mask the intended もの+が+き parse.
        if matches!(dominant, Some(PartOfSpeech::Auxiliary) | Some(PartOfSpeech::Verb)) {
            return dominant;
        }
        if matches!(dominant, Some(PartOfSpeech::Suffix)) {
            if !left_long && Self::right_reading_looks_like_particle_head(seg, dict) {
                // Downgrade to Noun so the (Noun, Noun) branch skips the
                // merge; the natural Particle-headed parse survives.
                return Some(PartOfSpeech::Noun);
            }
            return dominant;
        }
        // Reclass-to-Suffix only makes sense for readings whose dominant POS
        // misrecognises a compound suffix as a bunsetsu head (Particle) or a
        // stand-alone noun. Dominant Prefix means the reading has a real
        // Prefix role (しん→新 3979 > 心 Suffix 2252), which is about to
        // drive a Prefix+Noun merge with the next segment — swapping to a
        // weak Suffix candidate would fuse into the left compound instead
        // and lose the Prefix reading (bug: しん between 技術的 and 斎院
        // getting fused as 技術的心 rather than pairing with 斎院).
        if !matches!(
            dominant,
            Some(PartOfSpeech::Particle) | Some(PartOfSpeech::Noun)
        ) {
            return dominant;
        }
        let has_strong_suffix = seg.candidates.iter().any(|e| {
            e.pos == PartOfSpeech::Suffix && e.frequency >= SUFFIX_MERGE_THRESHOLD
        });
        if !has_strong_suffix {
            return dominant;
        }
        if !left_long && Self::right_reading_looks_like_particle_head(seg, dict) {
            return dominant;
        }
        // Particle-dominant readings whose Suffix homograph is a niche
        // proper-noun ending (だけ → 岳 as in 剣岳/穂高岳) should stay
        // Particle in modern-text conversion — reclassifying to Suffix
        // makes 件だけ fuse into 県岳 (min freq 2574) instead of a
        // 件+だけ bunsetsu. Only fires when the dominant Particle is much
        // stronger than the strongest Suffix candidate.
        if Self::particle_dominates_suffix(seg) {
            return dominant;
        }
        Some(PartOfSpeech::Suffix)
    }

    /// True when the segment's Particle candidate is dramatically stronger
    /// than its Suffix candidate — the ratio ≥ 3 filters だけ (9713 vs 2574,
    /// ×3.77) while leaving か (7424 vs 5054, ×1.47) and かい (4582 vs 3704,
    /// ×1.24) firing as before.
    fn particle_dominates_suffix(seg: &Segment) -> bool {
        let mut top_particle = 0u32;
        let mut top_suffix = 0u32;
        for e in &seg.candidates {
            match e.pos {
                PartOfSpeech::Particle if e.frequency > top_particle => {
                    top_particle = e.frequency;
                }
                PartOfSpeech::Suffix if e.frequency > top_suffix => {
                    top_suffix = e.frequency;
                }
                _ => {}
            }
        }
        top_particle >= 6000 && top_suffix > 0 && top_particle >= top_suffix * 3
    }

    /// True when the segment's reading decomposes cleanly as
    /// (strong Particle single char) + (content word). Used to veto the
    /// Noun→Suffix merge reclass so the natural Particle-headed parse
    /// wins instead of being fused into a compound noun.
    ///
    /// Particle threshold 7000 is chosen so common bunsetsu markers (が
    /// 9814 / を 9430 / に 9113 / は 8968) all pass while an incidental
    /// homograph like で→出 (Suffix 1) never does.
    fn right_reading_looks_like_particle_head(seg: &Segment, dict: &Dictionary) -> bool {
        const HEAD_PARTICLE_MIN: u32 = 7000;
        const TAIL_CONTENT_MIN: u32 = 2000;
        let mut it = seg.reading.chars();
        let first = match it.next() {
            Some(c) => c,
            None => return false,
        };
        let tail: String = it.collect();
        if tail.is_empty() {
            return false;
        }
        let head_r = first.to_string();
        let head_ents = dict.lookup(&head_r);
        let head_strong = head_ents.iter().any(|e| {
            e.pos == PartOfSpeech::Particle && e.frequency >= HEAD_PARTICLE_MIN
        });
        if !head_strong {
            return false;
        }
        let tail_ents = dict.lookup(&tail);
        tail_ents.iter().any(|e| {
            matches!(
                e.pos,
                PartOfSpeech::Noun | PartOfSpeech::Verb | PartOfSpeech::Adjective
            ) && e.frequency >= TAIL_CONTENT_MIN
        })
    }

    /// Reclassify the left segment's POS when a strong Prefix candidate would
    /// let the pair merge into an affix compound. IPADIC often has both a
    /// Prefix and a Noun surface for the same reading (さい: 際 Noun 4950 /
    /// 再 Prefix 4213; ふ: 歩 Noun 4764 / 不 Prefix 4503), and dominant_pos
    /// picks the Noun by raw freq. That kills the Prefix+Noun merge for
    /// さい+かくにん (再確認) and ふ+きょう (不況). A high 3500 threshold
    /// filters out marginal Prefix homographs like き→貴 (2087).
    fn effective_left_merge_pos(
        seg: &Segment,
        dominant: Option<PartOfSpeech>,
        right_pos: Option<PartOfSpeech>,
    ) -> Option<PartOfSpeech> {
        const PREFIX_MERGE_THRESHOLD: u32 = 3500;
        if !matches!(dominant, Some(PartOfSpeech::Noun)) {
            return dominant;
        }
        if !matches!(
            right_pos,
            Some(PartOfSpeech::Noun) | Some(PartOfSpeech::Suffix)
        ) {
            return dominant;
        }
        let has_strong_prefix = seg.candidates.iter().any(|e| {
            e.pos == PartOfSpeech::Prefix && e.frequency >= PREFIX_MERGE_THRESHOLD
        });
        if has_strong_prefix {
            Some(PartOfSpeech::Prefix)
        } else {
            dominant
        }
    }

    fn merge_affix_compounds(&self, mut segs: Vec<Segment>) -> Vec<Segment> {
        let mut i = 0;
        while i + 1 < segs.len() {
            let raw_cur = Self::dominant_pos(&segs[i]);
            let raw_nxt = Self::dominant_pos(&segs[i + 1]);
            // First pass: refine the right POS given the raw left POS. Then
            // refine the left POS given the refined right POS. Two passes let
            // Noun+Noun cases where BOTH sides need reclassification (さい+
            // かくにん where 再 Prefix is buried under 際 Noun and 確認 is a
            // straight Noun) still fire as Prefix+Noun.
            let nxt = Self::effective_right_merge_pos(
                &segs[i + 1],
                raw_nxt,
                raw_cur,
                &segs[i].reading,
                self,
            );
            let cur = Self::effective_left_merge_pos(&segs[i], raw_cur, nxt);
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
            let Some(synthetic_pos) = merge_right_pos else {
                i += 1;
                continue;
            };
            {
                let right = segs.remove(i + 1);
                let left = &mut segs[i];
                let merged_reading = format!("{}{}", left.reading, right.reading);
                // Which side plays which role in the synthetic surface? For
                // affix compounds we want to filter candidates to their role
                // (Prefix / Noun / Suffix) so the Cartesian product doesn't
                // pull in the same homograph that dominant_pos already
                // side-stepped (せいじ+か's か Particle at 7424).
                let (l_role, r_role) = match (cur, nxt) {
                    (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Suffix)) => {
                        (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Suffix))
                    }
                    (Some(PartOfSpeech::Prefix), Some(PartOfSpeech::Noun)) => {
                        (Some(PartOfSpeech::Prefix), Some(PartOfSpeech::Noun))
                    }
                    (Some(PartOfSpeech::Prefix), Some(PartOfSpeech::Suffix)) => {
                        (Some(PartOfSpeech::Prefix), Some(PartOfSpeech::Suffix))
                    }
                    // Noun+Particle: filter the right side to Particle POS
                    // only. Without this, take_top's top-5 pulls in the
                    // homograph Noun candidates for common trailing
                    // particles (は→刃/歯/覇/葉, の→之/野/盧) and the
                    // Cartesian product ships nonsense compounds like
                    // 後刃 / 私之 / 跡刃 as visible candidates. Left stays
                    // unfiltered so the dictionary's kana entry for the
                    // stem (わたし, あと …) keeps surfacing alongside the
                    // kanji form. Verb+Aux and Noun+Conj get the same
                    // treatment for the same reason (auxiliary / conjunction
                    // homographs would otherwise leak through).
                    (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Particle)) => {
                        (None, Some(PartOfSpeech::Particle))
                    }
                    (Some(PartOfSpeech::Verb), Some(PartOfSpeech::Auxiliary)) => {
                        (None, Some(PartOfSpeech::Auxiliary))
                    }
                    (Some(PartOfSpeech::Noun), Some(PartOfSpeech::Conjunction)) => {
                        (None, Some(PartOfSpeech::Conjunction))
                    }
                    _ => (None, None),
                };
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
                        // Synthetic: Cartesian product of top-5 candidates
                        // from each side, filtered by role when this is an
                        // affix compound. Filtering falls back to the raw
                        // top-5 if the role-filtered list is empty (defensive
                        // — shouldn't happen once effective_right_merge_pos
                        // fires, but keeps the merge productive either way).
                        //
                        // Additionally probe alternative 2-piece splits of
                        // the merged reading: when the DP picked さいかく+
                        // にん (才覚 4426 + 人 Suffix 1859) the primary
                        // cartesian top is 才覚人 at 1859, but the same
                        // reading also decomposes as さい+かくにん (再/際
                        // + 確認). Merging those alternatives into the
                        // candidate list keeps 再確認 / 何時間 reachable
                        // instead of disappearing under a suboptimal DP
                        // choice.
                        // Widened to 5 (was 3) so common Suffix homograph
                        // groups still surface their long-tail members: for
                        // か, top-5 Suffix reaches 家 (rank 5), keeping
                        // 政治家 / 芸術家 / 研究家 discoverable next to the
                        // 化 leader from a single reading.
                        fn take_top<'a>(
                            cands: &'a [DictionaryEntry],
                            role: Option<PartOfSpeech>,
                        ) -> Vec<&'a DictionaryEntry> {
                            if let Some(p) = role {
                                // Functional POS are near-closed classes
                                // where the top-frequency entry is the
                                // modern-Japanese default and the long
                                // tail is archaic/rare (の's Particle
                                // homograph 之, は's Particle homograph
                                // 巴, だ's Auxiliary homograph 抱 …).
                                // Enumerating those on the Cartesian
                                // product produces "後之" / "私之" style
                                // nonsense candidates. Cap to top-1 for
                                // Particle / Auxiliary / Conjunction; the
                                // open-class Suffix / Noun / Prefix roles
                                // still enumerate the top-5 so long-tail
                                // homographs (家 for か) stay reachable.
                                let cap = match p {
                                    PartOfSpeech::Particle
                                    | PartOfSpeech::Auxiliary
                                    | PartOfSpeech::Conjunction => 1,
                                    _ => 5,
                                };
                                let filtered: Vec<&DictionaryEntry> = cands
                                    .iter()
                                    .filter(|e| e.pos == p)
                                    .take(cap)
                                    .collect();
                                if !filtered.is_empty() {
                                    return filtered;
                                }
                            }
                            cands.iter().take(5).collect()
                        }
                        let l_tops = take_top(&left.candidates, l_role);
                        let r_tops = take_top(&right.candidates, r_role);
                        let mr = merged_reading.clone();
                        let mut cands: Vec<DictionaryEntry> = l_tops
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
                            .collect();
                        cands.extend(self.probe_2piece_alternatives(&merged_reading));
                        // Sort highest-freq first, then drop duplicate
                        // surfaces globally (not just adjacent — same-surface
                        // entries land far apart when many surfaces share the
                        // same freq band and would slip past dedup_by).
                        cands.sort_by(|a, b| b.frequency.cmp(&a.frequency));
                        let mut seen: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        cands.retain(|e| seen.insert(e.surface.clone()));
                        // 30 keeps the long-tail Suffix homograph reachable
                        // (for せいじか, top-5 Suffix × top-5 left Noun fills
                        // 25 slots at freq bands 5054/3255/2784/2682/2598 —
                        // 家 sits in the last band).
                        cands.truncate(30);
                        cands
                    }
                };
                *left = Segment {
                    reading: merged_reading,
                    start: left.start,
                    len: left.len + right.len,
                    candidates: merged_candidates,
                };
                // Don't advance i — the new merged segment may itself be mergeable.
            }
        }
        segs
    }

    /// Enumerate 2-piece decompositions of a merged reading whose left half
    /// resolves to a Noun/Prefix and right half to a Noun/Suffix, and return
    /// the top-scoring cartesian entries as extra candidates.
    ///
    /// The DP occasionally picks a longer left prefix that gives a low-freq
    /// compound (さいかく+人 = 才覚人 at 1859) when a shorter Prefix+Noun
    /// split of the same reading would produce a much stronger word (さい+
    /// かくにん = 再/際 + 確認 at ≥4213). Merging the alternative splits into
    /// the candidate pool keeps those forms discoverable without needing a
    /// bespoke dict entry per compound.
    fn probe_2piece_alternatives(&self, reading: &str) -> Vec<DictionaryEntry> {
        let chars: Vec<char> = reading.chars().collect();
        if chars.len() < 3 {
            // 2-char readings can only split as 1+1; the merge already
            // produced that cartesian, so alternatives contribute nothing.
            return Vec::new();
        }
        let mut byte_offsets = vec![0usize];
        for c in &chars {
            let prev = byte_offsets.last().copied().unwrap_or(0);
            byte_offsets.push(prev + c.len_utf8());
        }
        let mut out: Vec<DictionaryEntry> = Vec::new();
        for split in 1..chars.len() {
            let (lb, rb) = (byte_offsets[split], byte_offsets[chars.len()]);
            let left_r = &reading[..lb];
            let right_r = &reading[lb..rb];
            let left_char_count = split;
            let right_char_count = chars.len() - split;
            let l_ents = self.lookup(left_r);
            let r_ents = self.lookup(right_r);
            if l_ents.is_empty() || r_ents.is_empty() {
                continue;
            }
            for l in l_ents.iter().take(4) {
                if !matches!(l.pos, PartOfSpeech::Noun | PartOfSpeech::Prefix) {
                    continue;
                }
                for r in r_ents.iter().take(4) {
                    if !matches!(r.pos, PartOfSpeech::Noun | PartOfSpeech::Suffix) {
                        continue;
                    }
                    // Noun+Noun alternatives are only trustworthy when the
                    // right side looks like a real compound-forming word:
                    // multi-char AND high-freq (≥ 5000, matching entries
                    // like 時間 9523, 確認 5030, 学校 8376). Single-char
                    // right pieces are almost always Suffix-role in
                    // practice (し→市, か→化, き→気) and their Noun
                    // homographs degenerate into nonsense compounds like
                    // お願い+市 or 何+忍 that would sort above the
                    // legitimate Noun+Suffix product on raw freq alone.
                    if l.pos == PartOfSpeech::Noun
                        && r.pos == PartOfSpeech::Noun
                        && (right_char_count < 2 || r.frequency < 5000)
                    {
                        continue;
                    }
                    // Guard against 1-char Prefix+Noun where the "Prefix"
                    // is a weak homograph — same reasoning as the Noun+Noun
                    // rule above.
                    if l.pos == PartOfSpeech::Prefix
                        && left_char_count < 2
                        && l.frequency < 3000
                    {
                        continue;
                    }
                    out.push(DictionaryEntry {
                        reading: reading.to_string(),
                        surface: format!("{}{}", l.surface, r.surface),
                        pos: PartOfSpeech::Noun,
                        frequency: l.frequency.min(r.frequency),
                    });
                }
            }
        }
        out
    }

    /// Build candidate entries for a reading the user has forced into a single
    /// segment via a manual boundary resize.
    ///
    /// A flat [`lookup`](Self::lookup) returns nothing for a multi-token reading
    /// such as a particle glued onto a verb ("がふる"), so the segment would
    /// otherwise collapse to bare kana and could never rerank to the intended
    /// word. In that case we sub-segment the reading (reusing the DP segmenter +
    /// affix merge) and return the Cartesian product of each piece's top
    /// candidates, mirroring [`merge_affix_compounds`](Self::merge_affix_compounds)
    /// — so the forced segment still surfaces real words (が降る).
    pub fn candidates_for_unit(&self, reading: &str) -> Vec<DictionaryEntry> {
        // Real word or known compound — use the dictionary entries directly.
        let direct: Vec<DictionaryEntry> = self.lookup(reading).into_iter().cloned().collect();
        if !direct.is_empty() {
            return direct;
        }

        // Decompose the multi-token reading and recombine its pieces.
        let subs = self.segment(reading);
        if subs.len() < 2 {
            return direct; // can't decompose further; caller falls back to kana
        }

        const PER_SEG: usize = 3;
        const MAX_TOTAL: usize = 12;
        // Iteratively expand the Cartesian product, keeping the highest-frequency
        // combinations so the candidate list stays bounded.
        let mut combos: Vec<(String, u32)> = vec![(String::new(), u32::MAX)];
        for sub in &subs {
            let tops: Vec<&DictionaryEntry> = sub.candidates.iter().take(PER_SEG).collect();
            let mut next: Vec<(String, u32)> = Vec::new();
            if tops.is_empty() {
                // No dictionary entry for this piece — carry its kana through.
                for (acc, freq) in &combos {
                    next.push((format!("{}{}", acc, sub.reading), *freq));
                }
            } else {
                for (acc, freq) in &combos {
                    for t in &tops {
                        next.push((format!("{}{}", acc, t.surface), (*freq).min(t.frequency)));
                    }
                }
            }
            next.sort_by_key(|(_, freq)| std::cmp::Reverse(*freq));
            next.truncate(MAX_TOTAL);
            combos = next;
        }

        combos
            .into_iter()
            .map(|(surface, frequency)| DictionaryEntry {
                reading: reading.to_string(),
                surface,
                pos: PartOfSpeech::Other,
                frequency,
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
            .join("bonolith");
        Ok(data_dir.join("user_dict.json"))
    }

    /// Construct a dictionary attached to the default SQLite store at
    /// `~/.local/share/bonolith/dict.sqlite`, with all user entries loaded
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
        let overrides: std::collections::HashMap<(&str, &str), u32> = PRIORITY_OVERRIDES
            .iter()
            .map(|&(r, s, f)| ((r, s), f))
            .collect();

        // Reading/surface stems extracted from PRIORITY_OVERRIDES verb entries.
        // For each override that names a kanji-preferred verb base form (e.g.
        // つくる → 作る), we treat every inflection whose reading starts with
        // "つく" *and* whose surface starts with "作" as belonging to the same
        // family, and raise it to the base override's frequency. That lifts
        // 作った / 作って / 作っている / 作ります / 作られる in one shot,
        // instead of listing every form.
        let verb_families = collect_verb_stems(PRIORITY_OVERRIDES);

        // Precompute the top RAW kanji-form frequency per (reading, POS) for
        // Verb/Adjective/Noun. Used below to cap emphatic-katakana surface
        // variants (ツクった / カエった / ハサんだ / モノ / コト …) that
        // IPADIC assigns unrealistically low costs to; without the cap, the
        // compound ×2 boost or a marginal +50 lead over the kanji dominant
        // lets them dominate the canonical form of the same reading.
        //
        // The cap deliberately uses raw IPADIC freq (not the post-override
        // value), so a PRIORITY_OVERRIDES bump like つくる→作る at 12000 only
        // lifts the kanji form itself — the kata variant is capped just below
        // the raw kanji ceiling and ends up ranked below every hiragana form
        // in the family, not sandwiched between the elevated kanji and the
        // untouched hiragana.
        let mut max_kanji_freq: std::collections::HashMap<(&'static str, PartOfSpeech), u32> =
            std::collections::HashMap::new();
        for &(reading, surface, pos, freq) in builtin_dict::BUILTIN_ENTRIES {
            if !matches!(
                pos,
                PartOfSpeech::Verb | PartOfSpeech::Adjective | PartOfSpeech::Noun
            ) {
                continue;
            }
            if !surface_contains_kanji(surface) {
                continue;
            }
            max_kanji_freq
                .entry((reading, pos))
                .and_modify(|v| *v = (*v).max(freq))
                .or_insert(freq);
        }

        for &(reading, surface, pos, frequency) in builtin_dict::BUILTIN_ENTRIES {
            let mut frequency = overrides
                .get(&(reading, surface))
                .copied()
                .unwrap_or(frequency);

            if matches!(pos, PartOfSpeech::Verb | PartOfSpeech::Adjective) {
                // Family-based elevation: propagate the base override's freq
                // to every inflected kanji surface of the same verb family.
                for &(read_stem, surf_stem, boost) in &verb_families {
                    if reading.starts_with(read_stem) && surface.starts_with(surf_stem) {
                        if frequency < boost {
                            frequency = boost;
                        }
                        break;
                    }
                }
            }
            // Emphatic-katakana cap: keep the kata variant in candidates but
            // never above the reading's canonical kanji form. Applied to
            // Verb / Adjective / Noun — the same IPADIC artifact that makes
            // ツクった beat 作った also makes モノ (3302) beat 物 (3248) and
            // コト (2795) beat 事 (1188) for common everyday nouns.
            if matches!(
                pos,
                PartOfSpeech::Verb | PartOfSpeech::Adjective | PartOfSpeech::Noun
            ) && is_kata_dominant(surface)
            {
                if let Some(&kanji_top) = max_kanji_freq.get(&(reading, pos)) {
                    let cap = kanji_top.saturating_sub(200).max(500);
                    if frequency > cap {
                        frequency = cap;
                    }
                }
            }

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
            ("さんかく", &["△", "▲", "▽", "▼"]),
            ("しかく", &["□", "■", "◇", "◆"]),
            ("ほし", &["☆", "★"]),
            ("こめ", &["※"]),
            ("から", &["〜", "～"]),
            ("てん", &["・", "…", "‥", "、"]),
            // "まる" used to appear twice — one row for the shape group
            // (○◎●◯), one adding 。 to the front — so lookup returned
            // ○/◎/● at two frequency tiers each. Merge into one row with
            // 。 first (the sentence terminator is the more common intent
            // in Japanese text) followed by the shape variants.
            ("まる", &["。", "○", "◎", "●", "◯"]),
            ("かっこ", &["「」", "「", "」", "『』", "『", "』", "【】", "【", "】", "（）", "（", "）", "〔〕", "［］", "｛｝", "〈〉", "《》"]),
            ("かぎかっこ", &["「」", "「", "」", "『』", "『", "』"]),
            ("すみかっこ", &["【】", "【", "】"]),
            ("まるかっこ", &["（）", "（", "）"]),
            ("ゆうびん", &["〒"]),
        ];
        // Low base frequency (like emoji): a symbol shortcut must stay reachable
        // when its reading has no real word (やじるし, かっこ), but must never
        // outrank a common homophone word. At 8000 these clobbered everyday
        // words — e.g. した→↓ beat 舌/下/した, うえ→↑ beat 上, から→〜 beat から.
        const SYMBOL_BASE_FREQ: u32 = 500;
        for &(reading, surfaces) in symbols {
            for (i, &surface) in surfaces.iter().enumerate() {
                self.add_entry(DictionaryEntry {
                    reading: reading.to_string(),
                    surface: surface.to_string(),
                    pos: PartOfSpeech::Other,
                    // First candidate keeps the highest frequency within the group.
                    frequency: SYMBOL_BASE_FREQ.saturating_sub(i as u32 * 10),
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
            // Counter + か Particle ("何回か / 何人か / 何時間か") — the
            // approximate-quantity phrase. Without a direct entry the DP
            // merges 何回+か via Noun+Suffix reclass (か→化 5054) and the
            // surface becomes 何回化. Register these as high-freq Noun
            // compounds so the direct match beats the compound cartesian.
            ("なんかいか",   "何回か"),
            ("なんにんか",   "何人か"),
            ("なんにちか",   "何日か"),
            ("なんじかんか", "何時間か"),
            ("なんどか",     "何度か"),
            ("なんこか",     "何個か"),
            ("なんぼんか",   "何本か"),
            ("なんさつか",   "何冊か"),
            ("なんねんか",   "何年か"),
            ("なんかげつか", "何ヶ月か"),
            ("いくつか",     "いくつか"),
            ("いくらか",     "いくらか"),
            // Colloquial coordinating し ("〜だし、〜ですし") — IPADIC has
            // し only as a low-freq Particle (2158) buried under 市 5002 /
            // 氏 4646, so ですし segments as です+すし(寿司 4372)+... and
            // becomes "で鮨". Register the everyday compound forms so they
            // win the whole substring instead. だし alone is skipped: its
            // canonical noun reading is 出汁 (4201) / 山車 (4290), and a
            // 9000-freq hiragana entry would clobber that. だしね / だしよ /
            // だしよね are unambiguous colloquial forms and safe to boost.
            ("ですし", "ですし"),
            ("ですしね", "ですしね"),
            ("ですしよ", "ですしよ"),
            ("ですしよね", "ですしよね"),
            ("ですしから", "ですしから"),
            ("だしね", "だしね"),
            ("だしよ", "だしよ"),
            ("だしよね", "だしよね"),
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
            // Explanatory ending 準体助詞の+コピュラです. IPADIC's Particle→Auxiliary
            // bigram (の→です) costs 6.287, while ので(Particle 8664)→す(Noun) is only
            // 3.500, so のです mis-segmented as ので+す(→素). A single Auxiliary entry
            // wins outright; Verb/Adj→Aux connection (1.789/3.837) keeps 〜のです natural.
            ("のです", "のです"),
        ];
        for &(reading, surface) in formulaic {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Auxiliary,
                frequency: 9000,
            });
        }

        // Passive / honorific auxiliary される and its conjugations. IPADIC carries
        // these only as freq-2 hiragana surfaces, so サ変名詞+される (登録される,
        // 削除される, 表示される…) loses badly: とうろくされて mis-segmented as
        // 当路(4378)+腐れて(くされて Verb 5694). A high-frequency Auxiliary entry wins;
        // Noun→Aux connection (3.469) keeps the サ変名詞 boundary natural.
        let sareru_aux: &[(&str, &str)] = &[
            ("される", "される"),
            ("されて", "されて"),
            ("された", "された"),
            ("されない", "されない"),
            ("されます", "されます"),
            ("されました", "されました"),
        ];
        for &(reading, surface) in sareru_aux {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Auxiliary,
                frequency: 8500,
            });
        }

        // Godan さ-row verb past/te forms. IPADIC tags 話し / 出し / 探し only
        // under 連用形, and the standard compound generator emits た/て from
        // 連用形 for ichidan verbs only (five-dan さ has an identical stem but
        // no 連用タ接続 row in Verb.csv). As a result the auto-generated dict
        // contains 話します but no 話した / 話して — user typing はなした
        // used to reach only 話し手 (Noun) via segmentation.
        //
        // Frequencies are set high enough (≥ 8000) to survive segmentation:
        // the DP's user_scorer boost multiplies learned はなし|話 selections
        // by 10 (see engine::start_conversion), so a 3-char split cost easily
        // beats a 4-char single-unit match at freq ≤ 6500. 8500 puts single
        // unit cost at −35.2 vs split ≈ −35.0, keeping 話して as the top hit
        // for typical learning histories.
        //
        // Each row lists a base reading and every kanji surface that shares it.
        let godan_su_verbs: &[(&str, &[&str], u32)] = &[
            ("はなす",     &["話す"],                     8500),
            ("だす",       &["出す"],                     8500),
            ("かす",       &["貸す"],                     8000),
            ("おす",       &["押す"],                     8000),
            ("けす",       &["消す"],                     8000),
            ("なおす",     &["直す", "治す"],             8000),
            ("おこす",     &["起こす"],                   8000),
            ("うごかす",   &["動かす"],                   8000),
            ("おとす",     &["落とす"],                   8000),
            ("わたす",     &["渡す"],                     8000),
            ("さがす",     &["探す", "捜す"],             8500),
            ("しめす",     &["示す"],                     8000),
            ("うつす",     &["移す", "写す", "映す"],     8000),
            ("とおす",     &["通す"],                     8000),
            ("かえす",     &["返す"],                     8000),
            ("もどす",     &["戻す"],                     8000),
            ("まわす",     &["回す"],                     8000),
            ("ふやす",     &["増やす"],                   7500),
            ("へらす",     &["減らす"],                   7500),
            ("さす",       &["差す", "指す", "刺す"],     8000),
            ("かくす",     &["隠す"],                     7500),
            ("よごす",     &["汚す"],                     7500),
            ("あらわす",   &["表す", "現す"],             7500),
            ("たす",       &["足す"],                     7500),
            ("いかす",     &["生かす"],                   7500),
            ("あます",     &["余す"],                     7500),
            ("うながす",   &["促す"],                     7500),
            ("ためす",     &["試す"],                     8000),
            ("のこす",     &["残す"],                     8000),
            ("こわす",     &["壊す"],                     8000),
            ("たおす",     &["倒す"],                     7500),
            ("ころす",     &["殺す"],                     7500),
            ("なくす",     &["無くす", "亡くす"],         7500),
        ];
        const SU_TAILS: &[(&str, &str)] = &[
            ("した", "した"),
            ("して", "して"),
            ("している", "している"),
            ("しています", "しています"),
            ("していた", "していた"),
            ("していない", "していない"),
            ("しません", "しません"),  // negative polite
        ];
        for &(reading, surfaces, base_freq) in godan_su_verbs {
            let read_stem_len = reading.len() - 'す'.len_utf8();
            let read_stem = &reading[..read_stem_len];
            for &(read_suf, surf_suf) in SU_TAILS {
                let r = format!("{}{}", read_stem, read_suf);
                for (i, surf) in surfaces.iter().enumerate() {
                    let surf_stem_len = surf.len() - 'す'.len_utf8();
                    let surf_stem = &surf[..surf_stem_len];
                    self.add_entry(DictionaryEntry {
                        reading: r.clone(),
                        surface: format!("{}{}", surf_stem, surf_suf),
                        pos: PartOfSpeech::Verb,
                        frequency: base_freq.saturating_sub(i as u32 * 50),
                    });
                }
            }
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
            // 〜値 (value) suffix compounds — IPADIC ships 値 with reading
            // ち as Suffix at freq 1, so 期待値 / 平均値 / 最大値 segment
            // as きたい+ち, へいきん+ち etc. and the ち slot resolves to
            // 血/知/治 by dict rank. Register the common IT/math compounds
            // directly so the DP short-circuits to the correct kanji.
            ("きたいち",       "期待値"),
            ("へいきんち",     "平均値"),
            ("さいだいち",     "最大値"),
            ("さいしょうち",   "最小値"),
            ("ぜったいち",     "絶対値"),
            ("すうち",         "数値"),
            ("しきべつち",     "識別値"),
            ("しきべつし",     "識別子"),
            ("ひょうじゅんち", "標準値"),
            ("じっそくち",     "実測値"),
            ("かんそくち",     "観測値"),
            ("しょきち",       "初期値"),
            ("こていち",       "固定値"),
            // Counter/measure + だけ ("〜件だけ / 〜人だけ / 〜個だけ" —
            // "only N of them"). Without a direct entry the Noun+Particle
            // cartesian for けんだけ still ranks 県 (7717) above 件 (3627)
            // for the left slot; the direct compound short-circuits that.
            ("けんだけ",       "件だけ"),
            ("にんだけ",       "人だけ"),
            ("こだけ",         "個だけ"),
            ("ほんだけ",       "本だけ"),
            ("だいだけ",       "台だけ"),
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

        // General Japanese supplement — common words/forms absent from IPADIC
        // that appeared in user dictionaries and are useful for all users.
        let general_supplement: &[(&str, &str, u32)] = &[
            // Common nouns / expressions
            ("かのうせい",   "可能性",       8500), // extremely common; IPADIC omits hiragana surface
            ("さいげんせい", "再現性",       7500), // IPADIC lacks the compound; 再+厳正 narrowly beat 再現+性
            ("ざんりょう",   "残量",         7500),
            ("きに",         "気に",         7500),
            ("きょうどう",   "協働",         7500), // collaborative; IPADIC has 協同/共同 but not 協働
            ("とくのう",     "特濃",         7500), // e.g. 明治 特濃
            ("いしがま",     "石窯",         7500),
            ("にんにくのめ", "にんにくの芽", 7000),
            // Common verb/adj inflections missing as surface forms
            ("つくろう",   "作ろう",   7500), // volitional; 繕う(tsukurou) is a different word
            ("とろう",     "取ろう",   7500), // volitional; distinct from 徒労
            ("のこして",   "残して",   7500),
            ("よわく",     "弱く",     7500),
            ("いない",     "いない",   7300), // keep below 以内(7631) so 以内 still ranks first
            ("いなく",     "いなく",   7300),
            // Internet slang
            ("わら",   "(笑)",  8000),
            // Additional katakana (general/IT)
            ("まいぐれーしょん", "マイグレーション", 7500),
            ("いれぎゅらー",     "イレギュラー",     7500),
            ("いんすとーら",     "インストーラ",     7500),
            ("おーとちゃーじ",   "オートチャージ",   7000),
            // General katakana (UI / web slang)
            ("あぷり",       "アプリ",       8500),
            ("すまほ",       "スマホ",       8500),
            ("れいあうと",   "レイアウト",   8500),
            ("えびでんす",   "エビデンス",   8000),
            ("りさいず",     "リサイズ",     8000),
            ("もーだる",     "モーダル",     7500),
            ("ぷろふ",       "プロフ",       7500),
            ("いんぷれ",     "インプレ",     7500),
            // Proper nouns / brand names commonly typed by general users
            ("いぼのいと",     "揖保乃糸",     8000),
            ("しゃうえっせん", "シャウエッセン", 8000),
            ("りすぺりどん",   "リスペリドン", 7500),
            ("さるたな",       "サルタナ",     7000),
            ("せたがやく",     "世田谷区",     7500),
            ("にっせき",       "日赤",         7500), // IPADIC has 日夕(2067) only
            ("かんさい",       "関西",         8000), // IPADIC top is 簡裁(5392)
            // General nouns missing from IPADIC as single entries
            ("いっぽん",   "一本",     8500), // IPADIC splits as いっ+ぽん
            ("げっしょ",   "月初",     8000),
            ("ばくすい",   "爆睡",     7500),
            ("さくじつ",   "昨日",     6000), // IPADIC has 朔日(4378); keep below きのう if any
            ("かれい",     "加齢",     6000), // IPADIC top is 華麗(5161)
            // Adjective stems — help segmentation when ~い/~く/~さ follows
            ("あたらし", "新し",   7000), // IPADIC has 新(Noun,1)
            ("おいし",   "美味し", 7500),
            ("すずし",   "涼し",   7000), // IPADIC has 生絹(4378)
            ("やさし",   "優し",   7000),
            // 2-char い-adjective stems — needed for 連用形 (~く) / 名詞形 (~さ) /
            // 否定形 (~くない) / 中止形 (~くて). Without these つよく → ツヨ+く
            // and 強 never surfaces (only つよかった group + つよい are in IPADIC
            // as whole forms; 連用形 family is missing).
            ("つよ",     "強",     7000), // IPADIC top is ツヨ Noun 3102
            // Verb forms / adverbials
            ("いれ",     "淹れ",   4000), // brew (tea/coffee); above IPADIC 入れ(3307)
            ("おそく",   "遅く",   7500),
            // 連用形 whole forms — the stem entry alone leaves つよくなる → つ+よく+なる
            // because よく Adverb (10294) is dense enough to swallow the middle
            // syllable in the DP. Adding the 3-char 連用形 gives DP a longer,
            // cheaper alternative that wins over the よく split.
            //
            // Same fix applied to a batch of common i-adjectives whose 連用形
            // (~く) / 中止形 (~くて) / 否定形 (~くない) / 名詞形 (~さ) are
            // similarly unreachable via segmentation — IPADIC picks
            // nonsense homographs (たかく→多角, はやく→破約, ひろく→秘録,
            // うれしく→売れ+市区, …) or bad splits (せまく→せ+まく,
            // ながく→な+がく, わかく→わ+かく, …). Whole-form entries at
            // 7500 win the DP cleanly. Adjectives whose ~く is already
            // reachable via 2-segment stem+く (あかく=赤+く, くろく=黒+く,
            // あたらしく=新し+く, …) are intentionally left off — the user
            // can still cycle to kana on the second segment.
            ("つよく",   "強く",   7500),
            ("つよくて", "強くて", 7500),
            ("つよくない", "強くない", 7500),
            ("つよさ",   "強さ",   7500),
            ("たかく",   "高く",   7500),
            ("たかくて", "高くて", 7500),
            ("たかくない", "高くない", 7500),
            ("はやく",   "早く",   7500),
            ("はやくて", "早くて", 7500),
            ("はやくない", "早くない", 7500),
            // 速い family (IPADIC has 速い Adj 5231 but ships no 速く/速さ);
            // freq 7300 keeps 早く as the DP-picked default while surfacing
            // 速く as the immediate runner-up so it is one cycle away.
            ("はやく",   "速く",   7300),
            ("はやくて", "速くて", 7300),
            ("はやくない", "速くない", 7300),
            ("はやさ",   "速さ",   7300),
            // 早さ (temporal earliness) at 7100 — 速さ (speed) is by far the
            // more common reading for はやさ in modern text, so 速さ stays top.
            ("はやさ",   "早さ",   7100),
            ("ひろく",   "広く",   7500),
            ("ひろくて", "広くて", 7500),
            ("ひろくない", "広くない", 7500),
            ("ひろさ",   "広さ",   7500),
            ("せまく",   "狭く",   7500),
            ("せまくて", "狭くて", 7500),
            ("せまくない", "狭くない", 7500),
            ("せまさ",   "狭さ",   7500),
            ("ながく",   "長く",   7500),
            ("ながくて", "長くて", 7500),
            ("ながくない", "長くない", 7500),
            ("ながさ",   "長さ",   7500),
            ("ひくく",   "低く",   7500),
            ("ひくくて", "低くて", 7500),
            ("ひくくない", "低くない", 7500),
            ("ひくさ",   "低さ",   7500),
            ("しろく",   "白く",   7500),
            ("しろくて", "白くて", 7500),
            ("しろくない", "白くない", 7500),
            ("しろさ",   "白さ",   7500),
            ("おもく",   "重く",   7500),
            ("おもくて", "重くて", 7500),
            ("おもくない", "重くない", 7500),
            ("おもさ",   "重さ",   7500),
            ("とおく",   "遠く",   7500),
            ("とおくて", "遠くて", 7500),
            ("とおくない", "遠くない", 7500),
            ("とおさ",   "遠さ",   7500),
            ("わかく",   "若く",   7500),
            ("わかくて", "若くて", 7500),
            ("わかくない", "若くない", 7500),
            ("わかさ",   "若さ",   7500),
            ("みじかく", "短く",   7500),
            ("みじかくて", "短くて", 7500),
            ("みじかくない", "短くない", 7500),
            ("みじかさ", "短さ",   7500),
            ("すごく",   "凄く",   7500),
            ("すごくて", "凄くて", 7500),
            ("すごくない", "凄くない", 7500),
            ("すごさ",   "凄さ",   7500),
            ("うれしく", "嬉しく", 7500),
            ("うれしくて", "嬉しくて", 7500),
            ("うれしくない", "嬉しくない", 7500),
            ("うれしさ", "嬉しさ", 7500),
            ("かなしく", "悲しく", 7500),
            ("かなしくて", "悲しくて", 7500),
            ("かなしくない", "悲しくない", 7500),
            ("かなしさ", "悲しさ", 7500),
            ("たのしく", "楽しく", 7500),
            ("たのしくて", "楽しくて", 7500),
            ("たのしくない", "楽しくない", 7500),
            ("たのしさ", "楽しさ", 7500),
            ("さびしく", "寂しく", 7500),
            ("さびしくて", "寂しくて", 7500),
            ("さびしくない", "寂しくない", 7500),
            ("さびしさ", "寂しさ", 7500),
            ("よろしく", "宜しく", 7500),
            ("ふるさ",   "古さ",   7500),
            // Noun-stem + さ derivations
            ("たかさ", "高さ", 8000),
            ("ふとさ", "太さ", 7500),
            // Common nouns absent from / mis-ranked in IPADIC, surfaced from user
            // dictionaries (registered because the default conversion was wrong).
            ("きどく",       "既読",   8000), // IPADIC top is 奇特(4371); 既読 absent
            ("おまたせ",     "お待たせ", 7500), // お待たせ(しました); absent
            ("けつりゅう",   "血流",   7000), // absent
            ("ぼうまんかん", "膨満感", 7000), // absent
            ("けんこうこつ", "肩甲骨", 7000), // IPADIC has only old-form 肩胛骨(胛) 4378
            ("ふかぼり",     "深掘り", 7000), // absent (business jargon 深掘りする)
        ];
        for &(reading, surface, freq) in general_supplement {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Noun,
                frequency: freq,
            });
        }

        // Box-drawing characters (罫線) — useful for text tables and diagrams.
        let keisen: &[(&str, &str)] = &[
            ("けいせん", "└"), ("けいせん", "┘"), ("けいせん", "┌"), ("けいせん", "┐"),
            ("けいせん", "─"), ("けいせん", "│"), ("けいせん", "┼"), ("けいせん", "├"),
            ("けいせん", "┤"), ("けいせん", "┬"), ("けいせん", "┴"), ("けいせん", "┗"),
            ("けいせん", "┛"), ("けいせん", "┏"), ("けいせん", "┓"), ("けいせん", "━"),
            ("けいせん", "┃"), ("けいせん", "╋"), ("けいせん", "┣"), ("けいせん", "┫"),
            ("けいせん", "┳"), ("けいせん", "┻"), ("けいせん", "┿"), ("けいせん", "┝"),
            ("けいせん", "┥"), ("けいせん", "┯"), ("けいせん", "┰"),
            ("けいせん", "┷"), ("けいせん", "┸"), ("けいせん", "╂"),
        ];
        for &(reading, surface) in keisen {
            self.add_entry(DictionaryEntry {
                reading: reading.to_string(),
                surface: surface.to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 8000,
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

fn surface_contains_kanji(surface: &str) -> bool {
    surface.chars().any(|c| {
        ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c)
    })
}

/// Katakana-dominant surface: contains at least one katakana codepoint and no
/// kanji. Both pure katakana (ツクる) and mixed kata+hiragana suffix (ツクった)
/// match; kanji+katakana mixes (e.g., 挿ハサむ, hypothetical) do not.
fn is_kata_dominant(surface: &str) -> bool {
    let mut has_kata = false;
    for c in surface.chars() {
        if ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c) {
            return false;
        }
        if ('\u{30A1}'..='\u{30F6}').contains(&c) {
            has_kata = true;
        }
    }
    has_kata
}

const VERB_BASE_KANA: &[char] = &['る', 'む', 'ぶ', 'く', 'ぐ', 'す', 'つ', 'う', 'ぬ'];

/// Extract (reading_stem, surface_stem, freq) tuples from PRIORITY_OVERRIDES
/// entries that describe kanji-preferred verb base forms. A stem is the base
/// reading/surface with its trailing verb-base kana dropped, so that
/// starts_with matching catches every inflection of the same family.
///
/// Entries with a reading shorter than 3 chars (e.g. あう, みる) are skipped —
/// a 1-char stem would over-match unrelated verbs. Non-verb overrides (nouns,
/// hiragana-only surfaces, etc.) are also skipped.
fn collect_verb_stems(
    overrides: &'static [(&'static str, &'static str, u32)],
) -> Vec<(&'static str, &'static str, u32)> {
    let mut out = Vec::new();
    for &(reading, surface, freq) in overrides {
        let reading_chars: Vec<char> = reading.chars().collect();
        // Guarded: len >= 3, so last() is always Some.
        let Some(&last) = reading_chars.last() else {
            continue;
        };
        if reading_chars.len() < 3 {
            continue;
        }
        if !VERB_BASE_KANA.contains(&last) {
            continue;
        }
        if !surface_contains_kanji(surface) {
            continue;
        }
        let read_stem = &reading[..reading.len() - last.len_utf8()];
        // Trim the surface's trailing base kana if present (e.g., "見る" → "見",
        // "作る" → "作"); otherwise keep the whole surface.
        let surf_stem = match surface.chars().last() {
            Some(sc) if sc == last => &surface[..surface.len() - sc.len_utf8()],
            _ => surface,
        };
        out.push((read_stem, surf_stem, freq));
    }
    out
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
    fn priority_overrides_promote_everyday_words() {
        let dict = Dictionary::new();
        // Each buried everyday word must now top its homophone group; the
        // formerly-winning rare/abstract/literary surface must rank below it.
        for (reading, expected) in [
            // nouns
            ("みち", "道"), // beats 未知
            ("うみ", "海"), // beats 生み/膿/産み
            ("め", "目"),   // beats 眼/海布
            ("かわ", "川"),   // beats 皮
            ("かみ", "髪"),   // beats 加味
            ("りょうり", "料理"), // beats 良吏
            ("かぎ", "鍵"),   // beats 鈎
            ("ひと", "人"),   // beats 匪徒/費途
            ("こと", "事"),   // beats 古都/糊塗/殊/琴
            ("じしょ", "辞書"), // beats 字書/自署/地所
            // verbs
            ("みる", "見る"),     // beats 海松/水松
            ("いのる", "祈る"),   // beats 祷る
            ("およぐ", "泳ぐ"),   // beats 游ぐ
            ("とじる", "閉じる"), // beats 綴じる
            ("つくる", "作る"),   // beats 創る/造る
            ("あう", "会う"),     // beats 遭う/遇う
            ("さがす", "探す"),   // beats 捜す
        ] {
            let results = dict.lookup(reading);
            assert_eq!(
                results.first().map(|e| e.surface.as_str()),
                Some(expected),
                "expected {expected} to top the {reading} group, got {:?}",
                results.iter().take(3).map(|e| &e.surface).collect::<Vec<_>>(),
            );
        }
        // Context-dependent pairs are intentionally NOT force-ordered here —
        // the LLM decides these from context.
        let ame = dict.lookup("あめ");
        assert_eq!(ame.first().map(|e| e.surface.as_str()), Some("雨"));
    }

    #[test]
    fn verb_inflection_prefers_kanji_over_katakana() {
        // Every conjugated form of an override-listed verb family should top
        // its group with the kanji surface, and its emphatic-katakana variant
        // (IPADIC ships ツクった / サガした / トジた at 14–15K freq) must not.
        let dict = Dictionary::new();
        let cases: &[(&str, &str)] = &[
            // 作る family (godan ら, 促音便)
            ("つくった", "作った"),
            ("つくって", "作って"),
            ("つくっている", "作っている"),
            ("つくっています", "作っています"),
            ("つくります", "作ります"),
            ("つくりました", "作りました"),
            ("つくられる", "作られる"),
            // 探す family (godan さ) — past/te forms live in load_symbol_entries;
            // polite forms come from the generator.
            ("さがした", "探した"),
            ("さがして", "探して"),
            ("さがしている", "探している"),
            ("さがします", "探します"),
            // 泳ぐ family (godan が, イ音便 → 泳いだ/泳いで)
            ("およいだ", "泳いだ"),
            ("およいで", "泳いで"),
            ("およぎます", "泳ぎます"),
            // 閉じる family (ichidan)
            ("とじた", "閉じた"),
            ("とじて", "閉じて"),
            ("とじます", "閉じます"),
            // 祈る family (godan ら)
            ("いのった", "祈った"),
            ("いのって", "祈って"),
            ("いのります", "祈ります"),
        ];
        for &(reading, expected) in cases {
            let results = dict.lookup(reading);
            assert_eq!(
                results.first().map(|e| e.surface.as_str()),
                Some(expected),
                "expected {expected} to top {reading}; top-5: {:?}",
                results.iter().take(5).map(|e| e.surface.as_str()).collect::<Vec<_>>(),
            );
            // The kata variant, if present, must rank below every kanji form.
            let kata_pos = results
                .iter()
                .position(|e| is_kata_dominant(&e.surface));
            let last_kanji_pos = results
                .iter()
                .enumerate()
                .filter(|(_, e)| surface_contains_kanji(&e.surface))
                .map(|(i, _)| i)
                .max();
            if let (Some(k), Some(last_k)) = (kata_pos, last_kanji_pos) {
                assert!(
                    k > last_k,
                    "kata variant ranked at {k}, expected after all kanji forms (last at {last_k}) for {reading}",
                );
            }
        }
    }

    #[test]
    fn value_and_counter_compounds_short_circuit_dp() {
        // Regression: きたいち used to segment as きたい+ち (奇態+血) and
        // けんだけ merged Noun+Suffix into 県+岳 (min 2574). Direct
        // compound entries in compound_nouns short-circuit both.
        let dict = Dictionary::new();
        for &(reading, expected_top) in &[
            ("きたいち", "期待値"),
            ("へいきんち", "平均値"),
            ("さいだいち", "最大値"),
            ("しきべつし", "識別子"),
            ("けんだけ", "件だけ"),
            ("にんだけ", "人だけ"),
        ] {
            let segs = dict.segment(reading);
            let joined: Vec<&str> = segs.iter().map(|s| s.reading.as_str()).collect();
            assert_eq!(joined, vec![reading], "{reading} should merge to one seg");
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(top, Some(expected_top), "{reading} top mismatch");
        }
    }

    #[test]
    fn counter_plus_ka_is_particle_not_suffix() {
        // Regression: なんかいか used to segment as one merged 何回化
        // because か→化 Suffix (5054) triggers Noun+Suffix reclass on
        // 何回+か. Direct dict entries for the common counter+か phrases
        // let the DP short-circuit that merge and produce 何回か.
        let dict = Dictionary::new();
        for &(reading, expected_top) in &[
            ("なんかいか", "何回か"),
            ("なんにんか", "何人か"),
            ("なんじかんか", "何時間か"),
            ("いくつか", "いくつか"),
        ] {
            let segs = dict.segment(reading);
            let joined: Vec<&str> = segs.iter().map(|s| s.reading.as_str()).collect();
            assert_eq!(joined, vec![reading], "{reading} should merge into one seg");
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(top, Some(expected_top), "{reading} top mismatch");
        }
    }

    #[test]
    fn monogaki_stays_particle_split_in_context() {
        // Regression: ものがきに inside a sentence used to fuse into
        // Noun+Suffix compound (物書きに) even though the natural parse is
        // もの+が+き+に. IPADIC bumps がき→書き (Suffix 4400) above 餓鬼
        // (Noun 4353) via PRIORITY_OVERRIDES, so the DP treats がき as one
        // Suffix and the affix merge glues もの onto it. split_particle_
        // head_segments breaks がき back into が+き before the merge runs.
        let dict = Dictionary::new();
        for reading in ["ものがきになる", "まわりのものがきになる"] {
            let segs = dict.segment(reading);
            let joined: Vec<&str> =
                segs.iter().map(|s| s.reading.as_str()).collect();
            assert!(
                joined.iter().all(|r| !r.contains("ものがき")),
                "{reading}: no segment should contain ものがき, got {joined:?}",
            );
            // ものが and きに should each be a segment (Noun+Particle bunsetsu).
            assert!(
                joined.contains(&"ものが") && joined.contains(&"きに"),
                "{reading}: expected ものが/きに bunsetsu, got {joined:?}",
            );
        }
    }

    #[test]
    fn everyday_two_char_nouns_survive_particle_head_split() {
        // Regression (bug_list_fable_5_review_2026-08-28 #1): the
        // Particle-head gate over-fired on any 2-char reading whose
        // first kana was a strong Particle (と/に/は/の …), shredding
        // にく→に+苦, はし→は+市, とき→と+気, とち→と+血 and — once
        // the orphaned tail fused with the next Particle via Noun+
        // Particle merge — dropping 肉/橋/時/土地 from every parse.
        // The dominant-POS gate keeps these Noun-dominant segments
        // intact; only tail-Suffix-dominant readings (がき) still split.
        let dict = Dictionary::new();
        for (reading, want_substring) in [
            ("にくをたべる", "肉"),
            ("はしをわたる", "橋"),
            ("ときがきた", "時"),
            ("とちをかう", "土地"),
            ("のどがいたい", "喉"),
            ("にわをみる", "庭"),
        ] {
            let segs = dict.segment(reading);
            let joined: Vec<&str> =
                segs.iter().map(|s| s.reading.as_str()).collect();
            // The critical property is *reachability*: the correct
            // content-word kanji must live somewhere in some segment's
            // candidate list. Before the fix the wrong segmentation
            // dropped 肉/橋/時/土地/喉/庭 from every parse. Which slot
            // it occupies is a separate ranking concern.
            let mut reachable = false;
            for s in &segs {
                if s.candidates.iter().any(|e| e.surface.contains(want_substring)) {
                    reachable = true;
                    break;
                }
            }
            assert!(
                reachable,
                "{reading}: expected {want_substring:?} to be reachable \
                 in some segment's candidates, got segments {joined:?}",
            );
        }
    }

    #[test]
    fn common_noun_kata_capped_below_kanji() {
        // IPADIC ships モノ (3302) above 物 (3248) and コト (2795) above
        // 事 (1188). Extending the kata cap to Nouns keeps them in the
        // candidate list but never above the canonical kanji form.
        let dict = Dictionary::new();
        for &(reading, wrong_kata_top) in &[
            ("もの", "モノ"),
            ("こと", "コト"),
            ("とき", "トキ"),
        ] {
            let top = dict.lookup(reading).first().map(|e| e.surface.to_string());
            assert_ne!(
                top.as_deref(),
                Some(wrong_kata_top),
                "{reading}: {wrong_kata_top} should not top",
            );
        }
    }

    #[test]
    fn desushi_stays_hiragana_conjunction() {
        // Regression: ですしね used to segment as です+すし(寿司)+ね and
        // display "で鮨ね". The colloquial coordinator し is an IPADIC
        // Particle at freq 2158 — buried under 市/氏/寿司 — so the DP
        // preferred the 2-char noun over the 1-char particle. Add explicit
        // hiragana compound entries for the common ですし/だし forms.
        let dict = Dictionary::new();
        for &(reading, expected_top) in &[
            ("ですし", "ですし"),
            ("ですしね", "ですしね"),
            ("ですしよ", "ですしよ"),
            ("ですしよね", "ですしよね"),
            ("だしね", "だしね"),
            ("だしよ", "だしよ"),
        ] {
            let segs = dict.segment(reading);
            let joined: Vec<&str> = segs.iter().map(|s| s.reading.as_str()).collect();
            assert_eq!(
                joined,
                vec![reading],
                "{reading} should stay one segment"
            );
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(top, Some(expected_top), "{reading} top mismatch");
        }
        // だし alone stays ambiguous (dashi / 山車 / 出汁) — the compound
        // entry we skipped must NOT hijack it.
        let segs = dict.segment("だし");
        let top = segs[0].candidates.first().map(|e| e.surface.as_str());
        assert_ne!(top, Some("だし"), "だし alone must not top as hiragana");
    }

    #[test]
    fn shimasu_survives_masu_learning_boost() {
        // Real-world regression: once the user records ます|ます (score 0.228),
        // the DP for お願いします starts splitting します into し+ます — the
        // ×10 boost cuts split cost by 2.28 and edges out the base freq-7000
        // します Verb entry. The affix-compound merge then pulls お願い+し
        // into a Noun+Suffix compound and the top surface becomes お願い市.
        // PRIORITY_OVERRIDES bumps します-family to 9000-9500 so the single
        // unit stays comfortably below the boosted split.
        let dict = Dictionary::new();
        let scores: std::collections::HashMap<(&str, &str), f64> = [
            (("ます", "ます"), 0.228),
            (("さい", "再"), 0.455),
        ]
        .into_iter()
        .collect();
        let boost = |r: &str, entries: &[&DictionaryEntry]| {
            if r.chars().count() <= 1 {
                return 0.0;
            }
            let mut best = 0.0_f64;
            for e in entries {
                if let Some(&s) = scores.get(&(r, e.surface.as_str())) {
                    best = best.max(s);
                }
            }
            best * 10.0
        };
        for reading in ["おねがいします", "さいかくにんをおねがいします"] {
            let segs = dict.segment_with_boost(reading, boost);
            let joined: Vec<&str> = segs.iter().map(|s| s.reading.as_str()).collect();
            assert!(
                joined.iter().any(|r| *r == "します"),
                "expected します to stay as a single segment for {reading}, got {joined:?}",
            );
            for s in &segs {
                if s.reading == "します" {
                    assert_eq!(
                        s.candidates.first().map(|e| e.surface.as_str()),
                        Some("します"),
                        "expected します top for the します segment",
                    );
                }
            }
        }
    }

    #[test]
    fn affix_merge_survives_user_scorer_boost() {
        // Real-world regression: a user with さい|再 (3 uses) and かん|感 (1)
        // in their learning history saw the DP pick なんじ+かん (汝感) and
        // さいかく+にん (才覚人) — the desired 何時間 / 再確認 didn't even
        // appear in the candidate list. probe_2piece_alternatives now
        // enumerates the alternate split so the intended compound is
        // reachable.
        let dict = Dictionary::new();
        let scores: std::collections::HashMap<(&str, &str), f64> = [
            (("さい", "再"), 0.455),
            (("なん", "なん"), 0.365),
            (("なん", "何"), 0.228),
            (("にん", "人"), 0.228),
            (("かん", "感"), 0.228),
            (("かん", "間"), 0.228),
        ]
        .into_iter()
        .collect();
        for &(reading, expected_top) in &[
            ("なんじかん", "何時間"),
            ("さいかくにん", "再確認"),
            ("なんにん", "何人"),
        ] {
            let segs = dict.segment_with_boost(reading, |r, entries| {
                if r.chars().count() <= 1 {
                    return 0.0;
                }
                let mut best = 0.0_f64;
                for e in entries {
                    if let Some(&s) = scores.get(&(r, e.surface.as_str())) {
                        best = best.max(s);
                    }
                }
                best * 10.0
            });
            assert_eq!(segs.len(), 1, "{reading} should be a single merged segment");
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(top, Some(expected_top), "{reading} top mismatch under learned boost");
        }
    }

    #[test]
    fn affix_merge_reclassifies_particle_homograph_as_suffix() {
        // Regression for readings whose right-side dominant candidate is a
        // Particle but a strong Suffix homograph exists (か→化 5054 vs か
        // Particle 7424). Without effective_right_merge_pos the merge fired
        // as Noun+Particle → 政治+か (hiragana), instead of Noun+Suffix →
        // 政治化 (kanji).
        let dict = Dictionary::new();
        for &(reading, expected_top) in &[
            ("せいじか", "政治化"),
            ("かっせいか", "活性化"),
            ("じつようか", "実用化"),
            ("じどうか", "自動化"),
            ("みんしゅか", "民主化"),
            ("せんもんか", "専門化"),
        ] {
            let segs = dict.segment(reading);
            assert_eq!(segs.len(), 1, "{reading} should merge to a single segment, got {segs:?}");
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(top, Some(expected_top), "{reading} top mismatch");
            // The 家 alternative must still be discoverable within top-5
            // (widened cartesian keeps the long-tail Suffix homograph).
            let has_ka_ie = segs[0].candidates.iter().any(|e| e.surface.ends_with('家'));
            assert!(has_ka_ie, "{reading}: expected 家 form in candidate list");
        }
    }

    #[test]
    fn affix_merge_reclassifies_noun_homograph_as_suffix() {
        // にん→人 Suffix (1859) is buried under 忍 Noun (4206) as dominant.
        // Without the ≥1500 suffix threshold, なん+にん stayed two segments.
        let dict = Dictionary::new();
        for &(reading, expected_top) in &[
            ("なんにん", "何人"),
            ("なんじかん", "何時間"),
        ] {
            let segs = dict.segment(reading);
            assert_eq!(segs.len(), 1, "{reading} should merge to a single segment, got {segs:?}");
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(top, Some(expected_top), "{reading} top mismatch");
        }
    }

    #[test]
    fn affix_merge_reclassifies_prefix_homograph() {
        // さい dominant is 際 Noun (4950); 再 Prefix (4213) needs
        // effective_left_merge_pos to enable the Prefix+Noun merge for
        // 再確認. Similarly for ふ (歩 vs 不), み (未 already dominant).
        let dict = Dictionary::new();
        for &(reading, expected_top) in &[
            ("さいかくにん", "再確認"),
            ("ふかのう", "不可能"),
            ("ひこうかい", "非公開"),
            ("むかんけい", "無関係"),
            ("みかくにん", "未確認"),
        ] {
            let segs = dict.segment(reading);
            assert_eq!(segs.len(), 1, "{reading} should merge to a single segment, got {segs:?}");
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(top, Some(expected_top), "{reading} top mismatch");
        }
    }

    #[test]
    fn godan_su_verb_survives_learned_split_boost() {
        // Regression for the case that surfaced during install-and-test: the
        // user had recorded a はなし|話 selection (~3 times), so the DP's
        // ×10 boost pushed the はなし+て split cost below the single-unit
        // はなして match, and 話し手 (Noun 4378) reappeared as top. The
        // supplement's 8000+ base freq must keep the single unit winning.
        let dict = Dictionary::new();
        for &(reading, expected) in &[
            ("はなして", "話して"),
            ("はなした", "話した"),
            ("だして", "出して"),
            ("さがして", "探して"),
        ] {
            let segs = dict.segment_with_boost(reading, |r, entries| {
                if r.chars().count() <= 1 { return 0.0; }
                let mut best = 0.0_f64;
                for e in entries {
                    if r.len() == reading.len() - 'て'.len_utf8()
                        && surface_contains_kanji(&e.surface)
                    {
                        best = best.max(0.455);
                    }
                }
                best * 10.0
            });
            assert_eq!(segs.len(), 1, "{reading} should stay a single segment");
            let top = segs[0].candidates.first().map(|e| e.surface.as_str());
            assert_eq!(
                top,
                Some(expected),
                "expected {expected} top for {reading} even under a learned split boost, got {:?}",
                top,
            );
        }
    }

    #[test]
    fn canonical_katakana_verb_stays_on_top() {
        // Loanword verbs whose canonical form is katakana (サボる, ダブる, ググる)
        // have no kanji competitor, so the demotion path must NOT touch them.
        let dict = Dictionary::new();
        for &(reading, expected) in &[
            ("さぼった", "サボった"),
            ("さぼって", "サボって"),
            ("だぶった", "ダブった"),
        ] {
            let top = dict.lookup(reading).first().map(|e| e.surface.to_string());
            assert_eq!(
                top.as_deref(),
                Some(expected),
                "expected {expected} to remain top for {reading}, got {:?}",
                top,
            );
        }
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

    /// Regression: a symbol shortcut must not outrank a common homophone word.
    /// At base freq 8000, した→↓ beat 舌/下/した; lowering it (like emoji) keeps
    /// the arrow reachable without clobbering the word.
    #[test]
    fn symbol_shortcut_ranks_below_common_word() {
        let dict = Dictionary::new();
        let shita = dict.lookup("した");
        let arrow = shita.iter().find(|e| e.surface == "↓").map_or(0, |e| e.frequency);
        let top_word = shita
            .iter()
            .filter(|e| !matches!(e.surface.as_str(), "↓" | "⇓"))
            .map(|e| e.frequency)
            .max()
            .unwrap_or(0);
        assert!(
            top_word > arrow,
            "↓ ({arrow}) must rank below the top した word ({top_word})",
        );
    }

    /// Regression: 書き (compound tail: 下書き/落書き) is far commoner as input
    /// than 餓鬼; the override lifts it above 餓鬼 so がき defaults to 書き and the
    /// Noun+Suffix compound merge fires.
    #[test]
    fn gaki_kaki_outranks_gaki_demon_and_merges() {
        let dict = Dictionary::new();
        let gaki = dict.lookup("がき");
        let kaki = gaki.iter().find(|e| e.surface == "書き").map_or(0, |e| e.frequency);
        let demon = gaki.iter().find(|e| e.surface == "餓鬼").map_or(0, |e| e.frequency);
        assert!(kaki > demon, "書き ({kaki}) must outrank 餓鬼 ({demon})");

        // The compound suffix must merge into one bunsetsu, not split off 餓鬼.
        for (input, want) in [("したがき", "下書き"), ("らくがき", "落書き")] {
            let segs = dict.segment(input);
            assert_eq!(segs.len(), 1, "{input} should be one bunsetsu, got {:?}",
                       segs.iter().map(|s| &s.reading).collect::<Vec<_>>());
            let top = segs[0]
                .candidates
                .iter()
                .max_by_key(|e| e.frequency)
                .map(|e| e.surface.as_str())
                .unwrap_or("");
            assert_eq!(top, want, "{input} top candidate should be {want}");
        }
    }

    /// Regression: a heavily-learned short word (けん→件) must not fragment a
    /// longer compound it is a prefix of. Before the fix, the learning boost on
    /// けん let the DP split けんさく(検索) into 件＋柵, so "けんさくになります"
    /// rendered as 件柵になります even though "けんさく" alone stayed 検索.
    #[test]
    fn learned_prefix_does_not_fragment_longer_compound() {
        let dict = Dictionary::new();
        // Mimic production user learning: けん→件 selected several times. The
        // boost mirrors the engine closure (×10, single-char segments skipped).
        let boost = |reading: &str, entries: &[&DictionaryEntry]| -> f64 {
            if reading.chars().count() <= 1 {
                return 0.0;
            }
            let learned = reading == "けん"
                && entries.iter().any(|e| e.surface == "件");
            // ln(1+4)/ln(1+20) ≈ 0.53 — four selections of けん→件.
            if learned { 0.53 * 10.0 } else { 0.0 }
        };
        for input in ["けんさく", "けんさくになります"] {
            let segs = dict.segment_with_boost(input, &boost);
            let top = segs[0]
                .candidates
                .first()
                .map(|e| e.surface.as_str())
                .unwrap_or("");
            assert!(
                top.starts_with("検索"),
                "{input}: first bunsetsu should stay 検索, got {:?} ({})",
                segs.iter().map(|s| &s.reading).collect::<Vec<_>>(),
                top,
            );
        }
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

    /// Regression: ところ must offer 所 (everyday "place") as its top dictionary
    /// candidate, not the rare 野老 (3755, a plant / uncommon surname) that buried
    /// 所 (1685). PRIORITY_OVERRIDES lifts 所 to 4200.
    #[test]
    fn tokoro_defaults_to_tokoro_place() {
        let dict = Dictionary::new();
        let cands = &dict.segment("ところ")[0].candidates;
        assert_eq!(
            cands.first().map(|e| e.surface.as_str()),
            Some("所"),
            "got {:?}",
            cands.iter().map(|e| e.surface.as_str()).take(3).collect::<Vec<_>>(),
        );
    }

    /// Regression: つぎは must default to 次+は, not the archaic single word 継歯
    /// (4378) nor 継ぎ (3957, which outranked 次 3508). Noun+Particle merge makes
    /// つぎは one bunsetsu whose Cartesian candidates are frequency-ordered; the
    /// PRIORITY_OVERRIDES (次→4400, 継歯/継端→1500) must put 次は first.
    #[test]
    fn segmentation_tsugiha_defaults_to_next() {
        let dict = Dictionary::new();
        let segs = dict.segment("つぎは");
        assert_eq!(segs.len(), 1, "つぎは should be one bunsetsu, got {:?}",
            segs.iter().map(|s| s.reading.as_str()).collect::<Vec<_>>());
        let top = segs[0].candidates.first().map(|e| e.surface.as_str());
        assert_eq!(top, Some("次は"), "top candidate should be 次は, got {:?}",
            segs[0].candidates.iter().map(|e| e.surface.as_str()).take(5).collect::<Vec<_>>());
    }

    /// Regression: おねがいします must segment as お願い+します, not 尾根(おね 4435)+
    /// 害します (がいします Verb 5772, whose 5-char length bonus beat お願い 3459 once
    /// します was pulled in). PRIORITY_OVERRIDES raises お願い to 5800.
    #[test]
    fn segmentation_onegaishimasu() {
        let dict = Dictionary::new();
        let segs = dict.segment("おねがいします");
        let readings: Vec<&str> = segs.iter().map(|s| s.reading.as_str()).collect();
        assert_eq!(readings, vec!["おねがい", "します"], "got {:?}", readings);
        assert_eq!(
            segs[0].candidates.first().map(|e| e.surface.as_str()),
            Some("お願い"),
        );
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

    /// Regression [17]: the dict-management dialog used to hold a
    /// snapshot of user_entries at dialog-open time, then apply
    /// delete/edit by index onto that stale vec and push the mutated
    /// snapshot back through `replace_user_entries`. Any word registered
    /// between "show list" and "confirm delete" was clobbered by the
    /// replay of the pre-registration snapshot. The fix re-fetches
    /// `user_entries()` under the write lock at apply time and applies
    /// by (reading, surface) identity. This test locks in that pattern:
    /// a concurrent add survives an identity-based delete.
    #[test]
    fn identity_based_delete_preserves_concurrent_add() {
        let mut dict = Dictionary::new();
        let user_start = dict.user_start;
        let entry = |r: &str, s: &str| DictionaryEntry {
            reading: r.to_string(),
            surface: s.to_string(),
            pos: PartOfSpeech::Noun,
            frequency: 8000,
        };
        dict.add_entry(entry("あ", "A"));
        dict.add_entry(entry("い", "B"));
        dict.add_entry(entry("う", "C"));
        // Dialog opens and snapshots — for display only.
        let _dialog_snapshot: Vec<DictionaryEntry> = dict.user_entries().to_vec();
        // Concurrent register while dialog is still open.
        dict.add_entry(entry("え", "D"));
        // Apply "delete B" the fixed way: re-fetch live entries, drop
        // by identity, push back through replace_user_entries.
        let mut live: Vec<DictionaryEntry> = dict.user_entries().to_vec();
        let before = live.len();
        live.retain(|e| !(e.reading == "い" && e.surface == "B"));
        assert_eq!(live.len(), before - 1);
        dict.replace_user_entries(live);
        // D must survive.
        let surfaces: Vec<String> = dict
            .user_entries()
            .iter()
            .map(|e| e.surface.clone())
            .collect();
        assert_eq!(surfaces, vec!["A", "C", "D"]);
        // Built-in entries are still intact (user_start unchanged).
        assert_eq!(dict.user_start, user_start);
    }

    #[test]
    fn sync_and_load_via_store() {
        let dir = std::env::temp_dir().join("bonolith_test_dict_sync");
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
        let dir = std::env::temp_dir().join("bonolith_test_export_import");
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
        let dir = std::env::temp_dir().join("bonolith_test_trailing_comma");
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
        let dir = std::env::temp_dir().join("bonolith_test_bad_json");
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

    #[test]
    fn general_supplement_coverage() {
        let dict = Dictionary::new();
        // Verify each entry promoted from user dict to builtin appears as a
        // top-ranked candidate so it is safe to remove from the user dict.
        let cases: &[(&str, &str)] = &[
            ("かのうせい",   "可能性"),
            ("ざんりょう",   "残量"),
            ("きょうどう",   "協働"),
            ("つくろう",     "作ろう"),
            ("とろう",       "取ろう"),
            ("のこして",     "残して"),
            ("よわく",       "弱く"),
            ("いなく",       "いなく"),
            ("とくのう",     "特濃"),
            ("いしがま",     "石窯"),
            ("にんにくのめ", "にんにくの芽"),
            ("わら",         "(笑)"),
            ("まいぐれーしょん", "マイグレーション"),
            ("いれぎゅらー",     "イレギュラー"),
            ("いんすとーら",     "インストーラ"),
            ("おーとちゃーじ",   "オートチャージ"),
            ("けいせん",     "┌"),
            // Promoted from user_dict (2026-06-04 batch)
            ("あぷり",       "アプリ"),
            ("すまほ",       "スマホ"),
            ("れいあうと",   "レイアウト"),
            ("えびでんす",   "エビデンス"),
            ("りさいず",     "リサイズ"),
            ("もーだる",     "モーダル"),
            ("ぷろふ",       "プロフ"),
            ("いんぷれ",     "インプレ"),
            ("いぼのいと",     "揖保乃糸"),
            ("しゃうえっせん", "シャウエッセン"),
            ("りすぺりどん",   "リスペリドン"),
            ("さるたな",       "サルタナ"),
            ("せたがやく",     "世田谷区"),
            ("にっせき",       "日赤"),
            ("かんさい",       "関西"),
            ("いっぽん",   "一本"),
            ("げっしょ",   "月初"),
            ("ばくすい",   "爆睡"),
            ("さくじつ",   "昨日"),
            ("かれい",     "加齢"),
            ("あたらし",   "新し"),
            ("おいし",     "美味し"),
            ("すずし",     "涼し"),
            ("やさし",     "優し"),
            ("いれ",       "淹れ"),
            ("おそく",     "遅く"),
            ("つよ",       "強"),
            ("つよく",     "強く"),
            ("つよくて",   "強くて"),
            ("つよくない", "強くない"),
            ("つよさ",     "強さ"),
            ("たかく",     "高く"),
            ("たかくない", "高くない"),
            ("たかさ",     "高さ"),
            ("はやく",     "早く"),
            ("ひろく",     "広く"),
            ("ひろさ",     "広さ"),
            ("せまく",     "狭く"),
            ("ながく",     "長く"),
            ("ながさ",     "長さ"),
            ("ひくく",     "低く"),
            ("しろく",     "白く"),
            ("おもく",     "重く"),
            ("おもさ",     "重さ"),
            ("とおく",     "遠く"),
            ("わかく",     "若く"),
            ("わかさ",     "若さ"),
            ("みじかく",   "短く"),
            ("すごく",     "凄く"),
            ("うれしく",   "嬉しく"),
            ("うれしさ",   "嬉しさ"),
            ("かなしく",   "悲しく"),
            ("たのしく",   "楽しく"),
            ("たのしさ",   "楽しさ"),
            ("さびしく",   "寂しく"),
            ("よろしく",   "宜しく"),
            ("ふるさ",     "古さ"),
            ("ふとさ",     "太さ"),
            ("くらっかー", "🎉"),
        ];
        let mut failures = 0;
        for &(reading, expected_surface) in cases {
            let segs = dict.segment(reading);
            let all_surfaces: Vec<&str> = segs.iter()
                .flat_map(|s| s.candidates.iter().map(|c| c.surface.as_str()))
                .collect();
            if !all_surfaces.contains(&expected_surface) {
                failures += 1;
                eprintln!("FAIL {reading:20}: expected {expected_surface:?} in candidates {all_surfaces:?}");
            } else {
                eprintln!("pass {reading:20} → {expected_surface}");
            }
        }
        assert_eq!(failures, 0, "{failures} supplement entries missing from candidates");
    }

    /// Table-driven coverage: every listed i-adjective family MUST have
    /// all four standard conjugations (~く / ~くて / ~くない / ~さ)
    /// reachable via `segment()`. Unlike `general_supplement_coverage`
    /// which enumerates registered entries, this test enumerates the
    /// *expected* coverage set — so a missing form (e.g. Devin PR #3's
    /// gap "はやさ→早さ / ひくさ→低さ / しろさ→白さ / とおさ→遠さ /
    /// みじかさ→短さ / すごさ→凄さ") fails automatically the moment
    /// a family is added to the list without the matching supplement
    /// entries.
    ///
    /// Add a family here whenever you add a new i-adjective batch; keep
    /// it out only for adjectives with genuinely irregular coverage
    /// (よろし: 名詞形「宜しさ」は非自然、~くない も 現代語で稀).
    #[test]
    fn i_adjective_conjugation_coverage() {
        let dict = Dictionary::new();
        // (stem_reading, stem_surface). Kanji column is the primary
        // surface; secondary surfaces (早/速 both for はや) are checked
        // via `secondary_forms` below.
        let families: &[(&str, &str)] = &[
            ("つよ",   "強"),
            ("たか",   "高"),
            ("はや",   "早"),
            ("ひろ",   "広"),
            ("せま",   "狭"),
            ("なが",   "長"),
            ("ひく",   "低"),
            ("しろ",   "白"),
            ("おも",   "重"),
            ("とお",   "遠"),
            ("わか",   "若"),
            ("みじか", "短"),
            ("すご",   "凄"),
            ("うれし", "嬉し"),
            ("かなし", "悲し"),
            ("たのし", "楽し"),
            ("さびし", "寂し"),
        ];
        // (suffix_reading, suffix_surface): the four standard forms.
        let forms: &[(&str, &str)] = &[
            ("く",     "く"),
            ("くて",   "くて"),
            ("くない", "くない"),
            ("さ",     "さ"),
        ];
        // Secondary surface families that must also appear (adjacent
        // homograph readings that IPADIC lists as a distinct Adjective):
        // for はや, 速い は 別語なので 連用形/名詞形 も 速く/速さ で
        // 到達可能でなければならない。
        let secondary: &[(&str, &str)] = &[
            ("はや", "速"),
        ];
        let mut failures: Vec<(String, String)> = Vec::new();
        let check = |stem_r: &str, stem_s: &str, failures: &mut Vec<(String, String)>| {
            for &(suf_r, suf_s) in forms {
                let reading = format!("{stem_r}{suf_r}");
                let expected = format!("{stem_s}{suf_s}");
                let all_surfaces: Vec<String> = dict
                    .segment(&reading)
                    .iter()
                    .flat_map(|s| s.candidates.iter().map(|c| c.surface.clone()))
                    .collect();
                if !all_surfaces.contains(&expected) {
                    failures.push((reading, expected));
                }
            }
        };
        for &(stem_r, stem_s) in families {
            check(stem_r, stem_s, &mut failures);
        }
        for &(stem_r, stem_s) in secondary {
            check(stem_r, stem_s, &mut failures);
        }
        assert!(
            failures.is_empty(),
            "{} i-adjective conjugation(s) unreachable via segment():\n{}",
            failures.len(),
            failures
                .iter()
                .map(|(r, e)| format!("  {r:>12} → expected {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Regression: `てき` had 的 Suffix (7408) dominant over 敵 Noun (5474),
    /// so `split_particle_head_segments` fired and shredded the reading
    /// into て+き — 敵 then vanished from every reachable parse because
    /// `てき+を+叩く` became `て+きを+叩く` after Noun+Particle merge.
    /// The runner-up gate keeps 2-char readings whole when they carry a
    /// strong Noun/Verb/Adj candidate (freq ≥ 4500) alongside the Suffix
    /// leader, so 敵 stays reachable in the てき segment.
    #[test]
    fn split_particle_head_keeps_content_runnerup() {
        let dict = Dictionary::new();
        // 敵 (Noun 5474) alongside 的 (Suffix 7408) → keep てき whole.
        let segs = dict.segment("てきをたたく");
        let readings: Vec<&str> = segs.iter().map(|s| s.reading.as_str()).collect();
        assert_eq!(
            readings,
            vec!["てき", "を", "たたく"],
            "てき must stay whole so 敵 is reachable, got {:?}",
            readings,
        );
        let teki_surfaces: Vec<&str> = segs[0]
            .candidates
            .iter()
            .map(|e| e.surface.as_str())
            .collect();
        assert!(
            teki_surfaces.contains(&"敵"),
            "敵 must appear in てき candidates, got {:?}",
            teki_surfaces,
        );
        // Sanity: がき's runner-up 餓鬼 (Noun 4353) sits BELOW the 4500
        // gate on purpose, so the がき split still fires for がき+的
        // reclass (previous behaviour preserved — not asserted per-parse
        // here since surrounding-context segmentation is orthogonal).
    }

    /// Regression: several stem-particle readings had niche 1-word dict
    /// entries (きょうは→教派, かれは→枯れ葉, そらに→空似, やまが→
    /// 山家, これが→...→之が) winning by 1-seg cost advantage over the
    /// everyday Noun+Particle split. The demotions in PRIORITY_OVERRIDES
    /// plus the reading-length gate in surface_adjustment restore the
    /// modern default. The dominant_pos tie-breaker keeps やま Noun 4500
    /// ahead of やま Suffix 4500 so Noun+Particle merge fires for やまは.
    #[test]
    fn stem_particle_defaults_to_noun_plus_particle() {
        use crate::engine::{ConversionEngine, SharedCore};
        let shared = SharedCore::new_hermetic();
        // Romaji key-streams are used directly so combined kana (きょ) don't
        // get mistyped as ki+yo → 器用 by a naive kana→key converter.
        let cases = [
            ("kyouha", "今日は"),
            ("kyouga", "今日が"),
            ("kareha", "彼は"),
            ("sorani", "空に"),
            ("sorade", "空で"),
            ("yamaga", "山が"),
            ("yamaha", "山は"),
            ("mizuwo", "水を"),
        ];
        for (input, want) in cases {
            let mut e = ConversionEngine::with_shared(shared.clone());
            for ch in input.chars() {
                e.process_key(ch);
            }
            let state = e.start_conversion().expect("start_conversion");
            let top: String = state
                .segments
                .iter()
                .filter_map(|s| s.candidates.first().map(String::as_str))
                .collect::<Vec<_>>()
                .join("");
            assert_eq!(top, want, "{input}: expected {want}, got {top}");
        }

        // Lone particles must still get their +0.2 surface==reading bonus
        // (the reading-length gate only fires at 3+ chars).
        for (input, want) in [("ha", "は"), ("no", "の"), ("kara", "から"), ("made", "まで")] {
            let mut e = ConversionEngine::with_shared(shared.clone());
            for ch in input.chars() {
                e.process_key(ch);
            }
            let state = e.start_conversion().expect("start_conversion");
            let top = state.segments[0].candidates[0].as_str();
            assert_eq!(top, want, "{input}: lone particle must top its candidate list");
        }
    }

    /// The Noun+Particle merge in merge_affix_compounds used to pull the
    /// homograph Noun / archaic-Particle candidates for common trailing
    /// particles into the compound Cartesian product — visible symptom:
    /// "後は" showing "後刃" / "跡刃" as sibling candidates, "後の"
    /// showing "後之" / "跡之". The role filter now restricts the right
    /// side to Particle POS with a top-1 cap, so only the modern default
    /// "は" / "の" propagates.
    #[test]
    fn noun_particle_merge_hides_homograph_kanji() {
        let dict = Dictionary::new();
        // Nonsense surfaces that must NOT appear in the compound candidates.
        let cases: &[(&str, &[&str])] = &[
            ("あとは", &["後刃", "跡刃", "蹟刃", "痕刃"]),
            ("あとの", &["後之", "跡之", "蹟之", "痕之"]),
            ("わたしは", &["私刃", "わたし刃", "渡し刃"]),
            ("わたしの", &["私之", "わたし之", "渡し之"]),
        ];
        for &(reading, forbidden) in cases {
            let segs = dict.segment(reading);
            let all_surfaces: Vec<&str> = segs
                .iter()
                .flat_map(|s| s.candidates.iter().map(|c| c.surface.as_str()))
                .collect();
            for bad in forbidden {
                assert!(
                    !all_surfaces.contains(bad),
                    "{reading}: nonsense surface {bad:?} leaked into candidates {all_surfaces:?}",
                );
            }
            // Sanity: the modern-particle form must still be present.
            let modern = match reading.chars().last().unwrap() {
                'は' => format!("{}は", &reading[..reading.len() - "は".len()]),
                'の' => format!("{}の", &reading[..reading.len() - "の".len()]),
                _ => unreachable!(),
            };
            // At least one surface should end with the modern particle
            // over a kanji stem (e.g. "後は" / "私の").
            let has_kanji_particle = all_surfaces.iter().any(|s| {
                s.ends_with(modern.chars().last().unwrap())
                    && s.chars().count() >= 2
                    && !s.starts_with(reading.chars().next().unwrap())
            });
            assert!(
                has_kanji_particle,
                "{reading}: expected at least one kanji-stem+particle candidate in {all_surfaces:?}",
            );
        }
    }

    /// Devin PR #3 #2 regression: row-level user-entry mutations must
    /// NOT wipe rows the other frontend added since our startup.
    ///
    /// The prior `sync_user_entries_to_store` path called
    /// `DELETE FROM user_entries` + `INSERT` from this process's stale
    /// in-memory snapshot. Two processes sharing the DB (IBus and
    /// Fcitx5 both installed) triggered irreversible data loss: any
    /// delete/edit/add via one frontend flushed the other frontend's
    /// concurrent additions.
    ///
    /// Simulate the scenario:
    /// 1. Dict A opens store, adds E1 via the persist path.
    /// 2. "Other frontend" writes E2 directly to the store (bypassing
    ///    Dict A's memory — i.e., Dict A doesn't know E2 exists).
    /// 3. Dict A deletes E1 via the persist path.
    /// 4. Dict B opens the same store fresh — E2 MUST survive.
    #[test]
    fn row_level_persist_preserves_concurrent_frontend_entries() {
        use crate::core::store::DictStore;
        let dir = std::env::temp_dir().join("bonolith_test_row_level_persist");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("dict.sqlite");

        let store = Arc::new(DictStore::open(&db_path).unwrap());

        // Dict A: our process. Add E1 via the persist path.
        let mut dict_a = Dictionary::new();
        dict_a.attach_store(store.clone());
        dict_a
            .add_user_entry_and_persist(DictionaryEntry {
                reading: "aaa".to_string(),
                surface: "AAA".to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 8000,
            })
            .unwrap();

        // Other frontend: writes E2 directly to the DB. Dict A never
        // sees this — its in-memory snapshot only knows E1.
        store
            .upsert_user_entry(&DictionaryEntry {
                reading: "bbb".to_string(),
                surface: "BBB".to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 8000,
            })
            .unwrap();

        // Dict A: remove E1 via the persist path. The prior bug
        // wrote Dict A's stale (only-E1) snapshot back and wiped E2.
        let removed = dict_a
            .remove_user_entry_and_persist("aaa", "AAA")
            .unwrap();
        assert!(removed);

        // Dict B: fresh process on the same store. E2 must still exist.
        let loaded = store.load_user_entries().unwrap();
        assert_eq!(loaded.len(), 1, "loaded rows: {:?}", loaded);
        assert_eq!(loaded[0].reading, "bbb");
        assert_eq!(loaded[0].surface, "BBB");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Same contract for `update_user_entry_and_persist`: editing a
    /// row in one frontend must not wipe rows the other frontend added.
    #[test]
    fn update_persist_preserves_concurrent_frontend_entries() {
        use crate::core::store::DictStore;
        let dir = std::env::temp_dir().join("bonolith_test_update_persist");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("dict.sqlite");

        let store = Arc::new(DictStore::open(&db_path).unwrap());

        let mut dict_a = Dictionary::new();
        dict_a.attach_store(store.clone());
        dict_a
            .add_user_entry_and_persist(DictionaryEntry {
                reading: "old".to_string(),
                surface: "OLD".to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 8000,
            })
            .unwrap();

        // Other frontend adds a row Dict A doesn't know about.
        store
            .upsert_user_entry(&DictionaryEntry {
                reading: "keep".to_string(),
                surface: "KEEP".to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 8000,
            })
            .unwrap();

        // Dict A edits its row.
        let updated = dict_a
            .update_user_entry_and_persist("old", "OLD", "new", "NEW")
            .unwrap();
        assert!(updated);

        // Both the renamed row AND the concurrently-added row survive.
        let loaded = store.load_user_entries().unwrap();
        let readings: Vec<&str> = loaded.iter().map(|e| e.reading.as_str()).collect();
        assert!(readings.contains(&"new"), "renamed row missing: {loaded:?}");
        assert!(readings.contains(&"keep"), "concurrent row wiped: {loaded:?}");
        assert!(!readings.contains(&"old"), "old row survived: {loaded:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

