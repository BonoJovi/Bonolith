/// User learning scorer for Bonolith.
///
/// Records which (reading, surface) pairs the user selects and boosts
/// those pairs in future candidate ranking. Persists to the SQLite
/// store (`user_scores` table) when attached; otherwise keeps an
/// in-memory map only.

use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use crate::core::store::DictStore;

pub struct UserScorer {
    /// Map of "reading|surface" -> selection count
    counts: HashMap<String, u32>,
    /// Learned per-kana segmentation preferences. Stored as the boundary
    /// list (segment start positions, excluding 0). See
    /// [`DictStore::record_segmentation`] for the format.
    segmentations: HashMap<String, Vec<usize>>,
    /// Optional persistent store. When attached, every record() also
    /// writes through to the store.
    store: Option<Arc<DictStore>>,
}

impl UserScorer {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            segmentations: HashMap::new(),
            store: None,
        }
    }

    /// Construct a scorer backed by a persistent store. Loads any
    /// existing scores into memory; subsequent record() calls persist
    /// per-row updates immediately (cheap with SQLite).
    pub fn from_store(store: Arc<DictStore>) -> io::Result<Self> {
        let counts = store.load_user_scores()?;
        let segmentations = store.load_user_segmentations()?;
        Ok(Self {
            counts,
            segmentations,
            store: Some(store),
        })
    }

    /// Record a user selection for the given (reading, surface) pair.
    /// Called only for segments where the user explicitly chose a candidate,
    /// so no additional filtering is needed here.
    pub fn record(&mut self, reading: &str, surface: &str) {
        let key = Self::key(reading, surface);
        *self.counts.entry(key).or_insert(0) += 1;
        if let Some(store) = &self.store {
            if let Err(e) = store.increment_score(reading, surface) {
                log::warn!("failed to persist score for {}|{}: {}", reading, surface, e);
            }
        }
    }

    /// Clear all learning history from memory and the persistent store.
    /// Returns the number of rows deleted from the store (score + segmentation
    /// rows combined).
    pub fn clear_scores(&mut self) -> io::Result<usize> {
        self.counts.clear();
        self.segmentations.clear();
        if let Some(store) = &self.store {
            let scores = store.clear_user_scores()?;
            let segs = store.clear_user_segmentations()?;
            Ok(scores + segs)
        } else {
            Ok(0)
        }
    }

    /// Record a user-preferred segmentation for `kana`. `boundaries` is
    /// the segment start positions (char offsets), excluding 0. See
    /// [`DictStore::record_segmentation`] for the format. Called by the
    /// engine on commit when the final segmentation differs from what
    /// the DP segmenter originally produced.
    pub fn record_segmentation(&mut self, kana: &str, boundaries: Vec<usize>) {
        if let Some(store) = &self.store {
            if let Err(e) = store.record_segmentation(kana, &boundaries) {
                log::warn!(
                    "failed to persist segmentation for {}: {}",
                    kana,
                    e
                );
            }
        }
        self.segmentations.insert(kana.to_string(), boundaries);
    }

    /// Look up a learned segmentation for `kana`. Returns the boundary
    /// list (see [`record_segmentation`](Self::record_segmentation)) if
    /// the user has previously committed a non-default segmentation for
    /// exactly this kana string.
    pub fn lookup_segmentation(&self, kana: &str) -> Option<&[usize]> {
        self.segmentations.get(kana).map(|v| v.as_slice())
    }

    /// Drop a learned segmentation from memory and the store. Called
    /// when the learned layout has drifted back to what the DP
    /// segmenter now produces on its own (dictionary improvements,
    /// user resizes back to the default) so the row stops taking up
    /// space and reload cost forever — bug [23]. No-op if the entry
    /// was never recorded.
    pub fn forget_segmentation(&mut self, kana: &str) {
        if self.segmentations.remove(kana).is_none() {
            return;
        }
        if let Some(store) = &self.store {
            if let Err(e) = store.forget_segmentation(kana) {
                log::warn!(
                    "failed to forget segmentation for {}: {}",
                    kana,
                    e
                );
            }
        }
    }

    /// Score a (reading, surface) pair based on user history.
    /// Returns 0.0 if never selected. Uses absolute logarithmic scaling
    /// so that even a single selection provides meaningful signal.
    /// Saturates toward 1.0 around 20+ selections.
    pub fn score(&self, reading: &str, surface: &str) -> f64 {
        let key = Self::key(reading, surface);
        let count = match self.counts.get(&key) {
            Some(&c) => c,
            None => return 0.0,
        };

        // ln(1 + count) / ln(1 + 20) ≈ saturates at ~1.0 around 20 uses
        // 1 use → 0.23, 2 → 0.36, 5 → 0.59, 10 → 0.79, 20 → 1.0
        ((count as f64).ln_1p() / (20.0_f64).ln_1p()).min(1.0)
    }

    /// Default path for the legacy `user_scores.json` (used only by the
    /// migration path in `DictStore`). New writes go through SQLite.
    pub fn default_legacy_path() -> io::Result<std::path::PathBuf> {
        let data_dir = std::env::var("XDG_DATA_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                std::path::PathBuf::from(home).join(".local/share")
            })
            .join("bonolith");
        Ok(data_dir.join("user_scores.json"))
    }

    fn key(reading: &str, surface: &str) -> String {
        format!("{}|{}", reading, surface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> Arc<DictStore> {
        let dir = std::env::temp_dir().join(format!("bonolith_test_scorer_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Arc::new(DictStore::open(&dir.join("dict.sqlite")).unwrap())
    }

    #[test]
    fn unrecorded_score_is_zero() {
        let scorer = UserScorer::new();
        assert_eq!(scorer.score("きょう", "今日"), 0.0);
    }

    #[test]
    fn record_and_score() {
        let mut scorer = UserScorer::new();
        scorer.record("きょう", "今日");
        assert!(scorer.score("きょう", "今日") > 0.0);
    }

    #[test]
    fn more_selections_higher_score() {
        let mut scorer = UserScorer::new();
        for _ in 0..10 {
            scorer.record("きょう", "今日");
        }
        scorer.record("きょう", "京");
        assert!(scorer.score("きょう", "今日") > scorer.score("きょう", "京"));
    }

    #[test]
    fn single_selection_gives_boost() {
        let mut scorer = UserScorer::new();
        scorer.record("へんかん", "変換");
        // Even one selection should give a meaningful score
        assert!(scorer.score("へんかん", "変換") > 0.2);
    }

    #[test]
    fn kana_only_recorded() {
        let mut scorer = UserScorer::new();
        scorer.record("きょう", "きょう"); // same reading as surface — now recorded
        assert!(scorer.score("きょう", "きょう") > 0.0);
    }

    #[test]
    fn record_persists_through_store() {
        let store = temp_store("persist");
        let mut scorer = UserScorer::from_store(store.clone()).unwrap();
        scorer.record("きょう", "今日");
        scorer.record("きょう", "今日");
        let s1 = scorer.score("きょう", "今日");
        drop(scorer);

        let scorer2 = UserScorer::from_store(store).unwrap();
        let s2 = scorer2.score("きょう", "今日");
        assert!((s1 - s2).abs() < 1e-9);
    }

    #[test]
    fn from_store_starts_empty_when_db_fresh() {
        let store = temp_store("fresh");
        let scorer = UserScorer::from_store(store).unwrap();
        assert_eq!(scorer.score("anything", "anything"), 0.0);
    }
}
