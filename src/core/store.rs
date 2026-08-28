/// SQLite-backed persistent store for user dictionary entries and scores.
///
/// Replaces the v1.x `user_dict.json` + `user_scores.json` pair with a
/// single `dict.sqlite`. JSON is retained only as an import/export format.
///
/// ## Multi-process access
///
/// The store is opened in WAL (write-ahead log) mode so that fcitx5
/// (`fcitx5-bonolith.so`) and IBus (`ibus-engine-bonolith`) — which run in
/// separate processes — can both open the same database. Writes from
/// one process are durably persisted; the other process's `Connection`
/// will see them on the next read. However, **each process keeps its
/// own in-memory `Dictionary` cache**, which is loaded once at
/// `SharedCore::global()` init and not refreshed afterwards. This means
/// if a user registers a word in IBus and immediately switches to
/// fcitx5, the new word is in SQLite but won't appear in fcitx5's live
/// suggestions until fcitx5 is restarted. Live cross-process cache
/// invalidation (file watch / D-Bus signal) is left for future work;
/// the v2.0.0 contract is "durable single-source-of-truth, restart for
/// cache refresh".

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection};

use crate::core::dictionary::{DictionaryEntry, PartOfSpeech};

const SCHEMA_VERSION: i32 = 1;
const MIGRATION_FLAG: &str = "legacy_json_migrated";

pub struct DictStore {
    /// SQLite connection guarded behind a Mutex. Poison recovery via
    /// `unwrap_or_else(|e| e.into_inner())` is safe here: rusqlite protects
    /// the in-progress statement itself, so any panic that poisoned the
    /// lock left the connection either at a statement boundary or ended
    /// its transaction cleanly; retrying is preferable to letting a
    /// single panic wedge every subsequent read/write.
    conn: Mutex<Connection>,
}

impl DictStore {
    /// Open or create the database at the given path. Initializes the
    /// schema on first creation; safe to call repeatedly. Configures
    /// WAL journal mode so multiple processes (fcitx5 + IBus) can hold
    /// open connections simultaneously.
    pub fn open(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(sqlite_to_io)?;

        // PRAGMA journal_mode=WAL returns the mode that's actually in
        // effect — verify it took (read-only filesystem etc. would
        // fall back to other modes silently otherwise).
        let mode: String = conn
            .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
            .map_err(sqlite_to_io)?;
        if mode.to_lowercase() != "wal" {
            log::warn!(
                "dict.sqlite did not enter WAL mode (got {:?}); \
                 multi-process access may be impaired",
                mode
            );
        }
        // 5s busy_timeout absorbs brief writer contention (eg. fcitx5
        // and IBus both committing scores at the same instant).
        conn.execute_batch(
            "PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;",
        )
        .map_err(sqlite_to_io)?;

        let store = Self { conn: Mutex::new(conn) };
        store.init_schema()?;
        Ok(store)
    }

    /// Open the store at the default path, then run the legacy JSON
    /// migration if the per-DB flag isn't set yet. Used by both
    /// `SharedCore::global()` and the CLI/dialog flows so that legacy
    /// JSON files are imported regardless of which entry point first
    /// touches the store.
    pub fn open_default_with_migration() -> io::Result<Self> {
        use crate::core::dictionary::Dictionary;
        use crate::core::user_scorer::UserScorer;

        let store = Self::open(&Self::default_path()?)?;
        if let (Ok(dict_json), Ok(scores_json)) = (
            Dictionary::default_user_dict_path(),
            UserScorer::default_legacy_path(),
        ) {
            if let Err(e) = store.migrate_legacy_json(&dict_json, &scores_json) {
                log::warn!("legacy JSON migration failed: {}", e);
            }
        }
        Ok(store)
    }

    /// Default path: `$XDG_DATA_HOME/bonolith/dict.sqlite` (typically
    /// `~/.local/share/bonolith/dict.sqlite`).
    pub fn default_path() -> io::Result<PathBuf> {
        let data_dir = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                PathBuf::from(home).join(".local/share")
            })
            .join("bonolith");
        Ok(data_dir.join("dict.sqlite"))
    }

    fn init_schema(&self) -> io::Result<()> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS meta (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS user_entries (
               reading TEXT NOT NULL,
               surface TEXT NOT NULL,
               pos TEXT NOT NULL,
               frequency INTEGER NOT NULL,
               PRIMARY KEY (reading, surface)
             );
             CREATE INDEX IF NOT EXISTS idx_user_entries_reading
               ON user_entries(reading);
             CREATE TABLE IF NOT EXISTS user_scores (
               reading TEXT NOT NULL,
               surface TEXT NOT NULL,
               count INTEGER NOT NULL DEFAULT 0,
               PRIMARY KEY (reading, surface)
             );
             CREATE TABLE IF NOT EXISTS user_segmentations (
               kana TEXT PRIMARY KEY,
               boundaries TEXT NOT NULL,
               count INTEGER NOT NULL DEFAULT 1
             );
             COMMIT;",
        )
        .map_err(sqlite_to_io)?;
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value
               WHERE CAST(meta.value AS INTEGER) < CAST(excluded.value AS INTEGER)",
            params![SCHEMA_VERSION.to_string()],
        )
        .map_err(sqlite_to_io)?;
        Ok(())
    }

    /// Migrate `user_dict.json` and `user_scores.json` into the database
    /// if not already done. Idempotent: a meta flag prevents repeated
    /// runs. On success, the JSON files are renamed to `*.migrated` so
    /// the user can recover from them if needed.
    ///
    /// If the migration flag is already set but the JSON files exist,
    /// they are stale leftovers from a v1.x process that wrote after
    /// the upgrade — recover their content into SQLite (without double-
    /// counting) and rename them to `*.stale`.
    pub fn migrate_legacy_json(
        &self,
        dict_json: &Path,
        scores_json: &Path,
    ) -> io::Result<()> {
        if self.is_migrated()? {
            return self.recover_stale_jsons(dict_json, scores_json);
        }

        let entries = read_legacy_dict_json(dict_json)?;
        let scores = read_legacy_scores_json(scores_json)?;

        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction().map_err(sqlite_to_io)?;
        for entry in &entries {
            tx.execute(
                "INSERT OR IGNORE INTO user_entries (reading, surface, pos, frequency)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    entry.reading,
                    entry.surface,
                    pos_to_str(entry.pos),
                    entry.frequency
                ],
            )
            .map_err(sqlite_to_io)?;
        }
        for (key, count) in &scores {
            if let Some((reading, surface)) = key.split_once('|') {
                tx.execute(
                    "INSERT INTO user_scores (reading, surface, count)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(reading, surface) DO UPDATE SET count = excluded.count",
                    params![reading, surface, count],
                )
                .map_err(sqlite_to_io)?;
            }
        }
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, '1')",
            params![MIGRATION_FLAG],
        )
        .map_err(sqlite_to_io)?;
        tx.commit().map_err(sqlite_to_io)?;
        drop(conn);

        if dict_json.exists() {
            let _ = fs::rename(dict_json, append_extension(dict_json, ".migrated"));
        }
        if scores_json.exists() {
            let _ = fs::rename(scores_json, append_extension(scores_json, ".migrated"));
        }
        log::info!(
            "migrated {} user entries and {} scores from legacy JSON",
            entries.len(),
            scores.len()
        );
        Ok(())
    }

    /// Merge stale post-migration JSON files (`user_dict.json`,
    /// `user_scores.json`) into SQLite. Called only when the migration
    /// flag is already set, i.e. the JSON files are leftovers from a
    /// v1.x process that ran after the Bonolith upgrade.
    ///
    /// Semantics:
    /// - Dict: `INSERT OR IGNORE` so existing SQLite entries win on
    ///   conflict — the v2.0+ engine is the source of truth.
    /// - Scores: take `MAX(json_count, sqlite_count)` so we recover
    ///   keystrokes captured only by the stale process without double-
    ///   counting keystrokes already recorded in SQLite.
    ///
    /// Files are renamed to `*.stale` (not `*.migrated`, to preserve
    /// the distinction between the original migration backup and a
    /// post-migration leftover).
    fn recover_stale_jsons(
        &self,
        dict_json: &Path,
        scores_json: &Path,
    ) -> io::Result<()> {
        let dict_existed = dict_json.exists();
        let scores_existed = scores_json.exists();
        if !dict_existed && !scores_existed {
            return Ok(());
        }

        let entries = if dict_existed {
            read_legacy_dict_json(dict_json)?
        } else {
            Vec::new()
        };
        let scores = if scores_existed {
            read_legacy_scores_json(scores_json)?
        } else {
            HashMap::new()
        };

        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction().map_err(sqlite_to_io)?;

        let mut entries_added: usize = 0;
        for entry in &entries {
            let n = tx
                .execute(
                    "INSERT OR IGNORE INTO user_entries (reading, surface, pos, frequency)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        entry.reading,
                        entry.surface,
                        pos_to_str(entry.pos),
                        entry.frequency
                    ],
                )
                .map_err(sqlite_to_io)?;
            entries_added += n;
        }

        let mut scores_inserted: usize = 0;
        let mut scores_updated: usize = 0;
        for (key, json_count) in &scores {
            let Some((reading, surface)) = key.split_once('|') else {
                continue;
            };
            let existing: Option<u32> = tx
                .query_row(
                    "SELECT count FROM user_scores WHERE reading = ?1 AND surface = ?2",
                    params![reading, surface],
                    |row| row.get(0),
                )
                .ok();
            match existing {
                None => {
                    tx.execute(
                        "INSERT INTO user_scores (reading, surface, count)
                         VALUES (?1, ?2, ?3)",
                        params![reading, surface, json_count],
                    )
                    .map_err(sqlite_to_io)?;
                    scores_inserted += 1;
                }
                Some(c) if c < *json_count => {
                    tx.execute(
                        "UPDATE user_scores SET count = ?1
                         WHERE reading = ?2 AND surface = ?3",
                        params![json_count, reading, surface],
                    )
                    .map_err(sqlite_to_io)?;
                    scores_updated += 1;
                }
                _ => {}
            }
        }

        tx.commit().map_err(sqlite_to_io)?;
        drop(conn);

        if dict_existed {
            let _ = fs::rename(dict_json, append_extension(dict_json, ".stale"));
        }
        if scores_existed {
            let _ = fs::rename(scores_json, append_extension(scores_json, ".stale"));
        }

        if entries_added > 0 || scores_inserted > 0 || scores_updated > 0 {
            log::warn!(
                "recovered post-migration stale JSON: \
                 user_entries +{}, user_scores +{}/~{}; files renamed to *.stale",
                entries_added,
                scores_inserted,
                scores_updated,
            );
        } else {
            log::info!("renamed stale post-migration JSON to *.stale (no new data)");
        }
        Ok(())
    }

    fn is_migrated(&self) -> io::Result<bool> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let val: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![MIGRATION_FLAG],
                |row| row.get(0),
            )
            .ok();
        Ok(val.as_deref() == Some("1"))
    }

    pub fn load_user_entries(&self) -> io::Result<Vec<DictionaryEntry>> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn
            .prepare("SELECT reading, surface, pos, frequency FROM user_entries")
            .map_err(sqlite_to_io)?;
        let rows = stmt
            .query_map([], |row| {
                let reading: String = row.get(0)?;
                let surface: String = row.get(1)?;
                let pos_str: String = row.get(2)?;
                let frequency: u32 = row.get(3)?;
                Ok((reading, surface, pos_str, frequency))
            })
            .map_err(sqlite_to_io)?;
        let mut entries = Vec::new();
        for row in rows {
            let (reading, surface, pos_str, frequency) = row.map_err(sqlite_to_io)?;
            let pos = str_to_pos(&pos_str).unwrap_or(PartOfSpeech::Other);
            entries.push(DictionaryEntry {
                reading,
                surface,
                pos,
                frequency,
            });
        }
        Ok(entries)
    }

    pub fn upsert_user_entry(&self, entry: &DictionaryEntry) -> io::Result<()> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO user_entries (reading, surface, pos, frequency)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(reading, surface) DO UPDATE SET
               pos = excluded.pos,
               frequency = excluded.frequency",
            params![
                entry.reading,
                entry.surface,
                pos_to_str(entry.pos),
                entry.frequency
            ],
        )
        .map_err(sqlite_to_io)?;
        Ok(())
    }

    pub fn remove_user_entry(&self, reading: &str, surface: &str) -> io::Result<usize> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let n = conn
            .execute(
                "DELETE FROM user_entries WHERE reading = ?1 AND surface = ?2",
                params![reading, surface],
            )
            .map_err(sqlite_to_io)?;
        Ok(n)
    }

    pub fn clear_user_scores(&self) -> io::Result<usize> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let n = conn
            .execute("DELETE FROM user_scores", [])
            .map_err(sqlite_to_io)?;
        Ok(n)
    }

    pub fn load_user_scores(&self) -> io::Result<HashMap<String, u32>> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn
            .prepare("SELECT reading, surface, count FROM user_scores")
            .map_err(sqlite_to_io)?;
        let rows = stmt
            .query_map([], |row| {
                let reading: String = row.get(0)?;
                let surface: String = row.get(1)?;
                let count: u32 = row.get(2)?;
                Ok((format!("{}|{}", reading, surface), count))
            })
            .map_err(sqlite_to_io)?;
        let mut scores = HashMap::new();
        for row in rows {
            let (key, count) = row.map_err(sqlite_to_io)?;
            scores.insert(key, count);
        }
        Ok(scores)
    }

    /// Replace the entire user_entries table with the given list, in a
    /// single transaction. Used by FFI flows that operate via
    /// `Dictionary::replace_user_entries`.
    pub fn replace_all_user_entries(&self, entries: &[DictionaryEntry]) -> io::Result<()> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction().map_err(sqlite_to_io)?;
        tx.execute("DELETE FROM user_entries", [])
            .map_err(sqlite_to_io)?;
        for entry in entries {
            tx.execute(
                "INSERT OR REPLACE INTO user_entries (reading, surface, pos, frequency)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    entry.reading,
                    entry.surface,
                    pos_to_str(entry.pos),
                    entry.frequency
                ],
            )
            .map_err(sqlite_to_io)?;
        }
        tx.commit().map_err(sqlite_to_io)?;
        Ok(())
    }

    /// Increment the count for `(reading, surface)`. Inserts the row at
    /// count=1 if absent. Used per-commit by UserScorer.
    pub fn increment_score(&self, reading: &str, surface: &str) -> io::Result<()> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO user_scores (reading, surface, count) VALUES (?1, ?2, 1)
             ON CONFLICT(reading, surface) DO UPDATE SET count = count + 1",
            params![reading, surface],
        )
        .map_err(sqlite_to_io)?;
        Ok(())
    }

    /// Record a user-preferred segmentation. `boundaries` is the list of
    /// segment start positions (char offsets into `kana`) *excluding* 0 —
    /// so "きょう/は/いい/てんき" segmented over "きょうはいいてんき" is
    /// stored as `[3, 4, 6]`. An empty list means "single segment" (user
    /// dragged everything into one bunsetsu). The count is incremented on
    /// each re-record so repeated confirmations reinforce the entry;
    /// changing the segmentation for the same kana replaces `boundaries`.
    pub fn record_segmentation(&self, kana: &str, boundaries: &[usize]) -> io::Result<()> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let serialised = encode_boundaries(boundaries);
        conn.execute(
            "INSERT INTO user_segmentations (kana, boundaries, count)
                 VALUES (?1, ?2, 1)
             ON CONFLICT(kana) DO UPDATE SET
                 boundaries = excluded.boundaries,
                 count = count + 1",
            params![kana, serialised],
        )
        .map_err(sqlite_to_io)?;
        Ok(())
    }

    /// Load all learned segmentations into an in-memory map. Called once
    /// at UserScorer startup; subsequent updates go through
    /// `record_segmentation`.
    pub fn load_user_segmentations(&self) -> io::Result<HashMap<String, Vec<usize>>> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn
            .prepare("SELECT kana, boundaries FROM user_segmentations")
            .map_err(sqlite_to_io)?;
        let rows = stmt
            .query_map([], |row| {
                let kana: String = row.get(0)?;
                let boundaries: String = row.get(1)?;
                Ok((kana, boundaries))
            })
            .map_err(sqlite_to_io)?;
        let mut out = HashMap::new();
        for row in rows {
            let (kana, boundaries) = row.map_err(sqlite_to_io)?;
            // Silently drop rows whose boundaries don't parse cleanly.
            // Without this the old `filter_map` turned "3,x,6" into [3,6],
            // which then passed engine-side monotonicity checks and got
            // applied as a plausible-but-wrong learned layout. Falling
            // back to no entry means the DP segmenter runs — the safe
            // default when persisted state is corrupt.
            if let Some(bs) = decode_boundaries(&boundaries) {
                out.insert(kana, bs);
            } else {
                log::warn!(
                    "Discarding corrupt learned segmentation for '{}': {:?}",
                    kana,
                    boundaries,
                );
            }
        }
        Ok(out)
    }

    pub fn clear_user_segmentations(&self) -> io::Result<usize> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let n = conn
            .execute("DELETE FROM user_segmentations", [])
            .map_err(sqlite_to_io)?;
        Ok(n)
    }
}

fn encode_boundaries(boundaries: &[usize]) -> String {
    boundaries
        .iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Decode a comma-separated boundary list, all-or-nothing. Returns
/// `None` if any token fails to parse — the caller must then treat the
/// row as absent rather than salvaging the parseable tokens. Salvage
/// would produce a *plausible* boundary list ("3,x,6" → [3,6]) that
/// passes engine-side range / monotonicity checks and gets applied as
/// if it were correct, silently corrupting the user's learned layout.
fn decode_boundaries(s: &str) -> Option<Vec<usize>> {
    if s.is_empty() {
        return Some(Vec::new());
    }
    s.split(',').map(|p| p.parse().ok()).collect()
}

fn read_legacy_dict_json(path: &Path) -> io::Result<Vec<DictionaryEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data = fs::read_to_string(path)?;
    serde_json::from_str(&data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_legacy_scores_json(path: &Path) -> io::Result<HashMap<String, u32>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(path)?;
    serde_json::from_str(&data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Append a suffix to a path's filename, preserving the original
/// extension (e.g. `user_dict.json` → `user_dict.json.migrated`).
fn append_extension(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

fn pos_to_str(pos: PartOfSpeech) -> &'static str {
    use PartOfSpeech::*;
    match pos {
        Noun => "Noun",
        Verb => "Verb",
        Adjective => "Adjective",
        Adverb => "Adverb",
        Particle => "Particle",
        Auxiliary => "Auxiliary",
        Conjunction => "Conjunction",
        Interjection => "Interjection",
        Prefix => "Prefix",
        Suffix => "Suffix",
        Other => "Other",
    }
}

fn str_to_pos(s: &str) -> Option<PartOfSpeech> {
    use PartOfSpeech::*;
    Some(match s {
        "Noun" => Noun,
        "Verb" => Verb,
        "Adjective" => Adjective,
        "Adverb" => Adverb,
        "Particle" => Particle,
        "Auxiliary" => Auxiliary,
        "Conjunction" => Conjunction,
        "Interjection" => Interjection,
        "Prefix" => Prefix,
        "Suffix" => Suffix,
        "Other" => Other,
        _ => return None,
    })
}

fn sqlite_to_io(e: rusqlite::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bonolith_test_store_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("dict.sqlite")
    }

    /// Regression [24]: a corrupt row in user_segmentations must be
    /// dropped whole rather than salvaged into a plausible-but-wrong
    /// boundary list. Before the fix, "3,x,6" collapsed to [3,6] via
    /// filter_map — engine-side monotonicity / range checks then passed
    /// and the wrong layout got re-applied every time the user typed
    /// that kana.
    #[test]
    fn decode_boundaries_is_all_or_nothing() {
        assert_eq!(decode_boundaries(""), Some(Vec::<usize>::new()));
        assert_eq!(decode_boundaries("2,5"), Some(vec![2, 5]));
        // Any unparseable token invalidates the whole list.
        assert_eq!(decode_boundaries("3,x,6"), None);
        assert_eq!(decode_boundaries("x"), None);
        assert_eq!(decode_boundaries("2,,5"), None);
    }

    #[test]
    fn fresh_open_creates_empty_tables() {
        let path = temp_db_path("fresh");
        let store = DictStore::open(&path).unwrap();
        assert_eq!(store.load_user_entries().unwrap().len(), 0);
        assert_eq!(store.load_user_scores().unwrap().len(), 0);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn upsert_then_load() {
        let path = temp_db_path("upsert");
        let store = DictStore::open(&path).unwrap();
        store
            .upsert_user_entry(&DictionaryEntry {
                reading: "てすと".to_string(),
                surface: "テスト".to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 100,
            })
            .unwrap();
        let entries = store.load_user_entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].surface, "テスト");
        assert_eq!(entries[0].pos, PartOfSpeech::Noun);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn upsert_replaces_existing_row() {
        let path = temp_db_path("replace");
        let store = DictStore::open(&path).unwrap();
        let mut entry = DictionaryEntry {
            reading: "てすと".to_string(),
            surface: "テスト".to_string(),
            pos: PartOfSpeech::Noun,
            frequency: 100,
        };
        store.upsert_user_entry(&entry).unwrap();
        entry.frequency = 999;
        store.upsert_user_entry(&entry).unwrap();
        let entries = store.load_user_entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].frequency, 999);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn remove_entry() {
        let path = temp_db_path("remove");
        let store = DictStore::open(&path).unwrap();
        store
            .upsert_user_entry(&DictionaryEntry {
                reading: "てすと".to_string(),
                surface: "テスト".to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 100,
            })
            .unwrap();
        let n = store.remove_user_entry("てすと", "テスト").unwrap();
        assert_eq!(n, 1);
        assert_eq!(store.load_user_entries().unwrap().len(), 0);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn increment_score_inserts_then_increments() {
        let path = temp_db_path("score");
        let store = DictStore::open(&path).unwrap();
        store.increment_score("きょう", "今日").unwrap();
        store.increment_score("きょう", "今日").unwrap();
        store.increment_score("きょう", "今日").unwrap();
        let scores = store.load_user_scores().unwrap();
        assert_eq!(scores.get("きょう|今日"), Some(&3));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn reopen_preserves_data() {
        let path = temp_db_path("reopen");
        {
            let store = DictStore::open(&path).unwrap();
            store
                .upsert_user_entry(&DictionaryEntry {
                    reading: "テスト".to_string(),
                    surface: "Test".to_string(),
                    pos: PartOfSpeech::Noun,
                    frequency: 50,
                })
                .unwrap();
            store.increment_score("a", "あ").unwrap();
        }
        let store = DictStore::open(&path).unwrap();
        assert_eq!(store.load_user_entries().unwrap().len(), 1);
        assert_eq!(store.load_user_scores().unwrap().get("a|あ"), Some(&1));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn migrate_legacy_json_imports_and_renames() {
        let path = temp_db_path("migrate");
        let parent = path.parent().unwrap().to_path_buf();
        let dict_json = parent.join("user_dict.json");
        let scores_json = parent.join("user_scores.json");

        std::fs::write(
            &dict_json,
            r#"[{"reading":"くろーど","surface":"クロード","pos":"Noun","frequency":200}]"#,
        )
        .unwrap();
        std::fs::write(
            &scores_json,
            r#"{"きょう|今日":5,"へんかん|変換":1}"#,
        )
        .unwrap();

        let store = DictStore::open(&path).unwrap();
        store.migrate_legacy_json(&dict_json, &scores_json).unwrap();

        let entries = store.load_user_entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].surface, "クロード");

        let scores = store.load_user_scores().unwrap();
        assert_eq!(scores.get("きょう|今日"), Some(&5));
        assert_eq!(scores.get("へんかん|変換"), Some(&1));

        assert!(!dict_json.exists());
        assert!(parent.join("user_dict.json.migrated").exists());
        assert!(!scores_json.exists());
        assert!(parent.join("user_scores.json.migrated").exists());

        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn migrate_legacy_json_is_idempotent() {
        let path = temp_db_path("migrate_idempotent");
        let parent = path.parent().unwrap().to_path_buf();
        let dict_json = parent.join("user_dict.json");
        let scores_json = parent.join("user_scores.json");

        std::fs::write(
            &dict_json,
            r#"[{"reading":"a","surface":"あ","pos":"Other","frequency":1}]"#,
        )
        .unwrap();
        std::fs::write(&scores_json, "{}").unwrap();

        let store = DictStore::open(&path).unwrap();
        store.migrate_legacy_json(&dict_json, &scores_json).unwrap();
        assert_eq!(store.load_user_entries().unwrap().len(), 1);

        // Second call with no JSON files (already renamed) is a no-op.
        store.migrate_legacy_json(&dict_json, &scores_json).unwrap();
        assert_eq!(store.load_user_entries().unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn recovers_stale_post_migration_jsons() {
        let path = temp_db_path("recover_stale");
        let parent = path.parent().unwrap().to_path_buf();
        let dict_json = parent.join("user_dict.json");
        let scores_json = parent.join("user_scores.json");

        // First migration: empty initial state, just sets the flag and
        // renames source files away.
        std::fs::write(&dict_json, "[]").unwrap();
        std::fs::write(&scores_json, "{}").unwrap();
        let store = DictStore::open(&path).unwrap();
        store.migrate_legacy_json(&dict_json, &scores_json).unwrap();

        // Simulate a v2.0+ engine recording one score directly into
        // SQLite after migration finishes.
        store.increment_score("きょう", "今日").unwrap();
        store.increment_score("きょう", "今日").unwrap(); // count = 2
        store
            .upsert_user_entry(&DictionaryEntry {
                reading: "shared".to_string(),
                surface: "共有".to_string(),
                pos: PartOfSpeech::Noun,
                frequency: 100,
            })
            .unwrap();

        // Now a stale v1.x-style process drops fresh JSON files in
        // place: a new dict entry, plus scores where one overlaps and
        // one is unique to the JSON side.
        std::fs::write(
            &dict_json,
            r#"[
                {"reading":"shared","surface":"共有","pos":"Noun","frequency":100},
                {"reading":"くろーど","surface":"クロード","pos":"Noun","frequency":50}
            ]"#,
        )
        .unwrap();
        std::fs::write(
            &scores_json,
            // きょう|今日 in JSON has count 1, lower than SQLite's 2 → keep SQLite
            // へんかん|変換 only in JSON → recover
            r#"{"きょう|今日":1,"へんかん|変換":3}"#,
        )
        .unwrap();

        // Re-call migrate_legacy_json — flag is set, so this triggers
        // the stale-recovery path.
        store.migrate_legacy_json(&dict_json, &scores_json).unwrap();

        // Dict: shared was already there (skipped), クロード was added.
        let mut entries = store.load_user_entries().unwrap();
        entries.sort_by(|a, b| a.surface.cmp(&b.surface));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].surface, "クロード");
        assert_eq!(entries[1].surface, "共有");

        // Scores: 今日 stayed at 2 (max), 変換 inserted as 3.
        let scores = store.load_user_scores().unwrap();
        assert_eq!(scores.get("きょう|今日"), Some(&2));
        assert_eq!(scores.get("へんかん|変換"), Some(&3));

        // Files renamed to .stale (not .migrated, to preserve the
        // distinction).
        assert!(!dict_json.exists());
        assert!(parent.join("user_dict.json.stale").exists());
        assert!(!scores_json.exists());
        assert!(parent.join("user_scores.json.stale").exists());

        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn recovery_on_clean_install_is_noop() {
        // No JSON files anywhere → recover_stale_jsons must not fail
        // and must not create empty .stale artifacts.
        let path = temp_db_path("recover_clean");
        let parent = path.parent().unwrap().to_path_buf();
        let dict_json = parent.join("user_dict.json");
        let scores_json = parent.join("user_scores.json");

        let store = DictStore::open(&path).unwrap();
        // Set the migration flag manually so migrate_legacy_json takes
        // the recovery path on first call.
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, '1')",
                params![MIGRATION_FLAG],
            )
            .unwrap();
        }
        store.migrate_legacy_json(&dict_json, &scores_json).unwrap();

        assert!(!parent.join("user_dict.json.stale").exists());
        assert!(!parent.join("user_scores.json.stale").exists());

        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn wal_mode_active_after_open() {
        let path = temp_db_path("wal");
        let store = DictStore::open(&path).unwrap();
        let mode: String = {
            let conn = store.conn.lock().unwrap();
            conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))
                .unwrap()
        };
        assert_eq!(mode.to_lowercase(), "wal");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn two_connections_can_both_write_concurrently() {
        use std::thread;

        let path = temp_db_path("two_conns");
        // Two independent connections to the same file — simulates
        // fcitx5 and IBus processes both holding the store open.
        let store_a = std::sync::Arc::new(DictStore::open(&path).unwrap());
        let store_b = std::sync::Arc::new(DictStore::open(&path).unwrap());

        let a = store_a.clone();
        let h1 = thread::spawn(move || {
            for i in 0..50 {
                a.upsert_user_entry(&DictionaryEntry {
                    reading: format!("a{}", i),
                    surface: format!("A{}", i),
                    pos: PartOfSpeech::Noun,
                    frequency: 100,
                })
                .unwrap();
            }
        });

        let b = store_b.clone();
        let h2 = thread::spawn(move || {
            for i in 0..50 {
                b.upsert_user_entry(&DictionaryEntry {
                    reading: format!("b{}", i),
                    surface: format!("B{}", i),
                    pos: PartOfSpeech::Noun,
                    frequency: 100,
                })
                .unwrap();
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();

        // Both connections see all 100 rows (50 from each writer)
        assert_eq!(store_a.load_user_entries().unwrap().len(), 100);
        assert_eq!(store_b.load_user_entries().unwrap().len(), 100);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn concurrent_score_increments_accumulate() {
        use std::thread;

        let path = temp_db_path("score_conc");
        let store_a = std::sync::Arc::new(DictStore::open(&path).unwrap());
        let store_b = std::sync::Arc::new(DictStore::open(&path).unwrap());

        let a = store_a.clone();
        let h1 = thread::spawn(move || {
            for _ in 0..100 {
                a.increment_score("k", "v").unwrap();
            }
        });
        let b = store_b.clone();
        let h2 = thread::spawn(move || {
            for _ in 0..100 {
                b.increment_score("k", "v").unwrap();
            }
        });
        h1.join().unwrap();
        h2.join().unwrap();

        let scores = store_a.load_user_scores().unwrap();
        assert_eq!(scores.get("k|v"), Some(&200));

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn migrate_with_no_legacy_files_is_safe() {
        let path = temp_db_path("migrate_empty");
        let parent = path.parent().unwrap().to_path_buf();
        let store = DictStore::open(&path).unwrap();
        // No legacy files — migration just sets the flag and returns.
        store
            .migrate_legacy_json(&parent.join("user_dict.json"), &parent.join("user_scores.json"))
            .unwrap();
        assert_eq!(store.load_user_entries().unwrap().len(), 0);
        let _ = std::fs::remove_dir_all(&parent);
    }
}
