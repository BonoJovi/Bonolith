/// SQLite-backed persistent store for user dictionary entries and scores.
///
/// Replaces the v1.x `user_dict.json` + `user_scores.json` pair with a
/// single `dict.sqlite`. JSON is retained only as an import/export format.

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
    conn: Mutex<Connection>,
}

impl DictStore {
    /// Open or create the database at the given path. Initializes the
    /// schema on first creation; safe to call repeatedly.
    pub fn open(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(sqlite_to_io)?;
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

    /// Default path: `$XDG_DATA_HOME/jaim/dict.sqlite` (typically
    /// `~/.local/share/jaim/dict.sqlite`).
    pub fn default_path() -> io::Result<PathBuf> {
        let data_dir = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                PathBuf::from(home).join(".local/share")
            })
            .join("jaim");
        Ok(data_dir.join("dict.sqlite"))
    }

    fn init_schema(&self) -> io::Result<()> {
        let conn = self.conn.lock().unwrap();
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
             COMMIT;",
        )
        .map_err(sqlite_to_io)?;
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value
               WHERE meta.value < excluded.value",
            params![SCHEMA_VERSION.to_string()],
        )
        .map_err(sqlite_to_io)?;
        Ok(())
    }

    /// Migrate `user_dict.json` and `user_scores.json` into the database
    /// if not already done. Idempotent: a meta flag prevents repeated
    /// runs. On success, the JSON files are renamed to `*.migrated` so
    /// the user can recover from them if needed.
    pub fn migrate_legacy_json(
        &self,
        dict_json: &Path,
        scores_json: &Path,
    ) -> io::Result<()> {
        if self.is_migrated()? {
            return Ok(());
        }

        let entries = read_legacy_dict_json(dict_json)?;
        let scores = read_legacy_scores_json(scores_json)?;

        let mut conn = self.conn.lock().unwrap();
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

    fn is_migrated(&self) -> io::Result<bool> {
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
        let n = conn
            .execute(
                "DELETE FROM user_entries WHERE reading = ?1 AND surface = ?2",
                params![reading, surface],
            )
            .map_err(sqlite_to_io)?;
        Ok(n)
    }

    pub fn load_user_scores(&self) -> io::Result<HashMap<String, u32>> {
        let conn = self.conn.lock().unwrap();
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
        let mut conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO user_scores (reading, surface, count) VALUES (?1, ?2, 1)
             ON CONFLICT(reading, surface) DO UPDATE SET count = count + 1",
            params![reading, surface],
        )
        .map_err(sqlite_to_io)?;
        Ok(())
    }
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
        let dir = std::env::temp_dir().join(format!("jaim_test_store_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("dict.sqlite")
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

        // Drop a NEW JSON in place — second migration must be a no-op
        std::fs::write(
            &dict_json,
            r#"[{"reading":"b","surface":"い","pos":"Other","frequency":2}]"#,
        )
        .unwrap();
        store.migrate_legacy_json(&dict_json, &scores_json).unwrap();
        assert_eq!(store.load_user_entries().unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&parent);
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
