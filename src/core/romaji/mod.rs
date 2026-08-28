/// Romaji to Kana conversion state machine
///
/// Converts ASCII key input into Hiragana.
/// Handles: standard romaji, double consonants (っ), "n" ambiguity (ん),
/// compound sounds (しゃ, ちゅ, etc.), small kana (ぁ, っ via x/l prefix).

mod romaji_table;

use romaji_table::ROMAJI_TABLE;

pub struct RomajiConverter {
    buffer: String,
    output: String,
    /// Case-preserved raw input history, parallel to `output + buffer`
    /// but recording exactly what the user typed. Used by F9/F10 to
    /// round-trip "VIM"/"shi" as typed instead of deriving from kana
    /// ("ＶＩＭ"/"ＶＩＭ" vs "ｖｉｍ", "shi" vs "si"). `None` when
    /// tracking has been invalidated by `append_raw` or a `delete_last`
    /// that popped from `output` (we don't know how many raw chars
    /// produced the removed kana).
    raw_input: Option<String>,
}

impl RomajiConverter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            output: String::new(),
            raw_input: Some(String::new()),
        }
    }

    /// Process a single key input and return converted kana (if any).
    /// Case-folds the key for kana-table lookups so uppercase input
    /// ("VIM") still produces "ゔぃ"+"m"; the original case is kept in
    /// `raw_input` so F9/F10 can return "ＶＩＭ"/"VIM".
    pub fn process_key(&mut self, key: char) -> Option<String> {
        if let Some(raw) = self.raw_input.as_mut() {
            raw.push(key);
        }
        self.buffer.push(key.to_ascii_lowercase());
        self.try_convert()
    }

    fn try_convert(&mut self) -> Option<String> {
        // 1. Exact match in romaji table
        if let Some(kana) = exact_lookup(&self.buffer) {
            // Special case: "nn" followed by a vowel/y should be ん + n(vowel)
            // But "nn" at this point is a complete match, so it's fine —
            // the issue is handled by "n" + consonant logic below.
            self.buffer.clear();
            self.output.push_str(kana);
            return Some(kana.to_string());
        }

        // 2. Buffer is a prefix of some table entry — keep buffering
        if has_prefix_match(&self.buffer) {
            return None;
        }

        // 3. Double consonant → っ (or ん for nn) + keep rest
        //    Must check before n-rule so "nn" is handled as double-n
        if self.buffer.len() >= 2 {
            let bytes = self.buffer.as_bytes();
            if bytes[0] == bytes[1] && is_doubling_consonant(bytes[0] as char) {
                let geminate = if bytes[0] == b'n' { "ん" } else { "っ" };
                let is_nn = bytes[0] == b'n';
                self.output.push_str(geminate);
                if is_nn {
                    // "nn" → "ん", clear buffer entirely (Mozc-style)
                    self.buffer.clear();
                } else {
                    // "kk" → "っ", keep second consonant for next syllable
                    let rest = self.buffer[1..].to_string();
                    self.buffer = rest;
                }

                if !self.buffer.is_empty() {
                    if let Some(more_kana) = self.try_convert() {
                        return Some(format!("{}{}", geminate, more_kana));
                    }
                }
                return Some(geminate.to_string());
            }
        }

        // 4. "n" + non-matching char → ん + restart with remaining
        if self.buffer.len() >= 2 && self.buffer.starts_with('n') {
            let rest = self.buffer[1..].to_string();
            self.buffer = rest;
            self.output.push('ん');

            // Try to convert the remaining buffer recursively
            if let Some(more_kana) = self.try_convert() {
                return Some(format!("ん{}", more_kana));
            }
            return Some("ん".to_string());
        }

        // 5. Non-alpha passthrough (punctuation like '-')
        // Buffer is guaranteed non-empty by the len==1 guard.
        if let Some(ch) = self.buffer.chars().next().filter(|_| self.buffer.len() == 1) {
            if !ch.is_ascii_alphabetic() {
                if let Some(kana) = exact_lookup(&self.buffer) {
                    self.buffer.clear();
                    self.output.push_str(kana);
                    return Some(kana.to_string());
                }
            }
        }

        // 6. No match — discard buffer. Invalidate raw_input the same
        // way delete_last does when it can no longer preserve parity:
        // the discarded chars are already in raw_input but produced no
        // output, so a later F9/F10 that reached back into raw_input
        // would replay them as ghost text (kq + ka → preedit "か" but
        // raw_input "kqka", F10 → "kqka").
        if !self.buffer.is_empty() {
            self.raw_input = None;
        }
        self.buffer.clear();
        None
    }

    /// Flush remaining buffer (call on space/enter/commit)
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer == "n" {
            self.buffer.clear();
            self.output.push('ん');
            return Some("ん".to_string());
        }
        if !self.buffer.is_empty() {
            // Same story as rule-6 discard: the pending buffer never
            // produced kana, so raw_input's trailing chars are ghost
            // input that must not resurface via F9/F10. (saky + Space
            // → converted "さ", raw_input drops "ky" instead of
            // pasting "saky" back into F10.)
            self.raw_input = None;
            self.buffer.clear();
        }
        None
    }

    /// Reset the conversion buffer and output
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.output.clear();
        self.raw_input = Some(String::new());
    }

    /// Restore a previously-captured pending buffer — used when cancelling
    /// an F-key conversion that snapshotted the trailing consonant into
    /// the segment state at `start_kana_conversion` time. Rewriting it
    /// here lets the follow-up `preedit()` show kana+"m" as it did before
    /// F-key (bug [15]). No-op if the buffer already holds content — a
    /// concurrent keystroke would have already advanced past the cancel.
    pub fn restore_buffer(&mut self, pending: &str) {
        if !pending.is_empty() && self.buffer.is_empty() {
            self.buffer.push_str(pending);
        }
    }

    /// Get current romaji buffer (incomplete input)
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Case-preserved raw input history, or `None` if invalidated. See
    /// the `raw_input` field docs.
    pub fn raw_input(&self) -> Option<&str> {
        self.raw_input.as_deref()
    }

    /// Delete the last character from the buffer or output.
    /// Returns true if something was deleted, false if empty.
    pub fn delete_last(&mut self) -> bool {
        if !self.buffer.is_empty() {
            self.buffer.pop();
            if let Some(raw) = self.raw_input.as_mut() {
                raw.pop();
            }
            true
        } else if !self.output.is_empty() {
            self.output.pop();
            // Popping from output loses the raw→kana boundary info
            // (one kana may have been produced by 1-3 raw chars). We
            // can't accurately pop from raw_input, so invalidate it —
            // F9/F10 falls back to deriving romaji from the remaining
            // kana.
            self.raw_input = None;
            true
        } else {
            false
        }
    }

    /// Append a raw string directly to the output (bypassing romaji conversion).
    pub fn append_raw(&mut self, s: &str) {
        self.output.push_str(s);
        // Bypass mode — the output no longer has a 1:1 raw-input record.
        self.raw_input = None;
    }

    /// Get accumulated kana output
    pub fn output(&self) -> &str {
        &self.output
    }
}

/// Convert a hiragana string to full-width katakana.
/// Non-hiragana characters (ー, kanji, etc.) are left unchanged.
pub fn hiragana_to_katakana(s: &str) -> String {
    s.chars()
        .map(|c| {
            if ('\u{3041}'..='\u{3096}').contains(&c) {
                // Hiragana (ぁ-ゖ) → Katakana (ァ-ヶ): offset +0x60
                char::from_u32(c as u32 + 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Convert a hiragana string to half-width katakana.
/// Dakuten/handakuten are decomposed into separate characters.
pub fn hiragana_to_halfwidth_katakana(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            'あ' => result.push('ｱ'), 'い' => result.push('ｲ'), 'う' => result.push('ｳ'),
            'え' => result.push('ｴ'), 'お' => result.push('ｵ'),
            'か' => result.push('ｶ'), 'き' => result.push('ｷ'), 'く' => result.push('ｸ'),
            'け' => result.push('ｹ'), 'こ' => result.push('ｺ'),
            'さ' => result.push('ｻ'), 'し' => result.push('ｼ'), 'す' => result.push('ｽ'),
            'せ' => result.push('ｾ'), 'そ' => result.push('ｿ'),
            'た' => result.push('ﾀ'), 'ち' => result.push('ﾁ'), 'つ' => result.push('ﾂ'),
            'て' => result.push('ﾃ'), 'と' => result.push('ﾄ'),
            'な' => result.push('ﾅ'), 'に' => result.push('ﾆ'), 'ぬ' => result.push('ﾇ'),
            'ね' => result.push('ﾈ'), 'の' => result.push('ﾉ'),
            'は' => result.push('ﾊ'), 'ひ' => result.push('ﾋ'), 'ふ' => result.push('ﾌ'),
            'へ' => result.push('ﾍ'), 'ほ' => result.push('ﾎ'),
            'ま' => result.push('ﾏ'), 'み' => result.push('ﾐ'), 'む' => result.push('ﾑ'),
            'め' => result.push('ﾒ'), 'も' => result.push('ﾓ'),
            'や' => result.push('ﾔ'), 'ゆ' => result.push('ﾕ'), 'よ' => result.push('ﾖ'),
            'ら' => result.push('ﾗ'), 'り' => result.push('ﾘ'), 'る' => result.push('ﾙ'),
            'れ' => result.push('ﾚ'), 'ろ' => result.push('ﾛ'),
            'わ' => result.push('ﾜ'), 'を' => result.push('ｦ'), 'ん' => result.push('ﾝ'),
            // Dakuten (voiced): base + ﾞ
            'が' => { result.push('ｶ'); result.push('ﾞ'); }
            'ぎ' => { result.push('ｷ'); result.push('ﾞ'); }
            'ぐ' => { result.push('ｸ'); result.push('ﾞ'); }
            'げ' => { result.push('ｹ'); result.push('ﾞ'); }
            'ご' => { result.push('ｺ'); result.push('ﾞ'); }
            'ざ' => { result.push('ｻ'); result.push('ﾞ'); }
            'じ' => { result.push('ｼ'); result.push('ﾞ'); }
            'ず' => { result.push('ｽ'); result.push('ﾞ'); }
            'ぜ' => { result.push('ｾ'); result.push('ﾞ'); }
            'ぞ' => { result.push('ｿ'); result.push('ﾞ'); }
            'だ' => { result.push('ﾀ'); result.push('ﾞ'); }
            'ぢ' => { result.push('ﾁ'); result.push('ﾞ'); }
            'づ' => { result.push('ﾂ'); result.push('ﾞ'); }
            'で' => { result.push('ﾃ'); result.push('ﾞ'); }
            'ど' => { result.push('ﾄ'); result.push('ﾞ'); }
            'ば' => { result.push('ﾊ'); result.push('ﾞ'); }
            'び' => { result.push('ﾋ'); result.push('ﾞ'); }
            'ぶ' => { result.push('ﾌ'); result.push('ﾞ'); }
            'べ' => { result.push('ﾍ'); result.push('ﾞ'); }
            'ぼ' => { result.push('ﾎ'); result.push('ﾞ'); }
            'ゔ' => { result.push('ｳ'); result.push('ﾞ'); }
            // Handakuten (p-sounds): base + ﾟ
            'ぱ' => { result.push('ﾊ'); result.push('ﾟ'); }
            'ぴ' => { result.push('ﾋ'); result.push('ﾟ'); }
            'ぷ' => { result.push('ﾌ'); result.push('ﾟ'); }
            'ぺ' => { result.push('ﾍ'); result.push('ﾟ'); }
            'ぽ' => { result.push('ﾎ'); result.push('ﾟ'); }
            // Small kana
            'ぁ' => result.push('ｧ'), 'ぃ' => result.push('ｨ'), 'ぅ' => result.push('ｩ'),
            'ぇ' => result.push('ｪ'), 'ぉ' => result.push('ｫ'),
            'っ' => result.push('ｯ'),
            'ゃ' => result.push('ｬ'), 'ゅ' => result.push('ｭ'), 'ょ' => result.push('ｮ'),
            // Long vowel mark
            'ー' => result.push('ｰ'),
            // Punctuation and symbols (full-width → half-width)
            '。' => result.push('｡'), '、' => result.push('､'),
            '「' => result.push('｢'), '」' => result.push('｣'),
            '・' => result.push('･'),
            '！' => result.push('!'), '？' => result.push('?'),
            '（' => result.push('('), '）' => result.push(')'),
            '｛' => result.push('{'), '｝' => result.push('}'),
            '［' => result.push('['), '］' => result.push(']'),
            '＋' => result.push('+'), '－' => result.push('-'),
            '＝' => result.push('='), '＊' => result.push('*'),
            '／' => result.push('/'), '＼' => result.push('\\'),
            '＆' => result.push('&'), '＠' => result.push('@'),
            '＃' => result.push('#'), '＄' => result.push('$'),
            '％' => result.push('%'), '＾' => result.push('^'),
            '｜' => result.push('|'), '～' => result.push('~'),
            '＜' => result.push('<'), '＞' => result.push('>'),
            '：' => result.push(':'), '；' => result.push(';'),
            '＿' => result.push('_'), '＂' => result.push('"'),
            '＇' => result.push('\''),
            // Full-width digits → half-width
            '０'..='９' => result.push((c as u32 - '０' as u32 + '0' as u32) as u8 as char),
            // Full-width ASCII letters → half-width
            'Ａ'..='Ｚ' => result.push((c as u32 - 'Ａ' as u32 + 'A' as u32) as u8 as char),
            'ａ'..='ｚ' => result.push((c as u32 - 'ａ' as u32 + 'a' as u32) as u8 as char),
            // Pass through anything else unchanged
            _ => result.push(c),
        }
    }
    result
}

/// Convert a hiragana string back to half-width ASCII romaji.
/// Uses the romaji table in reverse. Characters not found (kanji, symbols, etc.)
/// are passed through after converting full-width forms to half-width.
pub fn hiragana_to_romaji(s: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Try 3-char, then 2-char, then 1-char match
        let mut matched = false;
        for len in (1..=3).rev() {
            if i + len > chars.len() {
                continue;
            }
            let slice: String = chars[i..i + len].iter().collect();
            if let Some(romaji) = kana_to_romaji_lookup(&slice) {
                result.push_str(romaji);
                i += len;
                matched = true;
                break;
            }
        }
        if !matched {
            let c = chars[i];
            // っ → double the next consonant, or "xtu" if at end
            if c == 'っ' {
                let mut doubled = false;
                if i + 1 < chars.len() {
                    for len in (1..=3).rev() {
                        if i + 1 + len > chars.len() {
                            continue;
                        }
                        let next_slice: String = chars[i + 1..i + 1 + len].iter().collect();
                        if let Some(romaji) = kana_to_romaji_lookup(&next_slice) {
                            if let Some(consonant) = romaji.chars().next() {
                                // Only double if the next mora actually starts
                                // with a consonant. Bare vowels (あいうえお)
                                // otherwise emitted "aa"/"ii"/... for "っあ" —
                                // e.g. F9/F10 turned "あっあ" into "aaa"
                                // instead of the correct "axtua".
                                let is_vowel = matches!(consonant, 'a' | 'e' | 'i' | 'o' | 'u');
                                if consonant.is_ascii_alphabetic() && !is_vowel {
                                    result.push(consonant);
                                    doubled = true;
                                }
                            }
                            break;
                        }
                    }
                }
                if !doubled {
                    result.push_str("xtu");
                }
                i += 1;
                continue;
            }
            // Full-width → half-width conversion for non-kana
            match c {
                '。' => result.push('.'),
                '、' => result.push(','),
                '！' => result.push('!'),
                '？' => result.push('?'),
                '（' => result.push('('),
                '）' => result.push(')'),
                '［' => result.push('['),
                '］' => result.push(']'),
                '｛' => result.push('{'),
                '｝' => result.push('}'),
                '＋' => result.push('+'),
                '＝' => result.push('='),
                '＊' => result.push('*'),
                '／' => result.push('/'),
                '＼' => result.push('\\'),
                '＆' => result.push('&'),
                '＠' => result.push('@'),
                '＃' => result.push('#'),
                '＄' => result.push('$'),
                '％' => result.push('%'),
                '＾' => result.push('^'),
                '｜' => result.push('|'),
                '～' => result.push('~'),
                '＜' => result.push('<'),
                '＞' => result.push('>'),
                '：' => result.push(':'),
                '；' => result.push(';'),
                '＿' => result.push('_'),
                '＂' => result.push('"'),
                '｀' => result.push('`'),
                'ー' => result.push('-'),
                _ => result.push(c),
            }
            i += 1;
        }
    }
    result
}

/// ASCII → full-width Japanese equivalent for a single character.
///
/// Shared by every front-end that inserts a punctuation/digit character
/// into the preedit while Bonolith is active (Fcitx5 via `bonolith_handle_key`,
/// IBus via `process_key_event`, F9 kana→romaji form). Previously duplicated
/// three times — the IBus and FFI copies also missed `'`, `-`, and ` ` that
/// the F9 form already handled.
///
/// Digits and letters map to their U+FF01-block equivalents; the punctuation
/// table below matches the Mozc/ATOK conventions users expect. Returns
/// `None` for characters we deliberately leave unchanged.
pub fn to_fullwidth_char(c: char) -> Option<char> {
    Some(match c {
        'a'..='z' => char::from_u32(c as u32 - 'a' as u32 + 'ａ' as u32)?,
        'A'..='Z' => char::from_u32(c as u32 - 'A' as u32 + 'Ａ' as u32)?,
        '0'..='9' => char::from_u32(c as u32 - '0' as u32 + '０' as u32)?,
        '!' => '！', '?' => '？', '.' => '。', ',' => '、',
        '(' => '（', ')' => '）', '[' => '［', ']' => '］',
        '{' => '｛', '}' => '｝', '+' => '＋', '=' => '＝',
        '*' => '＊', '/' => '／', '\\' => '＼', '&' => '＆',
        '@' => '＠', '#' => '＃', '$' => '＄', '%' => '％',
        '^' => '＾', '|' => '｜', '~' => '～', '<' => '＜',
        '>' => '＞', ':' => '：', ';' => '；', '_' => '＿',
        '"' => '＂', '\'' => '＇', '`' => '｀', '-' => 'ー',
        ' ' => '\u{3000}', // full-width space
        _ => return None,
    })
}

/// Convert a hiragana string to full-width ASCII romaji.
/// First converts to half-width romaji, then maps each ASCII char to its full-width equivalent.
pub fn hiragana_to_fullwidth_romaji(s: &str) -> String {
    let romaji = hiragana_to_romaji(s);
    romaji
        .chars()
        .map(|c| to_fullwidth_char(c).unwrap_or(c))
        .collect()
}

/// Reverse lookup: kana string → romaji.
/// Prefers shorter/common romaji forms, and only pure-alpha results
/// (avoids "n'" for ん — handled specially in the caller).
fn kana_to_romaji_lookup(kana: &str) -> Option<&'static str> {
    if kana == "ん" {
        return Some("n");
    }
    // っ is handled specially in hiragana_to_romaji (consonant doubling)
    if kana == "っ" {
        return None;
    }
    let mut best: Option<&'static str> = None;
    for &(romaji, k) in ROMAJI_TABLE.iter() {
        if k == kana {
            // Prefer shortest, alpha-only romaji
            let dominated = match best {
                None => false,
                Some(prev) => {
                    if romaji.len() < prev.len() {
                        false
                    } else if romaji.len() == prev.len() {
                        // Prefer alpha-only over forms with punctuation
                        !romaji.chars().all(|c| c.is_ascii_alphabetic())
                    } else {
                        true
                    }
                }
            };
            if !dominated {
                best = Some(romaji);
            }
        }
    }
    best
}

fn exact_lookup(s: &str) -> Option<&'static str> {
    ROMAJI_TABLE
        .binary_search_by_key(&s, |&(romaji, _)| romaji)
        .ok()
        .map(|i| ROMAJI_TABLE[i].1)
}

fn has_prefix_match(s: &str) -> bool {
    let pos = ROMAJI_TABLE.partition_point(|&(romaji, _)| romaji < s);
    if pos < ROMAJI_TABLE.len() && ROMAJI_TABLE[pos].0.starts_with(s) {
        return true;
    }
    // Also check the entry before (for cases where s falls between entries)
    if pos > 0 && ROMAJI_TABLE[pos - 1].0.starts_with(s) {
        return true;
    }
    false
}

fn is_doubling_consonant(c: char) -> bool {
    matches!(
        c,
        'b' | 'c' | 'd' | 'f' | 'g' | 'h' | 'j' | 'k' | 'm' | 'n' | 'p' | 'r' | 's' | 't'
            | 'v' | 'w' | 'z'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(input: &str) -> String {
        let mut conv = RomajiConverter::new();
        let mut result = String::new();
        for ch in input.chars() {
            if let Some(kana) = conv.process_key(ch) {
                result.push_str(&kana);
            }
        }
        if let Some(kana) = conv.flush() {
            result.push_str(&kana);
        }
        result
    }

    #[test]
    fn vowels() {
        assert_eq!(convert("a"), "あ");
        assert_eq!(convert("i"), "い");
        assert_eq!(convert("u"), "う");
        assert_eq!(convert("e"), "え");
        assert_eq!(convert("o"), "お");
    }

    #[test]
    fn basic_consonant_vowel() {
        assert_eq!(convert("ka"), "か");
        assert_eq!(convert("ki"), "き");
        assert_eq!(convert("ku"), "く");
        assert_eq!(convert("ke"), "け");
        assert_eq!(convert("ko"), "こ");
        assert_eq!(convert("sa"), "さ");
        assert_eq!(convert("ta"), "た");
        assert_eq!(convert("na"), "な");
        assert_eq!(convert("ha"), "は");
        assert_eq!(convert("ma"), "ま");
        assert_eq!(convert("ya"), "や");
        assert_eq!(convert("ra"), "ら");
        assert_eq!(convert("wa"), "わ");
    }

    #[test]
    fn shi_chi_tsu_fu() {
        assert_eq!(convert("shi"), "し");
        assert_eq!(convert("si"), "し");
        assert_eq!(convert("chi"), "ち");
        assert_eq!(convert("ti"), "ち");
        assert_eq!(convert("tsu"), "つ");
        assert_eq!(convert("tu"), "つ");
        assert_eq!(convert("fu"), "ふ");
        assert_eq!(convert("hu"), "ふ");
    }

    #[test]
    fn compound_sounds() {
        assert_eq!(convert("sha"), "しゃ");
        assert_eq!(convert("shu"), "しゅ");
        assert_eq!(convert("sho"), "しょ");
        assert_eq!(convert("cha"), "ちゃ");
        assert_eq!(convert("chu"), "ちゅ");
        assert_eq!(convert("cho"), "ちょ");
        assert_eq!(convert("ja"), "じゃ");
        assert_eq!(convert("ju"), "じゅ");
        assert_eq!(convert("jo"), "じょ");
        assert_eq!(convert("kya"), "きゃ");
        assert_eq!(convert("kyu"), "きゅ");
        assert_eq!(convert("kyo"), "きょ");
        assert_eq!(convert("nya"), "にゃ");
        assert_eq!(convert("nyu"), "にゅ");
        assert_eq!(convert("nyo"), "にょ");
    }

    #[test]
    fn double_consonant() {
        assert_eq!(convert("kka"), "っか");
        assert_eq!(convert("tta"), "った");
        assert_eq!(convert("ssa"), "っさ");
        assert_eq!(convert("ppa"), "っぱ");
        assert_eq!(convert("cchi"), "っち");
    }

    #[test]
    fn tchi() {
        assert_eq!(convert("tchi"), "っち");
    }

    #[test]
    fn n_handling() {
        assert_eq!(convert("na"), "な");
        assert_eq!(convert("ni"), "に");
        assert_eq!(convert("nn"), "ん");  // n+n → ん (via n-rule, second n flushed)
        assert_eq!(convert("n'"), "ん");
        // n before consonant
        assert_eq!(convert("nk"), "ん");
        assert_eq!(convert("kanji"), "かんじ");
        assert_eq!(convert("shinbun"), "しんぶん");
    }

    #[test]
    fn n_flush() {
        let mut conv = RomajiConverter::new();
        conv.process_key('n');
        assert_eq!(conv.flush(), Some("ん".to_string()));
    }

    #[test]
    fn small_kana() {
        assert_eq!(convert("xtu"), "っ");
        assert_eq!(convert("ltu"), "っ");
        assert_eq!(convert("xa"), "ぁ");
        assert_eq!(convert("la"), "ぁ");
    }

    #[test]
    fn long_vowel_mark() {
        assert_eq!(convert("-"), "ー");
    }

    #[test]
    fn word_toukyou() {
        assert_eq!(convert("toukyou"), "とうきょう");
    }

    #[test]
    fn word_gakkou() {
        assert_eq!(convert("gakkou"), "がっこう");
    }

    #[test]
    fn word_konnichiwa() {
        // "nn" clears buffer (Mozc-style): "konnichiwa" → "こんいちわ"
        // To type "こんにちわ", use "konnnichiwa" (3 n's)
        assert_eq!(convert("konnichiwa"), "こんいちわ");
        assert_eq!(convert("konnnichiwa"), "こんにちわ");
    }

    #[test]
    fn word_nihongo() {
        assert_eq!(convert("nihongo"), "にほんご");
    }

    #[test]
    fn word_senshuu() {
        assert_eq!(convert("senshuu"), "せんしゅう");
    }

    #[test]
    fn output_accumulates() {
        let mut conv = RomajiConverter::new();
        for ch in "toukyou".chars() {
            conv.process_key(ch);
        }
        assert_eq!(conv.output(), "とうきょう");
    }

    #[test]
    fn wo_particle() {
        assert_eq!(convert("wo"), "を");
    }

    #[test]
    fn wh_row() {
        assert_eq!(convert("wha"), "うぁ");
        assert_eq!(convert("whi"), "うぃ");
        assert_eq!(convert("whe"), "うぇ");
        assert_eq!(convert("who"), "うぉ");
    }

    #[test]
    fn hiragana_to_katakana_basic() {
        assert_eq!(hiragana_to_katakana("ぷろんぷと"), "プロンプト");
    }

    #[test]
    fn hiragana_to_katakana_mixed() {
        // Long vowel mark and non-hiragana pass through unchanged
        assert_eq!(hiragana_to_katakana("こーひー"), "コーヒー");
        assert_eq!(hiragana_to_katakana("あ"), "ア");
    }

    #[test]
    fn hiragana_to_katakana_empty() {
        assert_eq!(hiragana_to_katakana(""), "");
    }

    #[test]
    fn halfwidth_katakana_basic() {
        assert_eq!(hiragana_to_halfwidth_katakana("ぷろんぷと"), "ﾌﾟﾛﾝﾌﾟﾄ");
    }

    #[test]
    fn halfwidth_katakana_dakuten() {
        assert_eq!(hiragana_to_halfwidth_katakana("がぎぐげご"), "ｶﾞｷﾞｸﾞｹﾞｺﾞ");
        assert_eq!(hiragana_to_halfwidth_katakana("ぱぴぷぺぽ"), "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ");
    }

    #[test]
    fn halfwidth_katakana_small() {
        assert_eq!(hiragana_to_halfwidth_katakana("っ"), "ｯ");
        assert_eq!(hiragana_to_halfwidth_katakana("ゃゅょ"), "ｬｭｮ");
    }

    #[test]
    fn halfwidth_katakana_punctuation() {
        assert_eq!(hiragana_to_halfwidth_katakana("。、"), "｡､");
    }

    #[test]
    fn halfwidth_symbols() {
        assert_eq!(hiragana_to_halfwidth_katakana("！？"), "!?");
        assert_eq!(hiragana_to_halfwidth_katakana("（）"), "()");
    }

    #[test]
    fn romaji_basic() {
        assert_eq!(hiragana_to_romaji("か"), "ka");
        assert_eq!(hiragana_to_romaji("し"), "si");
        assert_eq!(hiragana_to_romaji("ち"), "ti");
        assert_eq!(hiragana_to_romaji("つ"), "tu");
    }

    #[test]
    fn romaji_words() {
        assert_eq!(hiragana_to_romaji("とうきょう"), "toukyou");
        assert_eq!(hiragana_to_romaji("にほんご"), "nihongo");
    }

    #[test]
    fn romaji_sokuon() {
        // っ + consonant → doubled consonant
        assert_eq!(hiragana_to_romaji("がっこう"), "gakkou");
        assert_eq!(hiragana_to_romaji("きって"), "kitte");
    }

    #[test]
    fn romaji_fullwidth_symbols() {
        assert_eq!(hiragana_to_romaji("。"), ".");
        assert_eq!(hiragana_to_romaji("、"), ",");
        assert_eq!(hiragana_to_romaji("！"), "!");
    }

    #[test]
    fn to_fullwidth_char_covers_shared_table() {
        // Punctuation & digits that every front-end used to duplicate.
        assert_eq!(to_fullwidth_char(','), Some('、'));
        assert_eq!(to_fullwidth_char('.'), Some('。'));
        assert_eq!(to_fullwidth_char('!'), Some('！'));
        assert_eq!(to_fullwidth_char('7'), Some('７'));
        // Letters and hyphen/apostrophe DO map (F9 form uses them);
        // the front-end wrappers filter these out so process_key still
        // owns romaji-input characters.
        assert_eq!(to_fullwidth_char('a'), Some('ａ'));
        assert_eq!(to_fullwidth_char('-'), Some('ー'));
        assert_eq!(to_fullwidth_char('\''), Some('＇'));
        assert_eq!(to_fullwidth_char(' '), Some('\u{3000}'));
        // Non-ASCII passthrough remains None.
        assert_eq!(to_fullwidth_char('あ'), None);
    }

    #[test]
    fn romaji_compound() {
        // Compound sounds should match as multi-char kana
        let sha = hiragana_to_romaji("しゃ");
        assert!(sha == "sya" || sha == "sha", "got: {}", sha);
        let cho = hiragana_to_romaji("ちょ");
        assert!(cho == "tyo" || cho == "cho", "got: {}", cho);
    }

    /// Regression (bug_list_fable_5_review_2026-08-28 #6a): a rule-6
    /// discard of an unmatched buffer prefix ("kq") must invalidate
    /// raw_input so a later F9/F10 does not replay the discarded chars
    /// as ghost text. Before the fix, `kq` + `ka` left output "か" but
    /// raw_input "kqka", so F10 rendered "kqka".
    #[test]
    fn rule6_discard_invalidates_raw_input() {
        let mut conv = RomajiConverter::new();
        conv.process_key('k');
        conv.process_key('q'); // rule-6 discard: "kq" has no prefix
        conv.process_key('k');
        conv.process_key('a'); // → か
        assert_eq!(conv.output(), "か");
        assert!(
            conv.raw_input().is_none(),
            "rule-6 discard should have invalidated raw_input; got {:?}",
            conv.raw_input(),
        );
    }

    /// Regression (bug_list_fable_5_review_2026-08-28 #6b): flush with
    /// a non-"n" pending buffer must invalidate raw_input the same way
    /// rule-6 discard does. Before the fix, `saky` + flush left output
    /// "さ" but raw_input "saky", so F10 rendered "saky".
    #[test]
    fn flush_non_n_discard_invalidates_raw_input() {
        let mut conv = RomajiConverter::new();
        for ch in "saky".chars() {
            conv.process_key(ch);
        }
        assert_eq!(conv.output(), "さ");
        assert_eq!(conv.buffer(), "ky");
        conv.flush();
        assert!(
            conv.raw_input().is_none(),
            "flush with non-n buffer should invalidate raw_input; got {:?}",
            conv.raw_input(),
        );
    }

    /// flush("n") is a legitimate produces-kana path — raw_input stays
    /// valid so F9/F10 on the resulting ん still round-trips to "n".
    #[test]
    fn flush_lone_n_preserves_raw_input() {
        let mut conv = RomajiConverter::new();
        conv.process_key('n');
        conv.flush();
        assert_eq!(conv.output(), "ん");
        assert_eq!(conv.raw_input(), Some("n"));
    }
}
