/// LLM-based contextual ranking and final validation
///
/// Uses a local quantized model (Qwen2.5-1.5B Q4_K_M) for:
/// 1. Disambiguating homophone candidates
/// 2. Final check on dictionary-converted sentences
/// 3. Context-aware reranking of candidates
///
/// Runs on a background thread with KV cache pre-warming
/// to minimize latency at conversion time.

mod http_scorer;

/// Neutral LLM score, returned when the model provides no usable signal
/// (empty context, missing probabilities). Used as the midpoint of the
/// scorer's 0.3–0.9 output band and as an explicit "no LLM opinion" value
/// for candidates we deliberately exclude from LLM influence.
pub const NEUTRAL_SCORE: f64 = 0.5;

/// Trait defining the LLM scoring interface.
/// Allows swapping between mock and real implementations.
pub trait LlmScorer: Send + Sync {
    /// Score a candidate sentence for contextual naturalness.
    /// Higher score = more natural. Range: 0.0–1.0.
    fn score(&self, context: &str, candidate: &str) -> f64;

    /// Pre-warm the cache with confirmed context (called during typing).
    fn warm_cache(&self, context: &str);
}

pub use http_scorer::HttpLlamaScorer;

/// LLM engine that uses a pluggable scorer for candidate reranking.
pub struct LlmEngine {
    scorer: Box<dyn LlmScorer>,
    /// Committed text context for scoring continuity
    context: String,
}

impl LlmEngine {
    /// Create with the best available scorer.
    /// Tries HTTP llama-server first, falls back to MockScorer.
    pub fn new() -> Self {
        let scorer: Box<dyn LlmScorer> = match HttpLlamaScorer::from_default_endpoint() {
            Some(s) => {
                log::info!("LlmEngine: using HttpLlamaScorer");
                Box::new(s)
            }
            None => {
                log::info!("LlmEngine: no LLM server available, using MockScorer");
                Box::new(MockScorer)
            }
        };
        Self {
            scorer,
            context: String::new(),
        }
    }

    /// Create with a custom scorer (for real LLM integration).
    pub fn with_scorer(scorer: Box<dyn LlmScorer>) -> Self {
        Self {
            scorer,
            context: String::new(),
        }
    }

    /// Update the committed context (called after each commit).
    pub fn update_context(&mut self, committed_text: &str) {
        self.context.push_str(committed_text);
        // Keep last 200 chars for context window
        if self.context.len() > 200 {
            let start = self.context.len() - 200;
            // Find a char boundary
            let start = self.context.ceil_char_boundary(start);
            self.context = self.context[start..].to_string();
        }
    }

    /// Score a candidate sentence for contextual naturalness.
    pub fn score_candidate(&self, candidate: &str) -> f64 {
        self.scorer.score(&self.context, candidate)
    }

    /// Score a candidate with explicit context (for background reranking).
    pub fn score_with_context(&self, context: &str, candidate: &str) -> f64 {
        self.scorer.score(context, candidate)
    }

    /// Rerank candidates by contextual naturalness.
    /// Returns candidates sorted by score (highest first).
    pub fn rerank(&self, candidates: &[String]) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = candidates
            .iter()
            .map(|c| {
                let score = self.scorer.score(&self.context, c);
                (c.clone(), score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if !scored.is_empty() {
            log::debug!(
                "LLM rerank: context='{}' top='{}' ({:.3}) from {} candidates",
                self.context,
                scored[0].0,
                scored[0].1,
                scored.len(),
            );
        }
        scored
    }

    /// Pre-warm KV cache with confirmed context (called during typing).
    pub fn warm_cache(&self) {
        self.scorer.warm_cache(&self.context);
    }

    /// Get current context.
    pub fn context(&self) -> &str {
        &self.context
    }
}

/// Mock scorer using heuristics for testing without a real LLM.
///
/// Scoring heuristics:
/// - Prefers kanji over pure hiragana (kanji = more specific conversion)
/// - Penalizes very short candidates (likely fragmented)
/// - Gives slight bonus for common patterns
///
/// This will be replaced by real LLM perplexity scoring.
pub struct MockScorer;

impl LlmScorer for MockScorer {
    fn score(&self, _context: &str, candidate: &str) -> f64 {
        let chars: Vec<char> = candidate.chars().collect();
        if chars.is_empty() {
            return 0.0;
        }

        let mut score: f64 = 0.5; // baseline

        // Kanji ratio bonus: more kanji = more specific = better
        let kanji_count = chars.iter().filter(|c| is_kanji(**c)).count();
        let kanji_ratio = kanji_count as f64 / chars.len() as f64;
        score += kanji_ratio * 0.3;

        // Length bonus: prefer reasonable length (2-10 chars)
        let len = chars.len();
        if len >= 2 && len <= 10 {
            score += 0.1;
        } else if len == 1 {
            score -= 0.1;
        }

        // Hiragana-only penalty for multi-char (likely unconverted)
        let all_hiragana = chars.iter().all(|c| is_hiragana(*c));
        if all_hiragana && len > 2 {
            score -= 0.15;
        }

        // Particle-only: very low score (not a useful conversion)
        if len == 1 && is_hiragana(chars[0]) {
            score -= 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    fn warm_cache(&self, _context: &str) {
        // No-op for mock
    }
}

fn is_kanji(c: char) -> bool {
    // CJK Unified Ideographs
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
        || ('\u{3400}'..='\u{4DBF}').contains(&c) // CJK Extension A
}

fn is_hiragana(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test engine with MockScorer (deterministic, no server dependency).
    fn mock_engine() -> LlmEngine {
        LlmEngine::with_scorer(Box::new(MockScorer))
    }

    #[test]
    fn mock_scorer_prefers_kanji() {
        let engine = mock_engine();
        let kanji_score = engine.score_candidate("今日");
        let hira_score = engine.score_candidate("きょう");
        assert!(kanji_score > hira_score);
    }

    #[test]
    fn mock_scorer_penalizes_single_char() {
        let engine = mock_engine();
        let long_score = engine.score_candidate("天気");
        let short_score = engine.score_candidate("て");
        assert!(long_score > short_score);
    }

    #[test]
    fn rerank_orders_by_score() {
        let engine = mock_engine();
        let candidates = vec![
            "きょう".to_string(),   // all hiragana
            "今日".to_string(),     // kanji
            "京".to_string(),       // single kanji
        ];
        let ranked = engine.rerank(&candidates);
        assert_eq!(ranked[0].0, "今日"); // kanji + good length should win
    }

    #[test]
    fn rerank_empty() {
        let engine = mock_engine();
        let ranked = engine.rerank(&[]);
        assert!(ranked.is_empty());
    }

    #[test]
    fn context_update() {
        let mut engine = mock_engine();
        engine.update_context("今日は");
        engine.update_context("天気が");
        assert_eq!(engine.context(), "今日は天気が");
    }

    #[test]
    fn context_truncation() {
        let mut engine = mock_engine();
        // Push a lot of text
        for _ in 0..100 {
            engine.update_context("あいうえおかきくけこ");
        }
        // Context should be truncated to ~200 chars
        assert!(engine.context().len() <= 600); // 200 chars * 3 bytes max
    }

    #[test]
    fn custom_scorer() {
        struct AlwaysHighScorer;
        impl LlmScorer for AlwaysHighScorer {
            fn score(&self, _ctx: &str, _candidate: &str) -> f64 {
                1.0
            }
            fn warm_cache(&self, _ctx: &str) {}
        }

        let engine = LlmEngine::with_scorer(Box::new(AlwaysHighScorer));
        assert_eq!(engine.score_candidate("anything"), 1.0);
    }

    #[test]
    fn homophone_disambiguation() {
        let engine = mock_engine();
        // 雨 vs 飴 — both are valid for あめ
        // Mock should prefer the kanji version equally, but both are kanji
        let ame_rain = engine.score_candidate("雨");
        let ame_candy = engine.score_candidate("飴");
        // Both are single kanji, so similar scores
        assert!((ame_rain - ame_candy).abs() < 0.01);
    }

    #[test]
    fn mixed_kanji_hiragana_scoring() {
        let engine = mock_engine();
        // 食べる (kanji+hiragana) vs たべる (all hiragana)
        let mixed = engine.score_candidate("食べる");
        let hira = engine.score_candidate("たべる");
        assert!(mixed > hira);
    }
}
