/// HTTP-based LLM scorer using a local llama-server instance.
///
/// Sends scoring requests to a llama-server process via HTTP.
/// This avoids the segfault issues of in-process llama.cpp usage
/// and isolates the model in a separate process.
///
/// The server should be started separately (e.g., via systemd):
///   llama-server -m ~/.local/share/bonolith/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
///     --host 127.0.0.1 --port 8080 --ctx-size 512

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

use super::LlmScorer;

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8080";
const SCORE_TIMEOUT: Duration = Duration::from_millis(500);
const HEALTH_TIMEOUT: Duration = Duration::from_millis(200);

/// LLM scorer that communicates with a local llama-server via HTTP.
pub struct HttpLlamaScorer {
    endpoint: String,
    agent: ureq::Agent,
    /// Suppress repeated warnings after first failure
    warned: AtomicBool,
}

#[derive(Serialize)]
struct CompletionRequest {
    prompt: String,
    n_predict: u32,
    temperature: f64,
    cache_prompt: bool,
    /// GBNF grammar. When set to a single-literal rule, it teacher-forces the
    /// model to emit exactly that string so we can read the per-token logprob
    /// of the candidate surface. Empty string = unconstrained (warm-up only).
    #[serde(skip_serializing_if = "String::is_empty")]
    grammar: String,
    /// Number of top token probabilities to return. Must be >= 1 for the
    /// server to populate `completion_probabilities`.
    n_probs: u32,
}

#[derive(Deserialize)]
struct CompletionResponse {
    /// Per-emitted-token info. With a grammar forcing the candidate string,
    /// each entry's `logprob` is the model's log P(token | context + prefix).
    #[serde(default)]
    completion_probabilities: Vec<TokenLogprob>,
}

#[derive(Deserialize)]
struct TokenLogprob {
    /// The emitted token's text. The grammar appends a trailing end-of-text
    /// token after the candidate literal, which comes back with an empty
    /// string here; its logprob is noise (probability of *stopping* after the
    /// surface) and must be excluded from the candidate's score.
    token: String,
    logprob: f64,
}

/// Escape a string into a GBNF double-quoted literal body.
fn gbnf_string_literal(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("root ::= \"{}\"", escaped)
}

/// Map a mean per-token log-probability (content tokens only) to the 0.3–0.9
/// scoring band used by the reranker. With the end-of-text token excluded, a
/// contextually strong surface scores around -1..-3; the band's lower end is
/// set to -13 because the default 1.5B model's logprobs run more negative than
/// the 0.5B's, and clamping correct-but-low candidates (~-11) to the floor
/// would collapse them into ties.
fn logprob_to_score(avg_logprob: f64) -> f64 {
    const LO: f64 = -13.0; // → 0.3
    const HI: f64 = -2.0; // → 0.9
    let t = ((avg_logprob - LO) / (HI - LO)).clamp(0.0, 1.0);
    0.3 + t * 0.6
}

impl HttpLlamaScorer {
    /// Connect to a llama-server at the given endpoint.
    /// Returns None if the server is not reachable.
    pub fn new(endpoint: &str) -> Option<Self> {
        let health_agent = ureq::Agent::config_builder()
            .timeout_connect(Some(HEALTH_TIMEOUT))
            .timeout_recv_body(Some(HEALTH_TIMEOUT))
            .build()
            .new_agent();

        // Health check
        let url = format!("{}/health", endpoint);
        match health_agent.get(&url).call() {
            Ok(resp) => {
                if resp.status() == 200 {
                    info!("HttpLlamaScorer: connected to {}", endpoint);
                } else {
                    warn!(
                        "HttpLlamaScorer: server at {} returned status {}",
                        endpoint,
                        resp.status()
                    );
                    return None;
                }
            }
            Err(e) => {
                info!("HttpLlamaScorer: server not available at {}: {}", endpoint, e);
                return None;
            }
        }

        // Use longer timeouts for actual scoring requests
        let agent = ureq::Agent::config_builder()
            .timeout_connect(Some(HEALTH_TIMEOUT))
            .timeout_recv_body(Some(SCORE_TIMEOUT))
            .build()
            .new_agent();

        Some(Self {
            endpoint: endpoint.to_string(),
            agent,
            warned: AtomicBool::new(false),
        })
    }

    /// Connect to the default endpoint, checking BONOLITH_LLM_ENDPOINT env var.
    pub fn from_default_endpoint() -> Option<Self> {
        let endpoint = std::env::var("BONOLITH_LLM_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        Self::new(&endpoint)
    }

    /// Score a candidate by its contextual log-probability.
    ///
    /// Sends the context as the prompt and a GBNF grammar that forces the model
    /// to emit exactly `candidate`, then reads back the per-token logprobs the
    /// server reports. Their mean is log P(candidate | context) per token — a
    /// genuine semantic signal that can disambiguate homophones (箸/橋/端,
    /// 機械/機会) the dictionary + connection-cost layer cannot.
    fn score_by_logprob(&self, context: &str, candidate: &str) -> f64 {
        // Fast-fail: once the server has been observed unreachable
        // (e.g., user ran `bonolith llm off` mid-session), skip the HTTP
        // round-trip entirely and return the neutral score so we
        // don't pay the connect-timeout per keystroke.
        if self.warned.load(Ordering::Relaxed) {
            return 0.5;
        }
        if candidate.is_empty() {
            return 0.5;
        }

        let url = format!("{}/completion", self.endpoint);
        // The grammar stops generation once the literal is complete; this is
        // just an upper bound. Kanji can take >1 token each, so budget room.
        let n_predict = (candidate.chars().count() as u32 * 3 + 4).min(48);

        let req = CompletionRequest {
            prompt: context.to_string(),
            n_predict,
            temperature: 0.0,
            cache_prompt: true,
            grammar: gbnf_string_literal(candidate),
            n_probs: 1,
        };

        let resp = match self.agent.post(&url).send_json(&req) {
            Ok(r) => r,
            Err(e) => {
                if !self.warned.swap(true, Ordering::Relaxed) {
                    warn!("HttpLlamaScorer: completion request failed: {}", e);
                } else {
                    debug!("HttpLlamaScorer: completion request failed: {}", e);
                }
                return 0.5;
            }
        };

        let body: CompletionResponse = match resp.into_body().read_json() {
            Ok(b) => b,
            Err(e) => {
                debug!("HttpLlamaScorer: failed to parse response: {}", e);
                return 0.5;
            }
        };

        // Sum only the candidate's own (non-empty) tokens. The trailing
        // end-of-text token the grammar emits comes back with an empty string
        // and a high-variance logprob unrelated to which reading is correct;
        // including it corrupts the ranking (e.g. 橋 losing to 端).
        let content: Vec<f64> = body
            .completion_probabilities
            .iter()
            .filter(|p| !p.token.is_empty())
            .map(|p| p.logprob)
            .collect();
        if content.is_empty() {
            // Server didn't return per-token probs (n_probs unsupported), or
            // only the end token came back; stay neutral so we neither help
            // nor hurt the ranking.
            return 0.5;
        }

        let avg = content.iter().sum::<f64>() / content.len() as f64;
        let score = logprob_to_score(avg);

        debug!(
            "HttpLlamaScorer: candidate='{}' ntok={} avg_logprob={:.3} score={:.3}",
            candidate,
            content.len(),
            avg,
            score,
        );

        score
    }
}

impl LlmScorer for HttpLlamaScorer {
    fn score(&self, context: &str, candidate: &str) -> f64 {
        self.score_by_logprob(context, candidate)
    }

    fn warm_cache(&self, context: &str) {
        if context.is_empty() || self.warned.load(Ordering::Relaxed) {
            return;
        }
        // Send a no-generation request to warm the server's KV cache
        let url = format!("{}/completion", self.endpoint);
        let req = CompletionRequest {
            prompt: context.to_string(),
            n_predict: 0,
            temperature: 0.0,
            cache_prompt: true,
            grammar: String::new(),
            n_probs: 0,
        };
        if let Err(e) = self.agent.post(&url).send_json(&req) {
            if !self.warned.swap(true, Ordering::Relaxed) {
                warn!("HttpLlamaScorer: warm_cache failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbnf_literal_wraps_and_escapes() {
        assert_eq!(gbnf_string_literal("箸"), "root ::= \"箸\"");
        // Backslashes and quotes must be escaped so the grammar stays valid.
        assert_eq!(gbnf_string_literal("a\"b\\c"), "root ::= \"a\\\"b\\\\c\"");
    }

    #[test]
    fn logprob_score_is_monotonic_and_banded() {
        // Higher (less negative) log-prob → higher score.
        assert!(logprob_to_score(-4.0) > logprob_to_score(-8.0));
        assert!(logprob_to_score(-8.0) > logprob_to_score(-12.0));
        // Clamped to the 0.3–0.9 band at the extremes.
        assert!((logprob_to_score(0.0) - 0.9).abs() < 1e-9);
        assert!((logprob_to_score(-100.0) - 0.3).abs() < 1e-9);
    }

    /// Live homophone discrimination against a running llama-server.
    /// Ignored by default (needs the server); run with:
    ///   cargo test --lib http_scorer -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_homophone_discrimination() {
        let scorer = match HttpLlamaScorer::from_default_endpoint() {
            Some(s) => s,
            None => {
                eprintln!("no llama-server reachable; skipping");
                return;
            }
        };
        // (context, correct surface, competing surface)
        let cases = [
            ("ご飯を食べるための", "箸", "橋"),
            ("川にかかった", "橋", "箸"),
            ("工場の", "機械", "機会"),
            ("またとない", "機会", "機械"),
            ("労働組合と会社が", "交渉", "高尚"),
        ];
        let mut correct = 0;
        for (ctx, good, bad) in cases {
            let sg = scorer.score(ctx, good);
            let sb = scorer.score(ctx, bad);
            let ok = sg > sb;
            correct += ok as usize;
            println!(
                "ctx={ctx:?} {good}={sg:.3} {bad}={sb:.3} -> {}",
                if ok { "OK" } else { "MISS" }
            );
        }
        // The 0.5B model isn't perfect, but it should clear a clear majority.
        assert!(correct >= 4, "only {correct}/5 homophone cases correct");
    }
}
