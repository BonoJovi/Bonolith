/// HTTP-based LLM scorer using a local llama-server instance.
///
/// Sends scoring requests to a llama-server process via HTTP.
/// This avoids the segfault issues of in-process llama.cpp usage
/// and isolates the model in a separate process.
///
/// The server should be started separately (e.g., via systemd):
///   llama-server -m ~/.local/share/jaim/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
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
}

#[derive(Deserialize)]
struct CompletionResponse {
    content: String,
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

    /// Connect to the default endpoint, checking JAIM_LLM_ENDPOINT env var.
    pub fn from_default_endpoint() -> Option<Self> {
        let endpoint = std::env::var("JAIM_LLM_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        Self::new(&endpoint)
    }

    /// Score using generation-based approach:
    /// Generate a completion from context, then measure character overlap with candidate.
    fn score_by_generation(&self, context: &str, candidate: &str) -> f64 {
        // Fast-fail: once the server has been observed unreachable
        // (e.g., user ran `jaim llm off` mid-session), skip the HTTP
        // round-trip entirely and return the neutral score so we
        // don't pay the connect-timeout per keystroke.
        if self.warned.load(Ordering::Relaxed) {
            return 0.5;
        }

        let url = format!("{}/completion", self.endpoint);
        let n_predict = (candidate.chars().count() as u32 + 5).min(30);

        let req = CompletionRequest {
            prompt: context.to_string(),
            n_predict,
            temperature: 0.0,
            cache_prompt: true,
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

        let generated = &body.content;
        if generated.is_empty() {
            return 0.5;
        }

        // Score: how well does the candidate match the generated continuation?
        let candidate_chars: Vec<char> = candidate.chars().collect();
        let generated_chars: Vec<char> = generated.chars().collect();

        if candidate_chars.is_empty() {
            return 0.5;
        }

        // Count matching characters from the start
        let match_len = candidate_chars
            .iter()
            .zip(generated_chars.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let match_ratio = match_len as f64 / candidate_chars.len() as f64;

        // Map to 0.3–0.9 range (never fully dominate or suppress)
        let score = 0.3 + match_ratio * 0.6;

        let gen_preview: String = generated.chars().take(20).collect();
        debug!(
            "HttpLlamaScorer: candidate='{}' generated='{}' match={}/{} score={:.3}",
            candidate, gen_preview, match_len, candidate_chars.len(), score,
        );

        score
    }
}

impl LlmScorer for HttpLlamaScorer {
    fn score(&self, context: &str, candidate: &str) -> f64 {
        self.score_by_generation(context, candidate)
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
        };
        if let Err(e) = self.agent.post(&url).send_json(&req) {
            if !self.warned.swap(true, Ordering::Relaxed) {
                warn!("HttpLlamaScorer: warm_cache failed: {}", e);
            }
        }
    }
}
