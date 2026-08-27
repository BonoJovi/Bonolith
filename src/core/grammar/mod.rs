/// Grammar validation engine (derived from Promps-Ent)
///
/// Scores conversion candidates by grammatical correctness.
/// Applies 9 Japanese grammar rules adapted from Promps-Ent's validation engine
/// to filter and rank IME conversion candidates.
///
/// Rules:
/// 1. ParticleWithoutNoun — particle must follow noun
/// 2. ConsecutiveParticles — no consecutive particles
/// 3. ConsecutiveNouns — consecutive nouns without particle (warning)
/// 4. VerbNotAtEnd — verb should be at end (SOV order)
/// 5. MissingSubject — no が marker with verb
/// 6. MissingObject — no を marker with verb
/// 7. ToutenAfterWo — 、cannot follow を
/// 8. ToutenNotAfterParticle — 、only after particles
/// 9. KutenNotAfterVerb — 。only after verbs

use super::dictionary::PartOfSpeech;

/// Grammar score for a conversion candidate
#[derive(Debug, Clone)]
pub struct GrammarScore {
    /// Overall score (0.0 = invalid, 1.0 = perfect)
    pub score: f64,
    /// Detected issues
    pub issues: Vec<GrammarIssue>,
}

#[derive(Debug, Clone)]
pub struct GrammarIssue {
    pub rule: GrammarRule,
    pub severity: Severity,
    pub position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarRule {
    /// Particle without preceding noun
    ParticleWithoutNoun,
    /// Consecutive particles
    ConsecutiveParticles,
    /// Consecutive nouns without particle
    ConsecutiveNouns,
    /// Verb not at end of sentence
    VerbNotAtEnd,
    /// Missing subject marker (が)
    MissingSubject,
    /// Missing object marker (を)
    MissingObject,
    /// Touten (、) after を
    ToutenAfterWo,
    /// Touten (、) not after particle
    ToutenNotAfterParticle,
    /// Kuten (。) not after verb
    KutenNotAfterVerb,
}

/// Token for grammar analysis — a word with its part of speech
#[derive(Debug, Clone)]
pub struct GrammarToken {
    pub surface: String,
    pub pos: PartOfSpeech,
}

pub struct GrammarEngine {}

impl GrammarEngine {
    pub fn new() -> Self {
        Self {}
    }

    /// Score a sequence of tokens for grammatical correctness.
    /// Returns a score (0.0–1.0) with detected issues.
    pub fn score(&self, tokens: &[GrammarToken]) -> GrammarScore {
        if tokens.is_empty() {
            return GrammarScore {
                score: 1.0,
                issues: Vec::new(),
            };
        }

        let mut issues = Vec::new();

        self.check_particle_without_noun(tokens, &mut issues);
        self.check_consecutive_particles(tokens, &mut issues);
        self.check_consecutive_nouns(tokens, &mut issues);
        self.check_verb_not_at_end(tokens, &mut issues);
        self.check_missing_subject(tokens, &mut issues);
        self.check_missing_object(tokens, &mut issues);
        self.check_touten_after_wo(tokens, &mut issues);
        self.check_touten_not_after_particle(tokens, &mut issues);
        self.check_kuten_not_after_verb(tokens, &mut issues);

        let score = calculate_score(&issues);

        GrammarScore { score, issues }
    }

    /// Rule 1: Particle must follow a noun (or pronoun/adverb)
    fn check_particle_without_noun(&self, tokens: &[GrammarToken], issues: &mut Vec<GrammarIssue>) {
        for (i, token) in tokens.iter().enumerate() {
            if token.pos == PartOfSpeech::Particle && i == 0 {
                issues.push(GrammarIssue {
                    rule: GrammarRule::ParticleWithoutNoun,
                    severity: Severity::Error,
                    position: i,
                });
            } else if token.pos == PartOfSpeech::Particle && i > 0 {
                let prev = &tokens[i - 1];
                if !can_precede_particle(prev.pos) {
                    issues.push(GrammarIssue {
                        rule: GrammarRule::ParticleWithoutNoun,
                        severity: Severity::Error,
                        position: i,
                    });
                }
            }
        }
    }

    /// Rule 2: No consecutive particles
    fn check_consecutive_particles(
        &self,
        tokens: &[GrammarToken],
        issues: &mut Vec<GrammarIssue>,
    ) {
        for i in 1..tokens.len() {
            if tokens[i].pos == PartOfSpeech::Particle
                && tokens[i - 1].pos == PartOfSpeech::Particle
            {
                issues.push(GrammarIssue {
                    rule: GrammarRule::ConsecutiveParticles,
                    severity: Severity::Error,
                    position: i,
                });
            }
        }
    }

    /// Rule 3: Consecutive nouns without particle (warning)
    fn check_consecutive_nouns(&self, tokens: &[GrammarToken], issues: &mut Vec<GrammarIssue>) {
        for i in 1..tokens.len() {
            if tokens[i].pos == PartOfSpeech::Noun && tokens[i - 1].pos == PartOfSpeech::Noun {
                issues.push(GrammarIssue {
                    rule: GrammarRule::ConsecutiveNouns,
                    severity: Severity::Warning,
                    position: i,
                });
            }
        }
    }

    /// Rule 4: Verb should be at the end (SOV order)
    fn check_verb_not_at_end(&self, tokens: &[GrammarToken], issues: &mut Vec<GrammarIssue>) {
        for (i, token) in tokens.iter().enumerate() {
            if token.pos == PartOfSpeech::Verb && i < tokens.len() - 1 {
                // Allow verb followed by auxiliary or punctuation
                let next = &tokens[i + 1];
                if next.pos != PartOfSpeech::Auxiliary
                    && !is_punctuation_surface(&next.surface)
                {
                    issues.push(GrammarIssue {
                        rule: GrammarRule::VerbNotAtEnd,
                        severity: Severity::Warning,
                        position: i,
                    });
                }
            }
        }
    }

    /// Rule 5: Missing subject marker (が) when verb is present
    fn check_missing_subject(&self, tokens: &[GrammarToken], issues: &mut Vec<GrammarIssue>) {
        let has_verb = tokens.iter().any(|t| t.pos == PartOfSpeech::Verb);
        if !has_verb {
            return;
        }
        let has_subject_marker = tokens
            .iter()
            .any(|t| t.pos == PartOfSpeech::Particle && is_subject_particle(&t.surface));
        if !has_subject_marker {
            // Only warn if the sentence has enough tokens to expect a subject
            if tokens.len() >= 3 {
                issues.push(GrammarIssue {
                    rule: GrammarRule::MissingSubject,
                    severity: Severity::Warning,
                    position: 0,
                });
            }
        }
    }

    /// Rule 6: Missing object marker (を) when verb is present
    fn check_missing_object(&self, tokens: &[GrammarToken], issues: &mut Vec<GrammarIssue>) {
        let has_verb = tokens.iter().any(|t| t.pos == PartOfSpeech::Verb);
        if !has_verb {
            return;
        }
        let has_object_marker = tokens
            .iter()
            .any(|t| t.pos == PartOfSpeech::Particle && t.surface == "を");
        if !has_object_marker {
            if tokens.len() >= 4 {
                issues.push(GrammarIssue {
                    rule: GrammarRule::MissingObject,
                    severity: Severity::Warning,
                    position: 0,
                });
            }
        }
    }

    /// Rule 7: Touten (、) cannot appear after を
    fn check_touten_after_wo(&self, tokens: &[GrammarToken], issues: &mut Vec<GrammarIssue>) {
        for i in 1..tokens.len() {
            if is_touten(&tokens[i].surface)
                && tokens[i - 1].pos == PartOfSpeech::Particle
                && tokens[i - 1].surface == "を"
            {
                issues.push(GrammarIssue {
                    rule: GrammarRule::ToutenAfterWo,
                    severity: Severity::Error,
                    position: i,
                });
            }
        }
    }

    /// Rule 8: Touten (、) should only appear after particles
    fn check_touten_not_after_particle(
        &self,
        tokens: &[GrammarToken],
        issues: &mut Vec<GrammarIssue>,
    ) {
        for i in 1..tokens.len() {
            if is_touten(&tokens[i].surface) && tokens[i - 1].pos != PartOfSpeech::Particle {
                // Allow after: nouns (listing/vocative), adverbs, conjunctions, interjections
                if !can_precede_touten(tokens[i - 1].pos) {
                    issues.push(GrammarIssue {
                        rule: GrammarRule::ToutenNotAfterParticle,
                        severity: Severity::Error,
                        position: i,
                    });
                }
            }
        }
    }

    /// Rule 9: Kuten (。) should only appear after verbs/auxiliaries
    fn check_kuten_not_after_verb(
        &self,
        tokens: &[GrammarToken],
        issues: &mut Vec<GrammarIssue>,
    ) {
        for i in 1..tokens.len() {
            if is_kuten(&tokens[i].surface) {
                let prev = &tokens[i - 1];
                if prev.pos != PartOfSpeech::Verb && prev.pos != PartOfSpeech::Auxiliary {
                    issues.push(GrammarIssue {
                        rule: GrammarRule::KutenNotAfterVerb,
                        severity: Severity::Error,
                        position: i,
                    });
                }
            }
        }
    }
}

/// Calculate overall score from issues.
/// Errors reduce score heavily, warnings reduce slightly.
fn calculate_score(issues: &[GrammarIssue]) -> f64 {
    let mut score: f64 = 1.0;
    for issue in issues {
        match issue.severity {
            Severity::Error => score -= 0.2,
            Severity::Warning => score -= 0.05,
        }
    }
    score.max(0.0)
}

/// Can this POS precede a particle?
///
/// Verbs and adjectives take particles all the time in real Japanese
/// (見たが / 楽しくて / 行きたいと …). Omitting them penalized correct
/// sentences in the legacy `convert()` scoring path.
fn can_precede_particle(pos: PartOfSpeech) -> bool {
    matches!(
        pos,
        PartOfSpeech::Noun
            | PartOfSpeech::Adverb
            | PartOfSpeech::Interjection
            | PartOfSpeech::Verb
            | PartOfSpeech::Adjective
    )
}

fn is_subject_particle(surface: &str) -> bool {
    matches!(surface, "が" | "は")
}

fn is_touten(surface: &str) -> bool {
    surface == "、" || surface == ","
}

fn is_kuten(surface: &str) -> bool {
    surface == "。" || surface == "."
}

/// Can this POS precede a touten (、)?
/// Allows: nouns (listing/vocative), adverbs, conjunctions, interjections
fn can_precede_touten(pos: PartOfSpeech) -> bool {
    matches!(
        pos,
        PartOfSpeech::Noun
            | PartOfSpeech::Adverb
            | PartOfSpeech::Conjunction
            | PartOfSpeech::Interjection
            | PartOfSpeech::Adjective
    )
}

fn is_punctuation_surface(surface: &str) -> bool {
    is_touten(surface) || is_kuten(surface) || surface == "！" || surface == "？"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(surface: &str, pos: PartOfSpeech) -> GrammarToken {
        GrammarToken {
            surface: surface.to_string(),
            pos,
        }
    }

    fn n(s: &str) -> GrammarToken {
        token(s, PartOfSpeech::Noun)
    }
    fn p(s: &str) -> GrammarToken {
        token(s, PartOfSpeech::Particle)
    }
    fn v(s: &str) -> GrammarToken {
        token(s, PartOfSpeech::Verb)
    }
    fn aux(s: &str) -> GrammarToken {
        token(s, PartOfSpeech::Auxiliary)
    }

    #[test]
    fn perfect_sentence() {
        // 私が本を読む (I read a book)
        let engine = GrammarEngine::new();
        let tokens = vec![n("私"), p("が"), n("本"), p("を"), v("読む")];
        let result = engine.score(&tokens);
        assert_eq!(result.score, 1.0);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn particle_without_noun() {
        // が本を読む (particle at start)
        let engine = GrammarEngine::new();
        let tokens = vec![p("が"), n("本"), p("を"), v("読む")];
        let result = engine.score(&tokens);
        assert!(result.score < 1.0);
        assert!(result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::ParticleWithoutNoun));
    }

    #[test]
    fn consecutive_particles() {
        // 私がを読む
        let engine = GrammarEngine::new();
        let tokens = vec![n("私"), p("が"), p("を"), v("読む")];
        let result = engine.score(&tokens);
        assert!(result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::ConsecutiveParticles));
    }

    #[test]
    fn consecutive_nouns_warning() {
        // 私本を読む (missing particle between nouns)
        let engine = GrammarEngine::new();
        let tokens = vec![n("私"), n("本"), p("を"), v("読む")];
        let result = engine.score(&tokens);
        let noun_issue = result
            .issues
            .iter()
            .find(|i| i.rule == GrammarRule::ConsecutiveNouns);
        assert!(noun_issue.is_some());
        assert_eq!(noun_issue.unwrap().severity, Severity::Warning);
    }

    #[test]
    fn verb_not_at_end() {
        // 読む私が本を (verb at start)
        let engine = GrammarEngine::new();
        let tokens = vec![v("読む"), n("私"), p("が"), n("本"), p("を")];
        let result = engine.score(&tokens);
        assert!(result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::VerbNotAtEnd));
    }

    #[test]
    fn verb_before_auxiliary_ok() {
        // 食べます (verb + auxiliary is fine)
        let engine = GrammarEngine::new();
        let tokens = vec![n("ご飯"), p("を"), v("食べる"), aux("ます")];
        let result = engine.score(&tokens);
        assert!(!result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::VerbNotAtEnd));
    }

    #[test]
    fn missing_subject_warning() {
        // 本を読む (no subject marker, 3 tokens with verb)
        let engine = GrammarEngine::new();
        let tokens = vec![n("本"), p("を"), v("読む")];
        let result = engine.score(&tokens);
        assert!(result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::MissingSubject));
    }

    #[test]
    fn touten_after_wo_error() {
        // 本を、読む
        let engine = GrammarEngine::new();
        let tokens = vec![
            n("本"),
            p("を"),
            token("、", PartOfSpeech::Other),
            v("読む"),
        ];
        let result = engine.score(&tokens);
        assert!(result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::ToutenAfterWo));
    }

    #[test]
    fn kuten_after_verb_ok() {
        // 読む。
        let engine = GrammarEngine::new();
        let tokens = vec![v("読む"), token("。", PartOfSpeech::Other)];
        let result = engine.score(&tokens);
        assert!(!result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::KutenNotAfterVerb));
    }

    #[test]
    fn kuten_after_noun_error() {
        // 本。
        let engine = GrammarEngine::new();
        let tokens = vec![n("本"), token("。", PartOfSpeech::Other)];
        let result = engine.score(&tokens);
        assert!(result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::KutenNotAfterVerb));
    }

    #[test]
    fn empty_tokens() {
        let engine = GrammarEngine::new();
        let result = engine.score(&[]);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn score_degrades_with_errors() {
        let engine = GrammarEngine::new();
        // Multiple errors should reduce score more
        let tokens = vec![p("が"), p("を"), v("読む")];
        let result = engine.score(&tokens);
        assert!(result.score < 0.8);
    }

    #[test]
    fn touten_after_noun_vocative_ok() {
        // 田中さん、大丈夫ですか (vocative + touten)
        let engine = GrammarEngine::new();
        let tokens = vec![
            n("田中さん"),
            token("、", PartOfSpeech::Other),
            token("大丈夫", PartOfSpeech::Adjective),
            aux("です"),
            p("か"),
        ];
        let result = engine.score(&tokens);
        assert!(!result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::ToutenNotAfterParticle));
    }

    #[test]
    fn touten_after_conjunction_ok() {
        // しかし、それは違う
        let engine = GrammarEngine::new();
        let tokens = vec![
            token("しかし", PartOfSpeech::Conjunction),
            token("、", PartOfSpeech::Other),
            n("それ"),
            p("は"),
            v("違う"),
        ];
        let result = engine.score(&tokens);
        assert!(!result
            .issues
            .iter()
            .any(|i| i.rule == GrammarRule::ToutenNotAfterParticle));
    }

    #[test]
    fn complex_valid_sentence() {
        // 今日は天気がいいです (Today the weather is nice)
        let engine = GrammarEngine::new();
        let tokens = vec![
            n("今日"),
            p("は"),
            n("天気"),
            p("が"),
            token("いい", PartOfSpeech::Adjective),
            aux("です"),
        ];
        let result = engine.score(&tokens);
        // Should have high score (only possible warning: missing を with no verb)
        assert!(result.score >= 0.9);
    }
}
