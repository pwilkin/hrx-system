//! Deterministic benchmark workloads.
//!
//! A benchmark is only comparable to another run of the same harness, so the
//! workloads are generated: a seeded word stream of ordinary English, in document
//! sizes a card's prefill rows name. The bytes match the Python launcher's
//! `loom-launch/text-v1` stream exactly. The prompt size is never estimated; each
//! request reports the prompt tokens the server counted.

pub const HEADER: &str = "Write a long, detailed essay discussing this documentation in depth, covering its purpose, its structure and its consequences.";

/// Ordinary words, so the tokenizer sees natural text rather than one repeated
/// token and the reported token count stays close to the intended size.
pub const WORDS: &str = "the of and to in a is that for it as was with be by on not he this are or his from at which but have an they you were her she there one all we can her would make like him into time has look two more write go see number no way could people my than first water been call who oil its now find long down day did get come made may part over new sound take only little work know place year live me back give most very after thing our just name good sentence man think say great where help through much before line right too mean old any same tell boy follow came want show also around form three small set put end does another well large must big even such because turn here why ask went men read need land different home us move try kind hand picture again change off play spell air away animal house point page letter mother answer found study still learn should world high every near add food between own below country plant last school father keep tree never start city earth eye light thought head under story saw left few while along might close something seem next hard open example begin life always those both paper together got group often run important until children side feet car mile night walk white sea began grow took river four carry state once book hear stop without second later miss idea enough eat face watch far really almost let above girl sometimes mountain cut young talk soon list song being leave family it under young";

/// The document identity stored with every record, so a comparison is honest.
pub const WORKLOAD_IDENTITY: &str = "loom-launch/text-v1";

/// Generates `count` words from a seeded xorshift stream, deterministically.
pub fn words_for(count: usize, seed: u32) -> String {
    if count == 0 {
        return String::new();
    }
    let mut state: u32 = if seed == 0 { 1 } else { seed };
    let vocabulary: Vec<&str> = WORDS.split_whitespace().collect();
    let size = vocabulary.len() as u32;
    let mut produced: Vec<String> = Vec::with_capacity(count);
    for index in 0..count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        produced.push(vocabulary[(state % size) as usize].to_string());
        if index != 0 && index % 12 == 0 {
            produced.last_mut().unwrap().push('.');
        }
    }
    produced.join(" ")
}

pub const SEED: u32 = 0x5EED;

/// A document sized in tokens; English runs about 0.75 words per token.
pub fn document(approximate_tokens: usize) -> String {
    if approximate_tokens == 0 {
        return String::new();
    }
    words_for((approximate_tokens as f64 * 0.75) as usize, SEED)
}

/// A document followed by a short instruction, for a prefill measurement.
pub fn prefill_prompt(approximate_tokens: usize) -> String {
    format!(
        "{HEADER}\n\n{}\n\nReply with one short sentence.",
        document(approximate_tokens)
    )
}

/// A decode request, optionally behind a document of the given size.
pub fn decode_prompt(approximate_tokens: usize) -> String {
    if approximate_tokens == 0 {
        return HEADER.to_string();
    }
    format!("{HEADER}\n\n{}", document(approximate_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_word_stream_matches_the_python_launcher() {
        assert_eq!(words_for(30, SEED), "life far our page good group you second tree his country my like. down always line on earth its far own than three page not. follow man above read its");
        let prompt = prefill_prompt(1000);
        assert_eq!(prompt.len(), 4177);
        assert!(prompt.ends_with("t at good few never two that down carry few give second. think start start this children\n\nReply with one short sentence."));
        assert_eq!(decode_prompt(2000).len(), 8081);
        assert_eq!(document(32000).len(), 127252);
        assert_eq!(decode_prompt(0), HEADER);
    }
}
