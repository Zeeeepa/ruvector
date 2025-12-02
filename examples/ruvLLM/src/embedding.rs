//! Embedding service for converting text to vectors

use crate::config::EmbeddingConfig;
use crate::error::{Error, Result};

/// Result of embedding a text
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Token count
    pub token_count: usize,
    /// Whether text was truncated
    pub truncated: bool,
}

/// Service for text embedding
pub struct EmbeddingService {
    /// Embedding dimension
    dimension: usize,
    /// Maximum tokens
    max_tokens: usize,
}

impl EmbeddingService {
    /// Create a new embedding service
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        Ok(Self {
            dimension: config.dimension,
            max_tokens: config.max_tokens,
        })
    }

    /// Embed a text string
    pub fn embed(&self, text: &str) -> Result<Embedding> {
        // Simple mock embedding - in production, this would use LFM2 encoder
        let token_count = text.split_whitespace().count();
        let truncated = token_count > self.max_tokens;

        // Generate deterministic embedding based on text hash
        let hash = self.hash_text(text);
        let vector = self.generate_embedding(hash);

        Ok(Embedding {
            vector,
            token_count: token_count.min(self.max_tokens),
            truncated,
        })
    }

    /// Embed multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn hash_text(&self, text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    fn generate_embedding(&self, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut vec: Vec<f32> = (0..self.dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter_mut().for_each(|x| *x /= norm);
        }

        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let config = EmbeddingConfig::default();
        let service = EmbeddingService::new(&config).unwrap();
        let embedding = service.embed("Hello world").unwrap();
        assert_eq!(embedding.vector.len(), config.dimension);
    }

    #[test]
    fn test_embedding_normalized() {
        let config = EmbeddingConfig::default();
        let service = EmbeddingService::new(&config).unwrap();
        let embedding = service.embed("Test text").unwrap();

        let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_same_text_same_embedding() {
        let config = EmbeddingConfig::default();
        let service = EmbeddingService::new(&config).unwrap();

        let e1 = service.embed("Same text").unwrap();
        let e2 = service.embed("Same text").unwrap();

        assert_eq!(e1.vector, e2.vector);
    }
}
