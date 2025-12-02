//! LFM2 inference pool for model management

use crate::config::InferenceConfig;
use crate::error::{Error, InferenceError, Result};
use crate::types::ModelSize;

use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature
    pub temperature: f32,
    /// Top-p (nucleus sampling)
    pub top_p: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

/// Result of generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Model used
    pub model_used: ModelSize,
    /// Whether KV cache was hit
    pub cache_hit: bool,
}

/// Pool of LFM2 models with lazy loading
pub struct InferencePool {
    /// Loaded models
    models: DashMap<ModelSize, Arc<MockModel>>,
    /// LRU tracking
    lru: RwLock<Vec<(ModelSize, Instant)>>,
    /// Configuration
    config: InferenceConfig,
}

/// Mock model for testing (in production, use llama.cpp or vLLM)
struct MockModel {
    size: ModelSize,
}

impl InferencePool {
    /// Create a new inference pool
    pub async fn new(config: &InferenceConfig) -> Result<Self> {
        Ok(Self {
            models: DashMap::new(),
            lru: RwLock::new(Vec::new()),
            config: config.clone(),
        })
    }

    /// Generate response from a model
    pub async fn generate(
        &self,
        model_size: ModelSize,
        prompt: &str,
        config: GenerationConfig,
        _session_key: Option<&str>,
    ) -> Result<GenerationResult> {
        // Get or load model
        let _model = self.get_or_load(model_size).await?;

        // Mock generation (in production, call actual LLM)
        let response = self.mock_generate(prompt, &config, model_size);

        Ok(GenerationResult {
            text: response,
            tokens_generated: config.max_tokens / 2, // Mock
            model_used: model_size,
            cache_hit: false,
        })
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthInfo> {
        Ok(HealthInfo {
            latency: 0.0,
            loaded_models: self.models.len(),
            available_memory: 0,
        })
    }

    async fn get_or_load(&self, size: ModelSize) -> Result<Arc<MockModel>> {
        // Check if already loaded
        if let Some(model) = self.models.get(&size) {
            self.update_lru(size);
            return Ok(model.clone());
        }

        // Evict if needed
        while self.models.len() >= self.config.max_loaded_models {
            if let Some((evict_size, _)) = self.get_lru_oldest() {
                self.models.remove(&evict_size);
            }
        }

        // Load model
        let model = Arc::new(MockModel { size });
        self.models.insert(size, model.clone());
        self.update_lru(size);

        Ok(model)
    }

    fn update_lru(&self, size: ModelSize) {
        let mut lru = self.lru.write();
        lru.retain(|(s, _)| *s != size);
        lru.push((size, Instant::now()));
    }

    fn get_lru_oldest(&self) -> Option<(ModelSize, Instant)> {
        let lru = self.lru.read();
        lru.first().cloned()
    }

    fn mock_generate(&self, prompt: &str, config: &GenerationConfig, model_size: ModelSize) -> String {
        // Simple mock response based on prompt
        let model_name = match model_size {
            ModelSize::M350 => "350M",
            ModelSize::M700 => "700M",
            ModelSize::B1_2 => "1.2B",
            ModelSize::B2_6 => "2.6B",
        };

        // Extract question from prompt
        let question = if let Some(q_start) = prompt.find("Question:") {
            let q = &prompt[q_start + 9..];
            if let Some(end) = q.find('\n') {
                q[..end].trim()
            } else {
                q.trim()
            }
        } else {
            "your question"
        };

        format!(
            "Based on the provided context, I can answer {}. \
            [This is a mock response from {} model with temperature {:.1}]",
            question, model_name, config.temperature
        )
    }
}

/// Health information
#[derive(Debug, Clone)]
pub struct HealthInfo {
    /// Check latency in ms
    pub latency: f32,
    /// Number of loaded models
    pub loaded_models: usize,
    /// Available memory in bytes
    pub available_memory: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_inference_pool_creation() {
        let config = InferenceConfig::default();
        let pool = InferencePool::new(&config).await.unwrap();
        assert_eq!(pool.models.len(), 0);
    }

    #[tokio::test]
    async fn test_generate() {
        let config = InferenceConfig::default();
        let pool = InferencePool::new(&config).await.unwrap();

        let result = pool.generate(
            ModelSize::M700,
            "Question: What is Rust?\n\nAnswer:",
            GenerationConfig::default(),
            None,
        ).await.unwrap();

        assert!(!result.text.is_empty());
        assert_eq!(result.model_used, ModelSize::M700);
    }

    #[tokio::test]
    async fn test_model_eviction() {
        let mut config = InferenceConfig::default();
        config.max_loaded_models = 2;
        let pool = InferencePool::new(&config).await.unwrap();

        // Load 3 models
        pool.generate(ModelSize::M350, "test", GenerationConfig::default(), None).await.unwrap();
        pool.generate(ModelSize::M700, "test", GenerationConfig::default(), None).await.unwrap();
        pool.generate(ModelSize::B1_2, "test", GenerationConfig::default(), None).await.unwrap();

        // Should only have 2 models loaded
        assert!(pool.models.len() <= 2);
    }
}
