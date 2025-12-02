//! Graph attention engine for context extraction

use crate::config::EmbeddingConfig;
use crate::error::{Error, Result};
use crate::memory::SubGraph;
use crate::types::MemoryNode;

/// Graph context after attention
#[derive(Debug, Clone)]
pub struct GraphContext {
    /// Output embedding
    pub embedding: Vec<f32>,
    /// Nodes ranked by attention
    pub ranked_nodes: Vec<MemoryNode>,
    /// Attention weights
    pub attention_weights: Vec<f32>,
    /// Summary statistics
    pub summary: GraphSummary,
}

/// Summary of graph attention
#[derive(Debug, Clone, Default)]
pub struct GraphSummary {
    /// Number of nodes attended
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Attention entropy
    pub attention_entropy: f32,
    /// Mean attention weight
    pub mean_attention: f32,
}

/// Graph attention engine
pub struct GraphAttentionEngine {
    /// Embedding dimension
    dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl GraphAttentionEngine {
    /// Create a new graph attention engine
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        let num_heads = 4;
        let head_dim = config.dimension / num_heads;

        Ok(Self {
            dim: config.dimension,
            num_heads,
            head_dim,
        })
    }

    /// Attend over subgraph
    pub fn attend(&self, query: &[f32], subgraph: &SubGraph) -> Result<GraphContext> {
        if subgraph.nodes.is_empty() {
            return Ok(GraphContext {
                embedding: query.to_vec(),
                ranked_nodes: vec![],
                attention_weights: vec![],
                summary: GraphSummary::default(),
            });
        }

        // Compute attention scores
        let mut scores: Vec<f32> = subgraph.nodes
            .iter()
            .map(|node| {
                // Scaled dot-product attention
                let score = dot_product(query, &node.vector) / (self.dim as f32).sqrt();
                score
            })
            .collect();

        // Apply edge-based modulation
        for edge in &subgraph.edges {
            // Find source node index
            if let Some(src_idx) = subgraph.nodes.iter().position(|n| n.id == edge.src) {
                // Boost based on edge weight and type
                let boost = edge.weight * 0.1;
                scores[src_idx] += boost;
            }
        }

        // Softmax
        let attention_weights = softmax(&scores);

        // Weighted sum of node embeddings
        let embedding = weighted_sum(&subgraph.nodes, &attention_weights, self.dim);

        // Rank nodes by attention
        let mut indexed: Vec<(usize, f32)> = attention_weights.iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ranked_nodes: Vec<MemoryNode> = indexed.iter()
            .map(|(i, _)| subgraph.nodes[*i].clone())
            .collect();
        let ranked_weights: Vec<f32> = indexed.iter()
            .map(|(_, w)| *w)
            .collect();

        // Compute summary
        let summary = GraphSummary {
            num_nodes: subgraph.nodes.len(),
            num_edges: subgraph.edges.len(),
            attention_entropy: entropy(&attention_weights),
            mean_attention: attention_weights.iter().sum::<f32>() / attention_weights.len() as f32,
        };

        Ok(GraphContext {
            embedding,
            ranked_nodes,
            attention_weights: ranked_weights,
            summary,
        })
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

fn weighted_sum(nodes: &[MemoryNode], weights: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];

    for (node, &weight) in nodes.iter().zip(weights.iter()) {
        for (i, &v) in node.vector.iter().take(dim).enumerate() {
            result[i] += v * weight;
        }
    }

    result
}

fn entropy(probs: &[f32]) -> f32 {
    -probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeType;
    use std::collections::HashMap;

    #[test]
    fn test_attention_empty_subgraph() {
        let config = EmbeddingConfig::default();
        let engine = GraphAttentionEngine::new(&config).unwrap();

        let query = vec![1.0; config.dimension];
        let subgraph = SubGraph {
            nodes: vec![],
            edges: vec![],
            center_ids: vec![],
        };

        let context = engine.attend(&query, &subgraph).unwrap();
        assert_eq!(context.embedding, query);
    }

    #[test]
    fn test_attention_single_node() {
        let config = EmbeddingConfig::default();
        let engine = GraphAttentionEngine::new(&config).unwrap();

        let query = vec![1.0; config.dimension];
        let node = MemoryNode {
            id: "test".into(),
            vector: vec![1.0; config.dimension],
            text: "Test".into(),
            node_type: NodeType::Document,
            source: "test".into(),
            metadata: HashMap::new(),
        };

        let subgraph = SubGraph {
            nodes: vec![node],
            edges: vec![],
            center_ids: vec!["test".into()],
        };

        let context = engine.attend(&query, &subgraph).unwrap();
        assert_eq!(context.ranked_nodes.len(), 1);
        assert_eq!(context.attention_weights.len(), 1);
        assert!((context.attention_weights[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax(&scores);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
