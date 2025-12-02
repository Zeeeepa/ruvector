//! Memory service wrapping Ruvector for vector + graph storage

use crate::config::MemoryConfig;
use crate::error::{Error, MemoryError, Result};
use crate::types::{EdgeType, MemoryEdge, MemoryNode, NodeType};

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Search result from memory
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Retrieved candidates
    pub candidates: Vec<SearchCandidate>,
    /// Expanded subgraph
    pub subgraph: SubGraph,
    /// Statistics
    pub stats: SearchStats,
}

/// Single search candidate
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    /// Node ID
    pub id: String,
    /// Distance to query
    pub distance: f32,
    /// Node data
    pub node: MemoryNode,
}

/// Subgraph from neighborhood expansion
#[derive(Debug, Clone)]
pub struct SubGraph {
    /// Nodes in subgraph
    pub nodes: Vec<MemoryNode>,
    /// Edges in subgraph
    pub edges: Vec<MemoryEdge>,
    /// Center node IDs
    pub center_ids: Vec<String>,
}

/// Search statistics
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Number of candidates
    pub k_retrieved: usize,
    /// Distance statistics
    pub distance_mean: f32,
    pub distance_std: f32,
    pub distance_min: f32,
    pub distance_max: f32,
    /// Graph depth
    pub graph_depth: usize,
}

/// Memory service providing vector search and graph operations
pub struct MemoryService {
    /// Vector storage (mock - in production use ruvector-core)
    vectors: DashMap<String, Vec<f32>>,
    /// Node storage
    nodes: DashMap<String, MemoryNode>,
    /// Edge storage (src_id -> edges)
    edges: DashMap<String, Vec<MemoryEdge>>,
    /// Configuration
    config: MemoryConfig,
}

impl MemoryService {
    /// Create a new memory service
    pub async fn new(config: &MemoryConfig) -> Result<Self> {
        Ok(Self {
            vectors: DashMap::new(),
            nodes: DashMap::new(),
            edges: DashMap::new(),
            config: config.clone(),
        })
    }

    /// Search with graph expansion
    pub async fn search_with_graph(
        &self,
        query: &[f32],
        k: usize,
        _ef_search: usize,
        max_hops: usize,
    ) -> Result<SearchResult> {
        // Simple linear search (in production, use HNSW)
        let mut candidates: Vec<SearchCandidate> = self.vectors
            .iter()
            .map(|entry| {
                let id = entry.key().clone();
                let vec = entry.value();
                let distance = cosine_distance(query, vec);
                let node = self.nodes.get(&id)
                    .map(|n| n.clone())
                    .unwrap_or_else(|| MemoryNode {
                        id: id.clone(),
                        vector: vec.clone(),
                        text: String::new(),
                        node_type: NodeType::Document,
                        source: String::new(),
                        metadata: HashMap::new(),
                    });
                SearchCandidate { id, distance, node }
            })
            .collect();

        // Sort by distance
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(k);

        // Expand neighborhood
        let center_ids: Vec<String> = candidates.iter().map(|c| c.id.clone()).collect();
        let subgraph = self.expand_neighborhood(&center_ids, max_hops)?;

        // Compute stats
        let stats = self.compute_stats(&candidates);

        Ok(SearchResult {
            candidates,
            subgraph,
            stats,
        })
    }

    /// Insert a node
    pub fn insert_node(&self, node: MemoryNode) -> Result<String> {
        let id = node.id.clone();
        self.vectors.insert(id.clone(), node.vector.clone());
        self.nodes.insert(id.clone(), node);
        Ok(id)
    }

    /// Insert an edge
    pub fn insert_edge(&self, edge: MemoryEdge) -> Result<String> {
        let id = edge.id.clone();
        self.edges
            .entry(edge.src.clone())
            .or_insert_with(Vec::new)
            .push(edge);
        Ok(id)
    }

    /// Update edge weight
    pub fn update_edge_weight(&self, src: &str, dst: &str, delta: f32) -> Result<()> {
        if let Some(mut edges) = self.edges.get_mut(src) {
            for edge in edges.iter_mut() {
                if edge.dst == dst {
                    edge.weight = (edge.weight + delta).clamp(0.0, 1.0);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Flush pending writes
    pub async fn flush(&self) -> Result<()> {
        // In production, this would persist to disk
        Ok(())
    }

    /// Expand neighborhood via graph traversal
    fn expand_neighborhood(&self, center_ids: &[String], max_hops: usize) -> Result<SubGraph> {
        let mut visited = std::collections::HashSet::new();
        let mut all_nodes = Vec::new();
        let mut all_edges = Vec::new();
        let mut frontier: Vec<String> = center_ids.to_vec();

        for _hop in 0..=max_hops {
            let mut next_frontier = Vec::new();

            for node_id in &frontier {
                if visited.contains(node_id) {
                    continue;
                }
                visited.insert(node_id.clone());

                // Get node
                if let Some(node) = self.nodes.get(node_id) {
                    all_nodes.push(node.clone());
                }

                // Get edges
                if let Some(edges) = self.edges.get(node_id) {
                    for edge in edges.iter() {
                        all_edges.push(edge.clone());
                        if !visited.contains(&edge.dst) {
                            next_frontier.push(edge.dst.clone());
                        }
                    }
                }
            }

            frontier = next_frontier;
        }

        Ok(SubGraph {
            nodes: all_nodes,
            edges: all_edges,
            center_ids: center_ids.to_vec(),
        })
    }

    fn compute_stats(&self, candidates: &[SearchCandidate]) -> SearchStats {
        if candidates.is_empty() {
            return SearchStats::default();
        }

        let distances: Vec<f32> = candidates.iter().map(|c| c.distance).collect();
        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
        let var = distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;

        SearchStats {
            k_retrieved: candidates.len(),
            distance_mean: mean,
            distance_std: var.sqrt(),
            distance_min: distances.iter().cloned().fold(f32::INFINITY, f32::min),
            distance_max: distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            graph_depth: 0,
        }
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        1.0 - dot / (norm_a * norm_b)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_insert_and_search() {
        let config = MemoryConfig::default();
        let memory = MemoryService::new(&config).await.unwrap();

        let node = MemoryNode {
            id: "test-1".into(),
            vector: vec![1.0, 0.0, 0.0],
            text: "Test document".into(),
            node_type: NodeType::Document,
            source: "test".into(),
            metadata: HashMap::new(),
        };

        memory.insert_node(node).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let result = memory.search_with_graph(&query, 10, 64, 2).await.unwrap();

        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].id, "test-1");
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(cosine_distance(&a, &b) < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 0.001);
    }
}
