//! Node.js bindings for RuVector MinCut
//!
//! Provides native Node.js API for dynamic minimum cut operations.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_mincut::{DynamicMinCut, MinCutBuilder, MinCutConfig};
use std::sync::{Arc, Mutex};

/// Edge representation for JavaScript
#[napi(object)]
pub struct JsEdge {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub weight: f64,
}

/// Statistics about the algorithm
#[napi(object)]
pub struct JsStats {
    pub insertions: u32,
    pub deletions: u32,
    pub queries: u32,
    pub avg_update_time_us: f64,
}

/// Minimum cut result
#[napi(object)]
pub struct JsMinCutResult {
    pub value: f64,
    pub is_exact: bool,
    pub approximation_ratio: f64,
}

/// Configuration for minimum cut
#[napi(object)]
pub struct JsMinCutConfig {
    pub approximate: Option<bool>,
    pub epsilon: Option<f64>,
    pub max_cut_size: Option<u32>,
}

/// Node.js wrapper for DynamicMinCut
#[napi]
pub struct MinCut {
    inner: Arc<Mutex<DynamicMinCut>>,
}

#[napi]
impl MinCut {
    /// Create a new empty minimum cut structure
    #[napi(constructor)]
    pub fn new(config: Option<JsMinCutConfig>) -> Result<Self> {
        let rust_config = if let Some(cfg) = config {
            MinCutConfig {
                approximate: cfg.approximate.unwrap_or(false),
                epsilon: cfg.epsilon.unwrap_or(0.1),
                max_cut_size: cfg.max_cut_size,
            }
        } else {
            MinCutConfig::default()
        };

        let mincut = MinCutBuilder::new()
            .with_config(rust_config)
            .build()
            .map_err(|e| Error::from_reason(format!("Failed to create MinCut: {}", e)))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(mincut)),
        })
    }

    /// Create from edges array
    #[napi(factory)]
    pub fn from_edges(edges: Vec<(u32, u32, f64)>, config: Option<JsMinCutConfig>) -> Result<Self> {
        let rust_config = if let Some(cfg) = config {
            MinCutConfig {
                approximate: cfg.approximate.unwrap_or(false),
                epsilon: cfg.epsilon.unwrap_or(0.1),
                max_cut_size: cfg.max_cut_size,
            }
        } else {
            MinCutConfig::default()
        };

        let mincut = MinCutBuilder::new()
            .with_config(rust_config)
            .from_edges(edges)
            .map_err(|e| Error::from_reason(format!("Failed to create MinCut from edges: {}", e)))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(mincut)),
        })
    }

    /// Insert an edge (returns new min cut value)
    #[napi]
    pub fn insert_edge(&self, u: u32, v: u32, weight: f64) -> Result<f64> {
        let mut mincut = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        mincut
            .insert_edge(u, v, weight)
            .map_err(|e| Error::from_reason(format!("Failed to insert edge: {}", e)))
    }

    /// Delete an edge (returns new min cut value)
    #[napi]
    pub fn delete_edge(&self, u: u32, v: u32) -> Result<f64> {
        let mut mincut = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        mincut
            .delete_edge(u, v)
            .map_err(|e| Error::from_reason(format!("Failed to delete edge: {}", e)))
    }

    /// Get minimum cut value
    #[napi(getter)]
    pub fn min_cut_value(&self) -> f64 {
        let mincut = self.inner.lock().unwrap();
        mincut.min_cut_value()
    }

    /// Get detailed minimum cut result
    #[napi]
    pub fn min_cut(&self) -> JsMinCutResult {
        let mincut = self.inner.lock().unwrap();
        let result = mincut.min_cut();

        JsMinCutResult {
            value: result.value,
            is_exact: result.is_exact,
            approximation_ratio: result.approximation_ratio,
        }
    }

    /// Get partition: returns { s: number[], t: number[] }
    #[napi]
    pub fn partition(&self) -> Result<serde_json::Value> {
        let mincut = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        let (s, t) = mincut.partition();

        let result = serde_json::json!({
            "s": s,
            "t": t
        });

        Ok(result)
    }

    /// Get cut edges
    #[napi]
    pub fn cut_edges(&self) -> Vec<JsEdge> {
        let mincut = self.inner.lock().unwrap();
        let edges = mincut.cut_edges();

        edges
            .into_iter()
            .map(|e| JsEdge {
                id: e.id,
                source: e.source,
                target: e.target,
                weight: e.weight,
            })
            .collect()
    }

    /// Get number of vertices
    #[napi(getter)]
    pub fn num_vertices(&self) -> u32 {
        let mincut = self.inner.lock().unwrap();
        mincut.num_vertices()
    }

    /// Get number of edges
    #[napi(getter)]
    pub fn num_edges(&self) -> u32 {
        let mincut = self.inner.lock().unwrap();
        mincut.num_edges()
    }

    /// Check if graph is connected
    #[napi]
    pub fn is_connected(&self) -> bool {
        let mincut = self.inner.lock().unwrap();
        mincut.is_connected()
    }

    /// Get algorithm statistics
    #[napi(getter)]
    pub fn stats(&self) -> JsStats {
        let mincut = self.inner.lock().unwrap();
        let stats = mincut.stats();

        JsStats {
            insertions: stats.insertions,
            deletions: stats.deletions,
            queries: stats.queries,
            avg_update_time_us: stats.avg_update_time_us,
        }
    }

    /// Reset statistics
    #[napi]
    pub fn reset_stats(&self) {
        let mut mincut = self.inner.lock().unwrap();
        mincut.reset_stats();
    }
}
