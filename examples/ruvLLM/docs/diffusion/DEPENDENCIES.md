# Recommended Rust Libraries and Dependencies

## Overview

This document provides research-backed recommendations for Rust libraries to use in implementing RuvDLLM. Each recommendation is based on active maintenance status, performance characteristics, and compatibility with our goals.

## Core ML Framework

### Primary Recommendation: [Candle](https://github.com/huggingface/candle)

**Why Candle:**
- Hugging Face's official Rust ML framework
- Designed for serverless inference (fast cold starts)
- Native support for LLMs (Mistral, LLaMA, Qwen, StarCoder2)
- Metal, CUDA, and CPU backends
- Active development and community
- Memory efficient (~500MB overhead vs PyTorch's 4GB)

```toml
[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"

[features]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
```

**Key Features for RuvDLLM:**
- Transformer building blocks already implemented
- GGUF/safetensors loading support
- Quantization utilities (4-bit, 8-bit)
- Device abstraction (CPU/GPU transparent)

### Alternative: [Burn](https://github.com/tracel-ai/burn)

**When to consider:**
- Need ONNX model import
- Want backend-agnostic code with swappable backends
- Require automatic differentiation for training
- Need WebGPU deployment

```toml
[dependencies]
burn = "0.15"
burn-wgpu = "0.15"  # WebGPU backend
burn-tch = "0.15"   # LibTorch backend
```

## SIMD Acceleration

### Primary: Rust std::arch + [SimSIMD](https://lib.rs/crates/simsimd)

**std::arch for custom kernels:**
```toml
# No external dependency - use std::arch
# Requires: #![feature(portable_simd)] on nightly for std::simd
```

**SimSIMD for optimized primitives:**
```toml
[dependencies]
simsimd = "0.4"
```

**Why SimSIMD:**
- 3-200x faster than naive implementations
- Mixed-precision support (crucial for ML)
- Supports x86 (AVX2, AVX-512) and ARM (NEON, SVE)
- Optimized for similarity/distance computations
- Actively maintained

**Key operations available:**
- Dot products (f32, f16, i8)
- Cosine similarity
- L2/L1 distances
- Hamming distance
- Jensen-Shannon divergence

### For Portable SIMD:

```rust
// Nightly feature for portable SIMD
#![feature(portable_simd)]
use std::simd::*;

// Or use safer wide crate on stable
use wide::*;
```

```toml
[dependencies]
wide = "0.7"  # Stable Rust SIMD wrapper
```

## GPU Acceleration

### CUDA: [cudarc](https://deepwiki.com/coreylowman/cudarc)

```toml
[dependencies]
cudarc = { version = "0.12", features = ["cuda-12050"] }
```

**Why cudarc:**
- Minimal, safe, ergonomic CUDA bindings
- Wraps cuBLAS, cuDNN
- Async execution support
- Used by Candle's CUDA backend

### Cross-Platform GPU: [CubeCL](https://github.com/tracel-ai/cubecl) + [wgpu](https://wgpu.rs/)

```toml
[dependencies]
cubecl = "0.4"
cubecl-wgpu = "0.4"
cubecl-cuda = { version = "0.4", optional = true }
wgpu = "23.0"
```

**Why CubeCL:**
- Write GPU kernels in Rust (not CUDA C/Metal Shading Language)
- Single codebase → CUDA, ROCm, WebGPU
- Zero-cost abstractions
- Active development (Burn team)

### Metal (Apple): Via Candle

```toml
[dependencies]
candle-core = { version = "0.8", features = ["metal"] }
metal = "0.30"  # Direct Metal access if needed
```

## Vector Database / HNSW

### Primary: [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)

```toml
[dependencies]
hnsw_rs = "0.3"
```

**Why hnswlib-rs:**
- Pure Rust HNSW implementation
- Multithreaded (parking_lot)
- Supports: L1, L2, Cosine, Jaccard, Hamming, Hellinger, Levenshtein
- Custom distance functions supported
- Serialization/persistence built-in

**Performance:**
- Sub-millisecond queries on 100K+ vectors
- Scales to millions with proper parameters

### Alternative: Integrate with Qdrant

```toml
[dependencies]
qdrant-client = "1.12"
```

**When to use Qdrant:**
- Need distributed vector search
- Require payload filtering
- Want hybrid search (dense + sparse)
- Production deployment at scale

## Federation & Networking

### Primary: [tokio](https://tokio.rs/) + [quinn](https://github.com/quinn-rs/quinn)

```toml
[dependencies]
tokio = { version = "1.42", features = ["full"] }
quinn = "0.11"  # QUIC protocol
rustls = "0.23" # TLS
```

**Why this stack:**
- tokio: Standard async runtime
- quinn: QUIC protocol (faster than TCP for real-time)
- rustls: Pure Rust TLS (no OpenSSL dependency)

### For Federated Learning: Build on [candle-fl](https://github.com/nfnt/candle-fl) patterns

```toml
# Reference implementation - may need customization
# See: https://github.com/nfnt/candle-fl
```

### Gossip Protocol: [memberlist](https://crates.io/crates/memberlist) or custom

```toml
[dependencies]
memberlist = "0.3"  # SWIM gossip protocol
```

## Cryptography & Privacy

### Encryption

```toml
[dependencies]
aes-gcm = "0.10"       # AES-256-GCM encryption
x25519-dalek = "2.0"   # Key exchange
ed25519-dalek = "2.1"  # Signatures
argon2 = "0.5"         # Key derivation
zeroize = "1.8"        # Secure memory clearing
```

### Differential Privacy

```toml
[dependencies]
rand = "0.8"
rand_distr = "0.4"     # Gaussian, Laplace distributions
# Custom DP implementation recommended for fine-grained control
```

## Serialization & Model Loading

```toml
[dependencies]
safetensors = "0.4"    # HuggingFace model format
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"        # Fast binary serialization
rmp-serde = "1.3"      # MessagePack (compact)
memmap2 = "0.9"        # Memory-mapped file access
```

## Configuration

```toml
[dependencies]
toml = "0.8"           # Config files
clap = { version = "4.5", features = ["derive"] }  # CLI
tracing = "0.1"        # Logging
tracing-subscriber = "0.3"
```

## Testing & Benchmarking

```toml
[dev-dependencies]
criterion = "0.5"      # Benchmarking
proptest = "1.5"       # Property-based testing
approx = "0.5"         # Float comparisons
```

## Complete Cargo.toml Template

```toml
[package]
name = "ruvdllm"
version = "0.1.0"
edition = "2024"

[features]
default = ["simd"]
simd = []
cuda = ["candle-core/cuda", "cudarc"]
metal = ["candle-core/metal"]
wgpu = ["cubecl-wgpu", "wgpu"]
federation = ["tokio", "quinn", "rustls"]
full = ["simd", "cuda", "metal", "wgpu", "federation"]

[dependencies]
# Core ML
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"

# SIMD
simsimd = "0.4"
wide = "0.7"

# GPU
cudarc = { version = "0.12", optional = true }
cubecl = { version = "0.4", optional = true }
cubecl-wgpu = { version = "0.4", optional = true }
wgpu = { version = "23.0", optional = true }

# Vector search
hnsw_rs = "0.3"

# Networking
tokio = { version = "1.42", features = ["full"], optional = true }
quinn = { version = "0.11", optional = true }
rustls = { version = "0.23", optional = true }

# Crypto
aes-gcm = "0.10"
x25519-dalek = "2.0"
ed25519-dalek = "2.1"
zeroize = "1.8"

# Serialization
safetensors = "0.4"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
memmap2 = "0.9"

# Utilities
thiserror = "2.0"
anyhow = "1.0"
tracing = "0.1"
rand = "0.8"
half = "2.4"  # f16 support
bytemuck = "1.19"

[dev-dependencies]
criterion = "0.5"
proptest = "1.5"
tokio-test = "0.4"

[[bench]]
name = "inference"
harness = false

[[bench]]
name = "talora"
harness = false
```

## Integration Notes

### Candle + SimSIMD Integration

For maximum performance, use SimSIMD for vector operations (dot products, similarity) and Candle for tensor operations:

```rust
use candle_core::{Tensor, Device};
use simsimd::SpatialSimilarity;

pub fn fast_similarity(a: &[f32], b: &[f32]) -> f32 {
    // SimSIMD for raw similarity
    f32::cosine(a, b).unwrap_or(0.0)
}

pub fn tensor_to_vec(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec1().unwrap()
}
```

### CubeCL for Custom GPU Kernels

When Candle's built-in ops aren't sufficient:

```rust
use cubecl::prelude::*;

#[cube(launch)]
fn denoise_kernel(
    x_t: &Tensor<f32>,
    noise_pred: &Tensor<f32>,
    alpha: f32,
    sigma: f32,
    output: &mut Tensor<f32>,
) {
    let idx = ABSOLUTE_POS;
    output[idx] = (x_t[idx] - sigma * noise_pred[idx]) / alpha;
}
```

### hnsw_rs for TALoRA Index

```rust
use hnsw_rs::prelude::*;

pub struct TALoRAIndex {
    index: Hnsw<f32, DistCosine>,
}

impl TALoRAIndex {
    pub fn new(dim: usize) -> Self {
        let index = Hnsw::new(
            16,    // M: neighbors per node
            1000,  // Initial capacity
            16,    // M0: neighbors for layer 0
            200,   // ef_construction
            DistCosine,
        );
        Self { index }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.index.search(query, k, 50)  // ef_search = 50
            .into_iter()
            .map(|n| (n.d_id, 1.0 - n.distance))  // Convert distance to similarity
            .collect()
    }
}
```

## Version Compatibility Matrix

| Library | Version | Rust Version | Notes |
|---------|---------|--------------|-------|
| Candle | 0.8.x | 1.75+ | Stable |
| CubeCL | 0.4.x | 1.75+ | GPU compute |
| hnsw_rs | 0.3.x | 1.70+ | HNSW index |
| SimSIMD | 0.4.x | 1.70+ | SIMD ops |
| tokio | 1.42.x | 1.70+ | Async runtime |
| quinn | 0.11.x | 1.75+ | QUIC |

## Sources

- [Candle - Hugging Face](https://github.com/huggingface/candle)
- [Burn Framework](https://github.com/tracel-ai/burn)
- [CubeCL](https://github.com/tracel-ai/cubecl)
- [SimSIMD](https://lib.rs/crates/simsimd)
- [hnsw_rs](https://github.com/jean-pierreBoth/hnswlib-rs)
- [cudarc](https://deepwiki.com/coreylowman/cudarc)
- [wgpu](https://wgpu.rs/)
- [Qdrant](https://qdrant.tech/)
- [candle-fl](https://github.com/nfnt/candle-fl)
- [RustFL](https://github.com/Sharvani1291/RustFL)
- [Rust SIMD Performance Guide](https://rust-lang.github.io/packed_simd/perf-guide/prof/mca.html)

---

**Last Updated:** December 2024
