# Subpolynomial-Time Dynamic Minimum Cut Demo

This example demonstrates the **ruvector-mincut** crate, which implements a state-of-the-art subpolynomial-time algorithm for maintaining minimum cuts in dynamic graphs.

## What This Example Demonstrates

### 1. **Basic Minimum Cut Computation**
- Creating graphs with the builder pattern
- Computing minimum cut values
- Extracting cut partitions and edges
- Analyzing graph statistics

### 2. **Dynamic Updates**
- Inserting edges with `O(n^{o(1)})` amortized time
- Deleting edges while maintaining the minimum cut
- Tracking algorithm performance metrics
- Building graphs incrementally from empty

### 3. **Exact vs Approximate Modes**
- **Exact algorithm**: Perfect minimum cut computation
- **Approximate algorithm**: (1+ε)-approximation with faster updates
- Performance trade-offs between accuracy and speed
- Configurable approximation ratio

### 4. **Real-Time Monitoring**
- Event-driven notifications when cuts change
- Threshold-based alerts (above/below)
- Multiple callback types:
  - `CutIncreased` / `CutDecreased`
  - `ThresholdCrossedBelow` / `ThresholdCrossedAbove`
  - `Connected` / `Disconnected`
  - `EdgeInserted` / `EdgeDeleted`
- Comprehensive metrics collection

### 5. **Network Resilience Analysis**
- Identifying critical edges (cut set)
- Measuring network robustness
- Failure analysis and partitioning
- Real-world application to infrastructure networks

### 6. **Performance Scaling**
- Benchmarking at different graph sizes
- Demonstrating subpolynomial scaling
- Constant-time queries
- Update time analysis

## How to Run

### From the Example Directory

```bash
cd examples/subpolynomial-time
cargo run --release
```

### From the Repository Root

```bash
cargo run --release --example subpolynomial-time-mincut-demo
```

**Note**: Use `--release` for accurate performance measurements!

## Expected Output

The demo produces detailed output for each section:

```
╔══════════════════════════════════════════════════════════════╗
║  Subpolynomial-Time Dynamic Minimum Cut Algorithm Demo      ║
║  ruvector-mincut v0.1.0                                      ║
╚══════════════════════════════════════════════════════════════╝

📊 DEMO 1: Basic Minimum Cut Computation
Creating a triangle graph with vertices 1, 2, 3...

Graph created:
  • Vertices: 3
  • Edges: 3
  • Connected: true

Minimum cut result:
  • Value: 2.0
  • Is exact: true
  • Approximation ratio: 1.0
  ...
```

Each demo section includes:
- Clear visual output with emoji indicators
- Detailed statistics and metrics
- Explanatory text about what's happening
- Performance measurements where relevant

## Key Concepts Illustrated

### Minimum Cut
The **minimum cut** of a graph is the smallest set of edges whose removal disconnects the graph. It's a fundamental measure of graph connectivity and has applications in:
- Network reliability analysis
- Clustering and community detection
- Image segmentation
- VLSI circuit design

### Subpolynomial Time
This implementation achieves `O(n^{o(1)})` amortized update time, which is faster than polynomial but slower than polylogarithmic. This is the current state-of-the-art for exact dynamic minimum cut.

### Dynamic Graphs
Unlike static algorithms that recompute from scratch, this maintains the minimum cut incrementally as edges are inserted and deleted, making it suitable for:
- Streaming graph data
- Real-time network monitoring
- Interactive graph editing
- Time-evolving networks

### Monitoring System
The event-driven monitoring system enables:
- Proactive alerting when connectivity degrades
- Historical tracking of network stability
- Integration with external monitoring tools
- Custom business logic based on cut values

## Code Structure

```rust
// Basic usage
let mincut = MinCutBuilder::new()
    .exact()
    .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0)])
    .build()?;

println!("Min cut: {}", mincut.min_cut_value());

// Dynamic updates
mincut.insert_edge(3, 1, 1.0)?;  // Now a triangle
mincut.delete_edge(1, 2)?;       // Break one edge

// Monitoring
let monitor = MonitorBuilder::new()
    .threshold_below(2.0, "critical")
    .on_event_type(EventType::CutDecreased, "alert", |event| {
        println!("Cut decreased to {}", event.new_value);
    })
    .build();

monitor.notify(old_cut, new_cut, None);
```

## Performance Characteristics

| Operation | Time Complexity | Description |
|-----------|----------------|-------------|
| Build | `O(m log n)` | Initial construction |
| Insert Edge | `O(n^{o(1)})` amortized | Subpolynomial |
| Delete Edge | `O(n^{o(1)})` amortized | Subpolynomial |
| Query | `O(1)` | Constant time |
| Get Partition | `O(n)` | Linear in vertices |

where `n` = number of vertices, `m` = number of edges

## Use Cases

### 1. Network Infrastructure
Monitor critical infrastructure (power grids, communication networks) for vulnerability to failures.

### 2. Social Networks
Identify communities and measure social cohesion.

### 3. Distributed Systems
Analyze partition tolerance in distributed databases and services.

### 4. Transportation Networks
Find critical routes and evaluate redundancy in road/rail networks.

### 5. Circuit Design
Optimize chip layouts by minimizing cross-partition connections.

## Advanced Features

### Approximate Mode
Trade accuracy for speed with the approximate algorithm:

```rust
let mincut = MinCutBuilder::new()
    .approximate(0.1)  // 10% approximation
    .build()?;
```

### Parallel Processing
Enable parallel computation for large graphs:

```rust
let mincut = MinCutBuilder::new()
    .parallel(true)
    .build()?;
```

### Custom Thresholds
Set multiple monitoring thresholds:

```rust
let monitor = MonitorBuilder::new()
    .threshold_below(1.0, "critical")
    .threshold_below(2.0, "warning")
    .threshold_above(10.0, "good")
    .build();
```

## Dependencies

- **ruvector-mincut**: Core minimum cut algorithm
- **rand**: Random graph generation for demos

## Further Reading

- [Dynamic Minimum Cut Algorithm Paper](https://arxiv.org/abs/2011.11264)
- [Graph Connectivity Algorithms](https://en.wikipedia.org/wiki/Connectivity_(graph_theory))
- [Network Reliability Theory](https://en.wikipedia.org/wiki/Network_reliability)

## License

This example is part of the ruvector project and follows the same license.

## Contributing

Suggestions for additional demos or use cases? Open an issue or PR at:
https://github.com/ruvnet/ruvector
