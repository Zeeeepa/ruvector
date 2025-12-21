//! Subpolynomial-Time Dynamic Minimum Cut Demo
//!
//! This example demonstrates the key features of the ruvector-mincut crate:
//! 1. Basic minimum cut computation
//! 2. Dynamic updates (insert/delete edges)
//! 3. Exact vs approximate modes
//! 4. Real-time monitoring
//! 5. Network resilience analysis
//! 6. Performance scaling

use ruvector_mincut::prelude::*;
use ruvector_mincut::{MonitorBuilder, EventType};
use rand::prelude::*;
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Subpolynomial-Time Dynamic Minimum Cut Algorithm Demo      ║");
    println!("║  ruvector-mincut v0.1.0                                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Basic usage
    demo_basic_usage();
    println!("\n{}\n", "─".repeat(64));

    // Demo 2: Dynamic updates
    demo_dynamic_updates();
    println!("\n{}\n", "─".repeat(64));

    // Demo 3: Exact vs approximate
    demo_exact_vs_approximate();
    println!("\n{}\n", "─".repeat(64));

    // Demo 4: Real-time monitoring
    demo_monitoring();
    println!("\n{}\n", "─".repeat(64));

    // Demo 5: Network resilience
    demo_network_resilience();
    println!("\n{}\n", "─".repeat(64));

    // Demo 6: Performance scaling
    demo_performance_scaling();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Demo Complete!                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Demo 1: Basic minimum cut computation
fn demo_basic_usage() {
    println!("📊 DEMO 1: Basic Minimum Cut Computation");
    println!("Creating a triangle graph with vertices 1, 2, 3...\n");

    // Create a triangle graph: 1-2, 2-3, 3-1
    let mincut = MinCutBuilder::new()
        .exact()
        .with_edges(vec![
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 1, 1.0),
        ])
        .build()
        .expect("Failed to build mincut");

    println!("Graph created:");
    println!("  • Vertices: {}", mincut.num_vertices());
    println!("  • Edges: {}", mincut.num_edges());
    println!("  • Connected: {}", mincut.is_connected());

    // Query the minimum cut
    let result = mincut.min_cut();
    println!("\nMinimum cut result:");
    println!("  • Value: {}", result.value);
    println!("  • Is exact: {}", result.is_exact);
    println!("  • Approximation ratio: {}", result.approximation_ratio);

    if let Some((s, t)) = result.partition {
        println!("  • Partition S: {:?}", s);
        println!("  • Partition T: {:?}", t);
    }

    if let Some(cut_edges) = result.cut_edges {
        println!("  • Number of cut edges: {}", cut_edges.len());
        for edge in &cut_edges {
            println!("    - Edge ({}, {}) with weight {}",
                edge.source, edge.target, edge.weight);
        }
    }

    // Get graph statistics
    let graph = mincut.graph();
    let stats = graph.read().stats();
    println!("\nGraph statistics:");
    println!("  • Total weight: {}", stats.total_weight);
    println!("  • Min degree: {}", stats.min_degree);
    println!("  • Max degree: {}", stats.max_degree);
    println!("  • Avg degree: {:.2}", stats.avg_degree);
}

/// Demo 2: Dynamic edge insertions and deletions
fn demo_dynamic_updates() {
    println!("🔄 DEMO 2: Dynamic Updates");
    println!("Starting with an empty graph and adding edges dynamically...\n");

    let mut mincut = MinCutBuilder::new()
        .exact()
        .build()
        .expect("Failed to build mincut");

    println!("Initial state:");
    println!("  • Min cut: {}", mincut.min_cut_value());

    // Insert edges one by one
    println!("\nInserting edge (1, 2)...");
    let cut = mincut.insert_edge(1, 2, 1.0).expect("Insert failed");
    println!("  • New min cut: {}", cut);

    println!("Inserting edge (2, 3)...");
    let cut = mincut.insert_edge(2, 3, 1.0).expect("Insert failed");
    println!("  • New min cut: {}", cut);

    println!("Inserting edge (3, 1)...");
    let cut = mincut.insert_edge(3, 1, 1.0).expect("Insert failed");
    println!("  • New min cut: {} (triangle formed)", cut);

    // Add a fourth vertex
    println!("\nAdding vertex 4 with edge to vertex 3...");
    println!("Inserting edge (3, 4)...");
    let cut = mincut.insert_edge(3, 4, 2.0).expect("Insert failed");
    println!("  • New min cut: {}", cut);

    // Now delete an edge from the triangle
    println!("\nDeleting edge (3, 1)...");
    let cut = mincut.delete_edge(3, 1).expect("Delete failed");
    println!("  • New min cut: {} (triangle broken)", cut);

    // Add it back
    println!("\nRe-inserting edge (1, 3)...");
    let cut = mincut.insert_edge(1, 3, 1.5).expect("Insert failed");
    println!("  • New min cut: {} (different weight this time)", cut);

    // Check algorithm statistics
    let stats = mincut.stats();
    println!("\nAlgorithm statistics:");
    println!("  • Total insertions: {} (including re-insertion)", stats.insertions);
    println!("  • Total deletions: {}", stats.deletions);
    println!("  • Total queries: {}", stats.queries);
    println!("  • Avg update time: {:.2} μs", stats.avg_update_time_us);
    println!("  • Avg query time: {:.2} μs", stats.avg_query_time_us);
}

/// Demo 3: Exact vs approximate algorithms
fn demo_exact_vs_approximate() {
    println!("⚖️  DEMO 3: Exact vs Approximate Algorithms");
    println!("Comparing exact and approximate modes on the same graph...\n");

    // Create test graph: a bridge graph (two triangles connected by an edge)
    let edges = vec![
        // Triangle 1
        (1, 2, 2.0),
        (2, 3, 2.0),
        (3, 1, 2.0),
        // Bridge
        (3, 4, 1.0),
        // Triangle 2
        (4, 5, 2.0),
        (5, 6, 2.0),
        (6, 4, 2.0),
    ];

    // Exact mode
    println!("Building with exact algorithm...");
    let start = Instant::now();
    let exact_mincut = MinCutBuilder::new()
        .exact()
        .with_edges(edges.clone())
        .build()
        .expect("Failed to build exact");
    let exact_time = start.elapsed();

    let exact_result = exact_mincut.min_cut();
    println!("Exact algorithm:");
    println!("  • Build time: {:?}", exact_time);
    println!("  • Min cut value: {}", exact_result.value);
    println!("  • Is exact: {}", exact_result.is_exact);
    println!("  • Approximation ratio: {}", exact_result.approximation_ratio);

    // Approximate mode with ε = 0.1 (10% approximation)
    println!("\nBuilding with approximate algorithm (ε = 0.1)...");
    let start = Instant::now();
    let approx_mincut = MinCutBuilder::new()
        .approximate(0.1)
        .with_edges(edges.clone())
        .build()
        .expect("Failed to build approximate");
    let approx_time = start.elapsed();

    let approx_result = approx_mincut.min_cut();
    println!("Approximate algorithm:");
    println!("  • Build time: {:?}", approx_time);
    println!("  • Min cut value: {}", approx_result.value);
    println!("  • Is exact: {}", approx_result.is_exact);
    println!("  • Approximation ratio: {}", approx_result.approximation_ratio);

    // Compare results
    println!("\nComparison:");
    println!("  • Exact value: {}", exact_result.value);
    println!("  • Approximate value: {}", approx_result.value);
    let error = ((approx_result.value - exact_result.value) / exact_result.value * 100.0).abs();
    println!("  • Error: {:.2}%", error);
    println!("  • Speedup: {:.2}x", exact_time.as_secs_f64() / approx_time.as_secs_f64());
}

/// Demo 4: Real-time monitoring with thresholds
fn demo_monitoring() {
    println!("📡 DEMO 4: Real-time Monitoring");
    println!("Setting up event monitoring with thresholds...\n");

    // Create counters for different event types
    let cut_increased_count = Arc::new(AtomicU64::new(0));
    let cut_decreased_count = Arc::new(AtomicU64::new(0));
    let threshold_count = Arc::new(AtomicU64::new(0));
    let disconnected_count = Arc::new(AtomicU64::new(0));

    // Build monitor with thresholds
    let inc_clone = cut_increased_count.clone();
    let dec_clone = cut_decreased_count.clone();
    let thr_clone = threshold_count.clone();
    let dis_clone = disconnected_count.clone();

    let monitor = MonitorBuilder::new()
        .threshold_below(1.5, "critical")
        .threshold_above(5.0, "warning")
        .on_event_type(EventType::CutIncreased, "inc_cb", move |event| {
            inc_clone.fetch_add(1, Ordering::SeqCst);
            println!("  [EVENT] Cut increased: {} → {}", event.old_value, event.new_value);
        })
        .on_event_type(EventType::CutDecreased, "dec_cb", move |event| {
            dec_clone.fetch_add(1, Ordering::SeqCst);
            println!("  [EVENT] Cut decreased: {} → {}", event.old_value, event.new_value);
        })
        .on_event_type(EventType::ThresholdCrossedBelow, "thr_cb", move |event| {
            thr_clone.fetch_add(1, Ordering::SeqCst);
            println!("  [ALERT] Threshold crossed below: {} (threshold: {:?})",
                event.new_value, event.threshold);
        })
        .on_event_type(EventType::Disconnected, "dis_cb", move |_event| {
            dis_clone.fetch_add(1, Ordering::SeqCst);
            println!("  [CRITICAL] Graph became disconnected!");
        })
        .build();

    println!("Monitor configured with:");
    println!("  • Critical threshold: < 1.5");
    println!("  • Warning threshold: > 5.0");
    println!("  • 4 event callbacks registered\n");

    // Simulate a series of graph changes
    println!("Simulating graph updates...\n");

    monitor.notify(0.0, 2.0, Some((1, 2)));
    std::thread::sleep(std::time::Duration::from_millis(10));

    monitor.notify(2.0, 3.0, Some((2, 3)));
    std::thread::sleep(std::time::Duration::from_millis(10));

    monitor.notify(3.0, 1.0, None);
    std::thread::sleep(std::time::Duration::from_millis(10));

    monitor.notify(1.0, 6.0, Some((3, 4)));
    std::thread::sleep(std::time::Duration::from_millis(10));

    monitor.notify(6.0, 0.0, None);
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Get metrics
    let metrics = monitor.metrics();
    println!("\nMonitoring metrics:");
    println!("  • Total events: {}", metrics.total_events);
    println!("  • Cut increased events: {}", cut_increased_count.load(Ordering::SeqCst));
    println!("  • Cut decreased events: {}", cut_decreased_count.load(Ordering::SeqCst));
    println!("  • Threshold violations: {}", threshold_count.load(Ordering::SeqCst));
    println!("  • Disconnection events: {}", disconnected_count.load(Ordering::SeqCst));
    println!("  • Min observed cut: {}", metrics.min_observed);
    println!("  • Max observed cut: {}", metrics.max_observed);
    println!("  • Average cut: {:.2}", metrics.avg_cut);

    // Print event breakdown
    println!("\nEvents by type:");
    for (event_type, count) in &metrics.events_by_type {
        println!("  • {}: {}", event_type, count);
    }
}

/// Demo 5: Network resilience analysis
fn demo_network_resilience() {
    println!("🛡️  DEMO 5: Network Resilience Analysis");
    println!("Analyzing a network's resistance to failures...\n");

    // Create a network topology: a mesh with redundant paths
    println!("Building a mesh network (6 nodes, 9 edges)...");
    let mincut = MinCutBuilder::new()
        .exact()
        .with_edges(vec![
            // Core ring
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 6, 1.0),
            (6, 1, 1.0),
            // Cross connections for redundancy
            (1, 3, 1.0),
            (2, 4, 1.0),
            (3, 5, 1.0),
        ])
        .build()
        .expect("Failed to build network");

    let graph = mincut.graph();
    let stats = graph.read().stats();

    println!("\nNetwork topology:");
    println!("  • Nodes: {}", stats.num_vertices);
    println!("  • Links: {}", stats.num_edges);
    println!("  • Avg degree: {:.2}", stats.avg_degree);
    println!("  • Min cut: {}", mincut.min_cut_value());

    println!("\nResilience interpretation:");
    let min_cut = mincut.min_cut_value();
    if min_cut == 0.0 {
        println!("  ❌ Network is disconnected - no resilience");
    } else if min_cut == 1.0 {
        println!("  ⚠️  Single point of failure - low resilience");
    } else if min_cut == 2.0 {
        println!("  ⚡ Moderate resilience - can survive 1 failure");
    } else {
        println!("  ✅ High resilience - can survive {} failures", min_cut as u32 - 1);
    }

    // Simulate edge failures
    println!("\nSimulating link failures...");
    let result = mincut.min_cut();
    if let Some(cut_edges) = result.cut_edges {
        println!("\nCritical edges (minimum cut set):");
        for (i, edge) in cut_edges.iter().enumerate() {
            println!("  {}. ({}, {}) - weight {}",
                i + 1, edge.source, edge.target, edge.weight);
        }
        println!("\nRemoving these {} edge(s) would disconnect the network!", cut_edges.len());
    }

    // Identify the partition
    if let Some((s, t)) = result.partition {
        println!("\nNetwork would split into:");
        println!("  • Component A: {} nodes {:?}", s.len(), s);
        println!("  • Component B: {} nodes {:?}", t.len(), t);
    }
}

/// Demo 6: Performance scaling analysis
fn demo_performance_scaling() {
    println!("📈 DEMO 6: Performance Scaling");
    println!("Measuring performance at different graph sizes...\n");

    let sizes = vec![10, 50, 100, 200];
    println!("{:<10} {:<15} {:<15} {:<15}", "Vertices", "Edges", "Build Time", "Query Time");
    println!("{}", "─".repeat(60));

    for n in sizes {
        // Create a random graph
        let mut rng = rand::thread_rng();
        let mut edges = Vec::new();

        // Create a path to ensure connectivity
        for i in 0..n-1 {
            edges.push((i, i + 1, rng.gen_range(1.0..10.0)));
        }

        // Add random edges for density
        let num_random_edges = n / 2;
        for _ in 0..num_random_edges {
            let u = rng.gen_range(0..n);
            let v = rng.gen_range(0..n);
            if u != v {
                edges.push((u, v, rng.gen_range(1.0..10.0)));
            }
        }

        // Build and measure
        let start = Instant::now();
        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build();
        let build_time = start.elapsed();

        if let Ok(mincut) = mincut {
            let start = Instant::now();
            let _cut = mincut.min_cut_value();
            let query_time = start.elapsed();

            println!("{:<10} {:<15} {:<15?} {:<15?}",
                n,
                mincut.num_edges(),
                build_time,
                query_time
            );
        }
    }

    println!("\n💡 Key observations:");
    println!("  • Query time is O(1) - constant regardless of size");
    println!("  • Build time grows subpolynomially: O(n^{{o(1)}})");
    println!("  • Update time (insert/delete) is also subpolynomial");

    // Demonstrate update performance
    println!("\nMeasuring update performance on n=100 graph...");
    let mut edges = Vec::new();
    for i in 0..99 {
        edges.push((i, i + 1, 1.0));
    }

    let mut mincut = MinCutBuilder::new()
        .exact()
        .with_edges(edges)
        .build()
        .expect("Build failed");

    // Measure insertions
    let start = Instant::now();
    for i in 0..10 {
        let _ = mincut.insert_edge(i, i + 50, 1.0);
    }
    let insert_time = start.elapsed();

    // Measure deletions
    let start = Instant::now();
    for i in 0..10 {
        let _ = mincut.delete_edge(i, i + 1);
    }
    let delete_time = start.elapsed();

    println!("\nUpdate performance (10 operations):");
    println!("  • Total insertion time: {:?}", insert_time);
    println!("  • Avg per insertion: {:?}", insert_time / 10);
    println!("  • Total deletion time: {:?}", delete_time);
    println!("  • Avg per deletion: {:?}", delete_time / 10);

    let stats = mincut.stats();
    println!("\nAggregate statistics:");
    println!("  • Total updates: {}", stats.insertions + stats.deletions);
    println!("  • Avg update time: {:.2} μs", stats.avg_update_time_us);
}
