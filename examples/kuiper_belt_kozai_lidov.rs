//! # Analysis Agent 11: Kozai-Lidov Mechanism Analysis
//!
//! Identifies objects with coupled eccentricity-inclination oscillations
//! caused by gravitational perturbations from a distant third body.
//!
//! Selection Criteria: e > 0.5, i > 30°, a > 50 AU
//!
//! Run with:
//! ```bash
//! cargo run --example kuiper_belt_kozai_lidov --all-features
//! ```

mod kuiper_belt {
    pub mod kuiper_cluster;
    pub mod kbo_data;
    pub mod inclination_analysis;
    pub mod aphelion_clustering;
    pub mod eccentricity_analysis;
    pub mod perihelion_analysis;
    pub mod kozai_lidov_mechanism;

    pub use kuiper_cluster::KuiperBeltObject;
    pub use kbo_data::get_kbo_data;
    pub use kozai_lidov_mechanism::{
        analyze_kozai_lidov_mechanism,
        get_kozai_analysis_report,
        KozaiLidovAnalysis,
    };
}

use kuiper_belt::{
    analyze_kozai_lidov_mechanism,
    get_kozai_analysis_report,
};
use std::fs;
use std::path::Path;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                ANALYSIS AGENT 11: KOZAI-LIDOV MECHANISM            ║");
    println!("║           Coupled Eccentricity-Inclination Oscillations            ║");
    println!("║              Powered by RuVector Vector Database                   ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    println!("🔍 Analyzing Kuiper Belt Objects for Kozai-Lidov Signatures...\n");

    // Run the analysis
    let analysis = analyze_kozai_lidov_mechanism();

    // Generate report
    let report = get_kozai_analysis_report(&analysis);

    // Display to console
    println!("{}", report);

    // Save report to file
    let report_path = "/home/user/ruvector/examples/kuiper_belt/KOZAI_LIDOV_ANALYSIS.txt";
    match fs::write(report_path, &report) {
        Ok(_) => {
            println!(
                "\n✓ Report saved to: {}",
                report_path
            );
        }
        Err(e) => {
            eprintln!("⚠️  Failed to save report: {}", e);
        }
    }

    // Save JSON data if serde_json is available
    if let Ok(json) = serde_json::to_string_pretty(&analysis) {
        let json_path = "/home/user/ruvector/examples/kuiper_belt/KOZAI_LIDOV_DATA.json";
        match fs::write(json_path, json) {
            Ok(_) => {
                println!("✓ Data saved to: {}", json_path);
            }
            Err(e) => {
                eprintln!("⚠️  Failed to save JSON: {}", e);
            }
        }
    }

    // Summary statistics
    println!("\n════════════════════════════════════════════════════════════════════");
    println!("ANALYSIS SUMMARY");
    println!("════════════════════════════════════════════════════════════════════\n");

    let stats = &analysis.summary;
    println!("Objects Identified (e > 0.5, i > 30°, a > 50 AU): {}", stats.count);
    println!(
        "  • Strong Kozai Signature (score > 0.7): {}",
        stats.strong_kozai_count
    );
    println!(
        "  • Moderate Kozai Signature (score > 0.5): {}",
        stats.moderate_kozai_count
    );

    if stats.count > 0 {
        println!("\nOrbital Parameters:");
        println!("  Eccentricity:  {:.3} - {:.3} (avg: {:.3})",
                 stats.e_distribution.min,
                 stats.e_distribution.max,
                 stats.e_distribution.mean);
        println!("  Inclination:   {:.1}° - {:.1}° (avg: {:.1}°)",
                 stats.i_distribution.min,
                 stats.i_distribution.max,
                 stats.i_distribution.mean);
        println!("  Semi-Major Axis: {:.1} AU (average)", stats.avg_a);
    }

    println!("\nEstimated Perturber Properties:");
    let perturber = &analysis.perturber_parameters;
    println!(
        "  Distance: {:.0} - {:.0} AU",
        perturber.distance_range.0, perturber.distance_range.1
    );
    println!(
        "  Mass: {:.1} - {:.1} Earth masses",
        perturber.mass_range.0, perturber.mass_range.1
    );
    println!(
        "  Inclination: {:.1}°",
        perturber.inclination_estimate
    );
    println!(
        "  Confidence: {:.0}%",
        perturber.overall_confidence * 100.0
    );

    println!("\nOscillation Characteristics:");
    let oscil = &analysis.oscillation_analysis;
    println!(
        "  Fundamental Period: {:.0} years",
        oscil.fundamental_period
    );
    println!(
        "  Eccentricity Amplitude: {:.3}",
        oscil.mean_e_amplitude
    );
    println!(
        "  Inclination Amplitude: {:.1}°",
        oscil.mean_i_amplitude
    );

    println!("\nCandidate Perturbers:");
    for candidate in &perturber.candidate_objects {
        println!(
            "  • {} (match: {:.0}%)",
            candidate.name,
            candidate.match_score * 100.0
        );
    }

    println!("\n════════════════════════════════════════════════════════════════════");
    println!("TOP KOZAI-LIDOV CANDIDATES (by Evidence Score)");
    println!("════════════════════════════════════════════════════════════════════\n");

    for (idx, obj) in analysis.kozai_candidates.iter().take(10).enumerate() {
        let badge = if obj.kozai_evidence_score > 0.7 {
            "★★★"
        } else if obj.kozai_evidence_score > 0.5 {
            "★★"
        } else {
            "★"
        };

        println!("{}. {} {}", idx + 1, badge, obj.name);
        println!(
            "   a={:.1} AU | e={:.3} | i={:.0}° | Kozai Score: {:.3}",
            obj.a, obj.e, obj.i, obj.kozai_evidence_score
        );
        println!(
            "   Expected Period: {:.0} years | ω Circulation: {:.2}",
            obj.estimated_kozai_period, obj.omega_circulation_indicator
        );
        println!();
    }

    println!("════════════════════════════════════════════════════════════════════");
    println!("Analysis Complete!");
    println!("════════════════════════════════════════════════════════════════════\n");
}
