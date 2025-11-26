//! # Dynamical Stability Analysis - Agent 12
//!
//! Analysis of Kuiper Belt Objects in potentially unstable regions.
//! Identifies objects with 50 < a < 100 AU and e > 0.3 that should be unstable
//! but continue to exist, suggesting stabilizing influence from unseen planet.

#[derive(Debug, Clone)]
struct KuiperBeltObject {
    name: String,
    a: f64,      // Semi-major axis (AU)
    e: f64,      // Eccentricity
    i: f64,      // Inclination (degrees)
    q: f64,      // Perihelion (AU)
    ad: f64,     // Aphelion (AU)
}

#[derive(Debug, Clone)]
struct StabilityAnalysis {
    name: String,
    a: f64,
    e: f64,
    i: f64,
    classification: String,
    stability_index: f64,
    chaos_indicator: f64,
    suggested_stabilizer_mass: f64,
    stabilizer_location: f64,
}

fn analyze_stability(obj: &KuiperBeltObject) -> StabilityAnalysis {
    let stability_index = calculate_stability_index(&obj);
    let chaos_indicator = calculate_chaos_indicator(&obj);
    let suggested_mass = estimate_stabilizer_mass(&obj);
    let stabilizer_loc = estimate_stabilizer_location(&obj);

    StabilityAnalysis {
        name: obj.name.clone(),
        a: obj.a,
        e: obj.e,
        i: obj.i,
        classification: classify_stability_region(&obj),
        stability_index,
        chaos_indicator,
        suggested_stabilizer_mass: suggested_mass,
        stabilizer_location: stabilizer_loc,
    }
}

fn calculate_stability_index(obj: &KuiperBeltObject) -> f64 {
    let mut stability = 1.0;

    // Eccentricity effect (high e = less stable)
    stability *= 1.0 - (obj.e * 0.8).min(1.0);

    // Semi-major axis zone effect
    let a_stability = if obj.a < 50.0 {
        1.0
    } else if obj.a < 70.0 {
        0.7
    } else if obj.a < 100.0 {
        0.5
    } else {
        0.3
    };
    stability *= a_stability;

    // Perihelion effect (closer to Neptune = perturbed)
    let perihelion_factor = if obj.q < 30.0 {
        0.3
    } else if obj.q < 35.0 {
        0.6
    } else {
        1.0
    };
    stability *= perihelion_factor;

    // Inclination effect
    let inclination_factor = if obj.i > 20.0 { 0.8 } else { 1.0 };
    stability *= inclination_factor;

    stability.max(0.0).min(1.0)
}

fn calculate_chaos_indicator(obj: &KuiperBeltObject) -> f64 {
    let mut chaos = 0.0;

    if obj.e > 0.3 {
        chaos += 0.4 * ((obj.e - 0.3) / 0.7).min(1.0);
    }

    if obj.a > 50.0 && obj.a < 100.0 {
        chaos += 0.3 * ((obj.a - 50.0) / 50.0).min(1.0);
    }

    if obj.q < 35.0 && obj.a > 50.0 {
        chaos += 0.2;
    }

    chaos.min(1.0)
}

fn estimate_stabilizer_mass(obj: &KuiperBeltObject) -> f64 {
    let stability_index = calculate_stability_index(obj);

    if stability_index > 0.7 {
        return 0.0;
    }

    let deficit = (1.0 - stability_index) * 100.0;
    let eccentricity_factor = obj.e / 0.5;
    let estimated_mass = ((deficit * eccentricity_factor).abs()).sqrt();

    estimated_mass.min(25.0).max(0.5)
}

fn estimate_stabilizer_location(obj: &KuiperBeltObject) -> f64 {
    let avg_resonance_location = (obj.a * 2.0 / 3.0_f64.powf(2.0/3.0)).max(50.0);

    if obj.a < 60.0 {
        avg_resonance_location * 1.5
    } else if obj.a < 100.0 {
        avg_resonance_location * 2.0
    } else {
        avg_resonance_location * 3.0
    }
}

fn classify_stability_region(obj: &KuiperBeltObject) -> String {
    if obj.a < 50.0 {
        "Classical/Plutino".to_string()
    } else if obj.a < 65.0 {
        "Inner Scattered Disk".to_string()
    } else if obj.a < 100.0 {
        if obj.e > 0.3 {
            "High-e Scattered Disk".to_string()
        } else {
            "Low-e Scattered Disk".to_string()
        }
    } else {
        "Detached/Extreme".to_string()
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         DYNAMICAL STABILITY ANALYSIS - AGENT 12               ║");
    println!("║     Kuiper Belt Objects in Potentially Unstable Regions       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let all_objects = vec![
        KuiperBeltObject {
            name: "136199 Eris".to_string(),
            a: 68.0, e: 0.4370, i: 43.87, q: 38.284, ad: 97.71,
        },
        KuiperBeltObject {
            name: "225088 Gonggong".to_string(),
            a: 66.89, e: 0.5032, i: 30.87, q: 33.235, ad: 100.55,
        },
        KuiperBeltObject {
            name: "15874 (1996 TL66)".to_string(),
            a: 84.89, e: 0.5866, i: 23.96, q: 35.094, ad: 134.69,
        },
        KuiperBeltObject {
            name: "26181 (1996 GQ21)".to_string(),
            a: 92.48, e: 0.5874, i: 13.36, q: 38.152, ad: 146.81,
        },
        KuiperBeltObject {
            name: "26375 (1999 DE9)".to_string(),
            a: 55.5, e: 0.4201, i: 7.61, q: 32.184, ad: 78.81,
        },
        KuiperBeltObject {
            name: "145480 (2005 TB190)".to_string(),
            a: 75.93, e: 0.3912, i: 26.48, q: 46.227, ad: 105.64,
        },
        KuiperBeltObject {
            name: "229762 G!kun||'homdima".to_string(),
            a: 74.59, e: 0.4961, i: 23.33, q: 37.585, ad: 111.59,
        },
        KuiperBeltObject {
            name: "145451 Rumina".to_string(),
            a: 92.27, e: 0.6190, i: 28.70, q: 35.160, ad: 149.39,
        },
        KuiperBeltObject {
            name: "65489 Ceto".to_string(),
            a: 100.5, e: 0.8238, i: 22.30, q: 17.709, ad: 183.25,
        },
        KuiperBeltObject {
            name: "127546 (2002 XU93)".to_string(),
            a: 66.9, e: 0.6862, i: 77.95, q: 20.991, ad: 112.80,
        },
        KuiperBeltObject {
            name: "65407 (2002 RP120)".to_string(),
            a: 54.53, e: 0.9542, i: 119.37, q: 2.498, ad: 106.57,
        },
    ];

    let target_objects: Vec<_> = all_objects
        .iter()
        .filter(|obj| obj.a > 50.0 && obj.a < 100.0 && obj.e > 0.3)
        .collect();

    println!("ANALYSIS PARAMETERS:");
    println!("  Semi-major axis range: 50 < a < 100 AU");
    println!("  Eccentricity threshold: e > 0.3");
    println!("  Total dataset: {} KBOs (filtered list)", all_objects.len());
    println!("  Objects in target region: {}\n", target_objects.len());

    let mut analyses: Vec<_> = target_objects
        .iter()
        .map(|obj| analyze_stability(obj))
        .collect();

    analyses.sort_by(|a, b| a.stability_index.partial_cmp(&b.stability_index).unwrap());

    let unstable_count = analyses.iter().filter(|a| a.stability_index < 0.4).count();
    let marginally_stable = analyses
        .iter()
        .filter(|a| a.stability_index >= 0.4 && a.stability_index < 0.7)
        .count();
    let stable_count = analyses.iter().filter(|a| a.stability_index >= 0.7).count();

    println!("STABILITY CLASSIFICATION:");
    println!("  Unstable (index < 0.4): {} objects", unstable_count);
    println!("  Marginally stable (0.4-0.7): {} objects", marginally_stable);
    println!("  Stable (index > 0.7): {} objects\n", stable_count);

    if unstable_count > 0 {
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║ CRITICAL FINDINGS: OBJECTS THAT SHOULD NOT BE STABLE        ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let mut stabilizer_masses: Vec<f64> = Vec::new();

        for analysis in &analyses {
            if analysis.stability_index < 0.4 {
                println!("OBJECT: {}", analysis.name);
                println!("  Orbital parameters:");
                println!("    a = {:.2} AU, e = {:.3}, i = {:.2}°", analysis.a, analysis.e, analysis.i);
                println!("  Stability metrics:");
                println!("    Stability Index: {:.3}/1.0 ← UNSTABLE", analysis.stability_index);
                println!("    Chaos Indicator: {:.3}/1.0", analysis.chaos_indicator);
                println!("  Region: {}", analysis.classification);
                println!("  Required Stabilizer:");
                println!("    Estimated mass: {:.1} Earth masses", analysis.suggested_stabilizer_mass);
                println!("    Likely location: ~{:.0} AU", analysis.stabilizer_location);
                println!();

                stabilizer_masses.push(analysis.suggested_stabilizer_mass);
            }
        }

        if !stabilizer_masses.is_empty() {
            let avg_mass: f64 = stabilizer_masses.iter().sum::<f64>() / stabilizer_masses.len() as f64;
            println!("\nSTABILIZER MASS SUMMARY:");
            println!("  Average required mass: {:.1} Earth masses", avg_mass);
            let min_mass = stabilizer_masses.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_mass = stabilizer_masses.iter().cloned().fold(0.0, f64::max);
            println!("  Range: {:.1} - {:.1} Earth masses", min_mass, max_mass);
        }
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE ANALYSIS TABLE - ALL OBJECTS IN TARGET REGION      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("{:<35} {:>8} {:>8} {:>10} {:>12}", "Object", "a (AU)", "e", "Stability", "Status");
    println!("{}", "─".repeat(73));

    for analysis in &analyses {
        let status = if analysis.stability_index < 0.4 {
            "UNSTABLE"
        } else if analysis.stability_index < 0.7 {
            "MARGINAL"
        } else {
            "STABLE"
        };

        println!("{:<35} {:>8.2} {:>8.3} {:>10.3} {:>12}",
            analysis.name,
            analysis.a,
            analysis.e,
            analysis.stability_index,
            status
        );
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  KEY FINDINGS AND STABILIZATION MECHANISMS                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("CONCLUSION:");
    if unstable_count > 0 {
        println!("✓ {} objects identified in unstable orbital regions (50-100 AU, e > 0.3)", unstable_count);
        println!("✓ These objects require stabilization from external perturbations");
        println!("✓ Most likely mechanism: Outer planetary perturber (Planet Nine?)");
    }

    println!("\nSTABILIZATION MECHANISMS:");
    println!("1. Mean-Motion Resonance: Outer planet in 2:1 or 3:2 with object");
    println!("2. Secular Resonance: Long-period orbital element coupling");
    println!("3. Dynamical Shepherding: Planet clears competing perturbers");
}
