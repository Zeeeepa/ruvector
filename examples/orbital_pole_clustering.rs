//! # ANALYSIS AGENT 15: ORBITAL POLE CLUSTERING
//!
//! Analyzes orbital pole vectors for clustering patterns in trans-neptunian objects.
//! Provides comprehensive statistical analysis of orbital pole distribution and
//! clustering strength as evidence of planetary perturbation.
//!
//! ## Theory
//! - Orbital pole = unit vector perpendicular to orbital plane
//! - Direction determined by Ω (longitude of ascending node) and i (inclination)
//! - Objects sharing same perturber show correlated orbital poles
//! - Clustering strength quantified by R-value and von Mises concentration parameter
//!
//! ## Mathematical Background
//!
//! Orbital pole vector in Cartesian coordinates:
//! ```
//! x = sin(i) * cos(Ω)
//! y = sin(i) * sin(Ω)
//! z = cos(i)
//! ```
//!
//! Where i is inclination and Ω is longitude of ascending node (both in radians).
//!
//! Clustering strength (R value):
//! ```
//! R = ||Σv_i|| / n
//! ```
//!
//! Von Mises concentration parameter κ (kappa):
//! - Measures how concentrated the distribution is around mean
//! - κ → 0: uniform distribution
//! - κ → ∞: perfect concentration
//!
//! ## Run
//! ```bash
//! cargo run --example orbital_pole_clustering
//! ```

use std::f32::consts::PI;
use std::collections::HashMap;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone)]
pub struct OrbitalElements {
    pub name: String,
    pub a: f32,        // Semi-major axis (AU)
    pub e: f32,        // Eccentricity
    pub i: f32,        // Inclination (degrees)
    pub omega: f32,    // Longitude of ascending node (degrees)
    pub w: f32,        // Argument of perihelion (degrees)
    pub q: Option<f32>, // Perihelion distance (AU)
    pub ad: Option<f32>, // Aphelion distance (AU)
}

#[derive(Debug, Clone, Copy)]
pub struct PoleVector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub magnitude: f32,
}

impl PoleVector {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        let magnitude = (x * x + y * y + z * z).sqrt();
        Self { x, y, z, magnitude }
    }

    pub fn dot_product(&self, other: &PoleVector) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> PoleVector {
        let mag = self.magnitude();
        if mag > 0.0 {
            PoleVector::new(self.x / mag, self.y / mag, self.z / mag)
        } else {
            PoleVector::new(0.0, 0.0, 0.0)
        }
    }
}

#[derive(Debug)]
pub struct PoleClustering {
    pub total_objects: usize,
    pub clustering_objects: usize,
    pub mean_pole_vector: PoleVector,
    pub resultant_vector_length: f32,
    pub clustering_strength: f32,  // R value: 0-1
    pub mean_residual_angle: f32,  // degrees
    pub concentration_parameter: f32, // κ (kappa)
    pub confidence_score: f32,     // 0-1
    pub clustering_significance: String, // "weak" | "moderate" | "strong" | "very_strong"
    pub mean_inclination: f32,
    pub mean_omega: f32,
}

#[derive(Debug)]
pub struct ClusterRegion {
    pub center: PoleVector,
    pub object_count: usize,
    pub objects: Vec<String>,
    pub radius: f32,
    pub mean_inclination: f32,
    pub mean_omega: f32,
}

// ============================================================================
// POLE VECTOR CONVERSIONS
// ============================================================================

/// Convert orbital elements (Ω, i) to unit pole vector (x, y, z)
pub fn convert_to_pole_vector(obj: &OrbitalElements) -> PoleVector {
    // Convert degrees to radians
    let omega_rad = obj.omega * PI / 180.0;
    let i_rad = obj.i * PI / 180.0;

    // Calculate Cartesian coordinates
    let sin_i = i_rad.sin();
    let cos_i = i_rad.cos();
    let cos_omega = omega_rad.cos();
    let sin_omega = omega_rad.sin();

    let x = sin_i * cos_omega;
    let y = sin_i * sin_omega;
    let z = cos_i;

    PoleVector::new(x, y, z)
}

// ============================================================================
// CLUSTERING ANALYSIS FUNCTIONS
// ============================================================================

/// Calculate mean pole vector from a set of pole vectors
pub fn calculate_mean_pole_vector(pole_vectors: &[PoleVector]) -> (PoleVector, f32) {
    if pole_vectors.is_empty() {
        return (PoleVector::new(0.0, 0.0, 0.0), 0.0);
    }

    // Sum all components
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;

    for pv in pole_vectors {
        sum_x += pv.x;
        sum_y += pv.y;
        sum_z += pv.z;
    }

    // Resultant length (before normalization)
    let resultant_length =
        (sum_x * sum_x + sum_y * sum_y + sum_z * sum_z).sqrt();

    // Normalize to unit vector
    let magnitude = resultant_length / pole_vectors.len() as f32;
    let x = sum_x / pole_vectors.len() as f32;
    let y = sum_y / pole_vectors.len() as f32;
    let z = sum_z / pole_vectors.len() as f32;

    (PoleVector::new(x, y, z), magnitude)
}

/// Calculate clustering strength (R value)
/// R = resultant vector length / n
/// Range: 0 (random) to 1 (perfect alignment)
pub fn calculate_clustering_strength(
    resultant_length: f32,
    n: usize,
) -> f32 {
    if n == 0 {
        0.0
    } else {
        resultant_length / n as f32
    }
}

/// Calculate concentration parameter κ (kappa) for von Mises distribution
pub fn calculate_concentration_parameter(r: f32) -> f32 {
    if r < 0.53 {
        2.0 * r + 8.0 * r.powi(3)
    } else if r < 0.85 {
        0.4 + 1.39 * r / (1.0 - r)
    } else {
        1.0 / (1.0 - r).ln() - 2.0 / (1.0 - r) + 1.0 / (1.0 - r).powi(2)
    }
}

/// Assess clustering significance from R value
pub fn assess_clustering_significance(r: f32) -> (String, f32) {
    if r < 0.3 {
        ("weak".to_string(), 0.2)
    } else if r < 0.5 {
        ("moderate".to_string(), 0.5)
    } else if r < 0.7 {
        ("strong".to_string(), 0.75)
    } else {
        ("very_strong".to_string(), 0.95)
    }
}

/// Calculate angular distance between two pole vectors (in degrees)
pub fn calculate_angular_distance(v1: &PoleVector, v2: &PoleVector) -> f32 {
    let dot_product = v1.dot_product(v2);
    let cos_angle = dot_product.max(-1.0).min(1.0);
    let angle_rad = cos_angle.acos();
    angle_rad * 180.0 / PI
}

/// Calculate mean residual angle from mean pole
pub fn calculate_mean_residual_angle(
    pole_vectors: &[PoleVector],
    mean_pole: &PoleVector,
) -> f32 {
    if pole_vectors.is_empty() {
        return 0.0;
    }

    let sum: f32 = pole_vectors
        .iter()
        .map(|pv| calculate_angular_distance(pv, mean_pole))
        .sum();

    sum / pole_vectors.len() as f32
}

/// Perform complete orbital pole clustering analysis
pub fn analyze_orbital_pole_clustering(
    objects: &[OrbitalElements],
    filter_min_a: f32,
) -> PoleClustering {
    // Filter objects by semi-major axis if requested
    let filtered: Vec<_> = if filter_min_a > 0.0 {
        objects.iter().filter(|obj| obj.a >= filter_min_a).collect()
    } else {
        objects.iter().collect()
    };

    // Convert to pole vectors
    let pole_vectors: Vec<PoleVector> = filtered
        .iter()
        .map(|obj| convert_to_pole_vector(obj))
        .collect();

    // Calculate mean pole
    let (mean_pole, resultant_length) = calculate_mean_pole_vector(&pole_vectors);

    // Calculate clustering metrics
    let n = pole_vectors.len();
    let clustering_strength = calculate_clustering_strength(resultant_length, n);
    let kappa = calculate_concentration_parameter(clustering_strength);
    let mean_residual_angle =
        calculate_mean_residual_angle(&pole_vectors, &mean_pole);
    let (significance, confidence) =
        assess_clustering_significance(clustering_strength);

    // Calculate mean orbital elements
    let mean_inclination: f32 =
        filtered.iter().map(|obj| obj.i).sum::<f32>() / n as f32;
    let mean_omega = calculate_circular_mean(
        &filtered.iter().map(|obj| obj.omega).collect::<Vec<_>>(),
    );

    PoleClustering {
        total_objects: objects.len(),
        clustering_objects: n,
        mean_pole_vector: mean_pole,
        resultant_vector_length: resultant_length,
        clustering_strength,
        mean_residual_angle,
        concentration_parameter: kappa,
        confidence_score: confidence,
        clustering_significance: significance,
        mean_inclination,
        mean_omega,
    }
}

/// Calculate circular mean of angles (in degrees)
fn calculate_circular_mean(angles: &[f32]) -> f32 {
    if angles.is_empty() {
        return 0.0;
    }

    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;

    for &angle in angles {
        let angle_rad = angle * PI / 180.0;
        sin_sum += angle_rad.sin();
        cos_sum += angle_rad.cos();
    }

    let mean_rad = sin_sum.atan2(cos_sum);
    let mean_deg = mean_rad * 180.0 / PI;

    if mean_deg < 0.0 {
        mean_deg + 360.0
    } else {
        mean_deg
    }
}

/// Identify pole clusters in 3D space
pub fn identify_pole_clusters(
    objects: &[OrbitalElements],
    pole_vectors: &[PoleVector],
    cluster_radius: f32,
) -> Vec<ClusterRegion> {
    if pole_vectors.is_empty() {
        return vec![];
    }

    let mut clusters: Vec<ClusterRegion> = vec![];
    let mut used = std::collections::HashSet::new();

    for (i, &pv) in pole_vectors.iter().enumerate() {
        if used.contains(&i) {
            continue;
        }

        let mut cluster = ClusterRegion {
            center: pv,
            object_count: 1,
            objects: vec![objects[i].name.clone()],
            radius: cluster_radius,
            mean_inclination: objects[i].i,
            mean_omega: objects[i].omega,
        };

        used.insert(i);

        // Find neighboring vectors
        for (j, &other) in pole_vectors.iter().enumerate() {
            if used.contains(&j) {
                continue;
            }

            let distance = calculate_angular_distance(&pv, &other);
            if distance < cluster_radius {
                cluster.object_count += 1;
                cluster.objects.push(objects[j].name.clone());
                used.insert(j);
            }
        }

        clusters.push(cluster);
    }

    clusters
}

// ============================================================================
// DISPLAY FUNCTIONS
// ============================================================================

fn format_analysis_results(result: &PoleClustering) -> String {
    let mut output = String::new();

    output.push_str("═══════════════════════════════════════════════════════════════════════\n");
    output.push_str("  ANALYSIS AGENT 15: ORBITAL POLE CLUSTERING\n");
    output.push_str("═══════════════════════════════════════════════════════════════════════\n\n");

    output.push_str(&format!("📊 Analysis Summary:\n"));
    output.push_str(&format!("   Total Objects Analyzed:     {}\n", result.total_objects));
    output.push_str(&format!("   Objects in Clustering Set:  {}\n\n", result.clustering_objects));

    output.push_str(&format!("🧭 Mean Pole Vector:\n"));
    output.push_str(&format!("   X: {:.4}\n", result.mean_pole_vector.x));
    output.push_str(&format!("   Y: {:.4}\n", result.mean_pole_vector.y));
    output.push_str(&format!("   Z: {:.4}\n", result.mean_pole_vector.z));
    output.push_str(&format!("   Magnitude: {:.4}\n\n", result.mean_pole_vector.magnitude));

    output.push_str(&format!("📈 Clustering Metrics:\n"));
    output.push_str(&format!(
        "   Resultant Vector Length:    {:.4}\n",
        result.resultant_vector_length
    ));
    output.push_str(&format!(
        "   Clustering Strength (R):    {:.4}\n",
        result.clustering_strength
    ));
    output.push_str(&format!(
        "   Concentration Parameter κ:  {:.2}\n",
        result.concentration_parameter
    ));
    output.push_str(&format!(
        "   Mean Residual Angle:        {:.2}°\n\n",
        result.mean_residual_angle
    ));

    output.push_str(&format!("⚡ Significance Assessment:\n"));
    output.push_str(&format!(
        "   Clustering Pattern:         {}\n",
        result.clustering_significance.to_uppercase()
    ));
    output.push_str(&format!(
        "   Confidence Score:           {:.1}%\n\n",
        result.confidence_score * 100.0
    ));

    output.push_str(&format!("🎯 Orbital Characteristics:\n"));
    output.push_str(&format!(
        "   Mean Inclination:           {:.2}°\n",
        result.mean_inclination
    ));
    output.push_str(&format!(
        "   Mean Ω (Ascending Node):    {:.2}°\n\n",
        result.mean_omega
    ));

    output.push_str("🔍 INTERPRETATION:\n");
    output.push_str(&format_interpretation(result));

    output.push_str("═══════════════════════════════════════════════════════════════════════\n");

    output
}

fn format_interpretation(result: &PoleClustering) -> String {
    let mut output = String::new();

    match result.clustering_significance.as_str() {
        "very_strong" => {
            output.push_str("   This indicates VERY STRONG orbital pole clustering.\n");
            output.push_str("   • Objects likely share a common dynamical origin\n");
            output.push_str("   • Signature consistent with gravitational shepherding\n");
            output.push_str("   • High probability of unidentified perturber presence\n");
            output.push_str("   • RECOMMENDATION: Perform numerical integration studies\n");
        }
        "strong" => {
            output.push_str("   This indicates STRONG orbital pole clustering.\n");
            output.push_str("   • Significant concentration in orbital pole direction\n");
            output.push_str("   • Objects show coherent dynamical behavior\n");
            output.push_str("   • Possible evidence of planetary perturbation\n");
            output.push_str("   • RECOMMENDATION: Detailed orbital family analysis\n");
        }
        "moderate" => {
            output.push_str("   This indicates MODERATE orbital pole clustering.\n");
            output.push_str("   • Some non-random clustering detected\n");
            output.push_str("   • May indicate partial dynamical coherence\n");
            output.push_str("   • Could represent collisional family or resonance\n");
            output.push_str("   • RECOMMENDATION: Expand dataset for confirmation\n");
        }
        "weak" => {
            output.push_str("   This indicates WEAK orbital pole clustering.\n");
            output.push_str("   • Objects are essentially randomly distributed\n");
            output.push_str("   • No strong evidence of common perturbation\n");
            output.push_str("   • Consider alternative dynamical mechanisms\n");
            output.push_str("   • RECOMMENDATION: Analyze other orbital parameters\n");
        }
        _ => {
            output.push_str("   Unknown significance level\n");
        }
    }

    output.push_str("\n");
    output
}

// ============================================================================
// SAMPLE DATA
// ============================================================================

fn get_sample_data() -> Vec<OrbitalElements> {
    vec![
        OrbitalElements {
            name: "Pluto".to_string(),
            a: 39.59,
            e: 0.2518,
            i: 17.15,
            omega: 110.29,
            w: 113.71,
            q: Some(29.619),
            ad: Some(49.56),
        },
        OrbitalElements {
            name: "Eris".to_string(),
            a: 68.0,
            e: 0.4370,
            i: 43.87,
            omega: 36.03,
            w: 150.73,
            q: Some(38.284),
            ad: Some(97.71),
        },
        OrbitalElements {
            name: "Haumea".to_string(),
            a: 43.01,
            e: 0.1958,
            i: 28.21,
            omega: 121.80,
            w: 240.89,
            q: Some(34.586),
            ad: Some(51.42),
        },
        OrbitalElements {
            name: "Makemake".to_string(),
            a: 45.51,
            e: 0.1604,
            i: 29.03,
            omega: 79.27,
            w: 297.08,
            q: Some(38.210),
            ad: Some(52.81),
        },
        OrbitalElements {
            name: "Gonggong".to_string(),
            a: 66.89,
            e: 0.5032,
            i: 30.87,
            omega: 336.84,
            w: 206.64,
            q: Some(33.235),
            ad: Some(100.55),
        },
        OrbitalElements {
            name: "Sedna".to_string(),
            a: 549.5,
            e: 0.8613,
            i: 11.93,
            omega: 144.48,
            w: 311.01,
            q: Some(76.223),
            ad: Some(1022.86),
        },
        OrbitalElements {
            name: "Quaoar".to_string(),
            a: 43.15,
            e: 0.0358,
            i: 7.99,
            omega: 188.96,
            w: 163.92,
            q: Some(41.601),
            ad: Some(44.69),
        },
        OrbitalElements {
            name: "Orcus".to_string(),
            a: 39.34,
            e: 0.2217,
            i: 20.56,
            omega: 268.39,
            w: 73.72,
            q: Some(30.614),
            ad: Some(48.06),
        },
        OrbitalElements {
            name: "Arawn".to_string(),
            a: 39.21,
            e: 0.1141,
            i: 3.81,
            omega: 144.74,
            w: 101.22,
            q: Some(34.734),
            ad: Some(43.68),
        },
        OrbitalElements {
            name: "Ixion".to_string(),
            a: 39.35,
            e: 0.2442,
            i: 19.67,
            omega: 71.09,
            w: 300.66,
            q: Some(29.740),
            ad: Some(48.96),
        },
        OrbitalElements {
            name: "Huya".to_string(),
            a: 39.21,
            e: 0.2729,
            i: 15.48,
            omega: 169.31,
            w: 67.51,
            q: Some(28.513),
            ad: Some(49.91),
        },
        OrbitalElements {
            name: "Lempo".to_string(),
            a: 39.72,
            e: 0.2298,
            i: 8.40,
            omega: 97.17,
            w: 295.82,
            q: Some(30.591),
            ad: Some(48.85),
        },
    ]
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("╔{}╗", "═".repeat(77));
    println!(
        "║{}  ORBITAL POLE CLUSTERING ANALYSIS - TRANS-NEPTUNIAN OBJECTS{}║",
        " ".repeat(5),
        " ".repeat(6)
    );
    println!("╚{}╝\n", "═".repeat(77));

    // Load sample data
    println!("📥 Loading sample TNO data...");
    let objects = get_sample_data();
    println!("   Loaded {} objects\n", objects.len());

    // Perform analysis
    println!("🔍 Analyzing orbital pole clustering...\n");
    let result = analyze_orbital_pole_clustering(&objects, 0.0);

    // Display results
    println!("{}", format_analysis_results(&result));

    // Convert to pole vectors and identify clusters
    let pole_vectors: Vec<PoleVector> = objects
        .iter()
        .map(|obj| convert_to_pole_vector(obj))
        .collect();

    let clusters =
        identify_pole_clusters(&objects, &pole_vectors, 30.0);

    println!("\n🎯 DETECTED POLE CLUSTERS:\n");
    println!(
        "   Identified {} clusters\n",
        clusters.len()
    );

    for (i, cluster) in clusters.iter().enumerate() {
        println!(
            "   Cluster {}: {} objects",
            i + 1,
            cluster.object_count
        );
        for (j, obj_name) in cluster.objects.iter().enumerate() {
            if j < 5 {
                print!(" {}", obj_name);
            }
        }
        if cluster.objects.len() > 5 {
            print!(" ... and {} more", cluster.objects.len() - 5);
        }
        println!();
    }

    println!(
        "\n═══════════════════════════════════════════════════════════════════════"
    );
    println!("✅ Analysis complete!\n");
}
