/**
 * ANALYSIS AGENT 15: ORBITAL POLE CLUSTERING
 *
 * Analyzes orbital pole vectors for clustering patterns in trans-neptunian objects.
 *
 * Theory:
 * - Orbital pole = vector perpendicular to orbital plane
 * - Direction defined by Ω (longitude of ascending node) and i (inclination)
 * - Objects perturbed by same planet show correlated orbital poles
 * - Clustering strength indicates perturbation significance
 *
 * Conversion Formula:
 * x = sin(i) * cos(Ω)
 * y = sin(i) * sin(Ω)
 * z = cos(i)
 */

export interface OrbitalElements {
  name: string;
  a: number;        // Semi-major axis (AU)
  e: number;        // Eccentricity
  i: number;        // Inclination (degrees)
  omega: number;    // Longitude of ascending node (degrees)
  w: number;        // Argument of perihelion (degrees)
  q?: number;       // Perihelion distance (AU)
  ad?: number;      // Aphelion distance (AU)
}

export interface PoleVector {
  name: string;
  x: number;
  y: number;
  z: number;
  magnitude: number;
}

export interface PoleClustering {
  totalObjects: number;
  clusteringObjects: number;
  meanPoleVector: PoleVector;
  resultantVectorLength: number;
  clusteringStrength: number; // 0-1: 0=random, 1=perfect alignment
  meanResidualAngle: number;  // degrees
  concentrationParameter: number; // κ (kappa) for von Mises distribution
  confidenceScore: number;    // Likelihood of perturbation
  clusteringSignificance: string; // "weak" | "moderate" | "strong" | "very_strong"
}

export interface ClusterRegion {
  centerX: number;
  centerY: number;
  centerZ: number;
  radius: number;
  objectCount: number;
  objects: string[];
  meanInclination: number;
  meanOmega: number;
}

/**
 * Convert orbital elements (Ω, i) to unit pole vector (x, y, z)
 */
export function convertToPoleVector(obj: OrbitalElements): PoleVector {
  // Convert degrees to radians
  const omega_rad = (obj.omega * Math.PI) / 180;
  const i_rad = (obj.i * Math.PI) / 180;

  // Calculate Cartesian coordinates
  const sin_i = Math.sin(i_rad);
  const cos_i = Math.cos(i_rad);
  const cos_omega = Math.cos(omega_rad);
  const sin_omega = Math.sin(omega_rad);

  const x = sin_i * cos_omega;
  const y = sin_i * sin_omega;
  const z = cos_i;

  // Calculate magnitude (should be ~1 for unit vector)
  const magnitude = Math.sqrt(x * x + y * y + z * z);

  return {
    name: obj.name,
    x,
    y,
    z,
    magnitude,
  };
}

/**
 * Calculate mean pole vector from a set of pole vectors
 */
export function calculateMeanPoleVector(
  poleVectors: PoleVector[]
): { mean: PoleVector; resultantLength: number } {
  if (poleVectors.length === 0) {
    return {
      mean: { name: "mean", x: 0, y: 0, z: 0, magnitude: 0 },
      resultantLength: 0,
    };
  }

  // Sum all components
  let sumX = 0,
    sumY = 0,
    sumZ = 0;
  for (const pv of poleVectors) {
    sumX += pv.x;
    sumY += pv.y;
    sumZ += pv.z;
  }

  // Resultant length (before normalization)
  const resultantLength = Math.sqrt(
    sumX * sumX + sumY * sumY + sumZ * sumZ
  );

  // Normalize to unit vector
  const magnitude = resultantLength / poleVectors.length;
  const x = sumX / poleVectors.length;
  const y = sumY / poleVectors.length;
  const z = sumZ / poleVectors.length;

  return {
    mean: {
      name: "mean_pole",
      x,
      y,
      z,
      magnitude,
    },
    resultantLength: magnitude,
  };
}

/**
 * Calculate clustering strength (R value from circular statistics)
 * R = resultant vector length / n
 * Range: 0 (random) to 1 (perfect alignment)
 */
export function calculateClusteringStrength(
  resultantLength: number,
  n: number
): number {
  if (n === 0) return 0;
  return resultantLength / n;
}

/**
 * Calculate concentration parameter κ (kappa) for von Mises distribution
 * Used to quantify clustering strength
 */
export function calculateConcentrationParameter(R: number): number {
  if (R < 0.53) {
    return 2 * R + 8 * R ** 3;
  } else if (R < 0.85) {
    return 0.4 + 1.39 * R / (1 - R);
  } else {
    return Math.log(1 / (1 - R)) - 2 / (1 - R) + 1 / (1 - R) ** 2;
  }
}

/**
 * Assess clustering significance from R value
 */
export function assessClusteringSignificance(
  R: number
): {
  significance: "weak" | "moderate" | "strong" | "very_strong";
  confidence: number;
} {
  if (R < 0.3) {
    return { significance: "weak", confidence: 0.2 };
  } else if (R < 0.5) {
    return { significance: "moderate", confidence: 0.5 };
  } else if (R < 0.7) {
    return { significance: "strong", confidence: 0.75 };
  } else {
    return { significance: "very_strong", confidence: 0.95 };
  }
}

/**
 * Calculate angular distance between two pole vectors
 */
export function calculateAngularDistance(
  v1: PoleVector,
  v2: PoleVector
): number {
  const dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const cosAngle = Math.max(-1, Math.min(1, dotProduct));
  const angleRad = Math.acos(cosAngle);
  return (angleRad * 180) / Math.PI;
}

/**
 * Calculate mean residual angle from mean pole
 */
export function calculateMeanResidualAngle(
  poleVectors: PoleVector[],
  meanPole: PoleVector
): number {
  if (poleVectors.length === 0) return 0;

  let sumAngles = 0;
  for (const pv of poleVectors) {
    sumAngles += calculateAngularDistance(pv, meanPole);
  }

  return sumAngles / poleVectors.length;
}

/**
 * Identify pole clusters in 3D space
 */
export function identifyPoleClusters(
  poleVectors: PoleVector[],
  clusterRadius: number = 30 // degrees
): ClusterRegion[] {
  if (poleVectors.length === 0) return [];

  const clusters: ClusterRegion[] = [];
  const used = new Set<string>();

  for (const pv of poleVectors) {
    if (used.has(pv.name)) continue;

    const cluster: ClusterRegion = {
      centerX: pv.x,
      centerY: pv.y,
      centerZ: pv.z,
      radius: clusterRadius,
      objectCount: 1,
      objects: [pv.name],
      meanInclination: 0,
      meanOmega: 0,
    };

    used.add(pv.name);

    // Find neighboring vectors
    for (const other of poleVectors) {
      if (used.has(other.name)) continue;
      const distance = calculateAngularDistance(pv, other);
      if (distance < clusterRadius) {
        cluster.objectCount++;
        cluster.objects.push(other.name);
        used.add(other.name);
      }
    }

    clusters.push(cluster);
  }

  return clusters;
}

/**
 * Perform complete orbital pole clustering analysis
 */
export function analyzeOrbitalPoleClustering(
  objects: OrbitalElements[],
  filterMinA: number = 0
): PoleClustering {
  // Filter objects by semi-major axis if requested
  const filtered = filterMinA > 0
    ? objects.filter(obj => obj.a >= filterMinA)
    : objects;

  // Convert to pole vectors
  const poleVectors = filtered.map(obj => convertToPoleVector(obj));

  // Calculate mean pole
  const { mean: meanPole, resultantLength } = calculateMeanPoleVector(poleVectors);

  // Calculate clustering metrics
  const n = poleVectors.length;
  const clusteringStrength = calculateClusteringStrength(resultantLength, n);
  const kappa = calculateConcentrationParameter(clusteringStrength);
  const meanResidualAngle = calculateMeanResidualAngle(
    poleVectors,
    meanPole
  );
  const { significance, confidence } = assessClusteringSignificance(
    clusteringStrength
  );

  return {
    totalObjects: objects.length,
    clusteringObjects: filtered.length,
    meanPoleVector: meanPole,
    resultantVectorLength: resultantLength,
    clusteringStrength,
    meanResidualAngle,
    concentrationParameter: kappa,
    confidenceScore: confidence,
    clusteringSignificance: significance,
  };
}

/**
 * Format analysis results for display
 */
export function formatAnalysisResults(result: PoleClustering): string {
  const lines = [
    "═══════════════════════════════════════════════════════════════════════",
    "  ANALYSIS AGENT 15: ORBITAL POLE CLUSTERING",
    "═══════════════════════════════════════════════════════════════════════",
    "",
    `📊 Analysis Summary:`,
    `   Total Objects Analyzed:     ${result.totalObjects}`,
    `   Objects in Clustering Set:  ${result.clusteringObjects}`,
    "",
    `🧭 Mean Pole Vector:`,
    `   X: ${result.meanPoleVector.x.toFixed(4)}`,
    `   Y: ${result.meanPoleVector.y.toFixed(4)}`,
    `   Z: ${result.meanPoleVector.z.toFixed(4)}`,
    `   Magnitude: ${result.meanPoleVector.magnitude.toFixed(4)}`,
    "",
    `📈 Clustering Metrics:`,
    `   Resultant Vector Length:    ${result.resultantVectorLength.toFixed(4)}`,
    `   Clustering Strength (R):    ${result.clusteringStrength.toFixed(4)}`,
    `   Concentration Parameter κ:  ${result.concentrationParameter.toFixed(2)}`,
    `   Mean Residual Angle:        ${result.meanResidualAngle.toFixed(2)}°`,
    "",
    `⚡ Significance Assessment:`,
    `   Clustering Pattern:         ${result.clusteringSignificance.toUpperCase()}`,
    `   Confidence Score:           ${(result.confidenceScore * 100).toFixed(1)}%`,
    "",
    `🔍 Interpretation:`,
    ...formatInterpretation(result),
    "═══════════════════════════════════════════════════════════════════════",
  ];

  return lines.join("\n");
}

function formatInterpretation(result: PoleClustering): string[] {
  const lines: string[] = [];

  switch (result.clusteringSignificance) {
    case "very_strong":
      lines.push(
        "   This indicates VERY STRONG orbital pole clustering.",
        "   • Objects likely share a common dynamical origin",
        "   • Signature consistent with gravitational shepherding",
        "   • High probability of unidentified perturber presence",
        "   • Recommended: Numerical integration and N-body modeling"
      );
      break;
    case "strong":
      lines.push(
        "   This indicates STRONG orbital pole clustering.",
        "   • Significant concentration in orbital pole direction",
        "   • Objects show coherent dynamical behavior",
        "   • Possible evidence of planetary perturbation",
        "   • Recommend deeper investigation of orbital families"
      );
      break;
    case "moderate":
      lines.push(
        "   This indicates MODERATE orbital pole clustering.",
        "   • Some non-random clustering detected",
        "   • May indicate partial dynamical coherence",
        "   • Could represent collisional family or resonance",
        "   • Further analysis with additional data recommended"
      );
      break;
    case "weak":
      lines.push(
        "   This indicates WEAK orbital pole clustering.",
        "   • Objects are essentially randomly distributed",
        "   • No strong evidence of common perturbation",
        "   • Could be statistical scatter or weak influence",
        "   • Consider alternative mechanisms (stellar encounters, etc.)"
      );
      break;
  }

  return lines;
}

/**
 * Generate detailed report with statistics
 */
export function generateDetailedReport(
  objects: OrbitalElements[],
  result: PoleClustering,
  clusters: ClusterRegion[]
): string {
  const lines: string[] = [
    formatAnalysisResults(result),
    "",
    "🎯 DETECTED POLE CLUSTERS:",
    "═══════════════════════════════════════════════════════════════════════",
    "",
  ];

  if (clusters.length === 0) {
    lines.push("   No distinct clusters identified with current parameters.");
  } else {
    for (let i = 0; i < clusters.length; i++) {
      const cluster = clusters[i];
      lines.push(`   Cluster ${i + 1}:`);
      lines.push(`   Center: (${cluster.centerX.toFixed(3)}, ${cluster.centerY.toFixed(3)}, ${cluster.centerZ.toFixed(3)})`);
      lines.push(`   Object Count: ${cluster.objectCount}`);
      lines.push(`   Members: ${cluster.objects.slice(0, 5).join(", ")}${cluster.objects.length > 5 ? "..." : ""}`);
      lines.push("");
    }
  }

  lines.push("═══════════════════════════════════════════════════════════════════════");

  return lines.join("\n");
}
