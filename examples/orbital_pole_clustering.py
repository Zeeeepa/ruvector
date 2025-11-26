#!/usr/bin/env python3
"""
ANALYSIS AGENT 15: ORBITAL POLE CLUSTERING

Analyzes orbital pole vectors for clustering patterns in trans-neptunian objects.
Provides comprehensive visualization and statistical analysis of orbital pole
distribution and clustering strength as evidence of planetary perturbation.

Theory:
- Orbital pole = unit vector perpendicular to orbital plane
- Direction determined by Ω (longitude of ascending node) and i (inclination)
- Objects sharing same perturber show correlated orbital poles
- Clustering strength quantified by R-value and von Mises concentration parameter

Run with:
    python3 orbital_pole_clustering.py
"""

import csv
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys
from enum import Enum


class SignificanceLevel(Enum):
    """Enum for clustering significance levels"""
    WEAK = ("weak", 0.2)
    MODERATE = ("moderate", 0.5)
    STRONG = ("strong", 0.75)
    VERY_STRONG = ("very_strong", 0.95)


@dataclass
class OrbitalElements:
    """Orbital elements for a trans-neptunian object"""
    name: str
    a: float        # Semi-major axis (AU)
    e: float        # Eccentricity
    i: float        # Inclination (degrees)
    omega: float    # Longitude of ascending node (degrees)
    w: float        # Argument of perihelion (degrees)
    q: Optional[float] = None    # Perihelion distance (AU)
    ad: Optional[float] = None   # Aphelion distance (AU)

    def to_pole_vector(self) -> List[float]:
        """Convert orbital elements to unit pole vector"""
        # Convert degrees to radians
        omega_rad = math.radians(self.omega)
        i_rad = math.radians(self.i)

        # Calculate Cartesian coordinates
        x = math.sin(i_rad) * math.cos(omega_rad)
        y = math.sin(i_rad) * math.sin(omega_rad)
        z = math.cos(i_rad)

        return [x, y, z]


@dataclass
class PoleClustering:
    """Results of orbital pole clustering analysis"""
    total_objects: int
    clustering_objects: int
    mean_pole_vector: List[float]
    resultant_vector_length: float
    clustering_strength: float  # R value: 0-1
    mean_residual_angle: float  # degrees
    concentration_parameter: float  # κ (kappa)
    confidence_score: float  # 0-1
    clustering_significance: str  # "weak" | "moderate" | "strong" | "very_strong"
    mean_inclination: float
    mean_omega: float


class OrbitalPoleAnalyzer:
    """Analyzes orbital pole clustering in TNO populations"""

    def __init__(self):
        self.objects: List[OrbitalElements] = []
        self.pole_vectors: Dict[str, np.ndarray] = {}
        self.results: Optional[PoleClustering] = None

    def add_object(self, obj: OrbitalElements) -> None:
        """Add an object to the analysis"""
        self.objects.append(obj)
        self.pole_vectors[obj.name] = obj.to_pole_vector()

    def add_from_dict(self, obj_dict: Dict) -> None:
        """Add object from dictionary"""
        obj = OrbitalElements(
            name=obj_dict.get('name'),
            a=float(obj_dict.get('a')),
            e=float(obj_dict.get('e')),
            i=float(obj_dict.get('i')),
            omega=float(obj_dict.get('omega')),
            w=float(obj_dict.get('w')),
            q=float(obj_dict.get('q')) if obj_dict.get('q') else None,
            ad=float(obj_dict.get('ad')) if obj_dict.get('ad') else None,
        )
        self.add_object(obj)

    def calculate_mean_pole(self) -> Tuple[List[float], float]:
        """Calculate mean pole vector and resultant length"""
        if not self.pole_vectors:
            return [0, 0, 0], 0.0

        # Sum all vectors
        vector_list = list(self.pole_vectors.values())
        vector_sum = [
            sum(v[0] for v in vector_list),
            sum(v[1] for v in vector_list),
            sum(v[2] for v in vector_list),
        ]

        # Resultant length
        vector_norm = math.sqrt(
            sum(x**2 for x in vector_sum)
        )
        resultant_length = vector_norm / len(self.pole_vectors)

        # Normalize
        if vector_norm > 0:
            mean_pole = [x / vector_norm for x in vector_sum]
        else:
            mean_pole = [0, 0, 0]

        return mean_pole, resultant_length

    def calculate_clustering_strength(self, mean_pole: List[float]) -> float:
        """Calculate R-value (clustering strength)"""
        if len(self.pole_vectors) == 0:
            return 0.0

        resultant_length = 0
        for vec in self.pole_vectors.values():
            # Project onto mean direction
            dot_product = sum(v * m for v, m in zip(vec, mean_pole))
            resultant_length += dot_product

        return resultant_length / len(self.pole_vectors)

    def calculate_concentration_parameter(self, R: float) -> float:
        """Calculate κ (kappa) for von Mises distribution"""
        if R < 0.53:
            return 2 * R + 8 * R**3
        elif R < 0.85:
            return 0.4 + 1.39 * R / (1 - R)
        else:
            return math.log(1 / (1 - R)) - 2 / (1 - R) + 1 / (1 - R)**2

    def assess_significance(self, R: float) -> Tuple[str, float]:
        """Assess clustering significance from R value"""
        if R < 0.3:
            return "weak", 0.2
        elif R < 0.5:
            return "moderate", 0.5
        elif R < 0.7:
            return "strong", 0.75
        else:
            return "very_strong", 0.95

    def calculate_angular_distance(
        self, v1: List[float], v2: List[float]
    ) -> float:
        """Calculate angular distance between two vectors (degrees)"""
        # Normalize vectors
        v1_norm_val = math.sqrt(sum(x**2 for x in v1))
        v2_norm_val = math.sqrt(sum(x**2 for x in v2))

        if v1_norm_val == 0 or v2_norm_val == 0:
            return 0.0

        v1_norm = [x / v1_norm_val for x in v1]
        v2_norm = [x / v2_norm_val for x in v2]

        # Calculate angle
        dot_product = sum(v * u for v, u in zip(v1_norm, v2_norm))
        dot_product = max(-1, min(1, dot_product))
        angle_rad = math.acos(dot_product)
        return math.degrees(angle_rad)

    def calculate_mean_residual_angle(self, mean_pole: List[float]) -> float:
        """Calculate mean residual angle from mean pole"""
        if not self.pole_vectors:
            return 0.0

        angles = []
        for vec in self.pole_vectors.values():
            angle = self.calculate_angular_distance(vec, mean_pole)
            angles.append(angle)

        return sum(angles) / len(angles) if angles else 0.0

    def analyze(self, filter_min_a: float = 0.0) -> PoleClustering:
        """Perform complete orbital pole clustering analysis"""
        # Filter objects if requested
        if filter_min_a > 0:
            filtered_objects = [obj for obj in self.objects if obj.a >= filter_min_a]
            filtered_vectors = {
                obj.name: self.pole_vectors[obj.name]
                for obj in filtered_objects
            }
        else:
            filtered_objects = self.objects
            filtered_vectors = self.pole_vectors

        n = len(filtered_vectors)

        # Calculate mean pole
        mean_pole, resultant_length = self.calculate_mean_pole()

        # Calculate clustering metrics
        clustering_strength = self.calculate_clustering_strength(mean_pole)
        kappa = self.calculate_concentration_parameter(clustering_strength)
        mean_residual = self.calculate_mean_residual_angle(mean_pole)
        significance, confidence = self.assess_significance(clustering_strength)

        # Calculate mean orbital elements
        inclinations = [obj.i for obj in filtered_objects]
        mean_inclination = sum(inclinations) / len(inclinations) if inclinations else 0.0
        mean_omega = self._calculate_circular_mean(
            [obj.omega for obj in filtered_objects]
        )

        results = PoleClustering(
            total_objects=len(self.objects),
            clustering_objects=n,
            mean_pole_vector=mean_pole,
            resultant_vector_length=resultant_length,
            clustering_strength=clustering_strength,
            mean_residual_angle=mean_residual,
            concentration_parameter=kappa,
            confidence_score=confidence,
            clustering_significance=significance,
            mean_inclination=mean_inclination,
            mean_omega=mean_omega,
        )

        self.results = results
        return results

    @staticmethod
    def _calculate_circular_mean(angles: List[float]) -> float:
        """Calculate circular mean of angles (in degrees)"""
        angles_rad = [math.radians(a) for a in angles]
        sin_sum = sum(math.sin(a) for a in angles_rad)
        cos_sum = sum(math.cos(a) for a in angles_rad)
        mean_rad = math.atan2(sin_sum, cos_sum)
        return math.degrees(mean_rad) % 360

    def identify_clusters(
        self, cluster_radius: float = 30.0
    ) -> List[Dict]:
        """Identify clusters of pole vectors in 3D space"""
        if not self.pole_vectors:
            return []

        clusters = []
        used = set()

        vectors_list = list(self.pole_vectors.items())

        for name, vector in vectors_list:
            if name in used:
                continue

            cluster = {
                "center": vector.copy(),
                "objects": [name],
                "mean_inclination": next(
                    obj.i for obj in self.objects if obj.name == name
                ),
            }

            used.add(name)

            # Find neighboring vectors
            for other_name, other_vector in vectors_list:
                if other_name in used:
                    continue

                distance = self.calculate_angular_distance(vector, other_vector)
                if distance < cluster_radius:
                    cluster["objects"].append(other_name)
                    used.add(other_name)

            clusters.append(cluster)

        return clusters

    def format_report(self) -> str:
        """Format analysis results as readable report"""
        if not self.results:
            return "No analysis results available. Run analyze() first."

        result = self.results
        report = []

        report.append("╔" + "═" * 77 + "╗")
        report.append("║" + " " * 15 + "ANALYSIS AGENT 15: ORBITAL POLE CLUSTERING" + " " * 19 + "║")
        report.append("╚" + "═" * 77 + "╝")
        report.append("")

        report.append("📊 ANALYSIS SUMMARY:")
        report.append(f"   Total Objects Analyzed:     {result.total_objects}")
        report.append(f"   Objects in Analysis Set:    {result.clustering_objects}")
        report.append("")

        report.append("🧭 MEAN POLE VECTOR:")
        report.append(f"   X: {result.mean_pole_vector[0]:8.4f}")
        report.append(f"   Y: {result.mean_pole_vector[1]:8.4f}")
        report.append(f"   Z: {result.mean_pole_vector[2]:8.4f}")
        report.append("")

        report.append("📈 CLUSTERING METRICS:")
        report.append(f"   Resultant Vector Length:    {result.resultant_vector_length:.4f}")
        report.append(f"   Clustering Strength (R):    {result.clustering_strength:.4f}")
        report.append(f"   Concentration Parameter κ:  {result.concentration_parameter:.2f}")
        report.append(f"   Mean Residual Angle:        {result.mean_residual_angle:.2f}°")
        report.append("")

        report.append("⚡ SIGNIFICANCE ASSESSMENT:")
        report.append(f"   Clustering Pattern:         {result.clustering_significance.upper()}")
        report.append(
            f"   Confidence Score:           {result.confidence_score * 100:.1f}%"
        )
        report.append("")

        report.append("🎯 ORBITAL CHARACTERISTICS:")
        report.append(f"   Mean Inclination:           {result.mean_inclination:.2f}°")
        report.append(f"   Mean Ω (Ascending Node):    {result.mean_omega:.2f}°")
        report.append("")

        # Interpretation
        report.append("🔍 INTERPRETATION:")
        report.extend(self._format_interpretation(result))

        report.append("═" * 79)

        return "\n".join(report)

    @staticmethod
    def _format_interpretation(result: PoleClustering) -> List[str]:
        """Generate interpretation text based on results"""
        lines = []

        if result.clustering_significance == "very_strong":
            lines.append(
                "   This indicates VERY STRONG orbital pole clustering."
            )
            lines.append("   • Objects likely share a common dynamical origin")
            lines.append(
                "   • Signature consistent with gravitational shepherding"
            )
            lines.append("   • High probability of unidentified perturber presence")
            lines.append(
                "   • RECOMMENDATION: Perform numerical integration studies"
            )
        elif result.clustering_significance == "strong":
            lines.append("   This indicates STRONG orbital pole clustering.")
            lines.append("   • Significant concentration in orbital pole direction")
            lines.append("   • Objects show coherent dynamical behavior")
            lines.append("   • Possible evidence of planetary perturbation")
            lines.append("   • RECOMMENDATION: Detailed orbital family analysis")
        elif result.clustering_significance == "moderate":
            lines.append("   This indicates MODERATE orbital pole clustering.")
            lines.append("   • Some non-random clustering detected")
            lines.append("   • May indicate partial dynamical coherence")
            lines.append("   • Could represent collisional family or resonance")
            lines.append("   • RECOMMENDATION: Expand dataset for confirmation")
        elif result.clustering_significance == "weak":
            lines.append("   This indicates WEAK orbital pole clustering.")
            lines.append("   • Objects are essentially randomly distributed")
            lines.append("   • No strong evidence of common perturbation")
            lines.append("   • Consider alternative dynamical mechanisms")
            lines.append(
                "   • RECOMMENDATION: Analyze other orbital parameters"
            )

        return lines


def create_sample_data() -> List[OrbitalElements]:
    """Create sample TNO data for analysis"""
    return [
        OrbitalElements("Pluto", 39.59, 0.2518, 17.15, 110.29, 113.71),
        OrbitalElements("Eris", 68.0, 0.4370, 43.87, 36.03, 150.73),
        OrbitalElements("Haumea", 43.01, 0.1958, 28.21, 121.80, 240.89),
        OrbitalElements("Makemake", 45.51, 0.1604, 29.03, 79.27, 297.08),
        OrbitalElements("Gonggong", 66.89, 0.5032, 30.87, 336.84, 206.64),
        OrbitalElements("Sedna", 549.5, 0.8613, 11.93, 144.48, 311.01),
        OrbitalElements("Quaoar", 43.15, 0.0358, 7.99, 188.96, 163.92),
        OrbitalElements("Orcus", 39.34, 0.2217, 20.56, 268.39, 73.72),
        OrbitalElements("Arawn", 39.21, 0.1141, 3.81, 144.74, 101.22),
        OrbitalElements("Ixion", 39.35, 0.2442, 19.67, 71.09, 300.66),
        OrbitalElements("Huya", 39.21, 0.2729, 15.48, 169.31, 67.51),
        OrbitalElements("Lempo", 39.72, 0.2298, 8.40, 97.17, 295.82),
        OrbitalElements("Achlys", 39.63, 0.1748, 13.55, 251.87, 14.40),
        OrbitalElements("Albion", 44.2, 0.0725, 2.19, 359.47, 6.89),
        OrbitalElements("Varuna", 43.18, 0.0525, 17.14, 97.21, 273.22),
    ]


def main():
    """Main analysis routine"""
    print("╔" + "═" * 77 + "╗")
    print(
        "║" + " " * 10 + "ORBITAL POLE CLUSTERING ANALYSIS - TRANS-NEPTUNIAN OBJECTS" + " " * 6 + "║"
    )
    print("╚" + "═" * 77 + "╝\n")

    # Create analyzer
    analyzer = OrbitalPoleAnalyzer()

    # Load sample data
    print("📥 Loading sample TNO data...")
    sample_data = create_sample_data()
    for obj in sample_data:
        analyzer.add_object(obj)
    print(f"   Loaded {len(sample_data)} objects\n")

    # Perform analysis
    print("🔍 Analyzing orbital pole clustering...")
    results = analyzer.analyze()

    # Display results
    print("\n" + analyzer.format_report() + "\n")

    # Identify clusters
    print("🎯 DETECTING POLE CLUSTERS:\n")
    clusters = analyzer.identify_clusters(cluster_radius=30.0)
    print(f"   Identified {len(clusters)} clusters\n")

    for i, cluster in enumerate(clusters, 1):
        print(f"   Cluster {i}:")
        print(f"   Objects ({len(cluster['objects'])}): {', '.join(cluster['objects'][:5])}")
        if len(cluster['objects']) > 5:
            print(f"             ... and {len(cluster['objects']) - 5} more")
        print()

    # Generate JSON report
    json_report = {
        "analysis": "orbital_pole_clustering",
        "results": {
            "total_objects": results.total_objects,
            "clustering_objects": results.clustering_objects,
            "mean_pole_vector": results.mean_pole_vector,
            "resultant_vector_length": float(results.resultant_vector_length),
            "clustering_strength": float(results.clustering_strength),
            "mean_residual_angle": float(results.mean_residual_angle),
            "concentration_parameter": float(results.concentration_parameter),
            "confidence_score": float(results.confidence_score),
            "clustering_significance": results.clustering_significance,
            "mean_inclination": float(results.mean_inclination),
            "mean_omega": float(results.mean_omega),
        },
        "clusters": clusters,
    }

    # Save JSON report
    with open("/tmp/orbital_pole_clustering_results.json", "w") as f:
        json.dump(json_report, f, indent=2)
    print("📊 Results saved to /tmp/orbital_pole_clustering_results.json\n")

    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
