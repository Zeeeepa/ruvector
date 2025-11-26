# Orbital Pole Clustering - Technical Specification

**Agent:** Analysis Agent 15
**Purpose:** Detect and quantify orbital pole clustering in TNO populations
**Algorithms:** Spherical vector statistics, von Mises distribution fitting
**Output Formats:** JSON, formatted text reports, statistical summaries

---

## 1. Mathematical Formulations

### 1.1 Pole Vector Conversion

**Input:** Orbital elements (Ω, i)
- Ω: Longitude of ascending node (degrees, 0-360)
- i: Inclination (degrees, 0-180)

**Conversion to Cartesian (x, y, z):**

```
Convert to radians:
  Ω_rad = Ω × π/180
  i_rad = i × π/180

Calculate components:
  x = sin(i_rad) × cos(Ω_rad)
  y = sin(i_rad) × sin(Ω_rad)
  z = cos(i_rad)

Normalize to unit length:
  magnitude = √(x² + y² + z²)
  x_norm = x / magnitude
  y_norm = y / magnitude
  z_norm = z / magnitude
```

**Output:** Unit pole vector **v** = [x_norm, y_norm, z_norm]

**Validity Conditions:**
- Magnitude should equal 1.0 (within floating-point precision)
- Components should satisfy: -1 ≤ x, y, z ≤ 1
- If x² + y² > sin²(i), orbital plane is tilted

---

### 1.2 Mean Pole Vector Calculation

**Input:** Set of unit vectors {v₁, v₂, ..., vₙ}

**Vector Sum:**
```
S_x = Σ v_i,x  (sum of x-components)
S_y = Σ v_i,y  (sum of y-components)
S_z = Σ v_i,z  (sum of z-components)

Vector sum magnitude:
||S|| = √(S_x² + S_y² + S_z²)
```

**Resultant Vector Length (R-value):**
```
R = ||S|| / n

Where n = number of objects
```

**Mean Pole Direction (normalized):**
```
v_mean,x = S_x / ||S||
v_mean,y = S_y / ||S||
v_mean,z = S_z / ||S||
```

**Physical Interpretation:**
- R = 0: Poles uniformly distributed (random)
- R = 1: All poles perfectly aligned
- Intermediate R: Partial clustering

---

### 1.3 Clustering Strength Assessment

**Clustering Strength = R-value** (from circular statistics)

```
Classification thresholds:
  R < 0.3:        WEAK         (confidence: 20%)
  0.3 ≤ R < 0.5:  MODERATE     (confidence: 50%)
  0.5 ≤ R < 0.7:  STRONG       (confidence: 75%)
  R ≥ 0.7:        VERY_STRONG  (confidence: 95%)
```

**Sample Size Correction:**

For small samples (n < 25), apply Rayleigh correction:

```
R_corrected = R × (1 - (3/(8κ)))  [for κ large]
```

---

### 1.4 Concentration Parameter (κ - Kappa)

The von Mises concentration parameter quantifies clustering strength.

**Three-region approximation:**

```
Region 1: R < 0.53
  κ = 2R + 8R³

Region 2: 0.53 ≤ R < 0.85
  κ = 0.4 + 1.39R/(1-R)

Region 3: R ≥ 0.85
  κ = ln(1/(1-R)) - 2/(1-R) + 1/(1-R)²
```

**Interpretation:**
- κ = 0: Uniform distribution
- κ = 1: Modest clustering
- κ = 10: Strong clustering
- κ = 100+: Extreme clustering (highly significant)

**Inverse Formula** (to recover R from κ):

```
For κ < 2:  R ≈ √(κ/(2-κ))
For κ ≥ 2:  R ≈ κ/(4 + κ - √(4 + 8κ))
```

---

### 1.5 Angular Distance Metric

**Input:** Two unit vectors v₁ and v₂

**Dot Product:**
```
cos_angle = v₁ · v₂ = v₁,x × v₂,x + v₁,y × v₂,y + v₁,z × v₂,z

Clamp to valid range:
cos_angle = max(-1, min(1, cos_angle))
```

**Angular Distance (radians):**
```
θ = arccos(cos_angle)
```

**Convert to Degrees:**
```
θ_degrees = θ × 180/π
```

**Properties:**
- 0° ≤ θ ≤ 180°
- 0° = perfect alignment
- 90° = perpendicular
- 180° = opposite directions

---

### 1.6 Mean Residual Angle

**Input:** Set of poles {v₁, ..., vₙ} and mean pole v_mean

**Residual Angles:**
```
For each object i:
  θ_i = angular_distance(v_i, v_mean)
```

**Mean Residual Angle:**
```
θ_mean = (Σ θ_i) / n
```

**Interpretation:**
- Small θ_mean (e.g., 10°): Tight clustering
- Large θ_mean (e.g., 45°): Loose clustering
- θ_mean ≈ 60°: Random (expected for uniform distribution)

---

### 1.7 Circular Mean Angle

For circular/spherical data (e.g., Ω values), must use circular mean.

**Standard Mean (WRONG for circular data):**
```
θ_simple = Σ θ_i / n
[Problem: atan(0°) + tan(350°) ≈ 175°, not 0°]
```

**Circular Mean (CORRECT):**
```
sin_sum = Σ sin(θ_i [rad])
cos_sum = Σ cos(θ_i [rad])

θ_circular = atan2(sin_sum, cos_sum)  [in radians]
θ_circular_deg = θ_circular × 180/π

Normalize to 0-360:
if θ_circular_deg < 0:
  θ_circular_deg += 360
```

---

## 2. Cluster Detection Algorithm

### 2.1 Distance-Based Clustering

**Method:** Simple greedy clustering using angular distance threshold

**Algorithm:**

```
INPUT: poles[] = array of pole vectors
       threshold = angular distance threshold (degrees, e.g., 30°)

OUTPUT: clusters[] = list of clusters

clusters = []
used = empty set

for i = 0 to poles.length:
  if i in used:
    continue

  cluster = {center: poles[i], members: [i]}
  used.add(i)

  for j = i+1 to poles.length:
    if j in used:
      continue

    distance = angular_distance(poles[i], poles[j])
    if distance < threshold:
      cluster.members.add(j)
      used.add(j)

  clusters.append(cluster)

return clusters
```

**Complexity:** O(n²) for n objects

**Threshold Selection:**
- 20°: Fine structure, many small clusters
- 30°: Balance, clear families
- 45°: Coarse structure, few large clusters

---

### 2.2 Cluster Statistics

For each cluster:

```
cluster_count = number of members

for each member j in cluster:
  inclination_j = get_inclination(object_j)
  omega_j = get_omega(object_j)

mean_inclination = Σ inclination_j / cluster_count
mean_omega = circular_mean([omega_j for all members])

cluster_vector_sum = Σ v_j (for all members)
cluster_concentration = ||cluster_vector_sum|| / cluster_count
```

---

## 3. Significance Testing

### 3.1 Rayleigh Test

Tests null hypothesis: "poles are uniformly distributed"

**Test Statistic:**
```
Z = n × R²
```

Where n = sample size, R = clustering strength

**P-value** (for large n):
```
p = exp(-Z) × (1 + (2Z - Z²)/(4n) - ...)
```

**Interpretation:**
- p < 0.001: Highly significant clustering
- p < 0.05: Significant clustering
- p > 0.05: Not significant (random distribution)

**Decision Rule:**
```
if p < 0.05:
  clustering is statistically significant
else:
  cannot reject null hypothesis of random distribution
```

---

### 3.2 Bootstrap Confidence Intervals

**Method:** Resample data with replacement, recalculate R each time

```
INPUT: poles[], num_iterations = 10000

bootstrap_R = []

for k = 0 to num_iterations:
  sample = random_sample_with_replacement(poles, n=len(poles))
  R_k = calculate_clustering_strength(sample)
  bootstrap_R.append(R_k)

sort(bootstrap_R)

confidence_interval_lower = bootstrap_R[2.5th percentile]
confidence_interval_upper = bootstrap_R[97.5th percentile]

return (lower, upper)
```

---

### 3.3 Monte Carlo Test

**Method:** Generate random orbital pole distributions, compare statistics

```
INPUT: observed_poles[], num_iterations = 1000

null_statistics = []

for k = 0 to num_iterations:
  random_poles = generate_random_poles(n=len(observed_poles))
  R_null = calculate_clustering_strength(random_poles)
  null_statistics.append(R_null)

R_observed = calculate_clustering_strength(observed_poles)

p_value = (number of R_null > R_observed) / num_iterations

return p_value
```

---

## 4. Filter Parameters

### 4.1 Semi-Major Axis Filter

**Purpose:** Focus analysis on distant objects (likely Planet Nine targets)

```
INPUT: objects[], min_a = threshold (e.g., 100 AU)

OUTPUT: filtered[] = objects where a ≥ min_a

for obj in objects:
  if obj.a >= min_a:
    filtered.append(obj)

return filtered
```

**Common Thresholds:**
- min_a = 0: All objects (default)
- min_a = 50: Beyond classical belt
- min_a = 100: Well beyond Neptune
- min_a = 250: Extreme objects (ETNO)

---

### 4.2 Data Quality Filters

```
Valid object criteria:
  - Orbital period > 0 (well-defined orbit)
  - 0 ≤ e < 1 (elliptical orbit)
  - 0 ≤ i ≤ 180 (valid inclination)
  - 0 ≤ Ω < 360 (valid ascending node)
  - 0 ≤ w < 360 (valid perihelion argument)
  - a > 0 (positive semi-major axis)
  - Uncertainty in a ≤ 1 AU (reasonable accuracy)
```

---

## 5. Output Formats

### 5.1 JSON Report Schema

```json
{
  "analysis": "orbital_pole_clustering",
  "metadata": {
    "timestamp": "2025-11-26T00:00:00Z",
    "version": "1.0",
    "data_source": "NASA JPL SBDB"
  },
  "parameters": {
    "filter_min_a": 0.0,
    "cluster_radius_degrees": 30.0,
    "total_objects_loaded": 1234,
    "objects_analyzed": 1200
  },
  "results": {
    "total_objects": 1200,
    "clustering_objects": 1200,
    "mean_pole_vector": [0.01, 0.12, 0.99],
    "resultant_vector_length": 0.94,
    "clustering_strength": 0.9415,
    "mean_residual_angle": 16.98,
    "concentration_parameter": 260.52,
    "confidence_score": 0.95,
    "clustering_significance": "very_strong",
    "mean_inclination": 17.99,
    "mean_omega": 109.61,
    "statistical_tests": {
      "rayleigh_z": 1330.8,
      "rayleigh_p_value": 2.3e-289,
      "monte_carlo_p_value": 0.001
    }
  },
  "clusters": [
    {
      "id": 1,
      "object_count": 850,
      "center": [0.01, 0.12, 0.99],
      "mean_inclination": 15.2,
      "mean_omega": 108.5,
      "members": ["Pluto", "Eris", ...]
    }
  ],
  "interpretation": {
    "summary": "Very strong orbital pole clustering detected...",
    "confidence_level": "95%",
    "physical_interpretation": "Evidence consistent with planetary perturbation...",
    "recommendations": [...]
  }
}
```

### 5.2 Text Report Format

```
═══════════════════════════════════════════════════════════════════════
  ANALYSIS AGENT 15: ORBITAL POLE CLUSTERING
═══════════════════════════════════════════════════════════════════════

📊 Analysis Summary:
   Total Objects Analyzed:     1200
   Objects in Clustering Set:  1200

🧭 Mean Pole Vector:
   X: 0.0112
   Y: 0.1224
   Z: 0.9924
   Magnitude: 1.0000

📈 Clustering Metrics:
   Resultant Vector Length:    0.9415
   Clustering Strength (R):    0.9415
   Concentration Parameter κ:  260.52
   Mean Residual Angle:        16.98°

⚡ Significance Assessment:
   Clustering Pattern:         VERY_STRONG
   Confidence Score:           95.0%

🎯 Orbital Characteristics:
   Mean Inclination:           17.99°
   Mean Ω (Ascending Node):    109.61°

🔍 Statistical Tests:
   Rayleigh Test Z:            1330.80
   Rayleigh p-value:           < 0.001
   Monte Carlo p-value:        0.001

═══════════════════════════════════════════════════════════════════════
```

---

## 6. Implementation Notes

### 6.1 Floating-Point Precision

**Considerations:**
- Use double-precision (64-bit) for accuracy
- Trigonometric functions sensitive to input precision
- Comparison of floating-point values use tolerance (e.g., 1e-10)

**Critical Operations:**
```
// Vector normalization
magnitude = sqrt(x² + y² + z²)
if magnitude < 1e-10:
  error("zero-magnitude vector")

// Angle calculation
cos_angle = max(-1.0, min(1.0, dot_product))  // Clamp
angle = acos(cos_angle)  // Avoid numerical errors
```

---

### 6.2 Edge Cases

```
1. Empty dataset (n = 0):
   - Return all metrics as 0 or NaN
   - Significance = "undefined"

2. Single object (n = 1):
   - R = 1.0 (perfect "clustering")
   - κ = ∞
   - Not statistically meaningful

3. Two objects (n = 2):
   - Maximum possible R = 1.0
   - κ potentially very large
   - Small sample correction needed

4. Collinear vectors:
   - R = 1.0, κ = ∞
   - Indicates perfect alignment (unlikely in real data)

5. Antipodal vectors:
   - Can cause cancellation in vector sum
   - Results in artificially low R
   - Detect and flag in output
```

---

### 6.3 Numerical Stability Issues

**Problem:** Small sample with high κ can cause overflow in κ calculation

**Solution:**
```
if r >= 0.9999:
  κ = 1e6  (cap at large value)

if r <= 0.0001:
  κ = 0    (cap at small value)
```

**Problem:** arccos() sensitive to floating-point errors

**Solution:**
```
// Instead of arccos(x) for x near ±1:
if x > 0.9999:
  θ = arccos(0.9999)
if x < -0.9999:
  θ = arccos(-0.9999)
```

---

## 7. Performance Characteristics

### 7.1 Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Pole conversion | O(n) | n trigonometric functions |
| Mean pole | O(n) | n vector additions |
| Clustering strength | O(n) | n dot products |
| Concentration parameter | O(1) | Closed-form formula |
| Cluster detection | O(n²) | Compare all pairs |
| Angular distances | O(n) | Per reference object |

**Total:** O(n²) dominated by cluster detection

### 7.2 Memory Usage

```
Array sizes:
  poles: n × 3 × 8 bytes = 24n bytes
  distances: n × n × 4 bytes = 4n² bytes (for full distance matrix)

For n = 10,000:
  poles: 240 KB
  distances: 400 MB (if computed all at once)

Optimized (compute on-demand): Just 240 KB
```

### 7.3 Typical Runtime (Python)

| Dataset Size | Runtime |
|---|---|
| 100 objects | < 1 ms |
| 1,000 objects | ~10 ms |
| 10,000 objects | ~1 second |
| 100,000 objects | ~100 seconds |

---

## 8. Validation Tests

### 8.1 Unit Tests

```python
# Test 1: Known distributions
def test_perfect_alignment():
    # All poles identical
    R = calculate_clustering_strength(poles)
    assert abs(R - 1.0) < 1e-6

# Test 2: Random distribution
def test_random_poles():
    poles = [random_unit_vector() for _ in range(1000)]
    R = calculate_clustering_strength(poles)
    assert R < 0.3  # Should be very small

# Test 3: Angular distance metric
def test_angular_distance():
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    d = angular_distance(v1, v2)
    assert abs(d - 90.0) < 1e-6  # Should be 90 degrees

# Test 4: Circular mean
def test_circular_mean():
    angles = [10, 20, 30]  # Degrees
    mean = circular_mean(angles)
    assert abs(mean - 20) < 1e-6
```

---

## 9. Future Enhancements

### 9.1 Planned Features

1. **Improved Clustering**
   - K-means clustering
   - Hierarchical clustering with dendrogram
   - DBSCAN with automatic epsilon selection

2. **Advanced Statistics**
   - Hypothesis testing for multiple clusters
   - Bayesian inference of number of clusters
   - Uncertainty propagation from orbital element errors

3. **Visualization**
   - 3D pole vector visualization (HEALPix)
   - Mollweide projection of pole distribution
   - Interactive cluster exploration

4. **Integration**
   - Direct SQL database queries
   - Streaming data support
   - API for external tools

---

**Last Updated:** 2025-11-26
**Version:** 1.0
**Status:** Production Ready
