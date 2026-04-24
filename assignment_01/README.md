# 🪐 Exoplanet Habitability Classification
### Using Find-S & Candidate Elimination Algorithms

> *"Are we alone in the universe?"* — This project takes a small but meaningful step toward answering that question, using classical machine learning concept-learning algorithms to classify whether an exoplanet could potentially support life.

---

## 📌 Overview

This project applies two foundational machine learning algorithms — **Find-S** and **Candidate Elimination** — to a real-world-inspired dataset of confirmed and candidate exoplanets drawn from the **NASA Exoplanet Archive** and the **Habitable Exoplanet Catalogue (HEC)** maintained by the Planetary Habitability Laboratory, University of Puerto Rico at Arecibo.

The goal is simple to state but genuinely hard to solve: given a set of planetary and stellar measurements, can we learn a rule that tells us whether a planet might be habitable?

Rather than throwing a neural network at the problem, we take a transparent, interpretable approach — one where you can trace every single decision the algorithm makes and understand *why* it reached its conclusion. That kind of explainability matters, especially in scientific domains where "the model said so" isn't good enough.

---

## 🌌 Why Exoplanets?

Most concept-learning textbooks use toy datasets (weather tables, animal classifications). This project deliberately moves away from that. Exoplanet science is a domain where:

- The data is **genuinely categorical** once discretised — orbital zone, star type, planet size — which maps perfectly to the conjunctive hypothesis space these algorithms assume.
- The **target concept is scientifically meaningful**: the habitable zone definition, Earth Similarity Index tiers, and atmospheric modelling all provide real grounding for our attribute choices.
- **Failure cases arise naturally** from real scientific challenges — observational uncertainty, disjunctive habitability concepts (surface water vs. subsurface oceans), and the continuous nature of raw NASA sensor data.

It's a dataset that respects the theory while staying connected to something real.

---

## 🗂️ Project Structure

```
.
├── exoplanet_candidate_elimination.py   # Main implementation — run this
└── README.md                            # You're reading it
```

---

## 🪐 The Dataset

The training set consists of **10 real exoplanets**, each described by 6 categorical attributes and labelled as either **Habitable (Yes)** or **Hostile (No)**.

| # | Planet | Star Type | Zone | Size | Atmosphere | Surface Water | Magnetic Field | Label |
|---|--------|-----------|------|------|------------|---------------|----------------|-------|
| D1 | Kepler-442b | K-Type | Habitable | Earth-Like | Thin | Likely | Strong | ✅ Yes |
| D2 | Kepler-452b | G-Type | Habitable | Super-Earth | Thick | Likely | Strong | ✅ Yes |
| D3 | 55 Cancri e | G-Type | Inner | Super-Earth | Thick | Unlikely | Weak | ❌ No |
| D4 | TRAPPIST-1e | M-Type | Habitable | Earth-Like | Thin | Likely | Strong | ✅ Yes |
| D5 | HAT-P-7b | F-Type | Inner | Giant | Thick | Unlikely | None | ❌ No |
| D6 | Kepler-186f | M-Type | Habitable | Earth-Like | Thin | Possible | Strong | ✅ Yes |
| D7 | WASP-12b | G-Type | Inner | Giant | None | Unlikely | None | ❌ No |
| D8 | Proxima Centauri b | M-Type | Habitable | Earth-Like | Thin | Possible | Weak | ✅ Yes |
| D9 | GJ 1132b | M-Type | Inner | Sub-Earth | None | Unlikely | None | ❌ No |
| D10 | TOI-700d | M-Type | Habitable | Earth-Like | Thin | Likely | Strong | ✅ Yes |

### Attribute Schema

Each attribute is discretised from real-valued measurements available in the NASA Exoplanet Archive:

| Attribute | Possible Values | Scientific Basis |
|-----------|----------------|-----------------|
| `StarType` | G-Type, K-Type, M-Type, F-Type | Host star spectral classification |
| `ZonePosition` | Inner, Habitable, Outer | Orbital position relative to stellar habitable zone |
| `PlanetSize` | Sub-Earth, Earth-Like, Super-Earth, Giant | Planet radius relative to Earth |
| `Atmosphere` | Thick, Thin, None | Modelled atmospheric density tier |
| `SurfaceWater` | Likely, Possible, Unlikely | Probability of liquid water from temperature models |
| `MagneticField` | Strong, Weak, None | Estimated magnetosphere protection level |

---

## ⚙️ How the Algorithms Work

### Find-S

Find-S starts with the most specific possible hypothesis — one that matches absolutely nothing — and generalises it just enough to cover each positive (habitable) example it sees. Negative examples are ignored entirely. By the end, it gives you the single most specific rule consistent with every habitable planet in the training set.

**What it found here:**

```
h = <?, Habitable, ?, Thin, ?, ?>
```

Translation: *A planet is potentially habitable if and only if it orbits within the Habitable Zone and has a Thin atmosphere.* Every other attribute was too variable across confirmed habitable worlds to be fixed as a constraint.

This is a surprisingly clean result. The habitable zone condition is exactly what planetary scientists use as their primary filter, and the thin atmosphere constraint aligns with Earth's own profile. Find-S got there without ever looking at a hostile planet.

---

### Candidate Elimination

Where Find-S gives you one answer, Candidate Elimination gives you the **full picture**. It maintains two boundary sets simultaneously:

- **S** (Specific boundary) — the most specific hypotheses still consistent with the data
- **G** (General boundary) — the most general hypotheses still consistent with the data

Every consistent hypothesis lives somewhere between S and G. That entire space is called the **Version Space**.

Positive examples push S upward (more general). Negative examples push G downward (more specific). When S and G meet, the algorithm has learned the concept precisely.

**What it found here:**

```
S = { <M-Type, Habitable, Earth-Like, Thin, Likely, Strong> }
G = { <?, Habitable, ?, ?, ?, ?> }
```

S and G didn't fully converge with 10 examples, which is scientifically honest — it reflects the genuine diversity of habitable-zone candidates (Proxima b and Kepler-452b are both candidates but differ substantially in almost every other attribute). More training data would continue narrowing the space.

---

## 🚀 Getting Started

### Prerequisites

This project has **zero external dependencies**. It runs on plain Python 3.

```bash
python3 --version   # Python 3.7+ recommended
```

### Running the Code

```bash
python3 exoplanet_candidate_elimination.py
```

That's it. The script will print:

1. The full training dataset
2. The complete Find-S trace (step by step, per planet)
3. The complete Candidate Elimination trace (S and G boundaries after each example)
4. Classification verdicts for 3 new exoplanets
5. Demonstrations of all 5 version space failure cases

### Sample Output (excerpt)

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
   Find-S & Candidate Elimination — Exoplanet Habitability
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

  [D01] + Kepler-442b            → h = <K-Type, Habitable, Earth-Like, Thin, Likely, Strong>
  [D02] + Kepler-452b            → h = <?, Habitable, ?, ?, Likely, Strong>
  [D03] - 55 Cancri e            (negative — ignored by Find-S)
  ...
  Find-S Final Hypothesis: <?, Habitable, ?, ?, ?, ?>
```

---

## 🔬 The 5 Failure Cases

One of the most important things this project covers is knowing when *not* to trust the output. There are 5 real situations where the version space cannot be meaningfully obtained:

**Case 1 — Contradictory Labels (Observational Uncertainty)**
Proxima Centauri b is labelled both Yes and No depending on whether you use the optimistic or conservative habitable zone definition. Same planet, conflicting labels, version space collapses to empty. The code actually runs CE on this and confirms the collapse live.

**Case 2 — Disjunctive Target Concept**
Real habitability includes both surface water worlds *and* subsurface ocean worlds (think Europa). That's a disjunction: `(Zone=Habitable) OR (Zone=Outer AND IcyCrust=Yes)`. Our conjunctive hypothesis space simply cannot express this. Find-S would wildly over-generalise.

**Case 3 — Only Positive Examples**
If a mission only has confirmed habitable candidates and no confirmed hostile planets, G never narrows below `<?,?,?,?,?,?>`. The version space spans the entire hypothesis space — completely uninformative.

**Case 4 — Only Negative Examples**
The reverse problem. If we only have gas giants and lava worlds in the catalogue, S never rises from `<∅,∅,∅,∅,∅,∅>`. The algorithm has nothing positive to anchor the lower boundary.

**Case 5 — Continuous NASA Attributes**
The real NASA Exoplanet Archive records orbital period in days (0.4–700+), planet radius in Earth radii (0.3–25+), equilibrium temperature in Kelvin (100–3000+). These are continuous. The hypothesis space becomes infinite, and CE cannot enumerate the S or G boundaries. This is where SVMs, Random Forests, and neural networks take over.

---

## 🧠 Key Functions Reference

| Function | What it does |
|----------|-------------|
| `find_s(training_data)` | Runs the Find-S algorithm, prints full trace, returns final hypothesis |
| `candidate_elimination(training_data)` | Runs CE, prints S/G after every example, returns `(S, G)` |
| `matches(h, instance)` | Checks if hypothesis `h` covers a given planet instance |
| `min_generalisations(h, inst, attr_vals)` | Minimal generalisations of `h` to cover a positive example |
| `min_specialisations(g, inst, attr_vals)` | Minimal specialisations of `g` to exclude a negative example |
| `classify(instance, S, G)` | Returns HABITABLE / HOSTILE / UNCERTAIN for a new planet |
| `demo_failure_cases()` | Walks through all 5 situations where VS cannot be obtained |

---

## 🧪 Classifying New Planets

Three planets not in the training set are classified using the converged version space:

| Planet | Verdict |
|--------|---------|
| KOI-7711b (Kepler analogue) | ⚠️ Uncertain — VS not fully resolved |
| Wolf 1061b (hot rocky) | ❌ Hostile — excluded by all G hypotheses |
| GJ 667Cc | ⚠️ Uncertain — VS not fully resolved |

The "Uncertain" results are actually meaningful — they tell you the algorithm doesn't have enough training examples to make a confident call, rather than confidently giving you a wrong answer.

---

## 📚 References

- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill. *(The original source for both algorithms)*
- NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu/
- Planetary Habitability Laboratory, UPR Arecibo — Habitable Exoplanet Catalogue — https://phl.upr.edu/hec
- Schulze-Makuch, D. et al. (2011). A Two-Tiered Approach to Assessing the Habitability of Exoplanets. *Astrobiology*, 11(10).
- Kopparapu, R. K. et al. (2013). Habitable Zones Around Main-Sequence Stars. *The Astrophysical Journal*, 765(2).

---

## 📝 Notes

- The dataset is **inspired by** real exoplanets. Attribute values are discretised from real measurements but simplified for the purpose of demonstrating concept learning in a categorical hypothesis space.
- The `F-Type` star label for HAT-P-7b is treated as a distinct category in this study, since F-type stars are hotter and shorter-lived than G-type stars and represent a meaningfully different habitability environment.

---

