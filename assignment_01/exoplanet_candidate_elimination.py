"""
============================================================
  Find-S & Candidate Elimination Algorithm
  Real-World Case Study: EXOPLANET HABITABILITY CLASSIFICATION
  Date: 24th April 2026
============================================================

Domain:
  NASA / Habitable Exoplanet Catalogue (HEC) inspired dataset.
  Goal: Learn a concept rule that classifies whether an exoplanet
  is POTENTIALLY HABITABLE based on 6 categorical attributes.

Attributes:
  1. StarType        : G-Type | K-Type | M-Type | F-Type
  2. ZonePosition    : Inner | Habitable | Outer
  3. PlanetSize      : Sub-Earth | Earth-Like | Super-Earth | Giant
  4. Atmosphere      : Thick | Thin | None
  5. SurfaceWater    : Likely | Possible | Unlikely
  6. MagneticField   : Strong | Weak | None

Target: LoanApproved → Yes (Habitable) | No (Hostile)

Real planets used:
  Kepler-442b, Kepler-452b, 55 Cancri e, TRAPPIST-1e,
  HAT-P-7b, Kepler-186f, WASP-12b, Proxima Centauri b,
  GJ 1132b, TOI-700d
============================================================
"""

from itertools import product

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
ATTRIBUTES = [
    "StarType", "ZonePosition", "PlanetSize",
    "Atmosphere", "SurfaceWater", "MagneticField"
]

# Each row: (StarType, ZonePosition, PlanetSize, Atmosphere, SurfaceWater, MagneticField, Label)
TRAINING_DATA = [
    # ID   Planet                StarType  Zone         Size           Atmos   Water      Mag      Label
    ("K-Type",  "Habitable", "Earth-Like",  "Thin",  "Likely",   "Strong", "Yes"),  # D1  Kepler-442b
    ("G-Type",  "Habitable", "Super-Earth", "Thick", "Likely",   "Strong", "Yes"),  # D2  Kepler-452b
    ("G-Type",  "Inner",     "Super-Earth", "Thick", "Unlikely", "Weak",   "No"),   # D3  55 Cancri e
    ("M-Type",  "Habitable", "Earth-Like",  "Thin",  "Likely",   "Strong", "Yes"),  # D4  TRAPPIST-1e
    ("F-Type",  "Inner",     "Giant",       "Thick", "Unlikely", "None",   "No"),   # D5  HAT-P-7b
    ("M-Type",  "Habitable", "Earth-Like",  "Thin",  "Possible", "Strong", "Yes"),  # D6  Kepler-186f
    ("G-Type",  "Inner",     "Giant",       "None",  "Unlikely", "None",   "No"),   # D7  WASP-12b
    ("M-Type",  "Habitable", "Earth-Like",  "Thin",  "Possible", "Weak",   "Yes"),  # D8  Proxima Centauri b
    ("M-Type",  "Inner",     "Sub-Earth",   "None",  "Unlikely", "None",   "No"),   # D9  GJ 1132b
    ("M-Type",  "Habitable", "Earth-Like",  "Thin",  "Likely",   "Strong", "Yes"),  # D10 TOI-700d
]

PLANET_NAMES = [
    "Kepler-442b", "Kepler-452b", "55 Cancri e", "TRAPPIST-1e",
    "HAT-P-7b", "Kepler-186f", "WASP-12b", "Proxima Centauri b",
    "GJ 1132b", "TOI-700d"
]

NUM_ATTRS = len(ATTRIBUTES)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def hyp_str(h):
    """Pretty-print hypothesis tuple."""
    return "<" + ", ".join("∅" if v is None else v for v in h) + ">"


def matches(h, instance):
    """True if hypothesis h covers the planet instance."""
    for hv, iv in zip(h, instance):
        if hv is None: return False   # ∅ matches nothing
        if hv == "?":  continue       # ? matches anything
        if hv != iv:   return False
    return True


def more_general_or_equal(h1, h2):
    """True if h1 is at least as general as h2."""
    for v1, v2 in zip(h1, h2):
        if v1 == "?":              continue
        if v2 == "?" and v1 != "?": return False
        if v1 != v2:               return False
    return True


def more_specific_or_equal(h1, h2):
    return more_general_or_equal(h2, h1)


def min_generalisations(h, inst, attr_vals):
    """
    Return minimal generalisations of h that cover positive instance inst.
    For each mismatching attribute, either adopt inst's value or widen to '?'.
    """
    mismatches = [i for i in range(NUM_ATTRS) if h[i] != inst[i] and h[i] != "?"]
    results = []
    for combo in product(*[[inst[i], "?"] for i in mismatches]):
        c = list(h)
        for pos, attr_i in enumerate(mismatches):
            c[attr_i] = combo[pos]
        results.append(tuple(c))
    return results or [h]


def min_specialisations(g, inst, attr_vals):
    """
    Return minimal specialisations of g that exclude negative instance inst.
    For each '?' in g, restrict it to any value ≠ inst[i].
    """
    results = []
    for i in range(NUM_ATTRS):
        if g[i] == "?":
            for val in attr_vals[i]:
                if val != inst[i]:
                    s = list(g); s[i] = val
                    results.append(tuple(s))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Find-S Algorithm
# ──────────────────────────────────────────────────────────────────────────────

def find_s(training_data):
    """
    Find-S: Find the most specific hypothesis consistent with all
    positive (habitable) exoplanet examples.
    """
    SEP = "─" * 70
    print(f"\n{SEP}")
    print("  FIND-S ALGORITHM  ─  Exoplanet Habitability")
    print(SEP)
    print(f"\nAttributes : {', '.join(ATTRIBUTES)}")
    print(f"Initial  h : {hyp_str(tuple([None]*NUM_ATTRS))}\n")

    h = tuple([None] * NUM_ATTRS)

    for idx, (example, name) in enumerate(zip(training_data, PLANET_NAMES), 1):
        inst  = example[:-1]
        label = example[-1]

        if label == "Yes":
            print(f"  [D{idx:02d}] + {name:<22} {inst}")
            new_h = list(h)
            for i in range(NUM_ATTRS):
                if new_h[i] is None:
                    new_h[i] = inst[i]
                elif new_h[i] != inst[i]:
                    new_h[i] = "?"
            h = tuple(new_h)
            print(f"          → h = {hyp_str(h)}\n")
        else:
            print(f"  [D{idx:02d}] - {name:<22} (negative — ignored by Find-S)\n")

    print(SEP)
    print(f"  Find-S Final Hypothesis: {hyp_str(h)}")
    print(f"\n  Interpretation:")
    print(f"    A planet is POTENTIALLY HABITABLE iff")
    for i, (attr, val) in enumerate(zip(ATTRIBUTES, h)):
        if val != "?":
            print(f"      {attr} = {val}")
    print(SEP)
    return h


# ──────────────────────────────────────────────────────────────────────────────
# Candidate Elimination Algorithm
# ──────────────────────────────────────────────────────────────────────────────

def candidate_elimination(training_data):
    """
    Candidate Elimination: Maintain S and G boundaries of the version space.
    Uses both positive (habitable) and negative (hostile) exoplanet examples.
    """
    SEP = "─" * 70
    print(f"\n{SEP}")
    print("  CANDIDATE ELIMINATION  ─  Exoplanet Habitability")
    print(SEP)

    # Collect all unique values per attribute for specialisation
    attr_vals = [set() for _ in range(NUM_ATTRS)]
    for ex in training_data:
        for i, v in enumerate(ex[:-1]):
            attr_vals[i].add(v)

    S = {tuple([None] * NUM_ATTRS)}
    G = {tuple(["?"]  * NUM_ATTRS)}

    print(f"\n  Initial S = {{ {hyp_str(list(S)[0])} }}")
    print(f"  Initial G = {{ {hyp_str(list(G)[0])} }}\n")

    for idx, (example, name) in enumerate(zip(training_data, PLANET_NAMES), 1):
        inst  = tuple(example[:-1])
        label = example[-1]
        sign  = "+" if label == "Yes" else "−"

        print(f"  [D{idx:02d}] {sign} {name}")
        print(f"       {inst}")

        if label == "Yes":
            # G: remove hypotheses inconsistent with this positive example
            G = {g for g in G if matches(g, inst)}

            # S: generalise to cover this positive example
            new_S = set()
            for s in S:
                if matches(s, inst):
                    new_S.add(s)
                else:
                    for g_new in min_generalisations(s, inst, attr_vals):
                        if any(more_general_or_equal(g, g_new) for g in G):
                            new_S.add(g_new)
            # Remove overly general members from S
            S = {s for s in new_S
                 if not any(s2 != s and more_general_or_equal(s, s2) for s2 in new_S)}

        else:
            # S: remove hypotheses inconsistent with this negative example
            S = {s for s in S if not matches(s, inst)}

            # G: specialise to exclude this negative example
            new_G = set()
            for g in G:
                if not matches(g, inst):
                    new_G.add(g)
                else:
                    for s_new in min_specialisations(g, inst, attr_vals):
                        if any(more_general_or_equal(s_new, s) for s in S):
                            new_G.add(s_new)
            # Remove overly specific members from G
            G = {g for g in new_G
                 if not any(g2 != g and more_specific_or_equal(g, g2) for g2 in new_G)}

        def fmt_set(boundary):
            if not boundary:
                return "∅  ← VERSION SPACE COLLAPSED"
            return "\n             ".join(hyp_str(h) for h in boundary)

        print(f"       S = {{ {fmt_set(S)} }}")
        print(f"       G = {{ {fmt_set(G)} }}\n")

    print(SEP)
    print("  FINAL VERSION SPACE")
    print(f"    S = {{ {' | '.join(hyp_str(s) for s in S) if S else '∅'} }}")
    print(f"    G = {{ {' | '.join(hyp_str(g) for g in G) if G else '∅'} }}")
    if S and G and S == G:
        print("\n  ✓ CONVERGED  (S = G)")
        print(f"\n  Learned Concept: A planet is POTENTIALLY HABITABLE iff")
        for attr, val in zip(ATTRIBUTES, list(S)[0]):
            if val not in ("?", None):
                print(f"      {attr} = {val}")
    elif not S or not G:
        print("\n  ✗ VERSION SPACE COLLAPSED — concept not learnable from this data")
    else:
        print("\n  ⚠ Partially converged (S ≠ G) — more examples needed")
    print(SEP)
    return S, G


# ──────────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────────

def classify(instance, S, G):
    """
    Classify a new exoplanet using the converged version space.
    Returns HABITABLE, HOSTILE, or UNCERTAIN.
    """
    if not S or not G:
        return "UNKNOWN — version space is empty"
    if all(matches(s, instance) for s in S):
        return "✓  POTENTIALLY HABITABLE  (all S hypotheses agree)"
    if all(not matches(g, instance) for g in G):
        return "✗  HOSTILE  (excluded by all G hypotheses)"
    return "?  UNCERTAIN  (version space not fully resolved for this instance)"


# ──────────────────────────────────────────────────────────────────────────────
# 5 Failure Cases
# ──────────────────────────────────────────────────────────────────────────────

def demo_failure_cases():
    SEP = "═" * 70
    print(f"\n{SEP}")
    print("  5 SITUATIONS WHERE VERSION SPACE CANNOT BE OBTAINED")
    print(f"{SEP}\n")

    cases = [
        (
            "Case 1 — Contradictory Labels (Observational Uncertainty)",
            [
                ("M-Type","Habitable","Earth-Like","Thin","Possible","Weak","Yes"),
                ("M-Type","Habitable","Earth-Like","Thin","Possible","Weak","No"),
            ],
            ["Proxima b (optimistic HZ)", "Proxima b (conservative HZ — tidal lock concern)"],
            "Same planet, conflicting labels due to HZ definition model.\n"
            "→ No hypothesis can be consistent with both.\n"
            "→ Version Space = ∅ (empty — concept unlearnable)"
        ),
        (
            "Case 2 — Disjunctive Target (Surface + Subsurface Habitability)",
            None,
            None,
            "True habitability concept:\n"
            "  (ZonePosition=Habitable) OR (ZonePosition=Outer AND IcyCrust=Yes)\n"
            "Conjunctive H cannot represent disjunctions.\n"
            "→ Find-S over-generalises to <?,...,?> (classifies everything as habitable)\n"
            "→ Version Space = entire H (meaningless)"
        ),
        (
            "Case 3 — Only Positive Examples (No Confirmed Hostile Planets)",
            [
                ("K-Type","Habitable","Earth-Like","Thin","Likely","Strong","Yes"),
                ("G-Type","Habitable","Super-Earth","Thick","Likely","Strong","Yes"),
                ("M-Type","Habitable","Earth-Like","Thin","Likely","Strong","Yes"),
            ],
            ["Kepler-442b","Kepler-452b","TRAPPIST-1e"],
            "G boundary never specialises below <?,...,?>.\n"
            "→ Version Space = entire hypothesis space between S and <?,?,?,?,?,?>\n"
            "→ Cannot distinguish habitable from gas giants — not informative"
        ),
        (
            "Case 4 — Only Negative Examples (No Confirmed Habitable Planet)",
            [
                ("G-Type","Inner","Super-Earth","Thick","Unlikely","Weak","No"),
                ("F-Type","Inner","Giant","Thick","Unlikely","None","No"),
                ("G-Type","Inner","Giant","None","Unlikely","None","No"),
                ("M-Type","Inner","Sub-Earth","None","Unlikely","None","No"),
            ],
            ["55 Cancri e","HAT-P-7b","WASP-12b","GJ 1132b"],
            "S stays at <∅,∅,∅,∅,∅,∅> — never generalised.\n"
            "G shrinks but S has no lower bound.\n"
            "→ Any new planet predicted UNKNOWN (S matches nothing)\n"
            "→ Version Space = ∅ from below — no learnable positive concept"
        ),
        (
            "Case 5 — Continuous Real-Valued NASA Attributes",
            None,
            None,
            "Real NASA Exoplanet Archive attributes:\n"
            "  OrbitalPeriod: 0.4 – 700+ days (continuous)\n"
            "  PlanetRadius : 0.3 – 25+ R⊕   (continuous)\n"
            "  EqTemperature: 100 – 3000+ K   (continuous)\n"
            "  StellarLumin : 0.001 – 1000+ L☉(continuous)\n\n"
            "Infinite threshold combinations → S and G have infinite members.\n"
            "CE cannot enumerate boundaries.\n"
            "→ Version Space = mathematically defined but computationally unobtainable\n"
            "→ Modern approaches (SVMs, Random Forests) handle this instead"
        ),
    ]

    for i, (title, data, names, explanation) in enumerate(cases, 1):
        print(f"  {'─'*66}")
        print(f"  {title}")
        print(f"  {'─'*66}")

        if data:
            print(f"\n  Training examples:")
            for j, (ex, nm) in enumerate(zip(data, names), 1):
                label = "✓ YES" if ex[-1] == "Yes" else "✗ NO"
                print(f"    D{j}: {nm:<30} → {label}")
                print(f"         {ex[:-1]}")

        print(f"\n  Analysis:\n")
        for line in explanation.split('\n'):
            print(f"    {line}")

        if data and title.startswith("Case 1"):
            # Actually run CE to show collapse
            print(f"\n  Running CE to demonstrate collapse...")
            attr_vals = [set() for _ in range(NUM_ATTRS)]
            for ex in data:
                for k, v in enumerate(ex[:-1]):
                    attr_vals[k].add(v)
            S = {tuple([None]*NUM_ATTRS)}
            G = {tuple(["?"]*NUM_ATTRS)}
            for ex in data:
                inst  = tuple(ex[:-1])
                label = ex[-1]
                if label == "Yes":
                    G = {g for g in G if matches(g, inst)}
                    new_S = set()
                    for s in S:
                        if matches(s, inst):
                            new_S.add(s)
                        else:
                            for gn in min_generalisations(s, inst, attr_vals):
                                if any(more_general_or_equal(g, gn) for g in G):
                                    new_S.add(gn)
                    S = {s for s in new_S if not any(
                        s2!=s and more_general_or_equal(s,s2) for s2 in new_S)}
                else:
                    S = {s for s in S if not matches(s, inst)}
                    new_G = set()
                    for g in G:
                        if not matches(g, inst):
                            new_G.add(g)
                        else:
                            for sn in min_specialisations(g, inst, attr_vals):
                                if any(more_general_or_equal(sn,s) for s in S):
                                    new_G.add(sn)
                    G = {g for g in new_G if not any(
                        g2!=g and more_specific_or_equal(g,g2) for g2 in new_G)}
            s_out = "{" + ", ".join(hyp_str(s) for s in S) + "}" if S else "∅"
            g_out = "{" + ", ".join(hyp_str(g) for g in G) + "}" if G else "∅"
            print(f"    Final S = {s_out}")
            print(f"    Final G = {g_out}")
            if not S:
                print("    ✗ CONFIRMED: S collapsed — Version Space = ∅")

        print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BANNER = "★" * 70
    print(f"\n{BANNER}")
    print("   Find-S & Candidate Elimination — Exoplanet Habitability")
    print(f"{BANNER}")

    # Print dataset
    print("\nTraining Dataset (NASA/HEC inspired):")
    print(f"  {'#':<4} {'Planet':<24} {'StarType':<9} {'Zone':<11} {'Size':<13} "
          f"{'Atm':<7} {'Water':<10} {'Mag':<8} Label")
    print("  " + "─" * 96)
    for i, (ex, nm) in enumerate(zip(TRAINING_DATA, PLANET_NAMES), 1):
        print(f"  D{i:<3} {nm:<24} {ex[0]:<9} {ex[1]:<11} {ex[2]:<13} "
              f"{ex[3]:<7} {ex[4]:<10} {ex[5]:<8} {ex[6]}")

    # Run algorithms
    h_finds = find_s(TRAINING_DATA)
    S_final, G_final = candidate_elimination(TRAINING_DATA)

    # Classify new planets
    print("\n" + "═" * 70)
    print("  CLASSIFYING NEW EXOPLANETS")
    print("═" * 70)
    new_planets = [
        (("K-Type","Habitable","Earth-Like","Thin","Likely","Strong"), "KOI-7711b (Kepler analogue)"),
        (("M-Type","Inner","Sub-Earth","None","Unlikely","None"),       "Wolf 1061b (hot rocky)"),
        (("G-Type","Habitable","Super-Earth","Thick","Possible","Weak"),"GJ 667Cc"),
    ]
    for inst, name in new_planets:
        result = classify(inst, S_final, G_final)
        print(f"\n  Planet : {name}")
        print(f"  Attrs  : {inst}")
        print(f"  Verdict: {result}")

    # Failure cases
    demo_failure_cases()

    print(f"\n{'★'*70}")
    print("  Execution Complete")
    print(f"{'★'*70}\n")
