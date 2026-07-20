from dataclasses import dataclass
from typing import List, Dict, Tuple


# ------------------------------------------------------------------------------
# Dataclass for Prognosis Recommendations
# ------------------------------------------------------------------------------
@dataclass
class PrognosisResult:
    """
    Dataclass representing the prognosis results for a patient's bone fracture.

    Attributes:
        rest_weeks_min (int): Minimum recommended rest duration in weeks.
        rest_weeks_max (int): Maximum recommended rest duration in weeks.
        cast_type (str): Type of cast, splint, or support recommended.
        plaster_required (bool): Flag indicating if a plaster/rigid cast is required.
        weight_bearing_status (str): Clinically indicated weight-bearing restriction.
        referral_flag (str): Recommended intervention pathway ('conservative' or 'surgical').
    """

    rest_weeks_min: int
    rest_weeks_max: int
    cast_type: str
    plaster_required: bool
    weight_bearing_status: str
    referral_flag: str


# ------------------------------------------------------------------------------
# AO Foundation Orthopedic Base Configurations
# ------------------------------------------------------------------------------
# Base healing times [min, max] weeks per bone and severity class
BASE_HEALING_WEEKS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "distal_radius": {
        "hairline": (3, 4),
        "simple": (6, 8),
        "displaced": (8, 10),
        "comminuted": (10, 12),
    },
    "clavicle": {
        "hairline": (3, 4),
        "simple": (6, 8),
        "displaced": (8, 10),
        "comminuted": (10, 12),
    },
    "ankle": {
        "hairline": (4, 6),
        "simple": (6, 8),
        "displaced": (8, 12),
        "comminuted": (12, 16),
    },
    "femur": {
        "hairline": (6, 8),
        "simple": (8, 12),
        "displaced": (12, 16),
        "comminuted": (16, 24),
    },
    "humerus": {
        "hairline": (3, 5),
        "simple": (6, 8),
        "displaced": (8, 12),
        "comminuted": (12, 16),
    },
    "metatarsal": {
        "hairline": (3, 4),
        "simple": (6, 8),
        "displaced": (8, 10),
        "comminuted": (10, 12),
    },
}

# Cast lookup type strings per bone and severity class
CAST_LOOKUP: Dict[str, Dict[str, str]] = {
    "distal_radius": {
        "hairline": "Removable Wrist Splint",
        "simple": "Short Arm Fiberglass Cast",
        "displaced": "Sugar-tong Splint (Pre-op)",
        "comminuted": "External Fixator / Rigid Post-op Splint",
    },
    "clavicle": {
        "hairline": "Figure-of-eight Sling",
        "simple": "Standard Shoulder Sling",
        "displaced": "Standard Shoulder Sling",
        "comminuted": "Surgical Clavicle Brace",
    },
    "ankle": {
        "hairline": "Controlled Ankle Movement (CAM) Boot",
        "simple": "Short Leg Fiberglass Cast",
        "displaced": "Short Leg Plaster Cast",
        "comminuted": "Post-op Rigid Leg Splint",
    },
    "femur": {
        "hairline": "Long Leg Knee Immobilizer",
        "simple": "Knee Immobilizer Splint",
        "displaced": "Skeletal Traction / Post-op Knee Brace",
        "comminuted": "Skeletal Traction / Post-op Knee Brace",
    },
    "humerus": {
        "hairline": "Coaptation Splint",
        "simple": "Sarmiento Brace",
        "displaced": "Hanging Arm Plaster Cast",
        "comminuted": "Post-op Shoulder Immobilizer Sling",
    },
    "metatarsal": {
        "hairline": "Post-op Stiff-soled Shoe",
        "simple": "Short Leg Walking Boot",
        "displaced": "Short Leg Plaster Cast (Non-weight bearing)",
        "comminuted": "Short Leg Plaster Cast (Non-weight bearing)",
    },
}


def get_prognosis(
    bone: str, severity: str, age: int, comorbidities: List[str]
) -> PrognosisResult:
    """
    Computes recovery prognosis recommendations using AO Foundation guidelines and modifier metrics.

    Args:
        bone (str): Affected bone (distal_radius, clavicle, ankle, femur, humerus, metatarsal).
        severity (str): Fracture severity (hairline, simple, displaced, comminuted).
        age (int): Patient age in years.
        comorbidities (List[str]): List of comorbidities (e.g., 'Osteoporosis', 'Diabetes').

    Returns:
        PrognosisResult: Structured clinical prognosis recommendations.
    """
    bone_clean = bone.strip().lower()
    severity_clean = severity.strip().lower()
    comorbidities_clean = [c.strip().lower() for c in comorbidities]

    # Validation / Fallbacks for inputs
    if bone_clean not in BASE_HEALING_WEEKS:
        bone_clean = "distal_radius"
    if severity_clean not in BASE_HEALING_WEEKS[bone_clean]:
        severity_clean = "simple"

    # 1. Fetch Base Values
    base_min, base_max = BASE_HEALING_WEEKS[bone_clean][severity_clean]
    cast_type = CAST_LOOKUP[bone_clean][severity_clean]

    # 2. Compute Modifiers (additively adjust healing speed multiplier)
    age_modifier = 0.0
    if age >= 60:
        age_modifier = 0.30
    elif age >= 40:
        age_modifier = 0.15
    elif age <= 14:
        age_modifier = -0.10

    comorbidity_modifier = 0.0
    has_osteoporosis = any("osteoporosis" in c for c in comorbidities_clean)
    has_diabetes = any("diabetes" in c for c in comorbidities_clean)

    if has_osteoporosis and has_diabetes:
        comorbidity_modifier = 0.45
    elif has_osteoporosis:
        comorbidity_modifier = 0.25
    elif has_diabetes:
        comorbidity_modifier = 0.20

    total_multiplier = 1.0 + age_modifier + comorbidity_modifier

    # Apply multiplier and round to nearest integer
    rest_weeks_min = max(1, round(base_min * total_multiplier))
    rest_weeks_max = max(1, round(base_max * total_multiplier))

    # 3. Determine Plaster Requirement
    # Plaster is typically required for conservative treatment of simple/displaced casts.
    plaster_required = "plaster" in cast_type.lower() or (
        "cast" in cast_type.lower() and severity_clean in ["simple", "displaced"]
    )

    # 4. Determine Weight-Bearing Status
    # Lower limb load-bearing bones: ankle, femur, metatarsal
    if bone_clean in ["ankle", "femur", "metatarsal"]:
        if severity_clean == "hairline":
            weight_bearing_status = "Full weight-bearing as tolerated"
        elif severity_clean == "simple":
            weight_bearing_status = (
                "Partial weight-bearing (touch-down) with assistive device"
            )
        else:
            weight_bearing_status = "Non-weight-bearing (strict wheelchair/crutches)"
    else:
        # Upper limb bones: distal_radius, clavicle, humerus
        weight_bearing_status = (
            "Upper extremity: Non-weight-bearing (No lifting/pushing)"
        )

    # 5. Determine Referral Flag (Surgical vs Conservative)
    # Displaced/comminuted bone fractures require reduction (surgery).
    # Femur fractures in adults almost always require surgical stabilization.
    if severity_clean in ["displaced", "comminuted"]:
        referral_flag = "surgical"
    elif bone_clean == "femur" and age > 14:
        referral_flag = "surgical"
    else:
        referral_flag = "conservative"

    return PrognosisResult(
        rest_weeks_min=rest_weeks_min,
        rest_weeks_max=rest_weeks_max,
        cast_type=cast_type,
        plaster_required=plaster_required,
        weight_bearing_status=weight_bearing_status,
        referral_flag=referral_flag,
    )


# ------------------------------------------------------------------------------
# Demonstration / Example Outputs
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        # Example 1: Tommy Shelby (14yo, hairline, wrist/distal_radius)
        {
            "bone": "distal_radius",
            "severity": "hairline",
            "age": 14,
            "comorbidities": [],
        },
        # Example 2: Arthur Dent (75yo, simple, leg/ankle, osteoporosis + diabetes)
        {
            "bone": "ankle",
            "severity": "simple",
            "age": 75,
            "comorbidities": ["Osteoporosis", "Diabetes"],
        },
        # Example 3: Bruce Wayne (45yo, displaced, femur, diabetes)
        {
            "bone": "femur",
            "severity": "displaced",
            "age": 45,
            "comorbidities": ["Diabetes"],
        },
    ]

    print("=" * 80)
    print("AO FOUNDATION FRACTURE PROGNOSIS GENERATION TESTS")
    print("=" * 80)

    for i, ex in enumerate(examples, 1):
        res = get_prognosis(ex["bone"], ex["severity"], ex["age"], ex["comorbidities"])
        print(
            f"\nExample {i}: Patient Profile - Age {ex['age']}, Bone: {ex['bone']}, Severity: {ex['severity']}, Comorbidities: {ex['comorbidities']}"
        )
        print(
            f"  -> Rest Weeks:       {res.rest_weeks_min} - {res.rest_weeks_max} weeks"
        )
        print(f"  -> Cast Type:        {res.cast_type}")
        print(f"  -> Plaster Required: {res.plaster_required}")
        print(f"  -> Weight Bearing:   {res.weight_bearing_status}")
        print(f"  -> Intervention:     {res.referral_flag.upper()} Referral")
        print("-" * 50)
    print("=" * 80)
