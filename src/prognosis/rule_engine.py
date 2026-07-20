from typing import Dict, List, Any


def calculate_prognosis(
    severity: str,
    confidence: float,
    age: int,
    comorbidities: List[str],
    bone_affected: str,
) -> Dict[str, Any]:
    """
    Evaluates clinical prognosis recommendations using a rules-based engine.

    Args:
        severity (str): Predicted severity ('hairline', 'simple', 'displaced', 'comminuted').
        confidence (float): Model classification confidence score (0.0 to 1.0).
        age (int): Patient age in years.
        comorbidities (List[str]): List of comorbidities (e.g., 'Osteoporosis', 'Diabetes').
        bone_affected (str): Bone/body part affected ('hand', 'leg', 'hip', 'shoulder', 'mixed', 'other').

    Returns:
        Dict[str, Any]: Recommendations matching the database prognosis schema:
            - rest_weeks_min (int)
            - rest_weeks_max (int)
            - cast_type (str)
            - plaster_required (bool)
            - weight_bearing_status (str)
            - referral_flag (str: 'conservative' or 'surgical')
    """
    severity_lower = severity.lower()
    bone_lower = bone_affected.lower()
    comorbidities_lower = [c.lower() for c in comorbidities]

    # 1. Determine Referral Flag (Surgical vs Conservative)
    # Comminuted (shattered) and Displaced (misaligned) fractures generally require surgical intervention.
    # Hip/Leg fractures in elderly patients with osteoporosis are also surgical high-risks.
    if severity_lower in ["comminuted", "displaced"]:
        referral_flag = "surgical"
    elif bone_lower in ["hip", "leg"] and age >= 65 and "osteoporosis" in comorbidities_lower:
        referral_flag = "surgical"
    else:
        referral_flag = "conservative"

    # 2. Determine Base Rest Weeks
    if severity_lower == "hairline":
        rest_min, rest_max = 3, 4
    elif severity_lower == "simple":
        rest_min, rest_max = 6, 8
    elif severity_lower == "displaced":
        rest_min, rest_max = 10, 12
    elif severity_lower == "comminuted":
        rest_min, rest_max = 12, 16
    else:
        # Default fallback
        rest_min, rest_max = 4, 6

    # Adjust rest weeks based on patient risk factors:
    # Elderly patients heal slower
    if age > 60:
        rest_min += 2
        rest_max += 2

    # Osteoporosis impairs bone rebuilding
    if "osteoporosis" in comorbidities_lower:
        rest_min += 2
        rest_max += 2

    # Diabetes slows circulation and cellular repair
    if "diabetes" in comorbidities_lower:
        rest_min += 1
        rest_max += 2

    # Lower extremity bone load-bearing increases recovery time
    if bone_lower in ["leg", "hip"] and severity_lower != "hairline":
        rest_min += 2
        rest_max += 2

    # 3. Determine Cast Type and Plaster Requirement
    if referral_flag == "surgical":
        cast_type = "Post-op Rigid Brace"
        plaster_required = False
    else:
        # Conservative treatments
        if severity_lower == "hairline":
            cast_type = "Removable Velcro Splint"
            plaster_required = False
        elif severity_lower == "simple":
            if bone_lower in ["leg", "hip"]:
                cast_type = "Short Leg Plaster Cast"
                plaster_required = True
            elif bone_lower in ["hand", "shoulder"]:
                cast_type = "Short Arm Plaster Cast"
                plaster_required = True
            else:
                cast_type = "Fiberglass Splint"
                plaster_required = False
        else:
            # Displaced/Comminuted undergoing initial conservative treatment
            if bone_lower in ["leg", "hip"]:
                cast_type = "Long Leg Plaster Cast"
                plaster_required = True
            else:
                cast_type = "Long Arm Plaster Cast"
                plaster_required = True

    # 4. Determine Weight Bearing Status
    if bone_lower in ["leg", "hip"]:
        if severity_lower == "hairline":
            weight_bearing_status = "Full weight bearing as tolerated with crutches"
        elif severity_lower == "simple":
            weight_bearing_status = "Partial weight bearing (touch-down) with crutches"
        else:
            weight_bearing_status = "Non-weight bearing (strict wheelchair/crutches)"
    else:
        # Upper extremity
        if severity_lower == "hairline":
            weight_bearing_status = "Full weight bearing (No lifting over 2 lbs)"
        else:
            weight_bearing_status = "Non-weight bearing for affected arm (Strict sling/elevation)"

    return {
        "rest_weeks_min": rest_min,
        "rest_weeks_max": rest_max,
        "cast_type": cast_type,
        "plaster_required": plaster_required,
        "weight_bearing_status": weight_bearing_status,
        "referral_flag": referral_flag,
    }
