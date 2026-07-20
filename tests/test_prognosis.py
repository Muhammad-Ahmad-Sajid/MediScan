import sys
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prognosis_engine import get_prognosis, PrognosisResult


@pytest.fixture
def default_patient_data():
    """Returns a fixture representing a healthy young adult (25yo, no comorbidities)."""
    return {"age": 25, "comorbidities": []}


# 1. Parameterized matrix of all 6 bones x 4 severities
@pytest.mark.parametrize(
    "bone", ["distal_radius", "clavicle", "ankle", "femur", "humerus", "metatarsal"]
)
@pytest.mark.parametrize("severity", ["hairline", "simple", "displaced", "comminuted"])
def test_all_bones_and_severities(bone, severity, default_patient_data):
    """Verifies that all combinations return a valid PrognosisResult with min < max rest weeks."""
    res = get_prognosis(bone, severity, **default_patient_data)

    assert isinstance(res, PrognosisResult)
    assert res.rest_weeks_min <= res.rest_weeks_max
    # Verify that rest weeks are positive integers
    assert res.rest_weeks_min >= 1
    assert isinstance(res.cast_type, str)
    assert isinstance(res.weight_bearing_status, str)
    assert res.referral_flag in ["conservative", "surgical"]


# 2. Age modifier increase test (65yo vs 30yo)
def test_age_modifier_increases_rest_time():
    """Verifies that age >= 60 correctly increases healing weeks compared to age 30."""
    res_65 = get_prognosis("distal_radius", "simple", age=65, comorbidities=[])
    res_30 = get_prognosis("distal_radius", "simple", age=30, comorbidities=[])

    assert res_65.rest_weeks_min > res_30.rest_weeks_min
    assert res_65.rest_weeks_max > res_30.rest_weeks_max


# 3. Comorbidity stacking (Osteoporosis + Diabetes)
def test_comorbidity_stacking_modifiers():
    """Verifies that osteoporosis (25%) and diabetes (20%) modifiers stack correctly to 45%."""
    res_none = get_prognosis("distal_radius", "simple", age=30, comorbidities=[])
    res_osteo = get_prognosis(
        "distal_radius", "simple", age=30, comorbidities=["Osteoporosis"]
    )
    res_diab = get_prognosis(
        "distal_radius", "simple", age=30, comorbidities=["Diabetes"]
    )
    res_both = get_prognosis(
        "distal_radius", "simple", age=30, comorbidities=["Osteoporosis", "Diabetes"]
    )

    # 6 base weeks:
    # none: 1.0x -> 6
    # diabetes: 1.2x -> 7.2 -> 7
    # osteoporosis: 1.25x -> 7.5 -> 8
    # both: 1.45x -> 8.7 -> 9
    assert res_both.rest_weeks_min > res_osteo.rest_weeks_min
    assert res_both.rest_weeks_min > res_diab.rest_weeks_min
    assert res_osteo.rest_weeks_min > res_none.rest_weeks_min
    assert res_diab.rest_weeks_min > res_none.rest_weeks_min

    assert res_both.rest_weeks_min == 9
    assert res_osteo.rest_weeks_min == 8
    assert res_diab.rest_weeks_min == 7
    assert res_none.rest_weeks_min == 6


# 4. Plaster required is False for hairline clavicle fracture
def test_hairline_clavicle_sling_only():
    """Verifies that a hairline clavicle fracture does not require rigid casting."""
    res = get_prognosis("clavicle", "hairline", age=30, comorbidities=[])
    assert res.plaster_required is False
    assert "sling" in res.cast_type.lower()


# 5. Referral flag is surgical for comminuted femur fractures
def test_comminuted_femur_is_surgical():
    """Verifies that a comminuted femur fracture always flags for surgical intervention."""
    res = get_prognosis("femur", "comminuted", age=30, comorbidities=[])
    assert res.referral_flag == "surgical"


# 6. Weight bearing status check for femur fractures
def test_femur_weight_bearing_status(default_patient_data):
    """Verifies weight bearing restrictions for femur fractures and checks for non-weight bearing statuses."""
    # Hairline femur
    res_hairline = get_prognosis("femur", "hairline", **default_patient_data)
    assert res_hairline.weight_bearing_status is not None
    assert "Full weight-bearing" in res_hairline.weight_bearing_status

    # Simple femur
    res_simple = get_prognosis("femur", "simple", **default_patient_data)
    assert res_simple.weight_bearing_status is not None
    assert "Partial weight-bearing" in res_simple.weight_bearing_status

    # Displaced femur (Non-weight-bearing)
    res_displaced = get_prognosis("femur", "displaced", **default_patient_data)
    assert res_displaced.weight_bearing_status is not None
    assert "Non-weight-bearing" in res_displaced.weight_bearing_status
