from ai_coaching_engine import get_ai_coaching

result = get_ai_coaching(
    user_level="beginner",
    exercise="squat",
    cycle_phase="luteal",
    form_score=65,
    reps=8,
    left_angle=120,
    right_angle=130,
    symmetry_diff=10,
    fatigue_level="high",
    performance_trend="declining"
)

print(result)