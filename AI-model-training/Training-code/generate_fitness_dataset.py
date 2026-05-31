import random
import pandas as pd

random.seed(42)

rows = []

user_levels = ["beginner", "intermediate", "advanced"]
exercises = ["squat", "lunge", "bicep_curl", "shoulder_press"]
cycle_phases = ["menstrual", "follicular", "ovulatory", "luteal", "skip"]
fatigue_levels = ["low", "medium", "high"]
performance_trends = ["improving", "stable", "declining"]

target_rows = 600

for _ in range(target_rows):
    user_level = random.choice(user_levels)
    exercise = random.choice(exercises)
    cycle_phase = random.choice(cycle_phases)
    fatigue_level = random.choice(fatigue_levels)
    performance_trend = random.choice(performance_trends)

    form_score = random.randint(40, 100)
    reps = random.randint(3, 20)
    left_angle = round(random.uniform(100, 170), 1)

    # keep right angle close most of the time
    right_angle = round(left_angle + random.uniform(-10, 10), 1)
    symmetry_diff = round(abs(left_angle - right_angle), 1)

    # Clear priority rules for coaching labels
    if cycle_phase in ["menstrual", "luteal"] and fatigue_level == "high" and performance_trend == "declining":
        coaching_label = "prioritise_recovery"

    elif symmetry_diff > 15:
        coaching_label = "improve_symmetry"

    elif form_score < 58:
        coaching_label = "reduce_intensity"

    elif form_score <= 72:
        coaching_label = "focus_control"

    elif form_score >= 82 and fatigue_level == "low":
        coaching_label = "increase_challenge"

    elif form_score >= 78 and fatigue_level in ["low", "medium"]:
        coaching_label = "progress_gradually"

    elif form_score >= 70 and symmetry_diff <= 8:
        coaching_label = "maintain_quality"

    else:
        coaching_label = "improve_range_of_motion"

    rows.append({
        "user_level": user_level,
        "exercise": exercise,
        "cycle_phase": cycle_phase,
        "form_score": form_score,
        "reps": reps,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "symmetry_diff": symmetry_diff,
        "fatigue_level": fatigue_level,
        "performance_trend": performance_trend,
        "coaching_label": coaching_label
    })

df = pd.DataFrame(rows)

print("Generated dataset label counts:")
print(df["coaching_label"].value_counts())

df.to_csv("fitness_coaching_dataset.csv", index=False)

print("\nSaved fitness_coaching_dataset.csv")