import joblib
import random
import pandas as pd

# load saved model files
model = joblib.load("ai_coach_model.pkl")
encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")


def normalise_cycle_phase(cycle_phase):
    if not cycle_phase:
        return "skip"

    cycle_phase = cycle_phase.lower().strip()

    aliases = {
        "menstration": "menstrual",
        "menstruation": "menstrual",
        "menstrual": "menstrual",
        "period": "menstrual",
        "ovulation": "ovulatory",
        "ovulatory": "ovulatory",
        "follicular": "follicular",
        "luteal": "luteal",
        "skip": "skip",
        "none": "skip"
    }

    return aliases.get(cycle_phase, "skip")


responses = {
    "reduce_intensity": [
        "Your form is dropping slightly, so reduce the intensity and focus on controlled movement.",
        "Slow the pace down for now. Your body positioning suggests quality should come before pushing harder.",
        "This session would benefit from lowering intensity and prioritising safe technique."
    ],
    "focus_control": [
        "Focus on slower, more controlled reps. Try not to rush through the movement.",
        "Your next goal should be control and consistency rather than speed.",
        "Keep the movement steady and focus on smooth technique from start to finish."
    ],
    "improve_symmetry": [
        "There is a noticeable difference between sides, so focus on keeping both sides balanced.",
        "Try to keep your left and right movement more even throughout each rep.",
        "Your symmetry could improve, so slow down and check that both sides are working equally."
    ],
    "increase_challenge": [
        "Your form is strong, so you may be ready to increase the challenge gradually.",
        "You are moving well. Consider progressing slightly next session if this still feels comfortable.",
        "Your performance suggests you can safely add a little more difficulty."
    ],
    "prioritise_recovery": [
        "Based on your fatigue and cycle context, a recovery-focused approach may be more suitable today.",
        "Your body may benefit from a lighter session, longer rest, and lower intensity movement.",
        "Prioritise recovery today. Focus on mobility, controlled reps, and listening to your energy levels."
    ],
    "maintain_quality": [
        "Your performance is stable. Keep this quality and avoid rushing the next set.",
        "You are working at a good level. Maintain this form and build consistency.",
        "Your technique looks steady, so continue focusing on quality reps."
    ],
    "progress_gradually": [
        "You are showing good consistency, so gradual progression would be appropriate.",
        "Keep building steadily. You can progress, but avoid increasing intensity too quickly.",
        "Your performance is improving, so aim for small, controlled progress next session."
    ],
    "improve_range_of_motion": [
        "Try to increase your range of motion slightly while keeping control.",
        "Focus on moving through a fuller range without sacrificing form.",
        "Your next focus should be improving movement depth and joint range safely."
    ]
}


cycle_notes = {
    "menstrual": [
        "Cycle-aware adjustment: because this is the menstrual phase, the session should prioritise comfort, controlled movement, hydration, and reduced intensity if cramps, fatigue, or low energy are present.",
        "Cycle-aware adjustment: during the menstrual phase, this system avoids pushing progression too aggressively and instead recommends technique quality, longer rest, and listening to fatigue signals."
    ],
    "luteal": [
        "Cycle-aware adjustment: because this is the luteal phase, the coaching focus shifts towards stability, control, and recovery, as some users may experience higher fatigue or reduced coordination.",
        "Cycle-aware adjustment: in the luteal phase, the system recommends avoiding sudden intensity jumps and focusing on steady reps, balance, and recovery between sets."
    ],
    "follicular": [
        "Cycle-aware adjustment: because this is the follicular phase, gradual progression may be suitable if form remains strong and fatigue stays low.",
        "Cycle-aware adjustment: in the follicular phase, the system allows more progression-focused coaching, but only if technique and recovery indicators remain positive."
    ],
    "ovulatory": [
        "Cycle-aware adjustment: because this is the ovulatory phase, higher intensity may be appropriate, but the system still monitors joint control, balance, and form quality.",
        "Cycle-aware adjustment: during the ovulatory phase, the system may support increased challenge when form is strong, while still avoiding unsafe movement patterns."
    ],
    "skip": [""]
}


def safe_encode(column, value):
    encoder = encoders[column]

    if value not in encoder.classes_:
        value = encoder.classes_[0]

    return encoder.transform([value])[0]


def get_ai_coaching(
    user_level,
    exercise,
    cycle_phase,
    form_score,
    reps,
    left_angle,
    right_angle,
    symmetry_diff,
    fatigue_level,
    performance_trend
):
    cycle_phase = normalise_cycle_phase(cycle_phase)

    input_data = pd.DataFrame([{
        "user_level": safe_encode("user_level", user_level),
        "exercise": safe_encode("exercise", exercise),
        "cycle_phase": safe_encode("cycle_phase", cycle_phase),
        "form_score": form_score,
        "reps": reps,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "symmetry_diff": symmetry_diff,
        "fatigue_level": safe_encode("fatigue_level", fatigue_level),
        "performance_trend": safe_encode("performance_trend", performance_trend)
    }])

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    coaching_label = encoders["coaching_label"].inverse_transform([prediction])[0]

    # stronger cycle-aware adjustment
    if cycle_phase == "menstrual" and fatigue_level in ["medium", "high"]:
        coaching_label = "prioritise_recovery"

    elif cycle_phase == "luteal" and performance_trend == "declining":
        coaching_label = "focus_control"

    elif cycle_phase == "follicular" and form_score >= 80 and fatigue_level == "low":
        coaching_label = "progress_gradually"

    elif cycle_phase == "ovulatory" and form_score >= 85 and fatigue_level == "low":
        coaching_label = "increase_challenge"

    feedback = random.choice(responses.get(
        coaching_label,
        ["Focus on controlled movement and maintain safe form."]
    ))

    cycle_note = random.choice(cycle_notes.get(cycle_phase, [""]))

    full_feedback = f"{feedback} {cycle_note}".strip()

    return {
        "coaching_label": coaching_label,
        "feedback": full_feedback
    }