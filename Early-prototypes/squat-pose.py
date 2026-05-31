import cv2
import mediapipe as mp
import numpy as np
import requests
from collections import Counter

# Mediapipe helpers for drawing landmarks and using pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Config for each exercise
# Includes which landmarks to track, rep thresholds, and whether it's upper or lower body
EXERCISE_CONFIG = {
    1: {  # sumo squat
        "name": "Sumo Squat",
        "left_landmarks": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "right_landmarks": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        "up_threshold": 165,
        "down_threshold": 120,
        "type": "lower_body"
    },
    2: {  # squat
        "name": "Squat",
        "left_landmarks": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "right_landmarks": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        "up_threshold": 165,
        "down_threshold": 120,
        "type": "lower_body"
    },
    3: {  # lunge
        "name": "Lunge",
        "left_landmarks": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "right_landmarks": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        "up_threshold": 170,
        "down_threshold": 110,
        "type": "lower_body"
    },
    4: {  # bicep curl
        "name": "Bicep Curl",
        "left_landmarks": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "right_landmarks": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        "up_threshold": 155,
        "down_threshold": 60,
        "type": "upper_body"
    },
    5: {  # shoulder press
        "name": "Shoulder Press",
        "left_landmarks": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "right_landmarks": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        "up_threshold": 160,
        "down_threshold": 80,
        "type": "upper_body"
    }
}

API_BASE_URL = "http://127.0.0.1:8000"
CAMERA_INDEX = 0


def get_auth_headers(token):
    # Build auth header for protected FastAPI routes
    return {
        "Authorization": f"Bearer {token}"
    }


def get_user_input():
    print("\nAvailable exercises:")
    for exercise_id, config in EXERCISE_CONFIG.items():
        print(f"{exercise_id}: {config['name']}")

    while True:
        try:
            user_id = int(input("\nEnter user ID: ").strip())
            break
        except ValueError:
            print("Please enter a valid numeric user ID.")

    while True:
        try:
            exercise_id = int(input("Enter exercise ID: ").strip())
            if exercise_id not in EXERCISE_CONFIG:
                print("Invalid exercise ID selected. Please choose one from the list.")
                continue
            break
        except ValueError:
            print("Please enter a valid numeric exercise ID.")

    token = input("Enter JWT access token: ").strip()

    cycle_phase = input(
        "Enter cycle phase (menstrual / follicular / ovulation / luteal) or press Enter to skip: "
    ).strip().lower()

    if cycle_phase == "":
        cycle_phase = None

    return user_id, exercise_id, token, cycle_phase


def create_workout_session(user_id, token, exercise_name="Workout Session"):
    # Data sent to FastAPI to create a new session row in the DB
    session_data = {
        "user_id": user_id,
        "session_name": exercise_name,
        "duration_minutes": 0,
        "calories_burned": 0,
        "notes": "Auto-created from pose detector"
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/workout_sessions",
            json=session_data,
            headers=get_auth_headers(token),
            timeout=10
        )
        response.raise_for_status()
        created_session = response.json()
        print("Created workout session ID:", created_session["id"])
        return created_session["id"]

    except requests.exceptions.RequestException as e:
        print("Failed to create workout session:", e)
        return None


def get_point(landmarks, point_name):
    # Get x and y for a landmark name like LEFT_KNEE
    return [
        landmarks[getattr(mp_pose.PoseLandmark, point_name).value].x,
        landmarks[getattr(mp_pose.PoseLandmark, point_name).value].y
    ]


def get_visibility(landmarks, point_name):
    # Get visibility/confidence for a landmark
    return landmarks[getattr(mp_pose.PoseLandmark, point_name).value].visibility


def calculate_angle(a, b, c):
    # Convert points into numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate the angle between the 3 points
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Keep the angle in a normal 0-180 range
    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_side_angles(landmarks, config):
    # Get the 3 left side landmarks for the chosen exercise
    left_a_name, left_b_name, left_c_name = config["left_landmarks"]

    # Get the 3 right side landmarks for the chosen exercise
    right_a_name, right_b_name, right_c_name = config["right_landmarks"]

    # Pull actual landmark coordinates for left side
    left_a = get_point(landmarks, left_a_name)
    left_b = get_point(landmarks, left_b_name)
    left_c = get_point(landmarks, left_c_name)

    # Pull actual landmark coordinates for right side
    right_a = get_point(landmarks, right_a_name)
    right_b = get_point(landmarks, right_b_name)
    right_c = get_point(landmarks, right_c_name)

    # Calculate left and right side angles
    left_angle = calculate_angle(left_a, left_b, left_c)
    right_angle = calculate_angle(right_a, right_b, right_c)

    # Also return the middle joint points so the angle text can be drawn there
    return left_angle, right_angle, left_b, right_b


def landmarks_are_visible(landmarks, config, min_visibility=0.5):
    # Check that all important tracked landmarks are visible enough before trusting the frame
    landmark_names = list(config["left_landmarks"]) + list(config["right_landmarks"])
    visibilities = [get_visibility(landmarks, name) for name in landmark_names]
    return min(visibilities) >= min_visibility


def smooth_angle(angle_history, new_angle, window_size=5):
    # Keep a short history of recent angles to reduce noisy frame spikes
    angle_history.append(new_angle)
    if len(angle_history) > window_size:
        angle_history.pop(0)

    return sum(angle_history) / len(angle_history)


def get_tracking_angle(left_angle, right_angle, exercise_name, exercise_type, tracking_side=None):
    # For lunges, lock to the chosen active leg for the whole rep
    if exercise_name == "Lunge":
        if tracking_side == "left":
            return left_angle
        if tracking_side == "right":
            return right_angle
        return min(left_angle, right_angle)

    # For lower body exercises like squats, use the average of both sides
    if exercise_type == "lower_body":
        return (left_angle + right_angle) / 2

    # For curls, use the average as well so one bad arm reading doesn't dominate
    if exercise_name == "Bicep Curl":
        return (left_angle + right_angle) / 2

    # For other upper body moves, keep it stricter
    return min(left_angle, right_angle)


def get_symmetry_thresholds(exercise_name):
    # Different exercises need different symmetry strictness
    if exercise_name == "Bicep Curl":
        return {
            "good": 15,
            "okay": 30,
            "bad_warning": 35
        }
    elif exercise_name == "Shoulder Press":
        return {
            "good": 10,
            "okay": 20,
            "bad_warning": 25
        }
    else:
        return {
            "good": 8,
            "okay": 15,
            "bad_warning": 20
        }


def get_symmetry_score(left_angle, right_angle, exercise_name):
    # Difference between left and right side angles
    diff = abs(left_angle - right_angle)

    # Lunges are naturally asymmetrical, so don't punish them the same way
    if exercise_name == "Lunge":
        if diff <= 20:
            return 100, diff
        elif diff <= 35:
            return 85, diff
        elif diff <= 50:
            return 70, diff
        else:
            return 55, diff

    thresholds = get_symmetry_thresholds(exercise_name)

    if diff <= thresholds["good"]:
        return 100, diff
    elif diff <= thresholds["okay"]:
        return 80, diff
    elif diff <= thresholds["bad_warning"]:
        return 60, diff
    else:
        return 40, diff


def calculate_elbow_drift(left_elbow_positions, right_elbow_positions):
    # Measure how much the elbows move around during a curl
    if len(left_elbow_positions) < 2 or len(right_elbow_positions) < 2:
        return 0

    left_start = np.array(left_elbow_positions[0])
    left_end = np.array(left_elbow_positions[-1])
    right_start = np.array(right_elbow_positions[0])
    right_end = np.array(right_elbow_positions[-1])

    left_drift = np.linalg.norm(left_end - left_start)
    right_drift = np.linalg.norm(right_end - right_start)

    return (left_drift + right_drift) / 2


def get_live_feedback(current_angle, left_angle, right_angle, stage, exercise_name, symmetry_diff):
    thresholds = get_symmetry_thresholds(exercise_name)

    # Squats
    if exercise_name in ["Sumo Squat", "Squat"]:
        if symmetry_diff > thresholds["bad_warning"]:
            return "Try to keep both sides even"

        if stage == "up" and current_angle > 150:
            return "Ready for next squat"
        elif current_angle < 125:
            return "Good squat depth"
        else:
            return "Go lower"

    # Lunges
    elif exercise_name == "Lunge":
        if current_angle < 115:
            return "Good lunge depth"
        elif current_angle < 130:
            return "Lower a little more"
        elif current_angle < 155:
            return "Keep dropping into the front leg"
        else:
            return "Step slightly further out and bend the front knee"

    # Curls
    elif exercise_name == "Bicep Curl":
        if symmetry_diff > thresholds["bad_warning"]:
            return "Try to keep both arms moving evenly"

        if current_angle < 65:
            return "Good squeeze at the top"
        elif current_angle > 145:
            return "Lower fully with control"
        else:
            return "Curl both arms higher"

    # Shoulder press
    elif exercise_name == "Shoulder Press":
        if symmetry_diff > thresholds["bad_warning"]:
            return "Try to press evenly on both sides"

        if left_angle > 160 and right_angle > 160:
            return "Good overhead lockout"
        elif current_angle < 90:
            return "Good starting position"
        else:
            return "Press higher overhead"

    return "Tracking..."


def stabilise_feedback(feedback_history, new_feedback, window_size=6):
    # Keep a short history of feedback messages and use the most common one
    feedback_history.append(new_feedback)
    if len(feedback_history) > window_size:
        feedback_history.pop(0)

    return Counter(feedback_history).most_common(1)[0][0]


def get_bad_form_flags(exercise_name, min_angle, max_angle, symmetry_diff, avg_angle, elbow_drift=0):
    flags = []
    thresholds = get_symmetry_thresholds(exercise_name)

    if exercise_name in ["Sumo Squat", "Squat"]:
        if symmetry_diff > thresholds["bad_warning"]:
            flags.append("Asymmetry detected")
        if min_angle > 125:
            flags.append("Not enough depth")
        if max_angle < 160:
            flags.append("Not standing fully upright")
        if avg_angle > 145:
            flags.append("Shallow overall squat range")

    elif exercise_name == "Lunge":
        if min_angle > 120:
            flags.append("Lunge too shallow")
        if max_angle < 160:
            flags.append("Not returning fully to standing")
        if avg_angle > 145:
            flags.append("Limited front-leg range")

    elif exercise_name == "Bicep Curl":
        if symmetry_diff > thresholds["bad_warning"]:
            flags.append("Asymmetry detected")
        if min_angle > 75:
            flags.append("Curl not high enough")
        if max_angle < 145:
            flags.append("Arm not extending fully")
        if avg_angle > 125:
            flags.append("Limited curl range")
        if elbow_drift > 0.08:
            flags.append("Elbows drifting too much")

    elif exercise_name == "Shoulder Press":
        if symmetry_diff > thresholds["bad_warning"]:
            flags.append("Asymmetry detected")
        if min_angle > 95:
            flags.append("Start position too high")
        if max_angle < 165:
            flags.append("Not pressing fully overhead")
        if avg_angle > 130:
            flags.append("Limited press range")

    return flags


def generate_feedback(min_angle, max_angle, avg_angle, rep_count, exercise_name, symmetry_diff, elbow_drift=0):
    if rep_count == 0:
        return f"No {exercise_name.lower()} reps detected."

    feedback = []
    thresholds = get_symmetry_thresholds(exercise_name)

    # Squats
    if exercise_name in ["Sumo Squat", "Squat"]:
        if symmetry_diff > thresholds["bad_warning"]:
            feedback.append("Your left and right sides look uneven. Try to move more symmetrically.")
        elif symmetry_diff > thresholds["okay"]:
            feedback.append("Slight side-to-side imbalance detected.")
        else:
            feedback.append("Good left-to-right symmetry overall.")

        if min_angle > 125:
            feedback.append("Go lower in your squat.")
        elif min_angle < 70:
            feedback.append("Very deep squat detected, stay controlled.")
        else:
            feedback.append("Good squat depth.")

        if max_angle < 160:
            feedback.append("Stand taller at the top.")
        else:
            feedback.append("Good lockout at the top.")

    # Lunges
    elif exercise_name == "Lunge":
        if min_angle > 120:
            feedback.append("Try lowering more into the front leg.")
        else:
            feedback.append("Good lunge depth.")

        if max_angle < 160:
            feedback.append("Push back up fully between reps.")
        else:
            feedback.append("Good return to standing.")

        if symmetry_diff > 45:
            feedback.append("Your stance may be unstable or too narrow.")

    # Curls
    elif exercise_name == "Bicep Curl":
        if symmetry_diff > thresholds["bad_warning"]:
            feedback.append("Your left and right sides look uneven. Try to curl both arms more evenly.")
        elif symmetry_diff > thresholds["okay"]:
            feedback.append("Slight side-to-side imbalance detected.")
        else:
            feedback.append("Good left-to-right symmetry overall.")

        if min_angle > 75:
            feedback.append("Curl the weight higher.")
        else:
            feedback.append("Good curl height.")

        if max_angle < 145:
            feedback.append("Lower the weight fully for full range.")
        else:
            feedback.append("Good arm extension.")

        if elbow_drift > 0.08:
            feedback.append("Keep your elbows closer to your sides.")

    # Shoulder press
    elif exercise_name == "Shoulder Press":
        if symmetry_diff > thresholds["bad_warning"]:
            feedback.append("Your press looks uneven side to side.")
        elif symmetry_diff > thresholds["okay"]:
            feedback.append("Slight side-to-side imbalance detected.")
        else:
            feedback.append("Good left-to-right symmetry overall.")

        if min_angle > 95:
            feedback.append("Lower your hands more before pressing.")
        else:
            feedback.append("Good starting depth.")

        if max_angle < 165:
            feedback.append("Press higher overhead.")
        else:
            feedback.append("Good overhead extension.")

    bad_form_flags = get_bad_form_flags(
        exercise_name,
        min_angle,
        max_angle,
        symmetry_diff,
        avg_angle,
        elbow_drift
    )

    if bad_form_flags:
        feedback.append("Main issues: " + ", ".join(bad_form_flags) + ".")

    return " ".join(feedback)


def calculate_form_score(min_angle, max_angle, avg_angle, rep_count, exercise_name, symmetry_diff, elbow_drift=0):
    if rep_count == 0:
        return 0

    score = 100
    thresholds = get_symmetry_thresholds(exercise_name)

    if exercise_name in ["Sumo Squat", "Squat"]:
        if symmetry_diff > thresholds["bad_warning"]:
            score -= 25
        elif symmetry_diff > thresholds["okay"]:
            score -= 10

        if min_angle > 125:
            score -= 20
        elif min_angle < 70:
            score -= 5
        if max_angle < 160:
            score -= 10

    elif exercise_name == "Lunge":
        # Much lighter symmetry penalty because lunges are naturally uneven
        if symmetry_diff > 45:
            score -= 10

        if min_angle > 120:
            score -= 20
        if max_angle < 160:
            score -= 10
        if avg_angle > 145:
            score -= 5

    elif exercise_name == "Bicep Curl":
        if symmetry_diff > thresholds["bad_warning"]:
            score -= 20
        elif symmetry_diff > thresholds["okay"]:
            score -= 8

        if min_angle > 75:
            score -= 15
        if max_angle < 145:
            score -= 12
        if elbow_drift > 0.08:
            score -= 12

    elif exercise_name == "Shoulder Press":
        if symmetry_diff > thresholds["bad_warning"]:
            score -= 25
        elif symmetry_diff > thresholds["okay"]:
            score -= 10

        if min_angle > 95:
            score -= 15
        if max_angle < 165:
            score -= 20

    return max(score, 0)


def get_rep_quality(min_angle, max_angle, exercise_name, symmetry_diff, elbow_drift=0):
    thresholds = get_symmetry_thresholds(exercise_name)

    if exercise_name in ["Sumo Squat", "Squat"]:
        if symmetry_diff > thresholds["bad_warning"]:
            return "Poor"

        if min_angle <= 125 and max_angle >= 160 and symmetry_diff <= thresholds["okay"]:
            return "Good"
        elif min_angle <= 135 and max_angle >= 150:
            return "Okay"
        else:
            return "Poor"

    elif exercise_name == "Lunge":
        # Do not use strict symmetry expectation for lunges
        if min_angle <= 120 and max_angle >= 160:
            return "Good"
        elif min_angle <= 130 and max_angle >= 150:
            return "Okay"
        else:
            return "Poor"

    elif exercise_name == "Bicep Curl":
        if symmetry_diff > thresholds["bad_warning"]:
            return "Poor"

        if min_angle <= 75 and max_angle >= 145 and symmetry_diff <= thresholds["okay"] and elbow_drift <= 0.08:
            return "Good"
        elif min_angle <= 90 and max_angle >= 130 and elbow_drift <= 0.12:
            return "Okay"
        else:
            return "Poor"

    elif exercise_name == "Shoulder Press":
        if symmetry_diff > thresholds["bad_warning"]:
            return "Poor"

        if min_angle <= 95 and max_angle >= 165 and symmetry_diff <= thresholds["okay"]:
            return "Good"
        elif min_angle <= 105 and max_angle >= 155:
            return "Okay"
        else:
            return "Poor"

    return "Unknown"


def analyse_completed_rep(rep_angles, rep_symmetry_diffs, exercise_name, rep_left_elbows=None, rep_right_elbows=None):
    # Analyse one completed rep using only the frames from that rep
    if not rep_angles:
        return "Unknown", 0, "No rep data.", None

    rep_min_angle = min(rep_angles)
    rep_max_angle = max(rep_angles)
    rep_avg_angle = sum(rep_angles) / len(rep_angles)
    rep_avg_symmetry = sum(rep_symmetry_diffs) / len(rep_symmetry_diffs) if rep_symmetry_diffs else 0

    elbow_drift = 0
    if exercise_name == "Bicep Curl" and rep_left_elbows and rep_right_elbows:
        elbow_drift = calculate_elbow_drift(rep_left_elbows, rep_right_elbows)

    rep_quality = get_rep_quality(
        min_angle=rep_min_angle,
        max_angle=rep_max_angle,
        exercise_name=exercise_name,
        symmetry_diff=rep_avg_symmetry,
        elbow_drift=elbow_drift
    )

    rep_score = calculate_form_score(
        min_angle=rep_min_angle,
        max_angle=rep_max_angle,
        avg_angle=rep_avg_angle,
        rep_count=1,
        exercise_name=exercise_name,
        symmetry_diff=rep_avg_symmetry,
        elbow_drift=elbow_drift
    )

    rep_feedback = generate_feedback(
        min_angle=rep_min_angle,
        max_angle=rep_max_angle,
        avg_angle=rep_avg_angle,
        rep_count=1,
        exercise_name=exercise_name,
        symmetry_diff=rep_avg_symmetry,
        elbow_drift=elbow_drift
    )

    rep_details = {
        "min_angle": round(float(rep_min_angle), 2),
        "max_angle": round(float(rep_max_angle), 2),
        "avg_angle": round(float(rep_avg_angle), 2),
        "avg_symmetry": round(float(rep_avg_symmetry), 2),
        "elbow_drift": round(float(elbow_drift), 4)
    }

    return rep_quality, rep_score, rep_feedback, rep_details


def estimate_duration_seconds(cap, fallback_fps=30):
    # Try to estimate workout duration from total frames processed
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 1:
            fps = fallback_fps

        duration_seconds = int(frame_count / fps)
        return max(duration_seconds, 1)
    except Exception:
        return 60


def estimate_calories_burned(exercise_name, reps, duration_seconds):
    # Very rough estimate just to populate the exercise_logs table
    duration_minutes = max(duration_seconds / 60, 1)

    if exercise_name in ["Squat", "Sumo Squat", "Lunge"]:
        return round(duration_minutes * 5.5, 2)
    elif exercise_name in ["Bicep Curl", "Shoulder Press"]:
        return round(duration_minutes * 4.0, 2)

    return round(duration_minutes * 4.5, 2)


def send_exercise_log(
    token,
    workout_session_id,
    exercise_id,
    reps,
    duration_seconds,
    calories_burned,
    sets=1,
    weight_kg=0
):
    # Send the session workout stats to the exercise_logs table
    data = {
        "workout_session_id": workout_session_id,
        "exercise_id": exercise_id,
        "sets": sets,
        "reps": reps,
        "weight_kg": weight_kg,
        "duration_seconds": duration_seconds,
        "calories_burned": calories_burned
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/exercise-logs",
            json=data,
            headers=get_auth_headers(token),
            timeout=10
        )
        response.raise_for_status()
        print("Exercise log saved:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Failed to send exercise log to FastAPI:", e)


def send_pose_data(
    counter,
    angles,
    exercise_name,
    symmetry_diffs,
    rep_scores,
    token,
    user_id,
    workout_session_id,
    exercise_id,
    rep_elbow_drifts=None
):
    # If no angle data was collected, don't send anything
    if not angles:
        print("No angle data to send.")
        return

    # Calculate overall stats from the whole set
    avg_angle = sum(angles) / len(angles)
    min_angle = min(angles)
    max_angle = max(angles)
    avg_symmetry_diff = sum(symmetry_diffs) / len(symmetry_diffs) if symmetry_diffs else 0
    overall_form_score = sum(rep_scores) / len(rep_scores) if rep_scores else 0
    avg_elbow_drift = sum(rep_elbow_drifts) / len(rep_elbow_drifts) if rep_elbow_drifts else 0

    # Generate the final coaching feedback
    feedback = generate_feedback(
        min_angle,
        max_angle,
        avg_angle,
        counter,
        exercise_name,
        avg_symmetry_diff,
        avg_elbow_drift
    )

    # Payload sent to the FastAPI backend
    data = {
        "user_id": user_id,
        "workout_session_id": workout_session_id,
        "exercise_id": exercise_id,
        "detected_reps": counter,
        "avg_joint_angle": round(avg_angle, 2),
        "min_angle": round(min_angle, 2),
        "max_angle": round(max_angle, 2),
        "form_score": round(overall_form_score, 2),
        "feedback": feedback
    }

    try:
        # Send final workout analysis to backend
        response = requests.post(
            f"{API_BASE_URL}/pose-analysis-logs",
            json=data,
            headers=get_auth_headers(token),
            timeout=10
        )
        response.raise_for_status()
        print("Pose analysis saved:", response.status_code, response.text)
        print("Average symmetry difference:", round(avg_symmetry_diff, 2))
        print("Average rep score:", round(overall_form_score, 2))
        if exercise_name == "Bicep Curl":
            print("Average elbow drift:", round(avg_elbow_drift, 4))

    except requests.exceptions.RequestException as e:
        print("Failed to send data to FastAPI:", e)


def get_recommendations(user_id, token, cycle_phase):
    try:
        response = requests.get(
            f"{API_BASE_URL}/users/{user_id}/recommendations",
            headers=get_auth_headers(token),
            params={"cycle_phase": cycle_phase} if cycle_phase else {},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        print("\n--- AI COACH RECOMMENDATIONS ---")
        for rec in data["recommendations"]:
            print("-", rec)

    except requests.exceptions.RequestException as e:
        print("Failed to fetch recommendations:", e)


def main():
    # Ask the user which account and exercise to use
    user_id, exercise_id, token, cycle_phase = get_user_input()
    config = EXERCISE_CONFIG[exercise_id]
    exercise_name = config["name"]
    exercise_type = config["type"]

    print("\nTracking tips:")
    print("- Stand side-on to the camera for best accuracy.")
    print("- Keep your full body visible on screen.")
    print("- Step back slightly if doing lunges.")
    print("- Press q when you want to finish the session.\n")

    # Create a new workout session before starting camera tracking
    workout_session_id = create_workout_session(user_id, token, f"{exercise_name} Session")
    if workout_session_id is None:
        print("Could not create workout session, stopping script.")
        return

    # ----- Start webcam -----
    # 0 = laptop webcam usually
    # 1 = phone / external cam if connected
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Main workout tracking variables
    counter = 0
    stage = None
    angles = []
    symmetry_diffs = []
    rep_scores = []
    rep_elbow_drifts = []
    live_feedback = "Tracking..."
    rep_quality = "N/A"

    # Small angle history buffers to smooth noisy readings
    left_angle_history = []
    right_angle_history = []
    feedback_history = []

    # Buffers for one rep at a time
    current_rep_angles = []
    current_rep_symmetry_diffs = []
    current_rep_left_elbows = []
    current_rep_right_elbows = []
    rep_in_progress = False

    # Lunge-specific tracking
    tracking_side = None
    rep_started = False

    # Stop if the camera can't be opened
    if not cap.isOpened():
        raise RuntimeError("Camera not found or not accessible")

    # Start Mediapipe pose model
    with mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Convert back to BGR for OpenCV drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Skip bad frames where important landmarks are not visible enough
                    if not landmarks_are_visible(landmarks, config, min_visibility=0.5):
                        cv2.putText(
                            image,
                            "Tracking lost - adjust position",
                            (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA
                        )
                        cv2.imshow("Mediapipe Feed", image)
                        if cv2.waitKey(10) & 0xFF == ord("q"):
                            break
                        continue

                    # Get raw left and right joint angles for the chosen exercise
                    raw_left_angle, raw_right_angle, left_joint, right_joint = get_side_angles(landmarks, config)

                    # Smooth noisy lower body too, not just curls
                    if exercise_name in ["Bicep Curl", "Lunge", "Squat", "Sumo Squat"]:
                        left_angle = smooth_angle(left_angle_history, raw_left_angle, window_size=5)
                        right_angle = smooth_angle(right_angle_history, raw_right_angle, window_size=5)
                    else:
                        left_angle = raw_left_angle
                        right_angle = raw_right_angle

                    # For lunges, detect active leg once the difference is clear enough
                    if exercise_name == "Lunge" and stage == "up" and not rep_started:
                        if abs(left_angle - right_angle) > 10:
                            rep_started = True
                            tracking_side = "left" if left_angle < right_angle else "right"
                            print(f"Tracking leg: {tracking_side}")

                    # Get the main angle used for rep tracking
                    tracking_angle = get_tracking_angle(
                        left_angle,
                        right_angle,
                        exercise_name,
                        exercise_type,
                        tracking_side
                    )
                    angles.append(tracking_angle)

                    # Work out how even the movement is left vs right
                    symmetry_score, symmetry_diff = get_symmetry_score(left_angle, right_angle, exercise_name)
                    symmetry_diffs.append(symmetry_diff)

                    # Generate live coaching text and stabilise it so it flickers less
                    raw_feedback = get_live_feedback(
                        tracking_angle,
                        left_angle,
                        right_angle,
                        stage,
                        exercise_name,
                        symmetry_diff
                    )
                    live_feedback = stabilise_feedback(feedback_history, raw_feedback, window_size=6)

                    # Convert joint positions from relative coords into screen coords
                    h, w, _ = image.shape
                    left_joint_position = tuple(np.multiply(left_joint, [w, h]).astype(int))
                    right_joint_position = tuple(np.multiply(right_joint, [w, h]).astype(int))

                    # If the user is back at the top position, mark stage as up
                    if tracking_angle > config["up_threshold"]:
                        stage = "up"
                        if not rep_in_progress:
                            rep_in_progress = True
                            current_rep_angles = []
                            current_rep_symmetry_diffs = []
                            current_rep_left_elbows = []
                            current_rep_right_elbows = []

                    # While a rep is in progress, store rep-only data
                    if rep_in_progress:
                        current_rep_angles.append(tracking_angle)
                        current_rep_symmetry_diffs.append(symmetry_diff)

                        if exercise_name == "Bicep Curl":
                            current_rep_left_elbows.append(left_joint)
                            current_rep_right_elbows.append(right_joint)

                    # Count a rep when they move from up to down past the threshold
                    if tracking_angle < config["down_threshold"] and stage == "up":
                        stage = "down"
                        counter += 1
                        print(f"{exercise_name} reps:", counter)

                        # Analyse only the data from that specific rep
                        rep_quality, rep_score, rep_feedback, rep_details = analyse_completed_rep(
                            current_rep_angles,
                            current_rep_symmetry_diffs,
                            exercise_name,
                            current_rep_left_elbows,
                            current_rep_right_elbows
                        )

                        rep_scores.append(rep_score)

                        if exercise_name == "Bicep Curl" and rep_details:
                            rep_elbow_drifts.append(rep_details["elbow_drift"])

                        print("Rep quality:", rep_quality)
                        print("Rep score:", rep_score)
                        print("Rep feedback:", rep_feedback)
                        print("Rep details:", rep_details)

                        # Reset lunge side detection for the next rep
                        if exercise_name == "Lunge":
                            tracking_side = None
                            rep_started = False

                        # Reset buffers for the next rep
                        current_rep_angles = []
                        current_rep_symmetry_diffs = []
                        current_rep_left_elbows = []
                        current_rep_right_elbows = []
                        rep_in_progress = False

                    # Draw left angle on screen
                    cv2.putText(
                        image,
                        f"L:{int(left_angle)}",
                        left_joint_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Draw right angle on screen
                    cv2.putText(
                        image,
                        f"R:{int(right_angle)}",
                        right_joint_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Orange box behind the main text UI
                    cv2.rectangle(image, (0, 0), (780, 240), (245, 117, 16), -1)

                    # Exercise label
                    cv2.putText(
                        image,
                        "EXERCISE",
                        (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        image,
                        exercise_name,
                        (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Rep count label
                    cv2.putText(
                        image,
                        "REPS",
                        (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        image,
                        str(counter),
                        (80, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Current stage label
                    cv2.putText(
                        image,
                        "STAGE",
                        (170, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        image,
                        str(stage),
                        (245, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Current tracking angle
                    cv2.putText(
                        image,
                        f"TRACKING ANGLE: {int(tracking_angle)}",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Left and right side values
                    cv2.putText(
                        image,
                        f"LEFT: {int(left_angle)}   RIGHT: {int(right_angle)}",
                        (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Symmetry difference value
                    cv2.putText(
                        image,
                        f"SYMMETRY DIFF: {int(symmetry_diff)}",
                        (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Show current tracked side for lunges
                    if exercise_name == "Lunge":
                        side_label = tracking_side.upper() if tracking_side else "DETECTING"
                        cv2.putText(
                            image,
                            f"TRACKING LEG: {side_label}",
                            (10, 195),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        coach_y = 225
                    else:
                        coach_y = 195

                    # Live coaching text
                    cv2.putText(
                        image,
                        f"COACH: {live_feedback}",
                        (10, coach_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Rep quality and symmetry score
                    cv2.putText(
                        image,
                        f"REP QUALITY: {rep_quality} | SYMMETRY SCORE: {symmetry_score}",
                        (380, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            except Exception as e:
                print("Pose error:", e)

            # Draw the full body pose skeleton if landmarks were found
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # Show webcam window
            cv2.imshow("Mediapipe Feed", image)

            # Press q to quit
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    # Estimate basic session stats for exercise_logs
    duration_seconds = estimate_duration_seconds(cap)
    calories_burned = estimate_calories_burned(exercise_name, counter, duration_seconds)

    # Clean up camera and windows
    cap.release()
    cv2.destroyAllWindows()

    # Save exercise-log summary first
    send_exercise_log(
        token=token,
        workout_session_id=workout_session_id,
        exercise_id=exercise_id,
        reps=counter,
        duration_seconds=duration_seconds,
        calories_burned=calories_burned,
        sets=1,
        weight_kg=0
    )

    # Save pose-analysis details
    send_pose_data(
        counter=counter,
        angles=angles,
        exercise_name=exercise_name,
        symmetry_diffs=symmetry_diffs,
        rep_scores=rep_scores,
        token=token,
        user_id=user_id,
        workout_session_id=workout_session_id,
        exercise_id=exercise_id,
        rep_elbow_drifts=rep_elbow_drifts
    )

    # Fetch AI recommendations from the backend
    get_recommendations(user_id, token, cycle_phase)


if __name__ == "__main__":
    main()