import time
import math
import numpy as np

from MangDang.mini_pupper.ESP32Interface import ESP32Interface

"""Calibration with IMU-assisted leveling.

This script keeps the core behavior of test_set_cali.py:
- Measure per-joint mechanical limits (abduction, hip, knee) for selected legs.
- Compare measured limits to JOINT_LIMIT_TARGETS.
- Compute offsets (measured - target) and apply them at neutral pose.
- Call esp32.save_calibration() so those offsets become the new calibration.

ADDITIONAL IMU STEPS
--------------------
1) IMU reference on belly:
   - User lays the robot belly-flat on a level surface.
   - We read raw IMU accelerometer data multiple times, average it, and
     compute pitch/roll. That becomes our "zero" reference orientation.

2) IMU check & leveling at neutral pose:
   - After calibration, we command a neutral standing pose (all servos 512,
     which after calibration should be the new mechanical neutral).
   - We again read IMU, average it, and compute pitch/roll.
   - Using the difference from the belly-flat reference, we apply a simple
     knee adjustment on a subset of legs to reduce pitch/roll.

IMU NOTES
---------
- imu_get_data() returns raw accelerometer and gyro: {ax, ay, az, gx, gy, gz}.
- For static poses (robot not moving), we can estimate pitch/roll directly
  from averaged accelerometer data; a full filter (e.g. complementary or
  Kalman) is not strictly necessary. Simple averaging is enough here.

Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left
Axis indices: 0=abduction, 1=hip/thigh, 2=knee
"""


# === USER-DEFINED CALIBRATION TARGETS ===
JOINT_LIMIT_TARGETS = np.array([
    [218, 799, 797, 220],  # abduction limits for legs 0..3
    [403, 613, 428, 624],  # hip/thigh limits for legs 0..3
    [837, 165, 828, 167],  # knee limits for legs 0..3
], dtype=int)

# Optional per-joint, per-leg sign for offsets.
OFFSET_SIGNS = np.ones_like(JOINT_LIMIT_TARGETS, dtype=int)


# Sweep parameters
LOAD_THRESHOLD = 175

# Per-joint, per-leg load thresholds during limit finding.
# Shape is [joint, leg] with:
#   joint 0 = abduction, 1 = hip/thigh, 2 = knee
#   leg   0 = front-right, 1 = front-left, 2 = back-right, 3 = back-left
LOAD_THRESHOLDS = [
    [200, 200, 200, 400],  # abduction joints, legs 0..3
    [200, 200, 200, 400],  # hip/thigh joints, legs 0..3
    [200, 200, 200, 400],  # knee joints, legs 0..3
]

STEP = 1
MAX_DELTA = 400
DWELL = 0.02

# Fixed movement shaping parameters (matching test_find_cali2/test_set_cali behavior)
KNEE_BACKOFF = 60      # servo counts to back off from knee hard stop between sweeps
HIP_BACKOFF = 60       # servo counts to back off from hip hard stop between sweeps
HIP_PRE_OFFSET = -100  # servo counts to move hip before starting knee sweep

# Number of times to repeat each joint sweep per leg when measuring limits.
JOINT_REPEAT_COUNT = 3

# Leveling gains: how many knee servo counts to change per degree of
# pitch/roll error (small values to keep motion gentle).
KNEE_LEVEL_GAIN_COUNTS_PER_DEG = 5.0
MAX_LEVEL_ANGLE_DEG = 10.0   # clamp pitch/roll used for leveling
MAX_LEVEL_KNEE_DELTA = 60    # max extra knee counts up/down from 512

# Direction in which increasing the knee command lowers the body for each leg.
# These are educated guesses; if you see the body move the wrong way, flip
# the sign for the affected leg.
KNEE_DOWN_DIR = {
    0: 1,   # front-right
    1: -1,  # front-left
    2: 1,   # back-right
    3: -1,  # back-left
}


def move_to_neutral_pose(esp32: ESP32Interface, hold_time: float = 1.0) -> None:
    """Move all servos to their neutral hardware command (512)."""
    print("Moving to neutral pose (all servos to 512)...")
    positions = [512] * 12
    esp32.servos_set_position(positions)
    time.sleep(hold_time)


def leg_joint_load_index(leg_index: int, axis_index: int) -> int:
    """Map (leg, axis_index) to index in 12-element load vector.

    Layout: [leg0_abd, leg0_hip, leg0_knee, leg1_abd, leg1_hip, leg1_knee, ...]
    axis_index: 0=abduction, 1=hip, 2=knee.
    """
    return leg_index * 3 + axis_index


def sweep_until_limit(
    esp32: ESP32Interface,
    positions: list[int],
    leg_index: int,
    axis_index: int,
    direction: int,
    load_threshold: int = LOAD_THRESHOLD,
    step: int = STEP,
    max_delta: int = MAX_DELTA,
    dwell: float = DWELL,
    monitor_knee_load: bool = False,
) -> int:
    """Sweep one joint in servo position until a load threshold or max_delta is reached.

    - positions is updated in-place and left at the final value.
    - Returns the final servo position as the measured limit.
    """
    axis_name = {0: "Abduction", 1: "Hip/Thigh", 2: "Knee"}[axis_index]

    servo_idx = leg_index * 3 + axis_index
    start_pos = positions[servo_idx]
    target_pos = start_pos + direction * max_delta

    print(f"\n=== Measuring {axis_name} limit on leg {leg_index} ===")
    print(f"Start pos: {start_pos}, target pos: {target_pos} (dir {direction:+d})")

    load_idx = leg_joint_load_index(leg_index, axis_index)

    # Optionally also monitor the knee load (axis 2) on this leg, so that
    # when sweeping the hip we stop if we push the knee harder into its
    # mechanical stop.
    knee_load_idx = None
    if monitor_knee_load and axis_index != 2:
        knee_load_idx = leg_joint_load_index(leg_index, 2)

    def within_range() -> bool:
        if direction < 0:
            return positions[servo_idx] >= target_pos
        else:
            return positions[servo_idx] <= target_pos

    while within_range():
        positions[servo_idx] += direction * step
        if positions[servo_idx] < 0:
            positions[servo_idx] = 0
        elif positions[servo_idx] > 1023:
            positions[servo_idx] = 1023

        esp32.servos_set_position(positions)
        time.sleep(dwell)

        loads = esp32.servos_get_load()
        if loads is None:
            print("Failed to read loads, stopping sweep.")
            break

        this_load = loads[load_idx]
        knee_load = loads[knee_load_idx] if knee_load_idx is not None else None

        stop_on_joint = abs(this_load) >= load_threshold
        stop_on_knee = knee_load is not None and abs(knee_load) >= load_threshold
        if stop_on_joint or stop_on_knee:
            reason = []
            if stop_on_joint:
                reason.append(f"{axis_name} load={this_load}")
            if stop_on_knee:
                reason.append(f"Knee load={knee_load}")
            print(
                f"Reached load threshold |load| >= {load_threshold} at pos {positions[servo_idx]} "
                f"({' and '.join(reason)})"
            )
            break

    # At the end of the sweep, read back the actual servo position from ESP32
    measured_positions = esp32.servos_get_position()
    if measured_positions is not None and len(measured_positions) > servo_idx:
        final_pos = measured_positions[servo_idx]
        print(f"Final {axis_name} measured limit on leg {leg_index} (from ESP32): {final_pos}")
    else:
        final_pos = positions[servo_idx]
        print(f"Final {axis_name} measured limit on leg {leg_index} (commanded only): {final_pos}")

    return int(final_pos)


def parse_leg_selection() -> list[int]:
    """Select which legs to calibrate: 'all', empty, or comma list like '0,2'."""
    print("Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left")
    s = input("Enter legs to calibrate (e.g. '0,2' or 'all'): ").strip().lower()
    if s == "all" or s == "":
        return [0, 1, 2, 3]
    legs: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            i = int(tok)
        except ValueError:
            continue
        if 0 <= i <= 3:
            legs.append(i)
    return sorted(set(legs))


# === IMU helpers ===

def read_imu_average(esp32: ESP32Interface, samples: int = 200, dt: float = 0.01) -> dict:
    """Read raw IMU data multiple times and return averaged ax, ay, az.

    We ignore the gyro for these static measurements; accelerometer alone
    is enough to get pitch/roll when the robot is stationary.
    """
    sum_ax = 0.0
    sum_ay = 0.0
    sum_az = 0.0
    valid_count = 0

    for _ in range(samples):
        data = esp32.imu_get_data()
        if data is None:
            time.sleep(dt)
            continue
        sum_ax += data["ax"]
        sum_ay += data["ay"]
        sum_az += data["az"]
        valid_count += 1
        time.sleep(dt)

    if valid_count == 0:
        raise RuntimeError("No valid IMU samples received.")

    return {
        "ax": sum_ax / valid_count,
        "ay": sum_ay / valid_count,
        "az": sum_az / valid_count,
    }


def accel_to_pitch_roll_deg(ax: float, ay: float, az: float) -> tuple[float, float]:
    """Convert accelerometer to pitch/roll in degrees.

    Uses a common aerospace convention:
    - roll: rotation about x-axis, right side down positive
    - pitch: rotation about y-axis, nose up positive

    For static cases, gravity dominates the accelerometer and this gives
    a reasonable orientation estimate.
    """
    # Protect against division by zero
    g = math.sqrt(ax * ax + ay * ay + az * az)
    if g == 0.0:
        return 0.0, 0.0

    # Normalize (not strictly required but can help numerical stability)
    ax_n = ax / g
    ay_n = ay / g
    az_n = az / g

    roll = math.degrees(math.atan2(ay_n, az_n))
    pitch = math.degrees(math.atan2(-ax_n, math.sqrt(ay_n * ay_n + az_n * az_n)))
    return pitch, roll


def get_reference_pitch_roll(esp32: ESP32Interface) -> tuple[float, float]:
    """Ask user to place robot belly-flat and measure reference pitch/roll."""
    print("\n=== IMU reference (belly-flat) ===")
    print("Lay the robot belly flat on a level surface and keep it still.")
    input("Press Enter when ready to capture IMU reference...")

    avg = read_imu_average(esp32, samples=200, dt=0.01)
    pitch, roll = accel_to_pitch_roll_deg(avg["ax"], avg["ay"], avg["az"])
    print(f"Reference pitch={pitch:.2f} deg, roll={roll:.2f} deg")
    return pitch, roll


def get_current_pitch_roll(esp32: ESP32Interface, message: str) -> tuple[float, float]:
    """Measure current pitch/roll after user positions the robot as instructed."""
    print("\n=== IMU measurement ===")
    print(message)
    input("Press Enter when ready to capture IMU data...")

    avg = read_imu_average(esp32, samples=200, dt=0.01)
    pitch, roll = accel_to_pitch_roll_deg(avg["ax"], avg["ay"], avg["az"])
    print(f"Measured pitch={pitch:.2f} deg, roll={roll:.2f} deg")
    return pitch, roll


def apply_leveling_adjustment(
    esp32: ESP32Interface,
    ref_pitch: float,
    ref_roll: float,
) -> None:
    """Use IMU to apply a small knee adjustment to reduce pitch/roll.

    This is a *simple* heuristic that:
    - assumes the robot is already on its feet for correction,
    - commands a neutral pose (512 for all joints),
    - reads IMU to get current pitch/roll (no extra user prompts here),
    - compares to the belly-flat reference,
    - adjusts knees on one or two legs based on the sign of the errors.

    You may tune KNEE_LEVEL_GAIN_COUNTS_PER_DEG, MAX_LEVEL_ANGLE_DEG,
    MAX_LEVEL_KNEE_DELTA and KNEE_DOWN_DIR to suit your robot.
    """
    # Move joints to neutral while the robot is already positioned
    # upright by the user.
    move_to_neutral_pose(esp32, hold_time=0.5)

    # Directly read IMU without additional prompts; we assume the robot
    # is stationary in the desired pose.
    avg = read_imu_average(esp32, samples=200, dt=0.01)
    cur_pitch, cur_roll = accel_to_pitch_roll_deg(avg["ax"], avg["ay"], avg["az"])
    print(f"Measured correction pose pitch={cur_pitch:.2f} deg, roll={cur_roll:.2f} deg")

    d_pitch = cur_pitch - ref_pitch
    d_roll = cur_roll - ref_roll
    print(f"Pitch error={d_pitch:.2f} deg, roll error={d_roll:.2f} deg")

    # Clamp angles used for leveling
    d_pitch_clamped = max(-MAX_LEVEL_ANGLE_DEG, min(MAX_LEVEL_ANGLE_DEG, d_pitch))
    d_roll_clamped = max(-MAX_LEVEL_ANGLE_DEG, min(MAX_LEVEL_ANGLE_DEG, d_roll))

    # Convert to knee servo deltas (counts)
    pitch_delta_counts = KNEE_LEVEL_GAIN_COUNTS_PER_DEG * d_pitch_clamped
    roll_delta_counts = KNEE_LEVEL_GAIN_COUNTS_PER_DEG * d_roll_clamped

    # Start from neutral commands
    positions = [512] * 12

    # Decide which legs to adjust based on pitch and roll.
    leg_deltas = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    # Pitch: nose up/down -> adjust front vs back legs
    if abs(pitch_delta_counts) > 0.5:
        if d_pitch_clamped > 0.0:
            # Nose up: lower front legs (0,1)
            leg_deltas[0] += abs(pitch_delta_counts)
            leg_deltas[1] += abs(pitch_delta_counts)
        else:
            # Nose down: lower back legs (2,3)
            leg_deltas[2] += abs(pitch_delta_counts)
            leg_deltas[3] += abs(pitch_delta_counts)

    # Roll: right side down/up -> adjust left vs right legs
    if abs(roll_delta_counts) > 0.5:
        if d_roll_clamped > 0.0:
            # Right side down (right legs lower): lower left legs (1,3)
            leg_deltas[1] += abs(roll_delta_counts)
            leg_deltas[3] += abs(roll_delta_counts)
        else:
            # Right side up (right legs higher): lower right legs (0,2)
            leg_deltas[0] += abs(roll_delta_counts)
            leg_deltas[2] += abs(roll_delta_counts)

    # Apply knee adjustments, respecting estimated "down" direction
    for leg, delta in leg_deltas.items():
        if delta <= 0.5:
            continue
        delta = min(delta, MAX_LEVEL_KNEE_DELTA)
        knee_idx = leg * 3 + 2
        down_dir = KNEE_DOWN_DIR.get(leg, 1)
        new_cmd = 512 + int(down_dir * delta)
        new_cmd = max(0, min(1023, new_cmd))
        positions[knee_idx] = new_cmd
        print(
            f"Leg {leg} knee adjusted by {down_dir * int(delta)} counts "
            f"-> command {new_cmd}"
        )

    esp32.servos_set_position(positions)
    print("Leveling adjustment applied (simple heuristic).")


def main():
    esp32 = ESP32Interface()

    if not JOINT_LIMIT_TARGETS.any():
        print(
            "WARNING: JOINT_LIMIT_TARGETS is all zeros. Please fill it "
            "with your target limits before running."
        )

    # Step 1: IMU reference with robot belly-flat (stationary)
    ref_pitch, ref_roll = get_reference_pitch_roll(esp32)

    # Give the user time to flip the robot onto its feet
    # before running the joint limit calibration.
    print(
        "\nNow flip the robot onto its feet in a safe position "
        "for joint calibration."
    )
    input(
        "Press Enter when the robot is on its feet and you are ready "
        "to start joint calibration..."
    )

    # Step 2: standard calibration procedure (as in test_set_cali.py)
    move_to_neutral_pose(esp32, hold_time=0.5)
    positions: list[int] = [512] * 12

    # Choose which legs to calibrate
    legs = parse_leg_selection()
    if not legs:
        print("No valid legs selected, exiting.")
        return

    # Measured limits: 3x4 array matching JOINT_LIMIT_TARGETS
    measured_limits = np.zeros((3, 4), dtype=int)

    for leg in legs:
        print(f"\n=== Measuring limits for leg {leg} ===")
        # Per-joint thresholds for this leg
        abd_load_threshold = LOAD_THRESHOLDS[0][leg]
        hip_load_threshold = LOAD_THRESHOLDS[1][leg]
        knee_load_threshold = LOAD_THRESHOLDS[2][leg]
        print(
            "Using load thresholds for leg "
            f"{leg}: abd={abd_load_threshold}, hip={hip_load_threshold}, "
            f"knee={knee_load_threshold}."
        )

        # Reset this leg's three joints to 512 commands
        for axis in (0, 1, 2):
            servo_idx = leg * 3 + axis
            positions[servo_idx] = 512
        esp32.servos_set_position(positions)
        time.sleep(0.2)

        # Convenience indices for this leg's joints
        abd_servo_idx = leg * 3 + 0  # axis 0 = abduction
        hip_servo_idx = leg * 3 + 1  # axis 1 = hip
        knee_servo_idx = leg * 3 + 2  # axis 2 = knee

        # Direction rules copied from your existing test script
        # Right legs (0, 2): knee_dir=+1, hip_dir=-1, abd_dir aligned with hip_dir
        # Left  legs (1, 3): knee_dir=-1, hip_dir=+1, abd_dir mostly hip_dir
        knee_dir = 1 if leg in (0, 2) else -1
        hip_dir = -1 if leg in (0, 2) else 1
        abd_dir = -1 if leg in (0, 3) else 1

        # Before moving the knee, move the hip a little first by
        # HIP_PRE_OFFSET counts. This value is signed:
        #   >0 moves along hip_dir, <0 moves opposite hip_dir.
        if HIP_PRE_OFFSET != 0:
            pre_pos = 512 + hip_dir * HIP_PRE_OFFSET
            if pre_pos < 0:
                pre_pos = 0
            elif pre_pos > 1023:
                pre_pos = 1023
            print(f"Pre-positioning hip on leg {leg} by {HIP_PRE_OFFSET} counts to pos {pre_pos}.")
            positions[hip_servo_idx] = pre_pos
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        knee_reps = np.zeros(JOINT_REPEAT_COUNT, dtype=int)
        hip_reps = np.zeros(JOINT_REPEAT_COUNT, dtype=int)
        abd_reps = np.zeros(JOINT_REPEAT_COUNT, dtype=int)

        # 1) Knee limit (axis 2)
        for rep in range(JOINT_REPEAT_COUNT):
            print(f"    Knee sweep repeat {rep + 1}/{JOINT_REPEAT_COUNT}")
            knee_pos_limit = sweep_until_limit(
                esp32=esp32,
                positions=positions,
                leg_index=leg,
                axis_index=2,
                direction=knee_dir,
                load_threshold=knee_load_threshold,
            )
            knee_reps[rep] = int(knee_pos_limit)

            if KNEE_BACKOFF > 0:
                backed_off_knee = knee_pos_limit - knee_dir * KNEE_BACKOFF
                print(
                    f"Backing off knee on leg {leg} by {KNEE_BACKOFF} "
                    f"counts to pos {backed_off_knee}."
                )
                positions[knee_servo_idx] = backed_off_knee
                esp32.servos_set_position(positions)
                time.sleep(0.2)

        # 2) Hip limit (axis 1), monitoring knee load
        for rep in range(JOINT_REPEAT_COUNT):
            print(f"    Hip sweep repeat {rep + 1}/{JOINT_REPEAT_COUNT}")
            hip_pos_limit = sweep_until_limit(
                esp32=esp32,
                positions=positions,
                leg_index=leg,
                axis_index=1,
                direction=hip_dir,
                load_threshold=hip_load_threshold,
                monitor_knee_load=True,
            )
            hip_reps[rep] = int(hip_pos_limit)

            if HIP_BACKOFF > 0:
                backed_off_hip = hip_pos_limit - hip_dir * HIP_BACKOFF
                print(
                    f"Backing off hip on leg {leg} by {HIP_BACKOFF} "
                    f"counts to pos {backed_off_hip}."
                )
                positions[hip_servo_idx] = backed_off_hip
                esp32.servos_set_position(positions)
                time.sleep(0.2)

        # 3) Abduction limit (axis 0)
        for rep in range(JOINT_REPEAT_COUNT):
            print(f"    Abduction sweep repeat {rep + 1}/{JOINT_REPEAT_COUNT}")
            abd_pos_limit = sweep_until_limit(
                esp32=esp32,
                positions=positions,
                leg_index=leg,
                axis_index=0,
                direction=abd_dir,
                load_threshold=abd_load_threshold,
            )
            abd_reps[rep] = int(abd_pos_limit)

            if HIP_BACKOFF > 0:
                backed_off_abd = abd_pos_limit - abd_dir * HIP_BACKOFF
                print(
                    f"Backing off abduction on leg {leg} by {HIP_BACKOFF} "
                    f"counts to pos {backed_off_abd}."
                )
                positions[abd_servo_idx] = backed_off_abd
                esp32.servos_set_position(positions)
                time.sleep(0.2)

        measured_limits[0, leg] = int(np.median(abd_reps))
        measured_limits[1, leg] = int(np.median(hip_reps))
        measured_limits[2, leg] = int(np.median(knee_reps))

        print(
            f"Leg {leg} repeats (abd, hip, knee):\n"
            f"  abduction: {abd_reps}\n"
            f"  hip      : {hip_reps}\n"
            f"  knee     : {knee_reps}"
        )
        print(
            f"Leg {leg} median limits: "
            f"abd={measured_limits[0, leg]}, "
            f"hip={measured_limits[1, leg]}, "
            f"knee={measured_limits[2, leg]}"
        )

        # After finishing this leg, return its joints to neutral (512)
        for axis in (0, 1, 2):
            servo_idx = leg * 3 + axis
            positions[servo_idx] = 512
        esp32.servos_set_position(positions)
        time.sleep(0.2)

    print("\nMeasured limits (servo positions):")
    print(measured_limits)
    print("Target limits (servo positions):")
    print(JOINT_LIMIT_TARGETS)

    raw_offsets = measured_limits - JOINT_LIMIT_TARGETS
    offsets = raw_offsets * OFFSET_SIGNS
    print("Computed offsets (measured - target) with sign correction:")
    print(offsets)

    print("\nPer-leg joint offsets (servo counts):")
    for leg in legs:
        abd_off = int(offsets[0, leg])
        hip_off = int(offsets[1, leg])
        knee_off = int(offsets[2, leg])
        print(
            f"Leg {leg}: abduction_offset={abd_off}, "
            f"hip_offset={hip_off}, knee_offset={knee_off}"
        )

    print("\nApplying offsets at neutral pose and saving calibration...")
    positions = [512] * 12
    for leg in legs:
        for axis in (0, 1, 2):
            servo_idx = leg * 3 + axis
            offset = int(offsets[axis, leg])
            cmd = 512 + offset
            if cmd < 0:
                cmd = 0
            elif cmd > 1023:
                cmd = 1023
            positions[servo_idx] = cmd

    print("Commanded positions for calibration pose:", positions)
    esp32.servos_set_position(positions)
    time.sleep(0.5)

    esp32.save_calibration()
    print("Calibration saved via ESP32.")

    # Optional: IMU-based leveling adjustment
    ans = input("\nApply simple IMU-based leveling adjustment now? [y/N]: ").strip().lower()
    if ans == "y":
        apply_leveling_adjustment(esp32, ref_pitch, ref_roll)
    else:
        print("Skipping IMU-based leveling adjustment.")


if __name__ == "__main__":
    main()
