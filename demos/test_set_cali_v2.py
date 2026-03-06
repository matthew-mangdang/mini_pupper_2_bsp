import time
from typing import Optional

import numpy as np

from MangDang.mini_pupper.ESP32Interface import ESP32Interface


"""Apply servo calibration offsets based on measured joint limits on Mini Pupper 2.

This script assumes you have already measured per-joint limit positions
(using test_find_cali_v2.py) and decided on target
limit values for each joint of each leg.

For selected legs, it will:
- Use hard-coded target limit positions in JOINT_LIMIT_TARGETS (3x4 array).
- Re-measure the current mechanical limits for each joint.
- Compute offsets in servo counts so that the measured limits map to the
  desired target limits.
- Command a calibration pose at neutral (512) plus these offsets.

Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left.
Axis indices: 0=abduction, 1=hip/thigh, 2=knee.
"""


JOINT_LIMIT_TARGETS = np.array(
    [
        [224, 800, 791, 221],  # abduction limits for legs 0..3
        [425, 579, 390, 604],  # hip/thigh limits for legs 0..3
        [822, 202, 827, 183],  # knee limits for legs 0..3
    ],
    dtype=int,
)

# Optional per-joint, per-leg sign for offsets.
OFFSET_SIGNS = np.ones_like(JOINT_LIMIT_TARGETS, dtype=int)


LOAD_THRESHOLD = 175

LOAD_THRESHOLDS = [
    [200, 200, 200, 200],  # abduction joints, legs 0..3
    [200, 200, 300, 200],  # hip/thigh joints, legs 0..3
    [150, 150, 200, 150],  # knee joints, legs 0..3
]

STEP = 1
MAX_DELTA = 400
DWELL = 0.02

KNEE_BACKOFF = 60
HIP_BACKOFF = 60
HIP_PRE_OFFSET = -100

JOINT_REPEAT_COUNT = 3


def move_to_neutral_pose(esp32: ESP32Interface, hold_time: float = 0.5) -> None:
    """Move all servos to their neutral software-calibrated position (512)."""
    print("Moving to neutral pose (all servos to 512)...")
    positions = [512] * 12
    esp32.servos_set_position(positions)
    time.sleep(hold_time)


def get_servo_id(leg_index: int, axis_index: int) -> int:
    """Return the servo channel index (0..11) for a leg/joint pair."""
    return leg_index * 3 + axis_index


def read_int(prompt: str, default: int, min_value: Optional[int] = None) -> int:
    """Read an int from stdin with a default and optional minimum clamp."""
    s = input(prompt).strip()
    if not s:
        value = default
    else:
        try:
            value = int(s)
        except ValueError:
            value = default
    if min_value is not None and value < min_value:
        value = min_value
    return value


def sweep_until_limit(
    esp32: ESP32Interface,
    positions: list[int],
    leg_index: int,
    axis_index: int,
    direction: int,
    load_threshold: int,
    step: int = STEP,
    max_delta: int = MAX_DELTA,
    dwell: float = DWELL,
    monitor_knee_load: bool = False,
) -> int:
    """Sweep one joint until a load threshold or max_delta is reached."""
    axis_name = {0: "Abduction", 1: "Hip/Thigh", 2: "Knee"}[axis_index]

    servo_idx = get_servo_id(leg_index, axis_index)
    start_pos = positions[servo_idx]
    target_pos = start_pos + direction * max_delta

    print(f"\n=== Measuring {axis_name} limit on leg {leg_index} ===")
    print(f"Start pos: {start_pos}, target pos: {target_pos} (dir {direction:+d})")

    load_idx = get_servo_id(leg_index, axis_index)

    knee_load_idx = None
    if monitor_knee_load and axis_index != 2:
        knee_load_idx = get_servo_id(leg_index, 2)

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


def main() -> None:
    esp32 = ESP32Interface()
    print(f"WARNING: All the parameters in this script are hard-coded. Please review and edit the script before running.")
    time.sleep(2.0)



    if not JOINT_LIMIT_TARGETS.any():
        print(
            "WARNING: JOINT_LIMIT_TARGETS is all zeros. "
            "Please fill it with your target limits before running."
        )

    move_to_neutral_pose(esp32, hold_time=0.5)
    positions: list[int] = [512] * 12   # Initializ position list

    legs = parse_leg_selection()
    if not legs:
        print("No valid legs selected, exiting.")
        return

    measured_limits = np.zeros((3, 4), dtype=int)

    for leg in legs:
        print(f"\n=== Measuring limits for leg {leg} ===")
        abd_load_threshold = LOAD_THRESHOLDS[0][leg]
        hip_load_threshold = LOAD_THRESHOLDS[1][leg]
        knee_load_threshold = LOAD_THRESHOLDS[2][leg]
        print(
            "Using load thresholds for leg "
            f"{leg}: abd={abd_load_threshold}, hip={hip_load_threshold}, "
            f"knee={knee_load_threshold}."
        )

        for axis in (0, 1, 2):
            servo_idx = get_servo_id(leg, axis)
            positions[servo_idx] = 512
        esp32.servos_set_position(positions)
        time.sleep(0.2)

        abd_servo_idx = get_servo_id(leg, 0)
        hip_servo_idx = get_servo_id(leg, 1)
        knee_servo_idx = get_servo_id(leg, 2)

        knee_dir = 1 if leg in (0, 2) else -1
        hip_dir = -1 if leg in (0, 2) else 1
        abd_dir = -1 if leg in (0, 3) else 1

        if HIP_PRE_OFFSET != 0:
            pre_pos = 512 + hip_dir * HIP_PRE_OFFSET
            if pre_pos < 0:
                pre_pos = 0
            elif pre_pos > 1023:
                pre_pos = 1023
            print(
                f"Pre-positioning hip on leg {leg} by {HIP_PRE_OFFSET} "
                f"counts to pos {pre_pos}."
            )
            positions[hip_servo_idx] = pre_pos
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        knee_reps = np.zeros(JOINT_REPEAT_COUNT, dtype=int)
        hip_reps = np.zeros(JOINT_REPEAT_COUNT, dtype=int)
        abd_reps = np.zeros(JOINT_REPEAT_COUNT, dtype=int)

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

        for axis in (0, 1, 2):
            servo_idx = get_servo_id(leg, axis)
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

    print("\nApplying offsets at neutral pose (not saving calibration)...")
    positions = [512] * 12
    for leg in legs:
        for axis in (0, 1, 2):
            servo_idx = get_servo_id(leg, axis)
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

    # esp32.save_calibration()
    print("Calibration not saved via ESP32.")


if __name__ == "__main__":
    main()
