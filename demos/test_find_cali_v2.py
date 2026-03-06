import time
from typing import Optional

import numpy as np

from MangDang.mini_pupper.ESP32Interface import ESP32Interface


"""Find calibrated servo position limits for abduction, knee and hip joints.

For each selected leg:
- Move all servos to hardware neutral (position 512).
- Sweep the knee joint (axis 2) in the negative direction until the load threshold is reached;
  record the corresponding servo position as the knee limit.
- With the knee fixed at that angle, sweep the hip/thigh joint (axis 1) negatively until
  the load threshold is reached; record that servo position as the hip limit.
- Sweep the abduction joint (axis 0) similarly to find its limit.
- Print the discovered limits for each leg.

Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left.
"""


LOAD_THRESHOLD_DEFAULT = 175
KNEE_BACKOFF_DEFAULT = 60  # servo counts to back off from knee hard stop before hip sweep
HIP_BACKOFF_DEFAULT = 60   # servo counts to back off from hip hard stop before abduction sweep
HIP_PRE_OFFSET_DEFAULT = -100  # servo counts to move hip before starting knee sweep

# Default number of times to repeat each joint sweep (knee/hip/abduction)
# once the joint is near its limit. This lets you "tap" the mechanical
# stop multiple times per joint without re-running the whole leg sequence.
JOINT_REPEAT_DEFAULT = 3

# Per-joint, per-leg load thresholds during limit finding.
# Shape is [joint, leg] with:
#   joint 0 = abduction, 1 = hip/thigh, 2 = knee
#   leg   0 = front-right, 1 = front-left, 2 = back-right, 3 = back-left
# Initialize all to reasonable defaults and customize individual entries as needed.
LOAD_THRESHOLDS = [
    [200, 200, 200, 200],  # abduction joints, legs 0..3
    [200, 200, 300, 200],  # hip/thigh joints, legs 0..3
    [150, 250, 200, 250],  # knee joints, legs 0..3
]


def move_to_neutral_pose(esp32: ESP32Interface, hold_time: float = 2.0) -> None:
    """Move all servos to their neutral software calibrated position (512)."""
    print("Moving to neutral pose (all servos to 512)...")
    positions = [512] * 12
    esp32.servos_set_position(positions)
    time.sleep(hold_time)


def get_servo_index(leg_index: int, axis_index: int) -> int:
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


def sweep_joint_negative_until_load(
    esp32: ESP32Interface,
    positions: list[int],
    leg_index: int,
    axis_index: int,
    load_threshold: int,
    direction: int,
    step: int = 2,
    max_delta: int = 400,
    dwell: float = 0.02,
    monitor_knee_load: bool = False,
) -> int:
    """Sweep one servo from its current value until a load threshold is reached."""
    axis_name = {0: "Abduction", 1: "Hip/Thigh", 2: "Knee"}[axis_index]

    servo_idx = get_servo_index(leg_index, axis_index)
    start_pos = positions[servo_idx]
    target_pos = start_pos + direction * max_delta

    print(f"\n=== Leg {leg_index} {axis_name} sweep (positions) ===")
    print(f"Start pos: {start_pos}, target pos: {target_pos} (direction {direction:+d})")

    load_idx = get_servo_index(leg_index, axis_index)

    # Optionally also monitor the knee load (axis 2) on this leg, so that
    # when sweeping the hip we stop if we push the knee harder into its
    # mechanical stop.
    knee_load_idx = None
    if monitor_knee_load and axis_index != 2:
        knee_load_idx = get_servo_index(leg_index, 2)

    # Helper for directional loop bound
    def within_range() -> bool:
        if direction < 0:
            return positions[servo_idx] >= target_pos
        else:
            return positions[servo_idx] <= target_pos

    while within_range():
        # Step one increment in the desired direction
        positions[servo_idx] += direction * step
        # Clamp so we don't overshoot beyond target_pos
        if direction < 0 and positions[servo_idx] < target_pos:
            positions[servo_idx] = target_pos
        elif direction > 0 and positions[servo_idx] > target_pos:
            positions[servo_idx] = target_pos
        esp32.servos_set_position(positions)
        time.sleep(dwell)

        loads = esp32.servos_get_load()
        if loads is None:
            print("Failed to read loads, stopping sweep.")
            break

        this_load = loads[load_idx]
        knee_load = loads[knee_load_idx] if knee_load_idx is not None else None
        #print(f"  pos={positions[servo_idx]:4d}, load={this_load:5d}, knee_load={knee_load if knee_load is not None else 0:5d}")

        # Stop if the swept joint hits its load threshold or, when
        # monitoring, if the knee on this leg does.
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
    # so we record the measured limit, not just the last commanded value.
    measured_positions = esp32.servos_get_position()
    if measured_positions is not None and len(measured_positions) > servo_idx:
        final_pos = measured_positions[servo_idx]
        print(f"Final {axis_name} servo pos (measured from ESP32): {final_pos}")
    else:
        final_pos = positions[servo_idx]
        print(f"Final {axis_name} servo pos (commanded only): {final_pos}")

    return int(final_pos)


def parse_leg_selection() -> list[int]:
    print("Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left")
    s = input("Enter legs to calibrate (e.g. '0,2' or 'all'): ").strip().lower()
    if s == "all" or s == "":
        return [0, 1, 2, 3]
    legs: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if 0 <= i <= 3:
            legs.append(i)
    return sorted(set(legs))


def main():
    esp32 = ESP32Interface()
    move_to_neutral_pose(esp32, hold_time=0.5)
    positions: list[int] = [512] * 12  # Initialize position list

    ###     Input handling      ###
    legs = parse_leg_selection()
    if not legs:
        print("No valid legs selected, exiting.")
        return

    knee_backoff = read_int(
        f"Enter knee backoff counts (default {KNEE_BACKOFF_DEFAULT}): ",
        KNEE_BACKOFF_DEFAULT,
        min_value=0,
    )
    print(f"Using knee backoff of {knee_backoff} counts.")

    hip_backoff = read_int(
        f"Enter hip backoff counts (default {HIP_BACKOFF_DEFAULT}): ",
        HIP_BACKOFF_DEFAULT,
        min_value=0,
    )
    print(f"Using hip backoff of {hip_backoff} counts.")

    hip_pre_offset = read_int(
        f"Enter hip pre-offset counts before knee sweep (default {HIP_PRE_OFFSET_DEFAULT}): ",
        HIP_PRE_OFFSET_DEFAULT,
    )
    print(f"Using hip pre-offset of {hip_pre_offset} counts (can be negative).")

    joint_repeat_count = read_int(
        "Enter number of repeats per joint when sweeping limits "
        f"(default {JOINT_REPEAT_DEFAULT}): ",
        JOINT_REPEAT_DEFAULT,
        min_value=1,
    )
    print(f"Using {joint_repeat_count} repeat(s) per joint sweep.")
    limits = np.zeros((3, 4), dtype=int)

    for leg in legs:
        print(f"\n=== Calibrating leg {leg} ===")
        # Per-joint thresholds for this leg
        abd_load_threshold = LOAD_THRESHOLDS[0][leg]
        hip_load_threshold = LOAD_THRESHOLDS[1][leg]
        knee_load_threshold = LOAD_THRESHOLDS[2][leg]
        print(
            "Using load thresholds for leg "
            f"{leg}: abd={abd_load_threshold}, hip={hip_load_threshold}, "
            f"knee={knee_load_threshold}."
        )
        # Arrays to hold repeated measurements for this leg's joints
        knee_reps = np.zeros(joint_repeat_count, dtype=int)
        hip_reps = np.zeros(joint_repeat_count, dtype=int)
        abd_reps = np.zeros(joint_repeat_count, dtype=int)

        # Ensure this leg's abduction, hip and knee start from neutral servo positions (512)
        abd_servo_idx = get_servo_index(leg, 0)  # axis 0 = abduction
        knee_servo_idx = get_servo_index(leg, 2)  # axis 2 = knee
        hip_servo_idx = get_servo_index(leg, 1)   # axis 1 = hip
        positions[abd_servo_idx] = 512
        positions[knee_servo_idx] = 512
        positions[hip_servo_idx] = 512
        esp32.servos_set_position(positions)
        time.sleep(0.2)

        # Define per-leg sweep directions so all joints move in a
        # consistent physical direction despite mirrored mounting.
        # Right legs (0, 2): negative direction in servo position
        # Left  legs (1, 3): positive direction in servo position
        knee_dir = 1 if leg in (0, 2) else -1
        hip_dir = -1 if leg in (0, 2) else 1
        abd_dir = -1 if leg in (0, 3) else 1

        # Before moving the knee, move the hip a little first by
        # hip_pre_offset counts. This value is signed:
        #   >0 moves along hip_dir, <0 moves opposite hip_dir.
        if hip_pre_offset != 0:
            pre_pos = 512 + hip_dir * hip_pre_offset
            # Clamp to valid servo range

            if pre_pos < 0:
                pre_pos = 0
            elif pre_pos > 1023:
                pre_pos = 1023
            print(f"Pre-positioning hip on leg {leg} by {hip_pre_offset} counts to pos {pre_pos}.")
            positions[hip_servo_idx] = pre_pos
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        # 1) Sweep knee (axis 2) in chosen direction until load threshold; repeat locally
        # and then HOLD knee near its limit (backed off by knee_backoff).
        for rep in range(joint_repeat_count):
            print(f"    Knee sweep repeat {rep + 1}/{joint_repeat_count}")
            knee_pos_limit = sweep_joint_negative_until_load(
                esp32=esp32,
                positions=positions,
                leg_index=leg,
                axis_index=2,  # knee
                load_threshold=knee_load_threshold,
                direction=knee_dir,
                step=1,
                max_delta=400,
                dwell=0.02,
            )
            knee_reps[rep] = int(knee_pos_limit)

            # After each hit, back off the knee slightly from its hard stop to
            # reduce holding torque/current. This also positions it for the
            # next repeat or for the upcoming hip sweep.
            if knee_backoff > 0:
                backed_off_pos = knee_pos_limit - knee_dir * knee_backoff
                print(
                    f"Backing off knee on leg {leg} by {knee_backoff} "
                    f"counts to pos {backed_off_pos}."
                )
                positions[knee_servo_idx] = backed_off_pos
                esp32.servos_set_position(positions)
                time.sleep(0.2)

        # 2) With knee fixed near its limit, sweep hip/thigh (axis 1); repeat locally
        for rep in range(joint_repeat_count):
            print(f"    Hip sweep repeat {rep + 1}/{joint_repeat_count}")
            hip_pos_limit = sweep_joint_negative_until_load(
                esp32=esp32,
                positions=positions,
                leg_index=leg,
                axis_index=1,  # hip/thigh
                load_threshold=hip_load_threshold,
                direction=hip_dir,
                step=1,
                max_delta=400,
                dwell=0.02,
                monitor_knee_load=True,
            )
            hip_reps[rep] = int(hip_pos_limit)
            
            # After each hit, back off the hip slightly from its hard stop to
            # reduce holding torque/current and prepare for the next repeat
            # or the abduction sweep.
            if hip_backoff > 0:
                backed_off_hip = hip_pos_limit - hip_dir * hip_backoff
                print(
                    f"Backing off hip on leg {leg} by {hip_backoff} "
                    f"counts to pos {backed_off_hip}."
                )
                positions[hip_servo_idx] = backed_off_hip
                esp32.servos_set_position(positions)
                time.sleep(0.2)

        # 3) Optionally sweep abduction (axis 0) using its own direction rule; repeat locally
        for rep in range(joint_repeat_count):
            print(f"    Abduction sweep repeat {rep + 1}/{joint_repeat_count}")
            abd_pos_limit = sweep_joint_negative_until_load(
                esp32=esp32,
                positions=positions,
                leg_index=leg,
                axis_index=0,  # abduction
                load_threshold=abd_load_threshold,
                direction=abd_dir,
                step=1,
                max_delta=400,
                dwell=0.02,
            )
            abd_reps[rep] = int(abd_pos_limit)

            # After each hit, back off the abduction joint slightly from its
            # hard stop to reduce holding torque/current and prepare for the
            # next repeat.
            if hip_backoff > 0:
                backed_off_abd = abd_pos_limit - abd_dir * hip_backoff
                print(
                    f"Backing off abduction on leg {leg} by {hip_backoff} "
                    f"counts to pos {backed_off_abd}."
                )
                positions[abd_servo_idx] = backed_off_abd
                esp32.servos_set_position(positions)
                time.sleep(0.2)

        # Take median across joint repeats for this leg and store in limits
        limits[0, leg] = int(np.median(abd_reps))
        limits[1, leg] = int(np.median(hip_reps))
        limits[2, leg] = int(np.median(knee_reps))

        print(
            f"Leg {leg} repeats (abd, hip, knee):\n"
            f"  abduction: {abd_reps}\n"
            f"  hip      : {hip_reps}\n"
            f"  knee     : {knee_reps}"
        )
        print(
            f"Leg {leg} median limits: "
            f"abd={limits[0, leg]}, "
            f"hip={limits[1, leg]}, "
            f"knee={limits[2, leg]}"
        )

        # After finishing this leg, return its joints to neutral (512)
        for axis in (0, 1, 2):
            servo_idx = get_servo_index(leg, axis)
            positions[servo_idx] = 512
        esp32.servos_set_position(positions)
        time.sleep(0.2)

    # Summary of discovered limits
    print("\n=== Calibration results (servo positions) ===")
    print("Limits array [joint, leg] (0=abd,1=hip,2=knee):")
    print(limits)

    for leg in legs:
        abd_lim = int(limits[0, leg])
        hip_lim = int(limits[1, leg])
        knee_lim = int(limits[2, leg])
        print(f"Leg {leg}: abd limit pos={abd_lim}, knee limit pos={knee_lim}, hip limit pos={hip_lim}")
        print(f"abd, knee, hip limit for leg {leg}: {abd_lim}, {knee_lim}, {hip_lim}")

    # Return robot to neutral pose at the end
    print("\nDone. Returning to neutral pose via ESP32.")
    move_to_neutral_pose(esp32)


if __name__ == "__main__":
    main()
