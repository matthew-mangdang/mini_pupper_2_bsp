import time
import numpy as np

from MangDang.mini_pupper.ESP32Interface import ESP32Interface


"""Auto-calibrate servo position limits for abduction, knee and hip joints.

Repeat each legs

For each selected leg:
- Move all servos to hardware neutral (position 512).
- Sweep the knee joint (axis 2) in the negative direction until the load threshold is reached;
  record the corresponding servo position as the knee limit.
- With the knee fixed at that angle, sweep the hip/thigh joint (axis 1) negatively until
  the load threshold is reached; record that servo position as the hip limit.
 - Optionally, sweep the abduction joint (axis 0) similarly to find its limit.
- Print the discovered limits for each leg.

Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left.
"""


LOAD_THRESHOLD_DEFAULT = 175
KNEE_BACKOFF_DEFAULT = 60  # servo counts to back off from knee hard stop before hip sweep
HIP_BACKOFF_DEFAULT = 60   # servo counts to back off from hip hard stop before abduction sweep
HIP_PRE_OFFSET_DEFAULT = -100  # servo counts to move hip before starting knee sweep

# Default number of times to repeat the limit-finding sequence per leg
NUM_LIMIT_MEASUREMENTS_DEFAULT = 5

# Default number of times to repeat each joint sweep (knee/hip/abduction)
# once the joint is near its limit. This lets you "tap" the mechanical
# stop multiple times per joint without re-running the whole leg sequence.
JOINT_REPEAT_DEFAULT = 3


def move_to_neutral_pose(esp32: ESP32Interface, hold_time: float = 2.0) -> None:
    """Move all servos to their neutral hardware position (512)."""
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
    """Sweep one servo position negatively from its current value until a load threshold.

    - Operates purely on ESP32 servo positions (no joint angles).
    - `positions` is updated in-place and left at the final value.
    - Returns the final servo position used as the limit.
    """
    axis_name = {0: "Abduction", 1: "Hip/Thigh", 2: "Knee"}[axis_index]

    # With the standard Mini Pupper mapping, ESP32 servo indices are:
    # leg*3 + axis (0=abd,1=hip,2=knee) → 0..11
    servo_idx = leg_index * 3 + axis_index
    start_pos = positions[servo_idx]
    target_pos = start_pos + direction * max_delta

    print(f"\n=== Leg {leg_index} {axis_name} sweep (positions) ===")
    print(f"Start pos: {start_pos}, target pos: {target_pos} (direction {direction:+d})")

    load_idx = leg_joint_load_index(leg_index, axis_index)

    # Optionally also monitor the knee load (axis 2) on this leg, so that
    # when sweeping the hip we stop if we push the knee harder into its
    # mechanical stop.
    knee_load_idx = None
    if monitor_knee_load and axis_index != 2:
        knee_load_idx = leg_joint_load_index(leg_index, 2)

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

    # Start by moving servos to their neutral hardware position using ESP32 positions
    print("Initializing all servos to pos=512 (neutral)...")
    positions: list[int] = [512] * 12
    esp32.servos_set_position(positions)
    time.sleep(0.5)


    ### Input handling
    legs = parse_leg_selection()
    if not legs:
        print("No valid legs selected, exiting.")
        return

    # Global load threshold selection (used as default for all legs)
    try:
        user_thr = input(f"Enter load threshold (default {LOAD_THRESHOLD_DEFAULT}): ").strip()
        load_threshold = int(user_thr) if user_thr else LOAD_THRESHOLD_DEFAULT
    except ValueError:
        load_threshold = LOAD_THRESHOLD_DEFAULT
    print(f"Using load threshold of {load_threshold} (will be default for each leg).")

    # Optional per-leg overrides for the load threshold
    per_leg_thresholds: dict[int, int] = {}
    for leg in legs:
        try:
            user_leg_thr = input(
                f"Enter load threshold for leg {leg} "
                f"(default {load_threshold}): "
            ).strip()
            leg_thr = int(user_leg_thr) if user_leg_thr else load_threshold
        except ValueError:
            leg_thr = load_threshold
        per_leg_thresholds[leg] = leg_thr
    print("Per-leg load thresholds:")
    for leg in legs:
        print(f"  leg {leg}: {per_leg_thresholds[leg]}")

    # Knee backoff selection (how many counts to move away from knee hard stop)
    try:
        user_backoff = input(f"Enter knee backoff counts (default {KNEE_BACKOFF_DEFAULT}): ").strip()
        knee_backoff = int(user_backoff) if user_backoff else KNEE_BACKOFF_DEFAULT
    except ValueError:
        knee_backoff = KNEE_BACKOFF_DEFAULT
    if knee_backoff < 0:
        knee_backoff = 0
    print(f"Using knee backoff of {knee_backoff} counts.")

    # Hip backoff selection (how many counts to move away from hip hard stop)
    try:
        user_hip_backoff = input(
            f"Enter hip backoff counts (default {HIP_BACKOFF_DEFAULT}): "
        ).strip()
        hip_backoff = int(user_hip_backoff) if user_hip_backoff else HIP_BACKOFF_DEFAULT
    except ValueError:
        hip_backoff = HIP_BACKOFF_DEFAULT
    if hip_backoff < 0:
        hip_backoff = 0
    print(f"Using hip backoff of {hip_backoff} counts.")

    # Hip pre-offset selection (signed counts to move hip before starting knee sweep)
    try:
        user_hip_offset = input(
            f"Enter hip pre-offset counts before knee sweep (default {HIP_PRE_OFFSET_DEFAULT}): "
        ).strip()
        hip_pre_offset = int(user_hip_offset) if user_hip_offset else HIP_PRE_OFFSET_DEFAULT
    except ValueError:
        hip_pre_offset = HIP_PRE_OFFSET_DEFAULT
    # hip_pre_offset is allowed to be negative to move in the opposite direction
    print(f"Using hip pre-offset of {hip_pre_offset} counts (can be negative).")

    # Number of repeated measurements per leg (user-configurable).
    # If set to 1, behavior matches the original script (single pass).
    try:
        user_reps = input(
            f"Enter number of repeated measurements per leg "
            f"(default {NUM_LIMIT_MEASUREMENTS_DEFAULT}): "
        ).strip()
        num_limit_measurements = int(user_reps) if user_reps else NUM_LIMIT_MEASUREMENTS_DEFAULT
    except ValueError:
        num_limit_measurements = NUM_LIMIT_MEASUREMENTS_DEFAULT
    if num_limit_measurements < 1:
        num_limit_measurements = 1
    print(f"Using {num_limit_measurements} measurement(s) per leg.")

    # Number of times to repeat each joint sweep (per trial).
    try:
        user_joint_reps = input(
            f"Enter number of repeats per joint when sweeping limits "
            f"(default {JOINT_REPEAT_DEFAULT}): "
        ).strip()
        joint_repeat_count = int(user_joint_reps) if user_joint_reps else JOINT_REPEAT_DEFAULT
    except ValueError:
        joint_repeat_count = JOINT_REPEAT_DEFAULT
    if joint_repeat_count < 1:
        joint_repeat_count = 1
    print(f"Using {joint_repeat_count} repeat(s) per joint sweep.")

    # Results storage: 3x4 array [joint, leg]
    # joint 0=abduction, 1=hip, 2=knee; leg 0..3
    # For robustness, we can measure each joint limit multiple
    # times per leg and take the median across trials.
    limits = np.zeros((3, 4), dtype=int)

    for leg in legs:
        print(f"\n=== Calibrating leg {leg} (repeating {num_limit_measurements} times) ===")
        leg_load_threshold = per_leg_thresholds.get(leg, load_threshold)
        print(f"Using load threshold {leg_load_threshold} for leg {leg}.")

        # Arrays to hold repeated measurements for this leg
        # Shape: [joint, trial]
        leg_trials = np.zeros((3, num_limit_measurements), dtype=int)

        for trial in range(num_limit_measurements):
            print(f"\n--- Leg {leg} trial {trial + 1}/{num_limit_measurements} ---")

            # Ensure this leg's abduction, hip and knee start from neutral servo positions (512)
            abd_servo_idx = leg * 3 + 0  # axis 0 = abduction
            knee_servo_idx = leg * 3 + 2  # axis 2 = knee
            hip_servo_idx = leg * 3 + 1   # axis 1 = hip
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
            knee_pos_limit = None
            for rep in range(joint_repeat_count):
                print(f"    Knee sweep repeat {rep + 1}/{joint_repeat_count}")
                knee_pos_limit = sweep_joint_negative_until_load(
                    esp32=esp32,
                    positions=positions,
                    leg_index=leg,
                    axis_index=2,  # knee
                    load_threshold=leg_load_threshold,
                    direction=knee_dir,
                    step=1,
                    max_delta=400,
                    dwell=0.02,
                )

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
            hip_pos_limit = None
            for rep in range(joint_repeat_count):
                print(f"    Hip sweep repeat {rep + 1}/{joint_repeat_count}")
                hip_pos_limit = sweep_joint_negative_until_load(
                    esp32=esp32,
                    positions=positions,
                    leg_index=leg,
                    axis_index=1,  # hip/thigh
                    load_threshold=leg_load_threshold,
                    direction=hip_dir,
                    step=1,
                    max_delta=400,
                    dwell=0.02,
                    monitor_knee_load=True,
                )
                
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

            # 3) Optionally sweep abduction (axis 0) using its own direction rule
            abd_pos_limit = sweep_joint_negative_until_load(
                esp32=esp32,
                positions=positions,
                leg_index=leg,
                axis_index=0,  # abduction
                load_threshold=leg_load_threshold,
                direction=abd_dir,
                step=1,
                max_delta=400,
                dwell=0.02,
            )

            # Store this trial's limits in [joint, trial] order: 0=abd, 1=hip, 2=knee
            leg_trials[0, trial] = int(abd_pos_limit)
            leg_trials[1, trial] = int(hip_pos_limit)
            leg_trials[2, trial] = int(knee_pos_limit)

            # After each trial, return this leg's joints to neutral (512)
            for axis in (0, 1, 2):
                servo_idx = leg * 3 + axis
                positions[servo_idx] = 512
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        # Take median across trials for this leg and store in limits
        limits[0, leg] = int(np.median(leg_trials[0, :]))
        limits[1, leg] = int(np.median(leg_trials[1, :]))
        limits[2, leg] = int(np.median(leg_trials[2, :]))

        print(f"Leg {leg} trial limits (abd, hip, knee):")
        for trial in range(num_limit_measurements):
            print(
                f"  trial {trial + 1}: "
                f"abd={leg_trials[0, trial]}, "
                f"hip={leg_trials[1, trial]}, "
                f"knee={leg_trials[2, trial]}"
            )
        print(
            f"Leg {leg} median limits: "
            f"abd={limits[0, leg]}, "
            f"hip={limits[1, leg]}, "
            f"knee={limits[2, leg]}"
        )

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
