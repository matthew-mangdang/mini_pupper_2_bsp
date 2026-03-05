import time
import numpy as np

from MangDang.mini_pupper.ESP32Interface import ESP32Interface


"""Auto-calibrate servo position limits for abduction, knee and hip joints.

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


LOAD_THRESHOLD_DEFAULT = 250
KNEE_BACKOFF_DEFAULT = 60  # servo counts to back off from knee hard stop before hip sweep
HIP_BACKOFF_DEFAULT = 60   # servo counts to back off from hip hard stop before abduction sweep
HIP_PRE_OFFSET_DEFAULT = -200  # servo counts to move hip before starting knee sweep


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
        print(f"  pos={positions[servo_idx]:4d}, load={this_load:5d}")

        if abs(this_load) >= load_threshold:
            print(f"Reached load threshold |load| >= {load_threshold} at pos {positions[servo_idx]}")
            break

    final_pos = positions[servo_idx]
    print(f"Final {axis_name} servo pos: {final_pos}")

    return final_pos


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

    # Load threshold selection
    try:
        user_thr = input(f"Enter load threshold (default {LOAD_THRESHOLD_DEFAULT}): ").strip()
        load_threshold = int(user_thr) if user_thr else LOAD_THRESHOLD_DEFAULT
    except ValueError:
        load_threshold = LOAD_THRESHOLD_DEFAULT
    print(f"Using load threshold of {load_threshold}.")

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

    # Results storage: 3x4 array [joint, leg]
    # joint 0=abduction, 1=hip, 2=knee; leg 0..3
    limits = np.zeros((3, 4), dtype=int)

    for leg in legs:
        print(f"\n=== Calibrating leg {leg} ===")

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
        """
        # Special Abduction direction for leg 3: 
        # legs 0,1,2 behave correctly with the
        # same rule as hip_dir, but leg 3 was reversed in practice.
        if leg == 3:
            abd_dir = -1 if hip_dir > 0 else 1
        else:
            abd_dir = hip_dir
        """
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

        # 1) Sweep knee (axis 2) in chosen direction in servo position until load threshold; HOLD knee there
        knee_pos_limit = sweep_joint_negative_until_load(
            esp32=esp32,
            positions=positions,
            leg_index=leg,
            axis_index=2,  # knee
            load_threshold=load_threshold,
            direction=knee_dir,
            step=1,
            max_delta=400,
            dwell=0.02,
        )

        # Optional: back off the knee slightly from its hard stop to
        # reduce holding torque/current before sweeping the hip.
        if knee_backoff > 0:
            backed_off_pos = knee_pos_limit - knee_dir * knee_backoff
            print(f"Backing off knee on leg {leg} by {knee_backoff} counts to pos {backed_off_pos}.")
            positions[knee_servo_idx] = backed_off_pos
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        # 2) With knee fixed, sweep hip/thigh (axis 1) in chosen direction in servo position until threshold
        hip_pos_limit = sweep_joint_negative_until_load(
            esp32=esp32,
            positions=positions,
            leg_index=leg,
            axis_index=1,  # hip/thigh
            load_threshold=load_threshold,
            direction=hip_dir,
            step=1,
            max_delta=400,
            dwell=0.02,
        )
        
        # Optional: back off the hip slightly from its hard stop to
        # reduce holding torque/current before sweeping abduction.
        if hip_backoff > 0:
            backed_off_hip = hip_pos_limit - hip_dir * hip_backoff
            print(f"Backing off hip on leg {leg} by {hip_backoff} counts to pos {backed_off_hip}.")
            positions[hip_servo_idx] = backed_off_hip
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        # 3) Optionally sweep abduction (axis 0) using its own direction rule
        abd_pos_limit = sweep_joint_negative_until_load(
            esp32=esp32,
            positions=positions,
            leg_index=leg,
            axis_index=0,  # abduction
            load_threshold=load_threshold,
            direction=abd_dir,
            step=1,
            max_delta=400,
            dwell=0.02,
        )

        # Store in [joint, leg] order: 0=abd, 1=hip, 2=knee
        limits[0, leg] = int(abd_pos_limit)
        limits[1, leg] = int(hip_pos_limit)
        limits[2, leg] = int(knee_pos_limit)

        # After finishing this leg, return its joints to neutral (512)
        for axis in (0, 1, 2):
            servo_idx = leg * 3 + axis
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
