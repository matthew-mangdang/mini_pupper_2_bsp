import time
import numpy as np

from MangDang.mini_pupper.ESP32Interface import ESP32Interface

"""Apply servo calibration offsets based on previously measured joint limits.

This script assumes you have already measured per-joint limit positions
(e.g. using auto_calibrate_limits.py) and decided on target limit values
for each joint of each leg.

It will:
- Hard-code those target limit positions in JOINT_LIMIT_TARGETS (3x4 array).
- Re-run the same limit-finding routine to measure the CURRENT limit
  positions for each joint.
- Compute offsets = target_limit - measured_limit (in servo counts).
- Move the robot to neutral pose (all commands = 512), add these offsets
  to each servo command, and send that pose.
- Call esp32.save_calibration(), so the ESP32 stores these offsets as
  the new calibration.

Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left
Axis indices: 0=abduction, 1=hip/thigh, 2=knee
"""


# === USER-DEFINED CALIBRATION TARGETS ===
# Fill this 3x4 array with the servo positions (0-1023) that
# you consider to be the desired mechanical limits for each
# joint of each leg.
#
# Row 0: abduction, Row 1: hip, Row 2: knee
# Col 0-3: legs 0..3
JOINT_LIMIT_TARGETS = np.array([
    [205, 816, 817, 312],  # abduction limits for legs 0..3 (to be filled)
    [329, 611, 388, 509],  # hip/thigh limits for legs 0..3 (to be filled)
    [833, 182, 829, 498],  # knee limits for legs 0..3 (to be filled)
], dtype=int)

# Optional per-joint, per-leg sign for offsets.
# By default all are +1; if a particular joint is mechanically reversed
# (so that a positive offset moves it in the opposite physical direction),
# you can flip its sign here.
OFFSET_SIGNS = np.ones_like(JOINT_LIMIT_TARGETS, dtype=int)

# Example: knees on leg 1 (front-left) and leg 2 (back-right)
# need inverted offset direction:
# joint index 2 = knee, leg indices 1 and 2
OFFSET_SIGNS[2, 1] = -1  # leg 1 knee
OFFSET_SIGNS[2, 2] = -1  # leg 2 knee


# Sweep parameters (no user input; adjust here if needed)
LOAD_THRESHOLD = 250
STEP = 1
MAX_DELTA = 400
DWELL = 0.02

# Fixed movement shaping parameters (matching test_find_cali.py behavior)
KNEE_BACKOFF = 60      # servo counts to back off from knee hard stop before hip sweep
HIP_BACKOFF = 60       # servo counts to back off from hip hard stop before abduction sweep
HIP_PRE_OFFSET = -150  # servo counts to move hip before starting knee sweep


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
            positions[servo_idx] = 1023.

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
    print(f"Final {axis_name} measured limit on leg {leg_index}: {final_pos}")
    return final_pos


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


def main():
    esp32 = ESP32Interface()

    if not JOINT_LIMIT_TARGETS.any():
        print("WARNING: JOINT_LIMIT_TARGETS is all zeros. Please fill it with your target limits before running.")

    # Start from neutral hardware commands
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

        # Direction rules copied from your test script
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
            # Clamp to valid servo range
            if pre_pos < 0:
                pre_pos = 0
            elif pre_pos > 1023:
                pre_pos = 1023
            print(f"Pre-positioning hip on leg {leg} by {HIP_PRE_OFFSET} counts to pos {pre_pos}.")
            positions[hip_servo_idx] = pre_pos
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        # 1) Knee limit (axis 2)
        measured_limits[2, leg] = sweep_until_limit(
            esp32=esp32,
            positions=positions,
            leg_index=leg,
            axis_index=2,
            direction=knee_dir,
        )

        # Back off the knee slightly from its hard stop to reduce
        # holding torque/current before sweeping the hip.
        if KNEE_BACKOFF > 0:
            backed_off_knee = measured_limits[2, leg] - knee_dir * KNEE_BACKOFF
            print(
                f"Backing off knee on leg {leg} by {KNEE_BACKOFF} "
                f"counts to pos {backed_off_knee}."
            )
            positions[knee_servo_idx] = backed_off_knee
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        # 2) Hip limit (axis 1), with knee held at its limit
        measured_limits[1, leg] = sweep_until_limit(
            esp32=esp32,
            positions=positions,
            leg_index=leg,
            axis_index=1,
            direction=hip_dir,
        )

        # Back off the hip slightly from its hard stop to reduce
        # holding torque/current before sweeping abduction.
        if HIP_BACKOFF > 0:
            backed_off_hip = measured_limits[1, leg] - hip_dir * HIP_BACKOFF
            print(
                f"Backing off hip on leg {leg} by {HIP_BACKOFF} "
                f"counts to pos {backed_off_hip}."
            )
            positions[hip_servo_idx] = backed_off_hip
            esp32.servos_set_position(positions)
            time.sleep(0.2)

        # 3) Abduction limit (axis 0), with knee+hip held
        measured_limits[0, leg] = sweep_until_limit(
            esp32=esp32,
            positions=positions,
            leg_index=leg,
            axis_index=0,
            direction=abd_dir,
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

    # Compute raw offsets in servo counts
    raw_offsets = JOINT_LIMIT_TARGETS - measured_limits
    # Apply per-joint/leg sign corrections (e.g. for reversed servos)
    offsets = raw_offsets * OFFSET_SIGNS
    print("Computed offsets (target - measured) with sign correction:")
    print(offsets)

    # Also print per-leg offsets, highlighting knee and hip, only for selected legs
    print("\nPer-leg joint offsets (servo counts):")
    for leg in legs:
        abd_off = int(offsets[0, leg])
        hip_off = int(offsets[1, leg])
        knee_off = int(offsets[2, leg])
        print(
            f"Leg {leg}: abduction_offset={abd_off}, "
            f"hip_offset={hip_off}, knee_offset={knee_off}"
        )

    # Apply offsets at neutral pose, then save calibration
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
    time.sleep(5.5)

    # Let ESP32 firmware capture current positions as new calibration
    #esp32.save_calibration()
    print("Calibration not saved via ESP32.")


if __name__ == "__main__":
    main()
