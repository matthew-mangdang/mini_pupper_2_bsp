import time
import numpy as np

from MangDang.mini_pupper.HardwareInterface2 import HardwareInterface
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
from MangDang.mini_pupper.Config import ServoParams

# Initial joint angles (rad), same shape as test_joint_limit.py
INITIAL_JOINT_ANGLES = np.array([
    [-0.00644456,  0.00644456, -0.00644456,  0.00644456],  # abduction
    [ 0.88270319,  0.88270319,  0.88270319,  0.88270319],  # hip/thigh
    [-0.69934338, -0.69934338, -0.69934338, -0.69934338],  # knee
])


def move_to_neutral_pose(esp32: ESP32Interface, hold_time: float = 2.0) -> None:
    """Move all servos to their neutral hardware position (512)."""
    print("Moving to neutral pose (all servos to 512)...")
    positions = [512] * 12
    esp32.servos_set_position(positions)
    time.sleep(hold_time)


def leg_joint_load_index(leg_index: int, axis_index: int) -> int:
    """
    Map (leg, axis_index) to index in 12-element load vector:
      [leg0_abd, leg0_hip, leg0_knee,
       leg1_abd, leg1_hip, leg1_knee, ...]
    axis_index: 0=abduction, 1=hip, 2=knee (same as angle array rows).
    """
    return leg_index * 3 + axis_index


def sweep_joint_negative_until_load(
    hw: HardwareInterface,
    esp32: ESP32Interface,
    joint_angles: np.ndarray,
    leg_index: int,
    axis_index: int,
    load_threshold: int = 60,
    step_deg: float = 1.0,
    max_delta_deg: float = 60.0,
    dwell: float = 0.1,
):
    """
    Sweep ONE joint (axis_index) of ONE leg (leg_index) negatively:
      - starting from its current angle in joint_angles
      - decrease angle in steps
      - after each step, read that servo's load
      - stop when |load| >= load_threshold or max_delta reached

    IMPORTANT: This function updates joint_angles IN PLACE and
    leaves the joint at the last angle where it stopped.
    """
    axis_name = {0: "Abduction", 1: "Hip/Thigh", 2: "Knee"}[axis_index]

    start_rad = joint_angles[axis_index, leg_index]
    start_deg = float(np.rad2deg(start_rad))
    target_min_deg = start_deg - max_delta_deg

    print(f"\n=== Leg {leg_index} {axis_name} negative sweep ===")
    print(f"Start angle: {start_deg:.1f} deg, target min: {target_min_deg:.1f} deg")

    current_deg = start_deg
    load_idx = leg_joint_load_index(leg_index, axis_index)

    while current_deg >= target_min_deg:
        # Update only this joint; keep others at whatever is in joint_angles
        joint_angles[axis_index, leg_index] = np.deg2rad(current_deg)
        hw.set_actuator_postions(joint_angles)
        time.sleep(dwell)

        loads = esp32.servos_get_load()
        if loads is None:
            print("Failed to read loads, stopping sweep.")
            break

        this_load = loads[load_idx]
        print(f"  angle={current_deg:6.1f} deg, load={this_load:5d}")

        if abs(this_load) >= load_threshold:
            print(f"Reached load threshold |load| >= {load_threshold} at {current_deg:.1f} deg")
            break

        current_deg -= step_deg

    # Joint remains at last commanded angle in joint_angles
    return joint_angles[axis_index, leg_index]


def parse_leg_selection() -> list[int]:
    print("Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left")
    s = input("Enter legs to test (e.g. '0,2' or 'all'): ").strip().lower()
    if s == "all" or s == "":
        return [0, 1, 2, 3]
    legs = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if 0 <= i <= 3:
            legs.append(i)
    return sorted(set(legs))


def main():
    hw = HardwareInterface(joint_checker_flag=False)
    esp32 = ESP32Interface()

    # Neutral joint angles corresponding to servo hardware neutral (position 512)
    servo_params = ServoParams()
    neutral_angles = servo_params.neutral_angles

    # Start by moving servos to their neutral hardware position
    move_to_neutral_pose(esp32)

    # Work on a mutable copy so we can hold knee position
    current_angles = INITIAL_JOINT_ANGLES.copy()

    legs = parse_leg_selection()
    if not legs:
        print("No valid legs selected, exiting.")
        return

    load_threshold = 600
    print(f"Using load threshold of {load_threshold} for all tests.")
    for leg in legs:
        print(f"\n=== Testing leg {leg} ===")

        # 1) Sweep knee (axis 2) negatively until load threshold; HOLD knee there
        knee_angle_final = sweep_joint_negative_until_load(
            hw=hw,
            esp32=esp32,
            joint_angles=current_angles,
            leg_index=leg,
            axis_index=2,         # knee
            load_threshold=load_threshold,
            step_deg=1.0,
            max_delta_deg=60.0,
            dwell=0.1,
        )
        print(f"Leg {leg} knee held at {np.rad2deg(knee_angle_final):.1f} deg")

        # 2) With knee fixed, sweep hip/thigh (axis 1) negatively until threshold
        thigh_angle_final = sweep_joint_negative_until_load(
            hw=hw,
            esp32=esp32,
            joint_angles=current_angles,
            leg_index=leg,
            axis_index=1,         # hip/thigh
            load_threshold=load_threshold,
            step_deg=1.0,
            max_delta_deg=60.0,
            dwell=0.1,
        )
        print(f"Leg {leg} thigh final angle {np.rad2deg(thigh_angle_final):.1f} deg")
        print(f"Leg {leg} knee held at {np.rad2deg(knee_angle_final):.1f} deg")

        # 3) Return this leg to the neutral joint angles (matching servo neutral)
        current_angles[0, leg] = neutral_angles[0, leg]  # abduction
        current_angles[1, leg] = neutral_angles[1, leg]  # hip
        current_angles[2, leg] = neutral_angles[2, leg]  # knee
        #hw.set_actuator_postions(current_angles)
        time.sleep(0.5)

    print("\nDone. Returning to neutral pose via ESP32.")
    move_to_neutral_pose(esp32)


if __name__ == "__main__":
    main()