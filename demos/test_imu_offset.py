from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

import numpy as np
import math, time

def get_roll_pitch(data):
    """Get the un-filtered raw pitch and roll from accelerometer data."""
    if len(data) != 6:
        raise ValueError("Expected 6 values (ax, ay, az, gx, gy, gz)")

    ax = data['ax']
    ay = data['ay']
    az = data['az']
    gx = data['gx']
    gy = data['gy']
    gz = data['gz']  # Assume in  deg/s 
    roll  = math.atan(ay / math.sqrt(ay * ay + az * az))
    pitch = math.atan(ax / math.sqrt(ax * ax + az * az))
    return pitch, roll ### Purposely flipped here, IMU is not aligned with MP2 axes

def get_orientation_offset(data, n_samples=30):
    """
    Get the roll and pitch offsets from the IMU data
    """
    print(f"Collecting {n_samples} samples to compute orientation offsets...")
    roll_samples = np.zeros(n_samples)
    pitch_samples = np.zeros(n_samples)
    for i in range(n_samples):
        roll_samples[i], pitch_samples[i] = get_roll_pitch(data)
        #print(f"Sample {i+1}/{n_samples}: Roll={numpy.degrees(roll_samples[i]):.4f} deg, Pitch={numpy.degrees(pitch_samples[i]):.4f} deg")
        time.sleep(0.05)  # 20 Hz
    roll_offset = np.median(roll_samples)
    pitch_offset = np.median(pitch_samples)   
    print(f"The orientation offsets roll & pitch are: {np.degrees(roll_offset):.3f} deg, {np.degrees(pitch_offset):.3f} deg")
    return roll_offset, pitch_offset

def get_orientation_norm(roll, pitch):
    """ Normalized roll & pitch into a normalized value"""
    quadrant_lim = np.pi/4
    roll_norm = 1 - (quadrant_lim - roll)/quadrant_lim
    pitch_norm = 1 - (quadrant_lim - pitch)/quadrant_lim
    return roll_norm, pitch_norm

def leg_lower_selection(roll_norm, pitch_norm, k=20.0, dz_max=40.0):
    """Compute per-leg lowering offsets from roll/pitch.

    Returns a list [dz_FR, dz_FL, dz_BR, dz_BL] in arbitrary units
    (e.g. servo counts), where positive dz means "lower this leg".

    At most two legs are non-zero; the selection and split depend on
    the direction and magnitude of the tilt.
    """
    # Tilt vector in "horizontal" plane: x = roll, y = pitch
    orientation_norm = np.array([roll_norm, pitch_norm], dtype=float)
    tilt_norm = np.linalg.norm(orientation_norm)
    if tilt_norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 0.0])

    # Total lowering factor with tilt magnitude and is clamped
    L = k * tilt_norm
    if L > dz_max:
        L = dz_max

    # Unit tilt vector; flipped so it points toward the "high" side
    tilt_hat = -orientation_norm / tilt_norm

    # Leg direction definition for vector computation (NumPy-based)
    leg_dirs = np.array([
        [+1.0, +1.0],   # front-right
        [-1.0, +1.0],   # front-left
        [+1.0, -1.0],   # back-right
        [-1.0, -1.0],   # back-left
    ], dtype=float)

    # Normalize each leg direction and compute dot-products in one shot
    leg_dirs_norm = leg_dirs / np.linalg.norm(leg_dirs, axis=1, keepdims=True)
    raw_weights = leg_dirs_norm @ tilt_hat # Dot product to get the weights
    #raw_weights = np.clip(raw_weights, 0.0, None)  # only legs on "downhill" side

    # Pick up to two legs with the largest positive weight using NumPy
    idx_sorted = np.argsort(raw_weights)[::-1]  # descending by weight
    positive_mask = raw_weights[idx_sorted] > 0.0
    active_idx = idx_sorted[positive_mask][:2]

    dz_legs = np.zeros(4, dtype=float)
    if active_idx.size == 0:
        return dz_legs.tolist()

    active_weights = raw_weights[active_idx]
    total_w = active_weights.sum()
    dz_legs[active_idx] = L * (active_weights / total_w)

    return dz_legs.tolist()

def main():
    mp2 = HardwareInterface()
    esp32 = ESP32Interface()
    previous_time = time.time()
    roll_offset, pitch_offset = get_orientation_offset(esp32.imu_get_data())

    while True:
        now = time.time()
        if (now - previous_time) < 0.05:  # 20 Hz
            continue
        previous_time = now
        roll, pitch = get_roll_pitch(esp32.imu_get_data())
        roll -= roll_offset
        pitch -= pitch_offset
        #print(f"\rRoll: {np.degrees(roll):.2f} deg, Pitch: {np.degrees(pitch):.2f} deg", end="", flush=True)
        print(f"Roll: {np.degrees(roll):.2f} deg, Pitch: {np.degrees(pitch):.2f} deg")

        # Example: compute which legs to lower (servo-count offsets)
        dz_legs = leg_lower_selection(roll, pitch)
        print(f"Leg lowering offsets [FR, FL, BR, BL]: {dz_legs}")
        
        
        


if __name__ == "__main__":
    main()