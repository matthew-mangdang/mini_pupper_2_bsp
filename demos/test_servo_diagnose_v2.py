from MangDang.mini_pupper.ESP32Interface import ESP32Interface

import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import statistics

"""
Auto diagnosis tool for mini pupper servos. This script performs systematic sweeps of each servo in both directions two times while 
recording load, voltage, and current data. The results are plotted and saved as .png files for visual analysis, 
and any servos with median load exceeding a defined threshold are flagged as potentially unhealthy.

CAUTIOUS!!! This script assume that the legs are already calibrated. 

Leg diagnose:
- Return the maximum and median load for each sweep and flag if median load exceeds a threshold (adjustable in code).

Plot explain:
- The top subplot shows the load readings for the servos in the same leg during the sweeps. Each servo's load is plotted in a different color, and vertical dashed lines indicate when we switch to a different servo or change sweep direction.
- The middle subplot shows the voltage readings from the board during the entire diagnosis process.
- The bottom subplot shows the current (amperage) readings from the board during the entire diagnosis process.

"""


### Configs
#OFFSET_STEPS = [25, 60, 60] * 4   # debug set up
OFFSET_STEPS = [110, 200, 180, 
                110, 200, 180, 
                110, 200, 180, 
                110, 200, 180]

DWELL_TIME = 0.01
STEP_SIZE = 1
NEUTRAL_POS = [512 for _ in range(12)]
ABS_MEDIAN_THRESHOLD = 160   # adjust to your robot

def sweep_servo(esp32, servo_id, num_steps, leg_servo_ids, start_pos=None, step_size=1,
                dwell_time=DWELL_TIME):
    if servo_id < 0 or servo_id >= len(NEUTRAL_POS):
        raise ValueError(f"servo_id {servo_id} out of range")

    cur_positions = esp32.servos_get_position()
    positions_cmd = cur_positions.copy() if cur_positions else NEUTRAL_POS.copy()
    if start_pos is None:
        start_pos = positions_cmd[servo_id]

    positions_history = []
    # Record loads only for the servos in the same leg
    load_history = {mid: [] for mid in leg_servo_ids}
    volt_history, amp_history = [], []

    for step in range(num_steps):
        pos = start_pos + step * step_size
        positions_cmd[servo_id] = pos
        esp32.servos_set_position(positions_cmd)
        time.sleep(dwell_time)

        loads = esp32.servos_get_load()
        power = esp32.get_power_status()

        positions_history.append(pos)
        for mid in leg_servo_ids:
            load_history[mid].append(loads[mid] if loads else None)

        if power:
            volt_history.append(power.get("volt"))
            amp_history.append(power.get("ampere"))
        else:
            volt_history.append(None)
            amp_history.append(None)

    positions_cmd[servo_id] = NEUTRAL_POS[servo_id]
    esp32.servos_set_position(positions_cmd)

    return {
        "servo_id": servo_id,
        "positions": positions_history,
        "loads": load_history,   # only same-leg servos
        "volts": volt_history,
        "amps": amp_history,
    }


def plot_servo_data(datasets, filename=None, title=None, dpi=150):
    if not datasets:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    all_servo_loads, all_volts, all_amps = {}, [], []

    for data in datasets:
        for mid, load_vals in data.get("loads", {}).items():
            all_servo_loads.setdefault(mid, []).extend(load_vals)
        all_volts.extend(data.get("volts", []))
        all_amps.extend(data.get("amps", []))

    for mid, load_vals in sorted(all_servo_loads.items()):
        axes[0].plot(load_vals, label=f"Servo {mid}", alpha=0.7)
    axes[1].plot(all_volts, label="Voltage")
    axes[2].plot(all_amps, label="Current")

    axes[0].set_ylabel("Load")
    axes[1].set_ylabel("Voltage (V)")
    axes[2].set_ylabel("Current (A)")
    axes[2].set_xlabel("Sample index")

    for ax in axes:
        ax.legend(loc="best")

    x_vlines = []
    running_len = 0
    for data in datasets:
        running_len += len(data.get("positions", []))
        if running_len > 0:
            x_vlines.append(running_len)

    for x_vline in x_vlines:
        for ax in axes:
            ax.axvline(x_vline, color='black', linestyle=':', linewidth=1, alpha=0.7)

    if title:
        fig.suptitle(title)

    if filename is None:
        filename = f"servo_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return filename

def diagnose_leg(esp32, leg_id, servo_ids, offset_steps, unhealthy_servos):
    ### The leg_id here is 1,2,3,4, only for display purposes.

    print(f"\n=== Diagnosing leg {leg_id} (servos {servo_ids}) ===")
    datasets = []
    for servo_id, steps in zip(servo_ids, offset_steps):
        sweeps = []
        # Two full cycles: forward/backward repeated
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=STEP_SIZE))
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=-STEP_SIZE))
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=-STEP_SIZE))
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=STEP_SIZE))
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=STEP_SIZE))
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=-STEP_SIZE))
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=-STEP_SIZE))
        sweeps.append(sweep_servo(esp32, servo_id, steps, servo_ids, step_size=STEP_SIZE))
        datasets.extend(sweeps)

        max_abs_loads, median_loads = [], []
        for sweep in sweeps:
            servo_loads = sweep["loads"][servo_id]
            max_abs = max(abs(x) for x in servo_loads)
            median_val = statistics.median(servo_loads)
            sweep["max_abs_load"] = max_abs
            sweep["median_load"] = median_val
            max_abs_loads.append(max_abs)
            median_loads.append(median_val)

        print(f"Servo {servo_id} max abs loads: {max_abs_loads}")
        print(f"Servo {servo_id} median loads: {median_loads}")
        if any(abs(m) > ABS_MEDIAN_THRESHOLD for m in median_loads):
            print(f"Cautious!!! Servo {servo_id} median load exceeds threshold {ABS_MEDIAN_THRESHOLD}. Not healthy.")
            unhealthy_servos.append(servo_id)

    filename = f"leg{leg_id}_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_servo_data(datasets, filename=filename, title=f"Leg {leg_id} Diagnostics")
    print(f"Saved plot for leg {leg_id} to {filename}")

def main():
    esp32 = ESP32Interface()
    esp32.servos_set_position(NEUTRAL_POS)
    print("Moved all servos to neutral. Starting diagnosis...")
    time.sleep(1.0)

    unhealthy_servos = []
    for leg_id in range(4):
        servo_ids = [leg_id*3 + i for i in range(3)]
        offset_steps = OFFSET_STEPS[leg_id*3:(leg_id+1)*3]
        diagnose_leg(esp32, leg_id+1, servo_ids, offset_steps, unhealthy_servos)

    print("\n=== Summary Report ===")
    if unhealthy_servos:
        print(f"Unhealthy servos detected, servo IDs: {unhealthy_servos}")
    else:
        print("All servos passed median load threshold check.")

if __name__ == "__main__":
    main()
