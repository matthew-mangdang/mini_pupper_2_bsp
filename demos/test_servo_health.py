from MangDang.mini_pupper.ESP32Interface import ESP32Interface

import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import statistics


### Configs
#OFFSET_STEPS = 120  # number of steps to move in one direction
OFFSET_STEPS = [110, 200, 180, 
                110, 200, 180, 
                110, 200, 180, 
                110, 200, 180]
DWELL_TIME = 0.01   # seconds to wait at each step
STEP_SIZE = 1     # position units to change per step

NEUTRAL_POS = [512 for _ in range(12)]
ACTIVE_MOTOR_ID = [0,1,2]  # indices of servos you want to test
SWEEP_MOTOR_ID = [0,1,2]


def sweep_servo(esp32, servo_id, num_steps=OFFSET_STEPS, start_pos=None, step_size=1,
                dwell_time=DWELL_TIME):
    """Sweep a single servo in one direction and record load and power.

    Args:
        esp32: An initialised ESP32Interface instance.
        servo_id: Index of the servo to move (0-11).
        start_pos: Starting position command for this servo. If None, the current
            servo position is read from the controller and used as the start.
        step_size: Increment in position units per step.
        num_steps: Number of steps to move in one direction.
        dwell_time: Time in seconds to wait at each step.

    Returns:
        dict with keys:
            "servo_id", "positions", "loads", "volts", "amps".
    """
    if servo_id < 0 or servo_id >= len(NEUTRAL_POS):
        raise ValueError(f"servo_id {servo_id} out of range")

    # Prefer the actual current positions if available so we don't disturb other servos
    cur_positions = esp32.servos_get_position()
    if cur_positions is not None:
        positions_cmd = cur_positions.copy()
    else:
        positions_cmd = NEUTRAL_POS.copy()

    # If no explicit start_pos provided, use the current commanded position for this servo
    if start_pos is None:
        start_pos = positions_cmd[servo_id]

    positions_history = []
    load_history = {mid: [] for mid in ACTIVE_MOTOR_ID}  # keyed by servo_id
    volt_history = []
    amp_history = []

    # Move in one direction from start_pos
    for step in range(num_steps):
        pos = start_pos + step * step_size
        positions_cmd[servo_id] = pos

        esp32.servos_set_position(positions_cmd)
        time.sleep(dwell_time)

        loads = esp32.servos_get_load()
        power = esp32.get_power_status()

        positions_history.append(pos)
        for mid in ACTIVE_MOTOR_ID:
            load_history[mid].append(loads[mid])

        #load_val = load_history[servo_id][-1]  # moving servo load for the print
        if power is not None:
            volt_val = power.get("volt")
            amp_val = power.get("ampere")
            volt_history.append(volt_val)
            amp_history.append(amp_val)
        else:
            volt_val = None
            amp_val = None
            volt_history.append(None)
            amp_history.append(None)

        # Per-step status print
        all_loads_str = ", ".join(f"{mid}:{load_history[mid][-1]}" for mid in ACTIVE_MOTOR_ID)
        #print(f"Servo {servo_id} step {step+1}/{num_steps}: pos={pos}, loads={{ {all_loads_str} }}, volt={volt_val}, amp={amp_val}")

    # Return servo to neutral position at the end of the sweep
    positions_cmd[servo_id] = NEUTRAL_POS[servo_id]
    esp32.servos_set_position(positions_cmd)

    return {
        "servo_id": servo_id,
        "positions": positions_history,
        "loads": load_history,
        "volts": volt_history,
        "amps": amp_history,
    }


def plot_servo_data(datasets, filename=None, title=None, dpi=150):
    """Plot loads, volts, and amps for one or more servo sweeps and save to file.

    Args:
        datasets: list of dicts as returned by sweep_servo().
        filename: Path to save the plot image. If None, a timestamped PNG is used.
        title: Optional overall title for the figure.
        dpi: Image DPI for saving.

    Returns:
        The path to the saved image file (string).
    """
    if not datasets:
        return None

    # Increase overall figure size so plotted data has more room to spread
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # Aggregate across all sweeps: concatenate head-to-tail per servo
    all_servo_loads = {}  # {servo_id: [load, load, ...]}
    all_volts = []
    all_amps = []

    for data in datasets:
        loads = data.get("loads", {})
        if isinstance(loads, dict):
            for mid, load_vals in loads.items():
                all_servo_loads.setdefault(mid, []).extend(load_vals)
        all_volts.extend(data.get("volts", []))
        all_amps.extend(data.get("amps", []))

    # One line per servo for loads (slight transparency for overlap readability)
    for mid, load_vals in sorted(all_servo_loads.items()):
        axes[0].plot(load_vals, label=f"Servo {mid}", alpha=0.7)

    axes[1].plot(all_volts, label="Voltage")
    axes[2].plot(all_amps, label="Current")

    axes[0].set_ylabel("Load (arb units)")
    axes[1].set_ylabel("Voltage (V)")
    axes[2].set_ylabel("Current (A)")
    axes[2].set_xlabel("Sample index")

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[2].legend(loc="best")

    # Draw separators at cumulative sweep lengths from each dataset's positions.
    # Example lengths [50, 50, 50, 50] -> lines at [50, 100, 150, 200].
    x_vlines = []
    running_len = 0
    for data in datasets:
        running_len += len(data.get("positions", []))
        if running_len > 0:
            x_vlines.append(running_len)

    for x_vline in x_vlines:
        for ax in axes:
            ax.axvline(x_vline, color='black', linestyle=':', linewidth=1, alpha=0.7)

    # Apply title if provided
    if title:
        fig.suptitle(title)

    # Determine filename
    if filename is None:
        filename = f"servo_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Ensure directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Improve spacing between subplots and leave room for suptitle
    #fig.subplots_adjust(hspace=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

    return filename


def main():
    esp32 = ESP32Interface()

    # Move all servos to neutral first
    esp32.servos_set_position(NEUTRAL_POS)
    print(f"Moved all servos to neutral position {NEUTRAL_POS}. Starting sweeps in 1 second...")
    time.sleep(1.0)

    # Dict keyed by servo_id; each value is a list of sweep dicts for that servo
    all_data = {servo_id: [] for servo_id in ACTIVE_MOTOR_ID}

    # Sweep each active servo one by one and collect data
    for servo_id in SWEEP_MOTOR_ID:
        data_1 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=STEP_SIZE) ## sweep in positive direction first
        all_data[servo_id].append(data_1)
        data_2 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=-STEP_SIZE) ## return to start
        all_data[servo_id].append(data_2)
        data_3 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=-STEP_SIZE) ## sweep in negative direction
        all_data[servo_id].append(data_3)
        data_4 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=STEP_SIZE) ## return to start
        all_data[servo_id].append(data_4)
        data_5 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=STEP_SIZE) ## sweep in positive direction first
        all_data[servo_id].append(data_5)
        data_6 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=-STEP_SIZE) ## return to start
        all_data[servo_id].append(data_6)
        data_7 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=-STEP_SIZE) ## sweep in negative direction
        all_data[servo_id].append(data_7)
        data_8 = sweep_servo(esp32, servo_id, OFFSET_STEPS[servo_id], step_size=STEP_SIZE) ## return to start
        all_data[servo_id].append(data_8)
        # Print when one motor finishes and sleep briefly
        print(f"Finished sweep for servo {servo_id}.")
        time.sleep(0.5)

        # For this moving servo only, record max(abs(load)) and median(load)
        sweeps = all_data[servo_id]
        max_abs_loads = []
        median_loads = []
        for sweep in sweeps:
            servo_loads = sweep["loads"][servo_id]
            max_abs_load = max(abs(x) for x in servo_loads)
            median_load = statistics.median(servo_loads)
            sweep["max_abs_load"] = max_abs_load
            sweep["median_load"] = median_load
            max_abs_loads.append(max_abs_load)
            median_loads.append(median_load)

        print(f"Servo {servo_id} max abs loads per sweep: {max_abs_loads}")
        print(f"Servo {servo_id} median loads per sweep: {median_loads}")

    # Flatten for plotting
    datasets = [sweep for sweeps in all_data.values() for sweep in sweeps]

    # Plot the collected data
    plot_servo_data(datasets)

    return 0


if __name__ == "__main__":
    main()