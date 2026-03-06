import time

from MangDang.mini_pupper.ESP32Interface import ESP32Interface


LEG_NAMES = {
	0: "front-right",
	1: "front-left",
	2: "back-right",
	3: "back-left",
}

AXIS_NAMES = {
	0: "Abduction",
	1: "Hip/Thigh",
	2: "Knee",
}


def leg_axis_to_servo_index(leg_index: int, axis_index: int) -> int:
	"""Map (leg, axis_index) to index in 12-element servo vector.

	Layout: [leg0_abd, leg0_hip, leg0_knee,
			 leg1_abd, leg1_hip, leg1_knee, ...]
	axis_index: 0=abduction, 1=hip, 2=knee.
	"""

	return leg_index * 3 + axis_index


def parse_legs() -> list[int]:
	print("Leg indices: 0=front-right, 1=front-left, 2=back-right, 3=back-left")
	s = input("Enter legs to test (e.g. '0,2' or 'all'): ").strip().lower()
	if s in ("all", ""):
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


def parse_axes() -> list[int]:
	print("Axis indices: 0=abduction, 1=hip/thigh, 2=knee")
	s = input("Enter axes to test (e.g. '1,2' or 'all'): ").strip().lower()
	if s in ("all", ""):
		return [0, 1, 2]

	axes: list[int] = []
	for tok in s.split(","):
		tok = tok.strip()
		if not tok:
			continue
		try:
			i = int(tok)
		except ValueError:
			continue
		if 0 <= i <= 2:
			axes.append(i)
	return sorted(set(axes))


def main() -> None:
	esp32 = ESP32Interface()

	# Neutral pose for all 12 servos
	NEUTRAL = 512
	positions: list[int] = [NEUTRAL] * 12
	print("Moving all servos to neutral (512)...")
	esp32.servos_set_position(positions)
	time.sleep(0.5)

	legs = parse_legs()
	if not legs:
		print("No valid legs selected, exiting.")
		return

	axes = parse_axes()
	if not axes:
		print("No valid axes selected, exiting.")
		return

	# Global sweep parameters for this run
	try:
		s = input("Start position [0-1023, default 512]: ").strip()
		start_pos = int(s) if s else NEUTRAL
	except ValueError:
		start_pos = NEUTRAL

	try:
		s = input("Target position [0-1023, default 800]: ").strip()
		target_pos = int(s) if s else 800
	except ValueError:
		target_pos = 800

	# Clamp to valid range
	start_pos = max(0, min(1023, start_pos))
	target_pos = max(0, min(1023, target_pos))

	if start_pos == target_pos:
		print("Start and target positions are equal; nothing to do.")
		return

	direction = 1 if target_pos > start_pos else -1

	try:
		s = input("Step size (counts, default 2): ").strip()
		step = int(s) if s else 2
	except ValueError:
		step = 2
	if step <= 0:
		step = 1

	try:
		s = input("Dwell between steps (seconds, default 0.05): ").strip()
		dwell = float(s) if s else 0.05
	except ValueError:
		dwell = 0.05

	try:
		s = input(
			"Optional load threshold to auto-stop (abs(load) >= thr). "
			"Press Enter to disable, or enter e.g. 300: "
		).strip()
		load_threshold = int(s) if s else None
	except ValueError:
		load_threshold = None

	print("\nStarting raw position sweep using ESP32 servo commands.")
	print("For each step we print: commanded_pos, measured_pos, load.")

	for leg in legs:
		for axis in axes:
			servo_idx = leg_axis_to_servo_index(leg, axis)
			leg_name = LEG_NAMES.get(leg, str(leg))
			axis_name = AXIS_NAMES.get(axis, str(axis))

			print(
				f"\n=== Sweeping leg {leg} ({leg_name}), axis {axis} ({axis_name}), "
				f"servo index {servo_idx} ==="
			)

			# Reset all to neutral, then set this servo to the chosen start position
			positions = [NEUTRAL] * 12
			positions[servo_idx] = start_pos
			esp32.servos_set_position(positions)
			time.sleep(0.2)

			pos = start_pos
			while True:
				# Command current position
				positions[servo_idx] = pos
				esp32.servos_set_position(positions)
				time.sleep(dwell)

				measured_positions = esp32.servos_get_position()
				loads = esp32.servos_get_load()

				measured_pos = (
					measured_positions[servo_idx]
					if measured_positions is not None
					else None
				)
				this_load = loads[servo_idx] if loads is not None else None

				print(
					f"cmd={pos:4d}, "
					f"meas={measured_pos if measured_pos is not None else '----'}, "
					f"load={this_load if this_load is not None else '----'}"
				)

				# Auto-stop on load threshold if requested
				if (
					load_threshold is not None
					and this_load is not None
					and abs(this_load) >= load_threshold
				):
					print(
						f"Reached load threshold |load| >= {load_threshold} at "
						f"cmd={pos} (leg {leg}, axis {axis})."
					)
					break

				# Next step or exit if we've reached/passed target
				if direction > 0:
					if pos >= target_pos:
						break
					pos = min(target_pos, pos + step)
				else:
					if pos <= target_pos:
						break
					pos = max(target_pos, pos - step)

			# After each servo sweep, go back to neutral for safety
			positions = [NEUTRAL] * 12
			esp32.servos_set_position(positions)
			time.sleep(0.2)

	print("\nDone. All servos returned to neutral (512).")


if __name__ == "__main__":
	main()

