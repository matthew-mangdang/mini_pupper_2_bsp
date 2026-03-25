#!/usr/bin/python
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time
import os

"""
Script to log the servo load history, just simply run it and it will create a load_history.log file in the same directory, and append the load history to it.

Usage:
python /home/ubuntu/mini_pupper_bsp/demos/load_logger.py

"""

def main():
    esp32 = ESP32Interface()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime("%H_%M_%S")
    #logfile = os.path.join(cur_dir, f"load_walk_forward_red_{timestamp}_t1.log")
    logfile = os.path.join(cur_dir, f"load_walk_forward_red_t1.log")

    with open(logfile, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        #f.write(f"{timestamp} Start logging...\n")
        #f.flush()
        print(f"[{timestamp}] Start logging...")
        time.sleep(1.0)
        
        try:
            while True:
                timestamp = time.strftime("%H:%M:%S")
                loads = esp32.servos_get_load()
                f.write(f"{timestamp} {loads}\n")
                f.flush()
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
