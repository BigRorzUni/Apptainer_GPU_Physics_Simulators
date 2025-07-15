import sys
import time
import mujoco
import numpy as np
from mujoco import viewer
import re

import timing_helper

def run_sim(xml_path, steps=100):
    print(f"\nLoading: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        print(f"Simulating {steps} steps")
        # Headless simulation
        time_start = time.time()
        for step in range(steps):
            mujoco.mj_step(model, data)
        time_taken = time.time() - time_start
        

        print(f"Completed {steps} steps for: {xml_path} in {time_taken} seconds")

        return time_taken
    
    except Exception as e:
        print(f"Failed to load or simulate {xml_path}: {e}")
        return None


def extract_n_from_filename(path):
    match = re.search(r'(\d+)', path)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_mujoco_batch.py scene1.xml scene2.xml ...")
        sys.exit(1)

    xml_paths = sys.argv[1:]
    print("Mujoco CPU:")

    n_vals = []
    timings = []

    for path in xml_paths:
        n = extract_n_from_filename(path)
        time_taken = run_sim(path)

        if time_taken is not None and n is not None:
            n_vals.append(n)
            timings.append(time_taken)

    print()
    
    timing_helper.send_times_csv(n_vals, timings, "data/mujoco_clutter.csv", "MuJoCo Time CPU", input_prefix="N")

    print()
