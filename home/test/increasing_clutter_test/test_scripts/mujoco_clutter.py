import mujoco
import mujoco_viewer

import numpy as np
import time
import multiprocessing
import math
import timing_helper 
import argparse
import re

parser = argparse.ArgumentParser(description="Run Newton Clutter Simulation")

parser.add_argument("steps", type=int, help="Simulation Steps")
parser.add_argument("xml_paths", nargs="+", help="List of scene XML files")

args = parser.parse_args()

steps = args.steps
xml_paths = args.xml_paths

def simulate_step(model, data, num_steps):
    for _ in range(num_steps):
        mujoco.mj_step(model, data)

def time_model(mj_model, steps):
    print()
    print("Warmup")
    num_cores = multiprocessing.cpu_count()

    mj_data_instances = [mujoco.MjData(mj_model) for _ in range(num_cores)]

    multiprocessing.set_start_method('spawn', force=True)

    print(f'Num total steps {steps}')
    print(f'num cores: {num_cores}')

    num_cpu_steps = math.ceil(steps / num_cores)
    print(f'Num CPU Steps: {num_cpu_steps}')

    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # warmup
        pool.starmap(simulate_step, [(mj_model, mj_data_instances[_], 200) for _ in range(num_cores)])
        print("Warmup done, now testing")

        # test
        t0 = time.perf_counter()
        pool.starmap(simulate_step, [(mj_model, mj_data_instances[_], num_cpu_steps) for _ in range(num_cores)])
        t1 = time.perf_counter()

    t = t1-t0 
    fps_per_env = round(1000 / t, 2)
    total_fps = fps_per_env * num_cores
    print(f'per env: {fps_per_env:,.2f} FPS')
    print(f'total  : {total_fps:,.2f} FPS')
    print(f'this took : {t1-t0} seconds')

    return t, fps_per_env, total_fps

def extract_n_from_filename(path):
    match = re.search(r'(\d+)', path)
    if match:
        return int(match.group(1))
    return None

def main():
    print(f'steps: {args.steps}')
    print(f'xml_path: {args.xml_paths}')
    print("-----------------------------")

    times = []
    fps_per_env = []
    total_fps = []
    n_vals = []

    for path in xml_paths:
        n = extract_n_from_filename(path)
        n_vals.append(n)

        print(f"Executing {steps} steps on a scene with {n} object(s)")

        mj_model = mujoco.MjModel.from_xml_path(path)

        mj_model.opt.solver = 1 
        mj_model.opt.timestep = 0.01

        t, e_fps, t_fps = time_model(mj_model, steps)

        times.append(t)
        fps_per_env.append(e_fps)
        total_fps.append(t_fps)
    
    timing_helper.send_times_csv(n_vals, times, "data/Mujoco/speed.csv", "MuJoCo Time CPU Parallel (s)", input_prefix="N")
    timing_helper.send_times_csv(n_vals, fps_per_env, "data/Mujoco/env_fps.csv", "MuJoCo FPS CPU Parallel", input_prefix="N")
    timing_helper.send_times_csv(n_vals, total_fps, "data/Mujoco/total_fps.csv", "MuJoCo FPS CPU Parallel", input_prefix="N")

if __name__ == "__main__":
    main()