import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

import re

import timing_helper
import argparse

parser = argparse.ArgumentParser(description="Run Genesis Contact Simulation")

parser.add_argument("steps", type=int, help="Simulation Steps")
parser.add_argument("xml_paths", nargs="+", help="List of scene XML files")
parser.add_argument("-B", type=int, default=2048) # batch size

args = parser.parse_args()

steps = args.steps
xml_paths = args.xml_paths
n_envs = args.B

def simulate_GPU(scene, total_steps):
    print()
    print("Warmup")

    num_batch_steps = int(total_steps / scene.n_envs)

    print(f'Num steps per Env: {num_batch_steps}')
    
    for i in range(200): 
        scene.step()

    print("Warmup done, now testing")

    t0 = time.perf_counter()
    for _ in range(num_batch_steps):
        scene.step()
    t1 = time.perf_counter()

    t = t1-t0 
    fps_per_env = round(1000 / t, 2)
    total_fps = fps_per_env * n_envs
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
    print(f"Batch Size: {args.B}")
    print("-----------------------------")

    times = []
    fps_per_env = []
    total_fps = []
    n_vals = []

    for path in xml_paths:
        n = extract_n_from_filename(path)
        n_vals.append(n)

        print(f"Executing {steps} steps on a scene with contacts between {n} free joint(s)")


        print("Setting up scene")

        gs.init(backend=gs.gpu)

        scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                constraint_solver=gs.constraint_solver.CG, # to match mjx
                tolerance=1e-8, # to match mjx
            ),
        )

        scene.add_entity(gs.morphs.MJCF(file=path))
        scene.build(n_envs)

        t, e_fps, t_fps = simulate_GPU(scene, steps)

        times.append(t)
        fps_per_env.append(e_fps)
        total_fps.append(t_fps)

        gs.destroy()

    timing_helper.send_times_csv(n_vals, times, f"data/Genesis/{n_envs}_speed.csv", "Speed (s)", input_prefix="N")
    timing_helper.send_times_csv(n_vals, fps_per_env, f"data/Genesis/{n_envs}_env_fps.csv", "FPS", input_prefix="N")
    timing_helper.send_times_csv(n_vals, total_fps, f"data/Genesis/{n_envs}_total_fps.csv", "FPS", input_prefix="N")

if __name__ == "__main__":
    main()