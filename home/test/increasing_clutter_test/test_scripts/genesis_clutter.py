import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

import sys

import re


import timing_helper

def simulate_GPU(scene, total_steps):
    print(f"Start of GPU simulation for", total_steps, "in a batch of", scene.n_envs)

    num_batch_steps = int(total_steps / scene.n_envs)

    time_start = time.time()
    for _ in range(num_batch_steps):
        scene.step()
    time_gpu = time.time() - time_start

    print("This took", time_gpu, "(s)")
    return time_gpu

def extract_n_from_filename(path):
    match = re.search(r'(\d+)', path)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python genesis_clutter.py steps scene1.xml scene2.xml ...")
        sys.exit(1)

    xml_paths = sys.argv[2:]
    steps = int(sys.argv[1])


    batch_sizes = [2048, 4096, 8192]

    n_vals = []
    time_gpu_parallel = [[] for _ in range(len(batch_sizes))] 

    for path in xml_paths:
        n = extract_n_from_filename(path)
        n_vals.append(n)

        print(f"Executing {steps} steps on a scene with {n} object(s)")
        for i, batch_size in enumerate(batch_sizes):
            gs.init(backend=gs.gpu)

            scene = gs.Scene(show_viewer=False) 
            entity = scene.add_entity(gs.morphs.MJCF(file=path))

            scene.build(n_envs=batch_size)

            t_gpu = simulate_GPU(scene, total_steps=steps)
            time_gpu_parallel[i].append(t_gpu)

            gs.destroy()

    timing_helper.send_times_csv(n_vals, time_gpu_parallel, "data/genesis_clutter.csv", "Genesis Time GPU", batch_sizes, input_prefix="N")