import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

import sys


import timing_helper

phys_params = gs.options.RigidOptions(
    dt=0.005,
    integrator=gs.integrator.Euler,
    gravity=(0.0, 0.0, -9.81),
    iterations=50,
)

def simulate_GPU(scene, total_steps):
    print(f"Start of GPU simulation for", total_steps, "in a batch of", scene.n_envs)

    num_batch_steps = int(total_steps / scene.n_envs)

    time_start = time.time()
    for _ in range(num_batch_steps):
        scene.step()
    time_gpu = time.time() - time_start

    print("This took", time_gpu, "(s)")
    return time_gpu


if __name__ == "__main__":
    lower_bound = int(sys.argv[1])
    upper_bound = int(sys.argv[2])
    points = int(sys.argv[3])
    inputs = np.logspace(lower_bound, upper_bound, points)
    inputs = [int(x) for x in inputs]
    
    batch_sizes = [2048, 4096, 8192]

    time_gpu_parallel = [[] for _ in range(len(batch_sizes))] 

    for size in inputs:
        for i, batch_size in enumerate(batch_sizes):
            gs.init(backend=gs.gpu)

            scene = gs.Scene(show_viewer=False) 
            entity = scene.add_entity(gs.morphs.MJCF(file="../xml/pendulum.xml"))
            scene.build(n_envs=batch_size)

            # Randomise position of joint
            rand_qpos = (torch.rand(batch_size, 1, device=gs.device) * 2 - 1) * torch.pi
            entity.set_qpos(rand_qpos)

            t_gpu = simulate_GPU(scene, total_steps=size)
            time_gpu_parallel[i].append(t_gpu)

            gs.destroy()

    timing_helper.send_times_csv(inputs, time_gpu_parallel, "data/genesis_pendulum_speed.csv", "Genesis Time GPU", batch_sizes)