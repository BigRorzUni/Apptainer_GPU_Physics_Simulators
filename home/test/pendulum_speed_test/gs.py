import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
import time



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
    inputs = np.logspace(1, 5, num = 5)
    inputs = [int(x) for x in inputs]
    batch_sizes = [2048, 4096, 8192]



    

    time_gpu_parallel = [[] for _ in range(len(batch_sizes))] 

    for size in inputs:
        for i, batch_size in enumerate(batch_sizes):
            gs.init(backend=gs.gpu)

            scene = gs.Scene(show_viewer=False)  # or True if needed
            entity = scene.add_entity(gs.morphs.MJCF(file="pendulum.xml"))
            scene.build(n_envs=batch_size)

            # Randomise position here once
            rand_pos = torch.rand(batch_size, 3, device=gs.device) * 0.5 + torch.tensor([0.0, 0.0, 0.5], device=gs.device)
            #entity.set_pos[:, 0, :] = rand_pos

            t_gpu = simulate_GPU(scene, total_steps=size)
            time_gpu_parallel[i].append(t_gpu)

            gs.destroy()

    timing_helper.send_times_csv(inputs, time_gpu_parallel, "genesis_pendulum_speed.csv", "Genesis Time GPU", batch_sizes)
