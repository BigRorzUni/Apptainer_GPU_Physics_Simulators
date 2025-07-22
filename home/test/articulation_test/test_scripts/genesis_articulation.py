import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

import argparse
import timing_helper
from pathlib import Path

parser = argparse.ArgumentParser(description="Run Genesis Articulation Simulation")

parser.add_argument("input_lb", type=int, help="Lower bound of input range")
parser.add_argument("input_ub", type=int, help="Upper bound of input range")
parser.add_argument("input_points", type=int, help="Number of input points")
parser.add_argument("-B", type=int, default=2048) # batch size

args = parser.parse_args()

n_envs = args.B

def simulate(scene, total_steps, franka):
    print()
    print("Warmup")

    t = torch.linspace(0, 2 * 3.14, total_steps, device=gs.device)
    path = 0.5 * torch.sin(t)

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

def main():
    print(f"Lower Bound: {args.input_lb}")
    print(f"Upper Bound: {args.input_ub}")
    print(f"Input Points: {args.input_points}")
    print(f"Batch Size: {args.B}")
    print("-----------------------------")

    inputs = np.logspace(args.input_lb, args.input_ub, args.input_points)
    inputs = [int(x) for x in inputs]

    print("Setting up scene")

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer   = True,
        rigid_options = gs.options.RigidOptions(
            dt                = 0.01,
        ),
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file='../xml/franka_emika_panda/panda.xml'),
    )
    
    scene.build(n_envs)

    franka.control_dofs_position(
        np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
    )

    times = []
    fps_per_env = []
    total_fps = []

    for steps in inputs:
        t, e_fps, t_fps = simulate(scene, steps, franka)

        times.append(t)
        fps_per_env.append(e_fps)
        total_fps.append(t_fps)

    timing_helper.send_times_csv(inputs, times, f"data/Genesis/{n_envs}_speed.csv", "Speed (s)")
    timing_helper.send_times_csv(inputs, fps_per_env, f"data/Genesis/{n_envs}_env_fps.csv", "FPS")
    timing_helper.send_times_csv(inputs, total_fps, f"data/Genesis/{n_envs}_total_fps.csv", "FPS")


if __name__ == "__main__":
    main()