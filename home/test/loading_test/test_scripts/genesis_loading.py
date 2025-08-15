import genesis as gs
import numpy as np
import time
import pynvml
import argparse
import timing_helper
import torch
import gc

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_memory_mb():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 ** 2)

parser = argparse.ArgumentParser(description="Run Genesis Loading Simulation")

parser.add_argument("-B", type=int, default=2048) # batch size

args = parser.parse_args()

n_envs = args.B


def main():
    N_TRIALS = 10

    print("Starting environment load + memory benchmark")
    

    avg_times = []
    avg_mem_per_env = []

    total_time = 0.0
    total_mem_per_env = 0.0

    for _ in range(N_TRIALS):
        gs.init(backend=gs.gpu, logging_level=None)

        scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(dt=0.01),
        )

        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.MJCF(file='../xml/franka_emika_panda/mjx_panda_free.xml'))

        mem_before = get_gpu_memory_mb()
        start_time = time.time()

        scene.build(n_envs)

        elapsed = time.time() - start_time
        mem_after = get_gpu_memory_mb()

        total_time += elapsed
        total_mem_per_env += (mem_after - mem_before) / n_envs

        del scene

        gs.destroy()
        time.sleep(0.05) 

        avg_time = total_time / N_TRIALS
        avg_mem_env = total_mem_per_env / N_TRIALS

        print(f"Env {n_envs}: {avg_time:.6f}s avg, {avg_mem_env:.4f} MB/env avg")

        avg_times.append(avg_time)
        avg_mem_per_env.append(avg_mem_env)
        
    print(sum(avg_times) / N_TRIALS)
    print(sum(avg_mem_per_env) / N_TRIALS)
    
if __name__ == "__main__":
    main()
