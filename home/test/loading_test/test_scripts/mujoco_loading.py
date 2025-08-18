import mujoco
import numpy as np
import time
import multiprocessing
import math
import timing_helper
import argparse
import gc
import psutil
import os

parser = argparse.ArgumentParser(description="Run MuJoCo CPU Loading Simulation")
parser.add_argument("-B", type=int, default=2048) # batch size
args = parser.parse_args()
n_envs = args.B

def get_mem_mb():
    """Return current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

def load_mujoco_envs(mj_model, n_envs):
    """
    Measures time and memory to create n_envs MjData instances on CPU.

    Returns:
        load_time_seconds, mem_per_env_MB
    """
    num_cores = multiprocessing.cpu_count()
    batch_size = n_envs

    gc.collect()
    mem_before = get_mem_mb()
    t0 = time.perf_counter()

    # Create MjData instances for each environment in the batch
    mj_data_instances = [mujoco.MjData(mj_model) for _ in range(batch_size)]

    t1 = time.perf_counter()
    mem_after = get_mem_mb()

    load_time = t1 - t0
    mem_per_env = (mem_after - mem_before) / batch_size

    # Cleanup
    del mj_data_instances
    gc.collect()

    return load_time, mem_per_env

def main():
    print(f"Batch size: {n_envs}")

    mj_model = mujoco.MjModel.from_xml_path("../xml/franka_emika_panda/scene.xml")
    mj_model.opt.solver = 1
    mj_model.opt.timestep = 0.01

    N_TRIALS = 100  # increase to average

    avg_times = []
    avg_mem_per_env = []

    for _ in range(N_TRIALS):
        load_time, mem_per_env = load_mujoco_envs(mj_model, n_envs)
        print(f"Load time: {load_time:.6f}s, Mem per env: {mem_per_env:.4f} MB")
        avg_times.append(load_time)
        avg_mem_per_env.append(mem_per_env)

    final_avg_time = sum(avg_times) / N_TRIALS
    final_avg_mem = sum(avg_mem_per_env) / N_TRIALS
    print(f"Final avg load time: {final_avg_time:.6f}s, avg mem per env: {final_avg_mem:.4f} MB")

    timing_helper.send_loading_times_csv([final_avg_time], [final_avg_mem],
                                         f"data/Mujoco/{n_envs}_load_info.csv")

if __name__ == "__main__":
    main()
