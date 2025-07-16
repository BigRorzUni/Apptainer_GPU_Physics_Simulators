import mujoco
from mujoco import mjx
import mujoco_viewer

import jax
from jax import numpy as jp
import numpy as np

jax.config.update("jax_enable_x64", False)

import time

import concurrent.futures
import multiprocessing

import math

import timing_helper 

import sys

import re


def simulate_step(model, data, num_steps):
    #mj_data = mujoco.MjData(model)  # Create mjData inside the function for each process
    for _ in range(num_steps):
        mujoco.mj_step(model, data)
    return data.qpos 

def simulate_GPU(mj_model, mj_data, total_steps, batch_size):

    print(f"Start of GPU parallel test for batch size {batch_size}")

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Tile the same data batch_size times (no randomisation)
    batch = jax.tree_util.tree_map(
        lambda x: jp.tile(x[None], [batch_size] + [1] * x.ndim),
        mjx_data
    )

    # JIT-compiled step function across batch
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    num_batch_steps = math.ceil(total_steps / batch_size)
    print(f'num batch steps: {num_batch_steps}')

    time_start = time.time()
    for _ in range(num_batch_steps):
        batch = jit_step(mjx_model, batch)
    time_gpu = time.time() - time_start

    return time_gpu

def time_model(mj_model, total_steps, gpu_batch_sizes):
    time_cpu_parallel = 0
    time_gpu = []

    # ------------------------------- General setup -----------------------------------------
    num_cores = multiprocessing.cpu_count()

    mj_data = mujoco.MjData(mj_model)

    mj_data_instances = [mujoco.MjData(mj_model) for _ in range(num_cores)]

    print(f'Num total steps {total_steps}')
    print(f'num cores: {num_cores}')

    num_cpu_steps = math.ceil(total_steps / num_cores)
    print(f'Num CPU Steps: {num_cpu_steps}')
    # ------------------------------- CPU parallel test --------------------------------------
    print("Start of CPU parallel test")
    multiprocessing.set_start_method('spawn', force=True)

    time_start = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(simulate_step, [(mj_model, mj_data_instances[_], num_cpu_steps) for _ in range(num_cores)])
    time_cpu_parallel = time.time() - time_start

    print(f"that took {time_cpu_parallel} (s)")   

    # ------------------------------- GPU parallel test --------------------------------------

    # For GPU we loop through different batch sizes and test
    for batch_size in gpu_batch_sizes:
        t = simulate_GPU(mj_model, mj_data, total_steps, batch_size)
        time_gpu.append(t)
        print(f"that took {t} (s)")

    return time_cpu_parallel, time_gpu

def extract_n_from_filename(path):
    match = re.search(r'(\d+)', path)
    if match:
        return int(match.group(1))
    return None


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()

    if len(sys.argv) < 3:
        print("Usage: python run_mujoco_batch.py steps scene1.xml scene2.xml ...")
        sys.exit(1)

    xml_paths = sys.argv[2:]
    steps = int(sys.argv[1])

    batch_sizes = [2048, 4096, 8192]

    n_vals = []
    time_cpu_serial = []
    time_cpu_parallel = []
    time_gpu_parallel = [[] for _ in range(len(batch_sizes))] 

    for path in xml_paths:
        mj_model = mujoco.MjModel.from_xml_path(path)
        n = extract_n_from_filename(path)

        print(f"Executing {steps} steps on a scene with {n} object(s)")
        
        t_cpu_p, t_gpu = time_model(mj_model, steps, batch_sizes)

        n_vals.append(n)
        time_cpu_parallel.append(t_cpu_p)
        for i, t in enumerate(t_gpu):
            time_gpu_parallel[i].append(t)


    timing_helper.send_times_csv(n_vals, time_cpu_parallel, "data/mujoco_clutter.csv", "MuJoCo Time CPU", input_prefix="N")
    timing_helper.send_times_csv(n_vals, time_gpu_parallel, "data/mjx_clutter.csv", "MJX Time GPU", batch_sizes, input_prefix="N")