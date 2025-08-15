import mujoco
from mujoco import mjx
import jax
from jax import numpy as jp
import numpy as np
import time
import pynvml
import argparse
import timing_helper
import gc
import os
import torch

jax.config.update("jax_enable_x64", False)

# NVML for GPU memory
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
def get_gpu_memory_mb():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 ** 2)

parser = argparse.ArgumentParser(description="Run MJX Loading Simulation")
parser.add_argument("-B", type=int, default=2048) # batch size
args = parser.parse_args()
n_envs = args.B

# Optional XLA performance flags
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

def mjx_load_time(mj_model):
    """
    Measures MJX “load time” for n_envs environments and memory used per environment.
    Returns: load_time_seconds, mem_per_env_MB
    """
    mem_before = get_gpu_memory_mb()

    t0 = time.perf_counter()

    # Create MjData and put into MJX
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # vmap across batch
    mjx_data = jax.vmap(lambda _: mjx_data)(np.arange(n_envs))

    t1 = time.perf_counter()
    mem_after = get_gpu_memory_mb()

    load_time = t1 - t0
    mem_per_env = (mem_after - mem_before) / n_envs

    return load_time, mem_per_env

def main():
    mj_model = mujoco.MjModel.from_xml_path("../xml/franka_emika_panda/mjx_scene.xml")

    avg_times = []
    avg_mem_per_env = []

    N_TRIALS = 100  

    for _ in range(N_TRIALS):
        load_time, mem_per_env = mjx_load_time(mj_model)
        print(f"Load time: {load_time:.6f}s, Mem per env: {mem_per_env:.4f} MB")
        avg_times.append(load_time)
        avg_mem_per_env.append(mem_per_env)
        gc.collect()

    # Average over trials
    final_avg_time = sum(avg_times) / N_TRIALS
    final_avg_mem = sum(avg_mem_per_env) / N_TRIALS
    print(f"Final avg load time: {final_avg_time:.6f}s, avg mem per env: {final_avg_mem:.4f} MB")

    timing_helper.send_loading_times_csv([final_avg_time], [final_avg_mem],
                                         f"data/MJX/{n_envs}_load_info.csv")

if __name__ == "__main__":
    main()
