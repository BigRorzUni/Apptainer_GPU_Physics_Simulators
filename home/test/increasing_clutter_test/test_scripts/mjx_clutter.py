import mujoco
from mujoco import mjx

import jax
from jax import numpy as jp
import numpy as np

import time
import math
import timing_helper 
import argparse

import os
import re

parser = argparse.ArgumentParser(description="Run Newton Clutter Simulation")

parser.add_argument("steps", type=int, help="Simulation Steps")
parser.add_argument("xml_paths", nargs="+", help="List of scene XML files")
parser.add_argument("-B", type=int, default=2048) # batch size

args = parser.parse_args()

steps = args.steps
xml_paths = args.xml_paths
n_envs = args.B

###### will boost performance by 30% according to mjx documentation ######
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

jit_steps = 1

def time_model(mj_model, steps):
    print()
    print("Warmup")

    mj_data = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = jax.vmap(lambda _: mjx_data)(np.arange(n_envs))

    print(f'Num total steps {steps}')
    print(f'num envs: {n_envs}')

    num_gpu_steps = math.ceil(steps / n_envs)
    print(f'Num Steps per Env: {num_gpu_steps}')

    def mjx_step_n(model, data):
        for _ in range(jit_steps):
            data = mjx.step(model, data)
        return data
    
    jit_step_n = jax.jit(jax.vmap(mjx_step_n, in_axes=(None, 0)), backend='gpu')

    curr_step = 0
    while True:
        mjx_data = jit_step_n(mjx_model, mjx_data)
        curr_step += jit_steps
        if curr_step >= 200:
            break
    print("Warmup done, now testing")

    t0 = time.perf_counter()
    curr_step = 0
    while True:
        mjx_data = jit_step_n(mjx_model, mjx_data)

        curr_step += jit_steps
        if curr_step >= num_gpu_steps:
            break

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

        print(f"Executing {steps} steps on a scene with {n} object(s)")

        mj_model = mujoco.MjModel.from_xml_path(path)

        mj_model.opt.solver = 1 
        mj_model.opt.timestep = 0.01

        t, e_fps, t_fps = time_model(mj_model, steps)

        times.append(t)
        fps_per_env.append(e_fps)
        total_fps.append(t_fps)
    
    timing_helper.send_times_csv(n_vals, times, f"data/MJX/{n_envs}_speed.csv", f"MJX Time GPU - Batch size {n_envs} (s)",input_prefix="N")
    timing_helper.send_times_csv(n_vals, fps_per_env, f"data/MJX/{n_envs}_env_fps.csv", f"MJX FPS GPU - Batch size {n_envs}",input_prefix="N")
    timing_helper.send_times_csv(n_vals, total_fps, f"data/MJX/{n_envs}_total_fps.csv", f"MJX FPS GPU - Batch size {n_envs}",input_prefix="N")


if __name__ == "__main__":
    main()