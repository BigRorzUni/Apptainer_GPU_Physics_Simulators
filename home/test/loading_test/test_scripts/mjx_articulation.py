import mujoco
from mujoco import mjx

import jax
from jax import numpy as jp
import numpy as np

jax.config.update("jax_enable_x64", False)

import time
import math
import timing_helper 
import argparse

import os

parser = argparse.ArgumentParser(description="Run MJX Articulation Simulation")

parser.add_argument("input_lb", type=int, help="Lower bound of input range")
parser.add_argument("input_ub", type=int, help="Upper bound of input range")
parser.add_argument("input_points", type=int, help="Number of input points")
parser.add_argument("-B", type=int, default=2048) # batch size

args = parser.parse_args()

###### will boost performance by 30% according to mjx documentation ######
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

n_envs = args.B
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

    # copied from free fall script, qpos when touching ground
    ref_pos = jp.tile(jp.array([ 0.33127472,  1.7633277 , -0.32836628, -0.23935018, -0.4920762 , 0.38396463,  0.39376438,  0.00876223,  0.03379252]), [n_envs, 1])

    mjx_data = mjx_data.replace(ctrl = ref_pos)

    curr_step = 0
    while True:
        mjx_data = jit_step_n(mjx_model, mjx_data)
        curr_step += jit_steps
        if curr_step >= 200:
            break
    print("Warmup done, now testing")

    rng = jax.random.PRNGKey(0)

    t0 = time.perf_counter()
    curr_step = 0
    while True:
        mjx_data = mjx_data.replace(ctrl = ref_pos + jax.random.uniform(rng, shape=(n_envs, 9)) * 0.4 - 0.2)
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

    

def main():
    print(f"Lower Bound: {args.input_lb}")
    print(f"Upper Bound: {args.input_ub}")
    print(f"Input Points: {args.input_points}")
    print(f"Batch Size: {args.B}")

    inputs = np.logspace(args.input_lb, args.input_ub, args.input_points)
    inputs = [int(x) for x in inputs]

    mj_model = mujoco.MjModel.from_xml_path("../xml/franka_emika_panda/mjx_scene.xml")

    mj_model.opt.solver = 1 
    mj_model.opt.timestep = 0.01

    times = []
    fps_per_env = []
    total_fps = []

    for steps in inputs:
        t, e_fps, t_fps = time_model(mj_model, steps)
        
        times.append(t)
        fps_per_env.append(e_fps)
        total_fps.append(t_fps)
    
    timing_helper.send_times_csv(inputs, times, f"data/MJX/{n_envs}_speed.csv", f"MJX Time GPU - Batch size {n_envs} (s)")
    timing_helper.send_times_csv(inputs, fps_per_env, f"data/MJX/{n_envs}_env_fps.csv", f"MJX FPS GPU - Batch size {n_envs}")
    timing_helper.send_times_csv(inputs, total_fps, f"data/MJX/{n_envs}_total_fps.csv", f"MJX FPS GPU - Batch size {n_envs}")

if __name__ == "__main__":
    main()