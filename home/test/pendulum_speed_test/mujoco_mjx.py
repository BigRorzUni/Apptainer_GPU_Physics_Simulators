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

# Now you can import your helper module
import timing_helper 


def simulate_step(model, data, num_steps):
    #mj_data = mujoco.MjData(model)  # Create mjData inside the function for each process
    for _ in range(num_steps):
        mujoco.mj_step(model, data)
    return data.qpos 

def simulate_GPU(mj_model, mj_data, total_steps, batch_size):

    print(f"Start of GPU parallel test for batch size {batch_size}")

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, batch_size)
    qpos_shape = mjx_data.qpos.shape
    batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, qpos_shape)))(rng)

    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    num_batch_steps = math.ceil(total_steps / batch_size)
    print(f'num batch steps: {num_batch_steps}')

    time_start = time.time()
    for i in range(num_batch_steps):
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

    # ------------------------------- GPU parallel test --------------------------------------

    # For GPU we loop through different batch sizes and test
    for batch_size in gpu_batch_sizes:
        time_gpu.append(simulate_GPU(mj_model, mj_data, total_steps, batch_size))

    return time_cpu_parallel, time_gpu


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()

    inputs = np.logspace(1, 5, num = 5)
    inputs = [int(x) for x in inputs]

    mj_model = mujoco.MjModel.from_xml_path("pendulum.xml")

    batch_sizes = [2048, 4096, 8192]

    time_cpu_serial = []
    time_cpu_parallel = []
    time_gpu_parallel = [[] for _ in range(len(batch_sizes))] 

    for size in inputs:
        t_cpu_p, t_gpu = time_model(mj_model, size, batch_sizes)
        time_cpu_parallel.append(t_cpu_p)

        for i, t in enumerate(t_gpu):
            time_gpu_parallel[i].append(t)

    timing_helper.send_times_csv(inputs, time_cpu_parallel, "mujoco_pendulum_speed.csv", "MuJoCo Time CPU Parallel")
    timing_helper.send_times_csv(inputs, time_gpu_parallel, "mjx_pendulum_speed.csv", "MJX Time GPU", batch_sizes)