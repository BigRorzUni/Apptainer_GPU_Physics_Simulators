import sys
import time
import mujoco
import numpy as np
import re

from mujoco import mjx
import jax
from jax import numpy as jp

jax.config.update("jax_enable_x64", False)

import timing_helper

def run_sim(xml_path, steps=100):
    print(f"\nLoading: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        mjx_model = mjx.put_model(model)
        mjx_data = mjx.put_data(mjx_model, data)

        print(f"Simulating {steps} steps")
        # Headless simulation
        N = 1

        # create N copies of the same initial state
        mjx_datas = jax.tree_util.tree_map(
            lambda x: jp.tile(x[None], [N] + [1] * x.ndim), mjx_data)

        @jax.jit
        def step_batch(data_batch):
            return jax.vmap(lambda d: mjx.step(mjx_model, d))(data_batch)

        @jax.jit
        def rollout_batch(data_batch):
            def step_fn(data):
                data = step_batch(data)
                return data # Log qpos
            
            jax.lax.scan(step_fn, data_batch, None, length=steps)

        # run batch simulation and log qpos
        start_time = time.time()
        rollout_batch(mjx_datas)
        time_taken = time.time() - start_time

        print(f"Completed {steps} steps for: {xml_path} in {time_taken} seconds")

        return time_taken
    
    except Exception as e:
        print(f"Failed to load or simulate {xml_path}: {type(e).__name__}: {str(e)[:200]}...")
        return None



def extract_n_from_filename(path):
    match = re.search(r'(\d+)', path)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_mjx_batch.py scene1.xml scene2.xml ...")
        sys.exit(1)

    xml_paths = sys.argv[1:]
    print("MJX GPU:")

    n_vals = []
    timings = []

    for path in xml_paths:
        n = extract_n_from_filename(path)
        time_taken = run_sim(path)

        if time_taken is not None and n is not None:
            n_vals.append(n)
            timings.append(time_taken)

    print()
    
    timing_helper.send_times_csv(n_vals, timings, "data/mjx_clutter.csv", "MJX Time GPU", input_prefix="N")

    print()
