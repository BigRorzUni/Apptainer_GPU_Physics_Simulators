import mujoco
import mujoco_viewer

import numpy as np
import time
import multiprocessing
import math
import timing_helper 
import sys
import argparse

parser = argparse.ArgumentParser(description="Run Mujoco Pendulum Simulation")

parser.add_argument("input_lb", type=int, help="Lower bound of input range")
parser.add_argument("input_ub", type=int, help="Upper bound of input range")
parser.add_argument("input_points", type=int, help="Number of input points")

args = parser.parse_args()

def simulate_step(model, data, num_steps):
    for _ in range(num_steps):
        mujoco.mj_step(model, data)

def time_model(mj_model, steps):
    print()
    print("Warmup")
    num_cores = multiprocessing.cpu_count()

    mj_data_instances = [mujoco.MjData(mj_model) for _ in range(num_cores)]

    multiprocessing.set_start_method('spawn', force=True)

    print(f'Num total steps {steps}')
    print(f'num cores: {num_cores}')

    num_cpu_steps = math.ceil(steps / num_cores)
    print(f'Num CPU Steps: {num_cpu_steps}')

    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # warmup
        pool.starmap(simulate_step, [(mj_model, mj_data_instances[_], 200) for _ in range(num_cores)])
        print("Warmup done, now testing")

        # test
        t0 = time.perf_counter()
        pool.starmap(simulate_step, [(mj_model, mj_data_instances[_], num_cpu_steps) for _ in range(num_cores)])
        t1 = time.perf_counter()

    t = t1-t0 
    fps_per_env = round(1000 / t, 2)
    total_fps = fps_per_env * num_cores
    print(f'per env: {fps_per_env:,.2f} FPS')
    print(f'total  : {total_fps:,.2f} FPS')
    print(f'this took : {t1-t0} seconds')

    return t, fps_per_env, total_fps

    

def main():
    print(f"Lower Bound: {args.input_lb}")
    print(f"Upper Bound: {args.input_ub}")
    print(f"Input Points: {args.input_points}")

    inputs = np.logspace(args.input_lb, args.input_ub, args.input_points)
    inputs = [int(x) for x in inputs]

    mj_model = mujoco.MjModel.from_xml_path("../xml/pendulum.xml")


    mj_model.opt.solver = 1 
    mj_model.opt.timestep = 0.01

    time = []
    fps_per_env = []
    total_fps = []

    for steps in inputs:
        t, e_fps, t_fps = time_model(mj_model, steps)

        time.append(t)
        fps_per_env.append(e_fps)
        total_fps.append(t_fps)
    
    timing_helper.send_times_csv(inputs, time, "data/Mujoco/speed.csv", "MuJoCo Time CPU Parallel (s)")
    timing_helper.send_times_csv(inputs, fps_per_env, "data/Mujoco/env_fps.csv", "MuJoCo FPS CPU Parallel")
    timing_helper.send_times_csv(inputs, total_fps, "data/Mujoco/total_fps.csv", "MuJoCo FPS CPU Parallel")

if __name__ == "__main__":
    main()