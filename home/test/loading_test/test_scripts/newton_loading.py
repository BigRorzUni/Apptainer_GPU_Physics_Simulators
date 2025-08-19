import numpy as np
import warp as wp
import newton
import newton.examples
import newton.sim
import newton.utils
import argparse
import time
import gc
import psutil
import timing_helper

def get_mem_mb():
    """Return current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

class Articulation:
    def __init__(self, num_envs=8):
        articulation_builder = newton.ModelBuilder()
        newton.utils.parse_mjcf("../xml/franka_emika_panda/panda.xml", articulation_builder)
        
        builder = newton.ModelBuilder()
        self.num_envs = num_envs

        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

def load_newton_envs(n_envs):
    """
    Measures Newton “load time” and memory usage for n_envs environments.
    Returns: load_time_seconds, mem_per_env_MB
    """
    gc.collect()
    mem_before = get_mem_mb()
    t0 = time.perf_counter()

    scene = Articulation(num_envs=n_envs)

    t1 = time.perf_counter()
    mem_after = get_mem_mb()

    load_time = t1 - t0
    mem_per_env = (mem_after - mem_before) / n_envs

    # Cleanup
    del scene
    gc.collect()

    return load_time, mem_per_env

def main():
    parser = argparse.ArgumentParser(description="Run Newton Loading Simulation")
    parser.add_argument("-B", type=int, default=2048) # batch size
    args = parser.parse_args()
    n_envs = args.B

    N_TRIALS = 1000

    avg_times = []
    avg_mem_per_env = []

    for _ in range(N_TRIALS):
        load_time, mem_per_env = load_newton_envs(n_envs)
        print(f"Load time: {load_time:.6f}s, Mem per env: {mem_per_env:.4f} MB")
        avg_times.append(load_time)
        avg_mem_per_env.append(mem_per_env)

    final_avg_time = sum(avg_times) / N_TRIALS
    final_avg_mem = sum(avg_mem_per_env) / N_TRIALS
    print(f"Final avg load time: {final_avg_time:.6f}s, avg mem per env: {final_avg_mem:.4f} MB")

    timing_helper.send_loading_times_csv([final_avg_time], [final_avg_mem],
                                         f"data/Newton/{n_envs}_load_info.csv")

if __name__ == "__main__":
    main()
