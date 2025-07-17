import numpy as np
import warp as wp

import newton
import newton.examples
import newton.sim
import newton.utils
import argparse

import timing_helper
import time

parser = argparse.ArgumentParser(description="Run Newton Pendulum Simulation")

parser.add_argument("input_lb", type=int, help="Lower bound of input range")
parser.add_argument("input_ub", type=int, help="Upper bound of input range")
parser.add_argument("input_points", type=int, help="Number of input points")
parser.add_argument("-B", type=int, default=2048) # batch size

args = parser.parse_args()

n_envs = args.B

class Pendulum:
    def __init__(self, stage_path="test.usd", num_envs=8):
        pendulum_builder = newton.ModelBuilder()
        newton.utils.parse_mjcf("../xml/pendulum.xml", pendulum_builder)
        
        builder = newton.ModelBuilder()

        self.num_envs = num_envs
        self.sim_dt = 0.01
        self.sim_time = 0.0

        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            builder.add_builder(pendulum_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        # finalize model
        self.model = builder.finalize()
        self.solver = newton.solvers.XPBDSolver(self.model)

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else: 
            self.simulate()

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.render_contacts(self.state_0, self.contacts, contact_point_radius=1e-2)
            self.renderer.end_frame()
            self.sim_time += self.sim_dt

def simulate_GPU(scene, total_steps):
    print()
    print("Warmup")

    num_batch_steps = int(total_steps / scene.num_envs)
    print(f'Num steps per Env: {num_batch_steps}')
    
    for _ in range(200): 
        scene.step()

    print("Warmup done, now testing")

    t0 = time.perf_counter()
    for _ in range(num_batch_steps):
        scene.step()
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
    print("-----------------------------")

    inputs = np.logspace(args.input_lb, args.input_ub, args.input_points)
    inputs = [int(x) for x in inputs]

    time = []
    fps_per_env = []
    total_fps = []


    for steps in inputs:
        with wp.ScopedDevice("cuda"):
            pendulum_test = Pendulum(stage_path=None, num_envs=n_envs)
            t, e_fps, t_fps = simulate_GPU(pendulum_test, steps)
            
            time.append(t)
            fps_per_env.append(e_fps)
            total_fps.append(t_fps)

    timing_helper.send_times_csv(inputs, time, f"data/Newton/{n_envs}_speed.csv", f"Newton Time GPU - Batch size {n_envs} (s)")
    timing_helper.send_times_csv(inputs, fps_per_env, f"data/Newton/{n_envs}_env_fps.csv", f"Newton FPS GPU - Batch size {n_envs}")
    timing_helper.send_times_csv(inputs, total_fps, f"data/Newton/{n_envs}_total_fps.csv", f"Newton FPS GPU - Batch size {n_envs}")


if __name__ == "__main__":
    main()