import numpy as np
import warp as wp

import newton
import newton.examples
import newton.sim
import newton.utils

import timing_helper
import time

import sys

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

        # Create random joint positions in a NumPy array
        random_q = np.random.uniform(-np.pi, np.pi, size=self.model.joint_q.shape).astype(np.float32)

        # Assign to model.joint_q
        wp.copy(self.model.joint_q, wp.array(random_q, dtype=wp.float32, device=self.model.joint_q.device))

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
    print(f"Start of GPU simulation for", total_steps, "in a batch of", scene.num_envs)

    num_batch_steps = int(total_steps / scene.num_envs)

    time_start = time.time()
    for _ in range(num_batch_steps):
        scene.step()
    time_gpu = time.time() - time_start

    print("This took", time_gpu, "(s)")
    return time_gpu



if __name__ == "__main__":
    lower_bound = int(sys.argv[1])
    upper_bound = int(sys.argv[2])
    points = int(sys.argv[3])
    inputs = np.logspace(lower_bound, upper_bound, points)
    inputs = [int(x) for x in inputs]

    batch_sizes = [2048, 4096, 8192]

    time_gpu_parallel = [[] for _ in range(len(batch_sizes))] 

    for size in inputs:
        for i, batch_size in enumerate(batch_sizes):
            with wp.ScopedDevice("cuda"):
                pendulum_test = Pendulum(stage_path=None, num_envs=batch_size)
                t_gpu = simulate_GPU(pendulum_test, size)
                time_gpu_parallel[i].append(t_gpu)

    timing_helper.send_times_csv(inputs, time_gpu_parallel, "data/newton_pendulum_speed.csv", "Newton Time GPU", batch_sizes)
