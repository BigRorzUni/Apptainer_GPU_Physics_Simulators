# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
# Users can pick bodies by right-clicking and dragging with the mouse.
#
###########################################################################

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.sim
import newton.utils


class Example:
    def __init__(self, stage_path="example_quadruped.usd", num_envs=8):
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_body_armature = 0.01
        articulation_builder.default_joint_cfg.armature = 0.01
        articulation_builder.default_joint_cfg.mode = newton.JOINT_MODE_TARGET_POSITION
        articulation_builder.default_joint_cfg.target_ke = 2000.0
        articulation_builder.default_joint_cfg.target_kd = 1.0
        articulation_builder.default_shape_cfg.ke = 1.0e4
        articulation_builder.default_shape_cfg.kd = 1.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e2
        articulation_builder.default_shape_cfg.mu = 1.0
        newton.utils.parse_mjcf("../../xml/franka_emika_panda/panda.xml", articulation_builder)
        articulation_builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        articulation_builder.joint_target[-12:] = articulation_builder.joint_q[-12:]

        jnt_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'finger_joint1',
            'finger_joint2',
        ]

        builder = newton.ModelBuilder()

        self.sim_time = 0.0
        fps = 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = 1

        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        builder.add_ground_plane()

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.XPBDSolver(self.model)
        # self.solver = newton.solvers.FeatherstoneSolver(self.model)
        # self.solver = newton.solvers.SemiImplicitSolver(self.model)
        # self.solver = newton.solvers.MuJoCoSolver(self.model)

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        print("Joint targets:", self.control.joint_target if self.control.joint_target is not None else None)
        print("Joint forces:", self.control.joint_f.numpy() if self.control.joint_f is not None else None)
        print(self.model.joint_dof_count)
        signals = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
        #self.control.joint_target = wp.array(signals, dtype=float)
        #print("Joint targets:", self.control.joint_target if self.control.joint_target is not None else None)
    
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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

             # Create target positions, for example base positions + small random noise
            base_targets = np.array([0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2, 0.4, 0.6])
            noise = np.random.uniform(low=-0.02, high=0.02, size=base_targets.shape)
            new_targets = base_targets + noise

            # Set the joint target positions
            self.control.joint_target = wp.array(new_targets, dtype=float)

            if self.renderer and hasattr(self.renderer, "apply_picking_force"):
                self.renderer.apply_picking_force(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.render_contacts(self.state_0.body_q, self.contacts, contact_point_radius=1e-2)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_quadruped.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=30000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=100, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()