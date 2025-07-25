import mujoco
from mujoco import mjx
import mujoco.viewer

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
# Get MuJoCo's standard humanoid model.
print('Getting MuJoCo humanoid XML description from GitHub:')
with open('mujoco/model/humanoid/humanoid.xml', 'r') as f:
  xml = f.read()

# Load the model, make two MjData's.
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
data2 = mujoco.MjData(model)

# Episode parameters.
duration = 3       # (seconds)
framerate = 60     # (Hz)
data.qpos[0:2] = [-.5, -.5]  # Initial x-y position (m)
data.qvel[2] = 4   # Initial vertical velocity (m/s)
ctrl_phase = 2 * np.pi * np.random.rand(model.nu)  # Control phase
ctrl_freq = 1     # Control frequency

# Visual options for the "ghost" model.
vopt2 = mujoco.MjvOption()
vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
pert = mujoco.MjvPerturb()  # Empty MjvPerturb object
# We only want dynamic objects (the humanoid). Static objects (the floor)
# should not be re-drawn. The mjtCatBit flag lets us do that, though we could
# equivalently use mjtVisFlag.mjVIS_STATIC
catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

# Simulate and render.
with mujoco.viewer.launch(model, data) as viewer:
    while data.time < duration and viewer.is_alive:
        # Apply sinusoidal control
        data.ctrl = np.sin(ctrl_phase + 2 * np.pi * data.time * ctrl_freq)

        # Step simulation
        mujoco.mj_step(model, data)

        # Update scene with main data
        mujoco.mjv_updateScene(model, data, viewer.opt, viewer.pert,
                               viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.scn)

        # Prepare ghost humanoid by offsetting qpos and forward kinematics
        data2.qpos[:] = data.qpos
        data2.qpos[0] += 1.5
        data2.qpos[1] += 1.0
        mujoco.mj_forward(model, data2)

        # Add ghost humanoid geoms to scene
        mujoco.mjv_addGeoms(model, data2, vopt2, pert, catmask, viewer.scn)

        # Render to screen
        viewer.render()