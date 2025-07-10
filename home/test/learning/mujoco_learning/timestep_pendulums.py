import mujoco
from mujoco import mjx
import mujoco_viewer

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt


chaotic_pendulum = """
<mujoco>
  <option timestep=".001">
    <flag energy="enable" contact="disable"/>
  </option>

  <default>
    <joint type="hinge" axis="0 -1 0"/>
    <geom type="capsule" size=".02"/>
  </default>

  <worldbody>
    <light pos="0 -.4 1"/>
    <camera name="fixed" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <body name="0" pos="0 0 .2">
      <joint name="root"/>
      <geom fromto="-.2 0 0 .2 0 0" rgba="1 1 0 1"/>
      <geom fromto="0 0 0 0 0 -.25" rgba="1 1 0 1"/>
      <body name="1" pos="-.2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="1 0 0 1"/>
      </body>
      <body name="2" pos=".2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 1 0 1"/>
      </body>
      <body name="3" pos="0 0 -.25">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(chaotic_pendulum)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)


PERTURBATION = 1e-7
SIM_DURATION = 10 # seconds
TIMESTEPS = np.power(10, np.linspace(-2, -4, 5))


plt.figure()
# prepare plotting axes
_, ax = plt.subplots(1, 1)

for dt in TIMESTEPS:
   # set timestep, print
    model.opt.timestep = dt

    # preallocate
    n_steps = int(SIM_DURATION / model.opt.timestep)
    sim_time = np.zeros(n_steps)
    energy = np.zeros(n_steps)


    # initialize
    mujoco.mj_resetData(model, data)
    data.qvel[0] = 10 # root joint velocity

    # simulate
    print('{} steps at dt = {:2.2g}ms'.format(n_steps, 1000*dt))
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        sim_time[i] = data.time
        energy[i] = data.energy[0] + data.energy[1]

    # plot
    ax.plot(sim_time, energy, label='timestep = {:2.2g}ms'.format(1000*dt))


# finalize plot
ax.set_title('total energy')
ax.set_ylabel('Joule')
ax.set_xlabel('second')
plt.legend()
plt.tight_layout()

plt.savefig('test3.png')