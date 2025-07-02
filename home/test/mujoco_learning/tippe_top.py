import mujoco
from mujoco import mjx
import mujoco_viewer

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt


tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>

  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""

# Make model and data
model = mujoco.MjModel.from_xml_string(tippe_top)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)


print('timestep', model.opt.timestep)
print('default gravity', model.opt.gravity)

print('positions', data.qpos)
print('velocities', data.qvel)

timevals = []
angular_velocity = []
stem_height = []

mujoco.mj_resetDataKeyframe(model, data, 0) 
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        timevals.append(data.time)
        angular_velocity.append(data.qvel[3:6].copy())
        stem_height.append(data.geom_xpos[2,2])
        viewer.render()
    else:
        break

plt.figure()

dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

ax[0].plot(timevals, angular_velocity)
ax[0].set_title('angular velocity')
ax[0].set_ylabel('radians / second')

ax[1].plot(timevals, stem_height)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('meters')

_ = ax[1].set_title('stem height')

plt.savefig('./test.png')

print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)

viewer.close()