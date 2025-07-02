import mujoco
from mujoco import mjx
import mujoco_viewer

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

MJCF = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="6 6"
     texuniform="true" reflectance=".2"/>
     <material name="wall" rgba='.5 .5 .5 1'/>
  </asset>

  <default>

    <geom type="box" size=".05 .05 .05" />
    <joint type="free"/>
  </default>

  <worldbody>
    <light name="light" pos="-.2 0 1"/>
    <geom name="ground" type="plane" size=".5 .5 10" material="grid"
     zaxis="-.3 0 1" friction=".1"/>
    <camera name="y" pos="-.1 -.6 .3" xyaxes="1 0 0 0 1 2"/>
    <body pos="0 0 .1">
      <joint/>
      <geom friction=".1"/>
    </body>
    <body pos="0 .2 .1">
      <joint/>
      <geom rgba="1 0 0 1" friction="1"/>
    </body>
  </worldbody>

</mujoco>
"""
n_steps = 5000

# load
model = mujoco.MjModel.from_xml_string(MJCF)
data = mujoco.MjData(model)

# Simulate and display video.

viewer = mujoco_viewer.MujocoViewer(model, data)

mujoco.mj_resetData(model, data)
for i in range(n_steps):
    if not viewer.is_alive:
        break
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()