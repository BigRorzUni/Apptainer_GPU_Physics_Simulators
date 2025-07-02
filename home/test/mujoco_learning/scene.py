import mujoco
from mujoco import mjx
import mujoco.viewer

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

import mediapy as media

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

print(mujoco.__version__)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

def get_geom_speed(model, data, geom_name):
  """Returns the speed of a geom."""
  geom_vel = np.zeros(6)
  geom_type = mujoco.mjtObj.mjOBJ_GEOM
  geom_id = data.geom(geom_name).id
  mujoco.mj_objectVelocity(model, data, geom_type, geom_id, geom_vel, 0)
  return np.linalg.norm(geom_vel)

def add_visual_capsule(scene, point1, point2, radius, rgba):
  """Adds one capsule to an mjvScene."""
  if scene.ngeom >= scene.maxgeom:
    return
  scene.ngeom += 1  # increment ngeom
  # initialise a new capsule, add it to the scene using mjv_connector
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
  mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                       mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                       point1, point2)

 # traces of time, position and speed
times = []
positions = []
speeds = []
offset = model.jnt_axis[0]/16  # offset along the joint axis

def modify_scene(scn):
  """Draw position trace, speed modifies width and colors."""
  if len(positions) > 1:
    for i in range(len(positions)-1):
      rgba=np.array((np.clip(speeds[i]/10, 0, 1),
                     np.clip(1-speeds[i]/10, 0, 1),
                     .5, 1.))
      radius=.003*(1+speeds[i])
      point1 = positions[i] + offset*times[i]
      point2 = positions[i+1] + offset*times[i+1]
      add_visual_capsule(scn, point1, point2, radius, rgba)

# Simulation callback (per frame)
def simulate_fn(model, data):
    positions.append(data.geom_xpos[data.geom("green_sphere").id].copy())
    times.append(data.time)
    speeds.append(get_geom_speed(model, data, "green_sphere"))

# Launch passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

  # Enable wireframe rendering of the entire scene.
  viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
  viewer.sync()

  while viewer.is_running():
    # Step the physics.
    mujoco.mj_step(model, data)

    # Add a 3x3x3 grid of variously colored spheres to the middle of the scene.
    viewer.user_scn.ngeom = 0
    i = 0
    for x, y, z in itertools.product(*((range(-1, 2),) * 3)):
      mujoco.mjv_initGeom(
          viewer.user_scn.geoms[i],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[0.02, 0, 0],
          pos=0.1*np.array([x, y, z]),
          mat=np.eye(3).flatten(),
          rgba=0.5*np.array([x + 1, y + 1, z + 1, 2])
      )
      i += 1
    viewer.user_scn.ngeom = i
    viewer.sync()