import mujoco
from mujoco import mjx
import mujoco_viewer

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from random import randint

# MJCF model: falling ball with gravity
mjcf = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.05" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Load CPU-side model and data for viewer
model = mujoco.MjModel.from_xml_string(mjcf)
data = mujoco.MjData(model)

# Put model on GPU for batched sim
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

N = 16  # Small batch size for visualization
timesteps = 500

# Create N copies of the same initial state
mjx_datas = jax.tree_util.tree_map(
    lambda x: jp.tile(x[None], [N] + [1] * x.ndim), mjx_data)

# Slightly perturb initial z positions across batch
mjx_datas = mjx_datas.replace(qpos = mjx_datas.qpos.at[:, 2].set(jp.linspace(1.0, 2.0, N)))

# Apply random initial velocities
key = jax.random.PRNGKey(42)
random_linear_vel = jax.random.uniform(key, (N, 3), minval=-10.0, maxval=10.0)
random_angular_vel = jax.random.uniform(key, (N, 3), minval=-5.0, maxval=5.0)
random_qvel = jp.concatenate([random_linear_vel, random_angular_vel], axis=1)
mjx_datas = mjx_datas.replace(qvel=random_qvel)

@jax.jit
def step_batch(data_batch):
    return jax.vmap(lambda d: mjx.step(mjx_model, d))(data_batch)

@jax.jit
def rollout_batch(data_batch):
    def step_fn(data, _):
        data = step_batch(data)
        return data, data.qpos  # Log qpos
    
    return jax.lax.scan(step_fn, data_batch, None, length=timesteps)

# Run batch simulation and log qpos
final_data, qpos_log = rollout_batch(mjx_datas)

qpos_log_np = np.array(qpos_log) 

idx = np.random.randint(N)  # random environment index

start_pos = qpos_log_np[0, idx, :3]  # initial x,y,z position

viewer = mujoco_viewer.MujocoViewer(model, data)

viewer.cam.lookat[:] = start_pos
viewer.cam.distance = 2.5
viewer.cam.elevation = -10
viewer.cam.azimuth = 90

for t in range(timesteps):
    if not viewer.is_alive:
        break
    data.qpos[:] = qpos_log_np[t, idx]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    viewer.render()

viewer.close()


# plot vertical position of the falling balls over time for all environments
for i in range(N):
    plt.plot(qpos_log_np[:, i, 2], label=f"env {i}")

plt.xlabel("Time step")
plt.ylabel("Z Position")
plt.title("Vertical positions over time")
plt.legend()
plt.savefig('testmjx.png')
