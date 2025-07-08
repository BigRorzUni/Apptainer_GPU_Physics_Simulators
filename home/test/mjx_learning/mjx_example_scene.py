import mujoco
from mujoco import mjx
import mujoco_viewer

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from random import randint

jax.config.update("jax_enable_x64", False)

# load CPU-side model and data for viewer
model = mujoco.MjModel.from_xml_path("../xml/ball_plane.xml")
data = mujoco.MjData(model)

# put model on GPU for batched sim
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

N = 16  
timesteps = 200

# create N copies of the same initial state
mjx_datas = jax.tree_util.tree_map(
    lambda x: jp.tile(x[None], [N] + [1] * x.ndim), mjx_data)

# apply random initial velocities
key = jax.random.PRNGKey(42)
random_qvel = jax.random.uniform(key, (N, 6), minval=-3.0, maxval=3.0)
mjx_datas = mjx_datas.replace(qvel=random_qvel)

#print("Combined qvel:\n", random_qvel)

@jax.jit
def step_batch(data_batch):
    return jax.vmap(lambda d: mjx.step(mjx_model, d))(data_batch)

@jax.jit
def rollout_batch(data_batch):
    def step_fn(data, _):
        data = step_batch(data)
        return data, data.qpos  # Log qpos
    
    return jax.lax.scan(step_fn, data_batch, None, length=timesteps)

# run batch simulation and log qpos
final_data, qpos_log = rollout_batch(mjx_datas)

qpos_log_np = np.array(qpos_log) 

# viewing
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
    mujoco.mj_forward(model, data)
    viewer.render()
viewer.close()

plt.figure(figsize=(10, 6))

# plot vertical position of the falling balls over time for all environments
for i in range(N):
    plt.plot(qpos_log_np[:, i, 2], label=f"env {i}")

plt.xlabel("Time step")
plt.ylabel("Z Position")
plt.title("Vertical positions over time (MJX)")
plt.grid(True)
plt.savefig('mjx_multi_env.png')
