import mujoco
import mujoco_viewer

import numpy as np
import matplotlib.pyplot as plt

N = 50
timesteps = 200

# Load the model once
model = mujoco.MjModel.from_xml_path("../xml/ball_plane.xml")

# Prepare to store logs: shape (timesteps, N, qpos_dim)
qpos_logs = np.zeros((timesteps, N, model.nq))

# Create N independent MjData instances and apply random initial velocities
datas = []
np.random.seed(42)
for i in range(N):
    data = mujoco.MjData(model)
    # Random initial velocities for 6 DOF, uniform [-3,3]
    data.qvel[:] = np.random.uniform(-3.0, 3.0, size=model.nv)
    datas.append(data)

# Run simulation for each environment independently and log qpos
for i, data in enumerate(datas):
    for t in range(timesteps):
        mujoco.mj_step(model, data)
        qpos_logs[t, i, :] = data.qpos

# Visualize one random environment with mujoco_viewer
idx = np.random.randint(N)
data_view = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data_view)

viewer.cam.lookat[:] = qpos_logs[0, idx, :3]
viewer.cam.distance = 2.5
viewer.cam.elevation = -10
viewer.cam.azimuth = 90

for t in range(timesteps):
    if not viewer.is_alive:
        break
    data_view.qpos[:] = qpos_logs[t, idx]
    mujoco.mj_forward(model, data_view)
    viewer.render()
viewer.close()

# Plot vertical position over time for all environments
for i in range(N):
    plt.plot(qpos_logs[:, i, 2], label=f"env {i}")

plt.xlabel("Time step")
plt.ylabel("Z Position")
plt.title("Vertical positions over time (mujoco CPU)")
plt.grid(True)
plt.savefig('mujoco_multi_env.png')
