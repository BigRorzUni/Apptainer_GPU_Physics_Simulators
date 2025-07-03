import genesis as gs
import matplotlib.pyplot as plt
import numpy as np

gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True)

# Load your MJCF describing plane + ball
entity = scene.add_entity(gs.morphs.MJCF(file='../xml/ball_plane.xml'))
ball_id = entity.get_link("ball").idx

# Build scene with 1 environment (single simulation)
envs = 5
scene.build(n_envs=envs)

z_vals = []
for i in range(100):
    scene.step()
    pos = entity.get_links_pos()  # shape (num_links, 3)

    ball_pos = pos[:, ball_id]
    z_pos = ball_pos[:,2]
    z_vals.append(z_pos.cpu().numpy())

z_vals_all_envs = np.array(z_vals)
num_envs = z_vals_all_envs.shape[1]
num_steps = z_vals_all_envs.shape[0]

plt.figure(figsize=(10, 6))

for env_idx in range(num_envs):
    plt.plot(range(num_steps), z_vals_all_envs[:, env_idx], label=f'Env {env_idx}')

plt.xlabel("Time step")
plt.ylabel("Z Position")
plt.title("Ball vertical positions over time for each environment")
plt.legend()
plt.grid(True)
plt.savefig('testgenesis_multi_env.png')


