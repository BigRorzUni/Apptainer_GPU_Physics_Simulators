import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch


fields = gs.options.SimOptions.model_fields
for name, field in fields.items():
    print(f"{name}: type={field.annotation}, default={field.default}")


phys_params = gs.options.RigidOptions(
    dt=0.005,
    integrator=gs.integrator.Euler,
    gravity=(0.0, 0.0, -9.81),
    iterations=50,
)

gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True, rigid_options=phys_params)

# Load your MJCF describing plane + ball
entity = scene.add_entity(gs.morphs.MJCF(file='../xml/ball_plane.xml'))

ball = entity.get_link("ball")
ball_id = ball.idx

ball = entity.get_link("ball")
ball_dof_start = ball.dof_start
ball_dof_end = ball.dof_end
n_dofs = entity.n_dofs
n_ball_dofs = ball_dof_end - ball_dof_start

#print(f"Total DOFs in entity: {n_dofs}")
#print(f"Ball DOFs from {ball_dof_start} to {ball_dof_end}")

# Build scene in given environments
N = 50
scene.build(n_envs=N)

vels = torch.zeros(N, n_dofs, device=gs.device)

# Random initial velocities for each env, uniform in [-1,1]
random_vels = ((2 * torch.rand(N, n_ball_dofs) - 1) * 3).to(gs.device)

# Assign these random velocities to the ball DOFs
vels[:, ball_dof_start:ball_dof_end] = random_vels

print("combined qvel:\n", vels)

entity.set_dofs_velocity(vels)

steps = 200

z_vals = []
for i in range(steps):
    scene.step()
    pos = entity.get_links_pos()  # shape (num_links, 3)

    ball_pos = pos[:, ball_id]
    z_pos = ball_pos[:,2]
    z_vals.append(z_pos.cpu().numpy())

z_vals_all_envs = np.array(z_vals)
num_steps = z_vals_all_envs.shape[0]

# free resources
gs.destroy()

plt.figure(figsize=(10, 6))

for env_idx in range(N):
    plt.plot(range(num_steps), z_vals_all_envs[:, env_idx], label=f'Env {env_idx}')

plt.xlabel("Time step")
plt.ylabel("Z Position")
plt.title("Vertical positions over time (GS)")
plt.grid(True)
plt.savefig('gs_multi_env.png')

