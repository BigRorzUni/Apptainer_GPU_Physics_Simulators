import mujoco
from mujoco import mjx
import jax
import jax.numpy as jp
import numpy as np

def test_mjx_working_GPU():
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

    # Load model and data
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)

    # Place model and data on GPU
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Confirm we're on GPU
    print("mj_data.qpos (CPU):", mj_data.qpos)
    print("mjx_data.qpos (GPU):", mjx_data.qpos, "Device:", mjx_data.qpos.devices())

    device = mjx_data.qpos.devices()
    assert any("cuda" in str(d).lower() for d in device), "JAX is not using a GPU!"

    # Create a batch of initial states (e.g., 8 different qpos)
    batch_size = 8
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, batch_size)

    def random_state(rng):
        return mjx_data.replace(qpos=jax.random.uniform(rng, shape=mjx_data.qpos.shape, minval=-0.1, maxval=0.1))

    batch = jax.vmap(random_state)(rngs)

    # Define a batched, jitted simulation step
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    # Run N simulation steps
    N = 10
    for i in range(N):
        batch = jit_step(mjx_model, batch)

    # Optional: bring results back to CPU and inspect
    final_qpos = jax.device_get(batch.qpos)
    print("Final qpos from batch after", N, "steps:\n", final_qpos)

    assert final_qpos.shape == (batch_size, mj_model.nq)

