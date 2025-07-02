import jax
import jax.numpy as jnp

def test_jax_on_GPU():
    devices = jax.devices()
    assert any("cuda" or "gpu" in str(device).lower() for device in devices), "JAX is not using a GPU!"
    x_gpu = jax.device_put(jnp.array([1.0, 2.0, 3.0]), device=jax.devices()[0])
    y_gpu = x_gpu * 2
    assert any("cuda" or "gpu" in str(y_gpu.device()).lower()), "JAX is not using a GPU!"
