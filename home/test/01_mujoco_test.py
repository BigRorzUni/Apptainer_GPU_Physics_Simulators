import mujoco
import mujoco_viewer

def test_mujoco_visualisation():

  model = mujoco.MjModel.from_xml_path("xml/pendulum.xml")
  data = mujoco.MjData(model)

  viewer = mujoco_viewer.MujocoViewer(model, data)

  for _ in range(10000):
      if viewer.is_alive:
          mujoco.mj_step(model, data)
          viewer.render()
      else:
          break

  viewer.close()

  assert 1 == 1