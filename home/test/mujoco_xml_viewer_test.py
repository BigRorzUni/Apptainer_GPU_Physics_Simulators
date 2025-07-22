import mujoco
import mujoco_viewer


model = mujoco.MjModel.from_xml_path("xml/Franka_emika_scenes_V1/Franka_and_cuboid.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

for _ in range(300):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

viewer.close()