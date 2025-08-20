import time
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## entities ##########################


# when loading an entity, you can specify its pose in the morph.
franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = '../../xml/pendulum.xml',
        pos   = (1.0, 1.0, 0.0),
        euler = (0, 0, 0),
    ),
)

########################## build ##########################
scene.build()

# PD control
steps = 1000
for i in range(steps):

    scene.step()