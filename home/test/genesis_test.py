import genesis as gs


def test_genesis_GPU_available():
    gs.init(backend=gs.gpu)

    # TODO add assert statment tos ee if geneiss working on GPU

    
# gs.init(backend=gs.cpu)

# scene = gs.Scene(show_viewer=True)
# plane = scene.add_entity(gs.morphs.Plane())
# franka = scene.add_entity(
#     gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
# )

# scene.build()

# for i in range(1000):
#     scene.step()