import genesis as gs

def test_genesis_GPU_available():
    gs.init(backend=gs.gpu)
    assert gs.backend.name == "cuda", "Genesis failed to initialise on the GPU"

    scene = gs.Scene()

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    scene.add_entity(gs.morphs.MJCF(file="xml/pendulum.xml"))

    scene.build()
    for i in range(1000):
        scene.step()
        scene.render()

    # free resources
    gs.destroy()

    print("memory freed")