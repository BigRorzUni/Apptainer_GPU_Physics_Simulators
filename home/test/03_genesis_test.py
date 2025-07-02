import genesis as gs

def test_genesis_GPU_available():
    gs.init(backend=gs.gpu)
    assert gs.backend.name == "cuda", "Genesis failed to initialise on the GPU"

    # free resources
    gs.destroy()

    print("memory freed")