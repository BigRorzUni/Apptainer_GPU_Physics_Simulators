import newton

import warp as wp

def test_newton_gpu_render():
    assert wp.is_cuda_available(), "CUDA GPU not available"

    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    model = builder.finalize(device="cuda")
    solver = newton.solvers.SemiImplicitSolver(model)

    renderer = newton.utils.SimRendererOpenGL(model, path="test.usd")

    fps = 60
    dt = 1.0 / fps
    sim_time = 0.0

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    N_steps = 10
    for _ in range(N_steps):
        state_0.clear_forces()

        contacts = model.collide(state_0)
        solver.step(state_in=state_0, state_out=state_1, control=control, contacts=contacts, dt=dt)

        # Rendering
        renderer.begin_frame(sim_time)
        renderer.render(state_0)
        renderer.render_contacts(state_0, contacts, contact_point_radius=1e-2)
        renderer.end_frame()

        sim_time += dt
        state_0, state_1 = state_1, state_0
    
    renderer.close()

    assert True