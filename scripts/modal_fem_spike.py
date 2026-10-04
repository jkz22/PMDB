"""Spike: prove dolfinx runs on Modal (decision D16). Run: modal run scripts/modal_fem_spike.py"""
import modal

TAG = "ghcr.io/fenics/dolfinx/dolfinx:v0.10.0"
image = (
    modal.Image.from_registry(TAG)
    .run_commands("pip install uv || python3 -m pip install uv")
    .run_commands("uv pip install --system numpy pandas")
)
app = modal.App("pmdb-fem-spike", image=image)


@app.function(cpu=2, memory=4096, timeout=600)
def spike():
    import sys
    import time

    import dolfinx
    import numpy as np
    import petsc4py
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import NonlinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc

    t0 = time.time()
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20, mesh.CellType.quadrilateral)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u = fem.Function(V, name="u")
    lam = fem.Constant(msh, PETSc.ScalarType(1.0))
    E, nu = 10.0, 0.3
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    I = ufl.Identity(2)
    Fl = lam * I
    F = ufl.variable(I + ufl.grad(u))
    Fe = F * ufl.inv(Fl)
    Je = ufl.det(Fe)
    Ic = ufl.tr(Fe.T * Fe)
    W = (mu / 2 * (Ic - 2) - mu * ufl.ln(Je) + la / 2 * ufl.ln(Je) ** 2) * ufl.det(Fl)
    P = ufl.diff(W, F)
    Fres = ufl.inner(P, ufl.grad(ufl.TestFunction(V))) * ufl.dx

    def left(x):
        return np.isclose(x[0], 0.0)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    fdim = msh.topology.dim - 1
    bcs = []
    for sub, loc in ((0, left), (1, bottom)):
        Vs, _ = V.sub(sub).collapse()
        facets = mesh.locate_entities_boundary(msh, fdim, loc)
        dofs = fem.locate_dofs_topological((V.sub(sub), Vs), fdim, facets)
        z = fem.Function(Vs)
        z.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(z, dofs, V.sub(sub)))

    problem = NonlinearProblem(
        Fres, u, bcs=bcs, petsc_options_prefix="spike_",
        petsc_options={
            "snes_type": "newtonls", "snes_linesearch_type": "basic",
            "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 30,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
        },
    )
    its = []
    for k in range(1, 6):
        lam.value = 1.0 + 0.2 * k / 5
        problem.solve()
        its.append(int(problem.solver.getIterationNumber()))
        assert problem.solver.getConvergedReason() > 0, problem.solver.getConvergedReason()

    DG = fem.functionspace(msh, ("DG", 0))
    Jf = fem.Function(DG)
    Jf.interpolate(fem.Expression(ufl.det(I + ufl.grad(u)), DG.element.interpolation_points))
    Pf = fem.Function(DG)
    Pf.interpolate(fem.Expression(ufl.sqrt(ufl.inner(P, P)), DG.element.interpolation_points))
    return {
        "dolfinx": dolfinx.__version__, "petsc4py": petsc4py.__version__,
        "python": sys.version, "newton_its": its, "mean_J": float(np.mean(Jf.x.array)),
        "max_abs_P": float(np.max(np.abs(Pf.x.array))), "wall_s": time.time() - t0,
    }


@app.local_entrypoint()
def main():
    print(spike.remote())
