# FEM Modal spike report

Result: SUCCESS on first run. Nothing committed. File: scripts/modal_fem_spike.py (new, uncommitted).

## Image
- Tag: `ghcr.io/fenics/dolfinx/dolfinx:v0.10.0` (exists; Modal accepted it with no add_python).
- Recipe: `modal.Image.from_registry(TAG).run_commands("pip install uv || python3 -m pip install uv").run_commands("uv pip install --system numpy pandas")`
- `--system` worked; no issues. Image reported python 3.12.3.

## Returned numbers
- dolfinx 0.10.0, petsc4py 3.24.0, python 3.12.3 (GCC 13.3.0)
- Newton iterations per step (5 steps, lambda 1.04..1.2): [4, 4, 4, 4, 4]
- mean J = 1.44 (expected 1.44)
- max |P| = 1.7e-14 (expected ~0, < 1e-8*E)
- in-function wall time 1.92 s (includes assembly/JIT compile)

## dolfinx 0.10 API used
- `mesh.create_unit_square(MPI.COMM_WORLD, 20, 20, mesh.CellType.quadrilateral)`
- `fem.functionspace(msh, ("Lagrange", 1, (2,)))` (vector space via shape tuple)
- `fem.Constant(msh, PETSc.ScalarType(1.0))`; update with `lam.value = ...`
- BCs per component: `Vs,_ = V.sub(i).collapse()`; `mesh.locate_entities_boundary(msh, fdim, marker)`; `fem.locate_dofs_topological((V.sub(i), Vs), fdim, facets)`; `fem.dirichletbc(zero_Function_on_Vs, dofs, V.sub(i))`
- `from dolfinx.fem.petsc import NonlinearProblem`;
  `NonlinearProblem(F_form, u, bcs=bcs, petsc_options_prefix="spike_", petsc_options={...})` (J defaults to automatic derivative; no J arg given)
  options: snes_type newtonls, snes_linesearch_type basic, snes_rtol 1e-12, snes_atol 1e-12, snes_max_it 30, ksp_type preonly, pc_type lu, pc_factor_mat_solver_type mumps
- Solve: `problem.solve()` (returns u; in 0.10 it is the SNES-based problem). Iterations: `problem.solver.getIterationNumber()`; reason: `problem.solver.getConvergedReason()` (>0 converged).
- Post-processing: `fem.Expression(ufl_expr, DG.element.interpolation_points)` then `Function.interpolate(expr)` (in 0.10 `interpolation_points` is a property, not a call; it worked as written).
- UFL: `F = ufl.variable(I + grad(u))`, `Fe = F*inv(lam*I)`, W = (mu/2(tr(FeT Fe)-2) - mu ln Je + la/2 ln(Je)^2)*det(Fl), `P = ufl.diff(W, F)` (2D, plane-strain-like with 3D-style Lame la).

## Cost
Container time not measured exactly; run completed in well under 2 min including image build (first build) and container start. Compute itself ~2 s. Assumed rates $0.0000131/core-s, $0.00000222/GiB-s: even 60 s billed = 2*60*1.31e-5 + 4*60*2.22e-6 = ~$0.0021. Far under $2 cap. (Pricing page not checked.)

## Issues
- First attempt failed only because a combined heredoc+modal bash command was refused by the worktree sandbox; split into Write + plain command. No code issues.
- Note: `modal run` creates the app under workspace "main" env (kyy82/main).
