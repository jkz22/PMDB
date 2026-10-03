"""Plane-strain finite-strain neo-Hookean solver with eigenstretches (dolfinx 0.10, serial).

Only this module imports dolfinx / ufl / petsc4py / mpi4py (P1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from pmdb.fem.materials import ARTEFACT, BINDER, GRAPHITE, PORE, SI, PhaseProps, lame
from pmdb.fem.result import FIELD_KEYS, SimResult

LABELS = (BINDER, SI, GRAPHITE, PORE, ARTEFACT)


@dataclass(frozen=True)
class BCSpec:
    orientation: Literal["bottom", "top"]
    lateral: Literal["both", "left"] = "both"  # "left" = test-only roller support


def _interp_points(V0):
    ip = V0.element.interpolation_points
    return ip() if callable(ip) else ip


def simulate(labels: np.ndarray, px_um: float, props_fn: Callable[[float], dict[int, PhaseProps]],
             bc: BCSpec, solver_opts: dict, frames: np.ndarray, extra_targets: Sequence[float] = (),
             log: Callable[[dict], None] | None = None) -> SimResult:
    assert MPI.COMM_WORLD.size == 1, "the FEM run is serial"
    t_start = time.time()
    labels = np.asarray(labels, dtype=np.uint8)
    H, W = labels.shape
    h = float(px_um)
    frames = np.asarray(frames, dtype=float)

    msh = mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([W * h, H * h])], [W, H],
        mesh.CellType.quadrilateral)
    tdim = msh.topology.dim
    ncell = msh.topology.index_map(tdim).size_local
    mid = mesh.compute_midpoints(msh, tdim, np.arange(ncell, dtype=np.int32))
    col = np.floor(mid[:, 0] / h).astype(np.int64)
    row = (H - 1 - np.floor(mid[:, 1] / h)).astype(np.int64)
    pix = row * W + col
    assert ncell == H * W and np.array_equal(np.sort(pix), np.arange(H * W)), "pixel map is not a permutation"

    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    V0 = fem.functionspace(msh, ("DG", 0))
    u = fem.Function(V, name="u")
    mu_f, lam_f, lx_f, lz_f, ly_f = (fem.Function(V0) for _ in range(5))

    cell_dofs = np.asarray(V0.dofmap.list)[:, 0]
    label_by_dof = np.empty(ncell, dtype=np.uint8)
    label_by_dof[cell_dofs] = labels.ravel()[pix]

    def set_coeffs(s: float) -> None:
        props = props_fn(s)
        tab = np.zeros((5, 5))
        for lab in LABELS:
            pp = props[lab]
            mu, lam = lame(pp.E, pp.nu)
            tab[lab] = (mu, lam, pp.stretch[0], pp.stretch[1], pp.stretch[2])
        vals = tab[label_by_dof]
        for k, f in enumerate((mu_f, lam_f, lx_f, lz_f, ly_f)):
            f.x.array[:] = vals[:, k]

    # --- kinematics and energy (P6)
    F2 = ufl.variable(ufl.Identity(2) + ufl.grad(u))
    Fl2 = ufl.as_matrix([[lx_f, 0], [0, lz_f]])
    Fe2 = F2 * ufl.inv(Fl2)
    Jlam = lx_f * lz_f * ly_f
    Je = ufl.det(Fe2) / ly_f
    lnJe = ufl.ln(Je)
    trCe = ufl.tr(Fe2.T * Fe2) + 1.0 / ly_f**2
    psi = Jlam * (mu_f / 2 * (trCe - 3) - mu_f * lnJe + lam_f / 2 * lnJe**2)
    P = ufl.diff(psi, F2)
    dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": int(solver_opts["quadrature_degree"])})
    F_res = ufl.inner(P, ufl.grad(ufl.TestFunction(V))) * dx

    # --- boundary conditions (P10, P12 pattern)
    fdim = tdim - 1
    bcs = []

    def add_bc(sub: int, locator) -> None:
        Vs, _ = V.sub(sub).collapse()
        facets = mesh.locate_entities_boundary(msh, fdim, locator)
        dofs = fem.locate_dofs_topological((V.sub(sub), Vs), fdim, facets)
        z = fem.Function(Vs)
        z.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(z, dofs, V.sub(sub)))

    add_bc(0, lambda x: np.isclose(x[0], 0.0))
    if bc.lateral == "both":
        add_bc(0, lambda x: np.isclose(x[0], W * h))
    if bc.orientation == "bottom":
        add_bc(1, lambda x: np.isclose(x[1], 0.0))
    else:
        add_bc(1, lambda x: np.isclose(x[1], H * h))

    problem = NonlinearProblem(
        F_res, u, bcs=bcs, petsc_options_prefix="fem_",
        petsc_options={
            "snes_type": "newtonls", "snes_linesearch_type": str(solver_opts["linesearch"]),
            "snes_rtol": float(solver_opts["snes_rtol"]), "snes_atol": float(solver_opts["snes_atol"]),
            "snes_max_it": int(solver_opts["snes_max_it"]), "snes_stol": 0.0,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 100,
        })

    # --- post-processing expressions (P13)
    Be2 = Fe2 * Fe2.T
    Je_s = Je
    be33 = 1.0 / ly_f**2
    expr = {
        "J": ufl.det(F2),
        "sxx": (mu_f * (Be2[0, 0] - 1) + lam_f * lnJe) / Je_s,
        "szz": (mu_f * (Be2[1, 1] - 1) + lam_f * lnJe) / Je_s,
        "sxz": mu_f * Be2[0, 1] / Je_s,
        "syy": (mu_f * (be33 - 1) + lam_f * lnJe) / Je_s,
    }
    ip = _interp_points(V0)
    expr_c = {k: fem.Expression(v, ip) for k, v in expr.items()}
    out_f = {k: fem.Function(V0) for k in expr}

    X = V.tabulate_dof_coordinates()
    ncol = np.rint(X[:, 0] / h).astype(np.int64)
    nrow = H - np.rint(X[:, 1] / h).astype(np.int64)

    def snapshot() -> tuple[np.ndarray, dict[str, np.ndarray]]:
        uu = u.x.array.reshape(-1, 2)
        un = np.full((H + 1, W + 1, 2), np.nan, dtype=np.float32)
        un[nrow, ncol, :] = uu
        fl = {}
        for k, e in expr_c.items():
            out_f[k].interpolate(e)
            a = np.empty(H * W)
            a[pix] = out_f[k].x.array[cell_dofs]
            fl[k] = a.reshape(H, W)
        fl["vm"] = np.sqrt(0.5 * ((fl["sxx"] - fl["szz"]) ** 2 + (fl["szz"] - fl["syy"]) ** 2
                                  + (fl["syy"] - fl["sxx"]) ** 2) + 3.0 * fl["sxz"] ** 2)
        return un, {k: v.astype(np.float32) for k, v in fl.items()}

    nfr = len(frames)
    u_nodes = np.full((nfr, H + 1, W + 1, 2), np.nan, dtype=np.float32)
    fields = {k: np.full((nfr, H, W), np.nan, dtype=np.float32) for k in FIELD_KEYS}
    converged = np.zeros(nfr, dtype=bool)
    substeps: list[dict] = []

    def attempt(s_try: float, ds: float) -> tuple[bool, int, int]:
        set_coeffs(s_try)
        t0 = time.time()
        reason, its, fnorm, err = -99, 0, float("nan"), None
        try:
            problem.solve()
            reason = int(problem.solver.getConvergedReason())
            its = int(problem.solver.getIterationNumber())
            fnorm = float(problem.solver.getFunctionNorm())
        except Exception as exc:  # PETSc / SNES failure
            err = repr(exc)[:500]
            try:
                reason = int(problem.solver.getConvergedReason())
                its = int(problem.solver.getIterationNumber())
                fnorm = float(problem.solver.getFunctionNorm())
            except Exception:
                pass
            if reason > 0:
                reason = -99
        ok = reason > 0 and bool(np.all(np.isfinite(u.x.array)))
        rec = {"s": float(s_try), "ds": float(ds), "its": its, "reason": reason, "ok": bool(ok),
               "wall_s": time.time() - t0, "fnorm": fnorm, "error": err}
        substeps.append(rec)
        if log is not None:
            log(rec)
        return ok, its, reason

    targets = sorted(set(float(x) for x in frames) | set(float(x) for x in extra_targets) | {0.0})
    ds_max = float(solver_opts["ds_max"])
    ds_min = float(solver_opts["ds_min"])
    grow_le = int(solver_opts["grow_if_its_le"])

    def frame_index(s: float) -> int | None:
        hit = np.nonzero(np.isclose(frames, s, atol=1e-12))[0]
        return int(hit[0]) if len(hit) else None

    failed_at_s = float("nan")
    # state at s = 0 (always solved; zero iterations)
    u.x.array[:] = 0.0
    ok, _, _ = attempt(0.0, 0.0)
    s_n, s_nm1 = 0.0, None
    u_n = u.x.array.copy()
    u_nm1 = None
    if not ok:
        failed_at_s = 0.0
    else:
        fi = frame_index(0.0)
        if fi is not None:
            u_nodes[fi], f0 = snapshot()
            for k in FIELD_KEYS:
                fields[k][fi] = f0[k]
            converged[fi] = True
        ds = min(0.05, ds_max)
        pred_ok = True
        for target in targets:
            if target <= 0.0:
                continue
            while s_n < target - 1e-12 and not np.isfinite(failed_at_s):
                step = min(ds, target - s_n)
                if target - (s_n + step) < 1e-12:
                    step = target - s_n
                s_try = s_n + step
                use_pred = (pred_ok and u_nm1 is not None and s_nm1 is not None and s_n > s_nm1)
                if use_pred:
                    u.x.array[:] = u_n + (step / (s_n - s_nm1)) * (u_n - u_nm1)
                else:
                    u.x.array[:] = u_n
                ok, its, _ = attempt(s_try, step)
                pred_ok = True
                if ok:
                    u_nm1, s_nm1 = u_n, s_n
                    u_n = u.x.array.copy()
                    s_n = target if abs(target - s_try) < 1e-12 else s_try
                    if its <= grow_le:
                        ds = min(ds_max, 2 * ds)
                else:
                    u.x.array[:] = u_n
                    if use_pred:  # predictor can overshoot into inverted pores: retry this step from u_n
                        pred_ok = False
                        continue
                    new_step = step / 2.0
                    if new_step < ds_min * (1.0 - 1e-9):
                        failed_at_s = float(s_try)
                    else:
                        ds = new_step
            if np.isfinite(failed_at_s):
                break
            fi = frame_index(target)
            if fi is not None:
                set_coeffs(target)
                u.x.array[:] = u_n
                un, fl = snapshot()
                u_nodes[fi] = un
                for k in FIELD_KEYS:
                    fields[k][fi] = fl[k]
                converged[fi] = True

    return SimResult(labels=labels, px_um=h, s=frames, converged=converged, u_nodes=u_nodes,
                     fields=fields, failed_at_s=failed_at_s, substeps=substeps,
                     wall_s=time.time() - t_start)
