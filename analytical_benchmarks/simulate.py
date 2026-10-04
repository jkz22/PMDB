"""Image-based charge/discharge + cycling simulator (illustrative, literature parameters, not calibrated).

Input: one segmented SEM image (a PMDB site, or any BSE .tif). The image itself evolves cycle by cycle:
  per half-cycle  constant current (C-rate): the Li flux is shared by the Si surface that is not yet full
                  (charge) or empty (discharge) and diffuses inside each particle (implicit finite
                  differences); Li trapped in particle cores carries over to the next cycle; graphite/binder
                  lithiates uniformly. (--li erfc: older instant-surface model, c = erfc(d / 2 sqrt(D t)).) Each pixel expands
                  isotropically with its Li content; a 2-D plane-strain finite-element solve (Q1 pixel
                  elements, coating clamped below and laterally, free top) gives stress, pore closure and
                  thickness change.
  per cycle       SEI grows on all active Si surfaces (delta = k sqrt(cycles since exposure)), locks Li and
                  fills adjacent pores; Si particles crack with Weibull probability from their peak tensile
                  stress (crack plane normal to the max principal stress, crack pixels become pore and expose
                  new surface); fragments < A_MIN or without contact to the matrix become inactive.
  KPIs            recomputed on every evolved image with the QC pipeline's own particle code.
Labels: 0 pore, 1 graphite/binder, 2 Si (active), 3 SEI, 4 Si (inactive).
Stresses are elastic (no plasticity, small-strain theory at up to 56% strain): use them as a relative
indicator. Rows are taken as the through-thickness direction (top = separator side), which is an assumption.

  python3 simulate.py Batch_1/5n1q8atc Batch_2/epqdaau9 --cycles 50 --seeds 3
  python3 simulate.py --image my_bse.tif --nm-per-px 25
  python3 simulate.py --image my_bse.tif --cycles 100 --seeds 1 --render   # + render_<name>.npz for evolve_video.py
"""
import argparse, json, os, numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
from scipy import ndimage as ndi
from scipy.sparse.linalg import splu
from scipy.special import erf, erfc
from skimage import draw, measure
from seg import segment
from kpis import particles

PX = 0.1          # µm per pixel of the evolving image
MB = 2            # mechanics element = MB x MB pixels (0.2 µm)
PAR = dict(
    D_si=1e-3,                      # Li diffusivity in Si, µm²/s (1e-11 cm²/s; literature 1e-14..1e-10 cm²/s)
    eps_si=3.8 ** (1 / 3) - 1,      # linear strain of Si -> Li15Si4 (+280% volume)
    eps_gr=1.10 ** (1 / 3) - 1,     # graphite -> LiC6 (+10% volume), treated as isotropic
    E={0: 0.01, 1: 10.0, 2: 80.0, 3: 1.0, 4: 80.0}, nu=0.25,   # Young's moduli, GPa
    Q_si=3579 * 2.33, Q_gr=372 * 2.26,                          # mAh per cm³ of phase
    q_sei=1500.0,                   # Li locked per cm³ of SEI, mAh/cm³ (LiF/Li2CO3-like)
    k_sei=0.05,                     # SEI growth, µm per sqrt(cycle)
    sigma0=25.0, m=4.0,             # Weibull scale (GPa, elastic indicator) and modulus for Si cracking
    a_min=0.25,                     # fragments below this area (µm²) lose contact -> inactive
    li="cc",                        # "cc": constant-current Li model, "erfc": instant surface switch (older)
)
NSUB = 120                          # Li time steps per half-cycle
DMG = (12, 30, 60, 90, 120)         # steps (of NSUB) at which stress is evaluated for damage
CMAP = np.array([[30, 60, 200], [90, 90, 90], [255, 150, 0], [40, 200, 90], [220, 30, 30]], np.uint8)
S4 = ndi.generate_binary_structure(2, 1)


# ---------------- finite elements: Q1 plane strain on a pixel grid ----------------
class Mesh:
    def __init__(self, ny, nx, h, nu):
        D = np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]]) / ((1 + nu) * (1 - 2 * nu))
        xi = np.array([-1, 1, 1, -1]); et = np.array([-1, -1, 1, 1])     # bl, br, tr, tl

        def Bm(s, t):
            dx = xi * (1 + et * t) / 4 * 2 / h; dy = et * (1 + xi * s) / 4 * 2 / h
            B = np.zeros((3, 8)); B[0, 0::2] = dx; B[1, 1::2] = dy; B[2, 0::2] = dy; B[2, 1::2] = dx
            return B
        g = 1 / np.sqrt(3); self.K0 = np.zeros((8, 8)); self.f0 = np.zeros(8)
        for s in (-g, g):
            for t in (-g, g):
                B = Bm(s, t); self.K0 += B.T @ D @ B * (h / 2) ** 2; self.f0 += B.T @ D @ [1, 1, 0] * (h / 2) ** 2
        self.D, self.Bc, self.ny, self.nx, self.h = D, Bm(0, 0), ny, nx, h
        nid = lambda i, j: i * (nx + 1) + j
        r, c = [a.ravel() for a in np.mgrid[0:ny, 0:nx]]
        n = np.stack([nid(r + 1, c), nid(r + 1, c + 1), nid(r, c + 1), nid(r, c)], 1)
        self.edof = np.stack([2 * n, 2 * n + 1], 2).reshape(-1, 8)
        self.ndof = 2 * (ny + 1) * (nx + 1); I = np.arange(ny + 1); J = np.arange(nx + 1)
        fixed = np.unique(np.r_[2 * nid(ny, J) + 1, 2 * nid(I, 0), 2 * nid(I, nx)])  # bottom uy, sides ux
        self.free = np.setdiff1d(np.arange(self.ndof), fixed); self.top_y = 2 * nid(0, J) + 1
        self.rows = np.repeat(self.edof, 8, 1).ravel(); self.cols = np.tile(self.edof, (1, 8)).ravel()

    def factor(self, E):
        K = sp.coo_matrix((np.outer(E, self.K0.ravel()).ravel(), (self.rows, self.cols)), shape=(self.ndof,) * 2).tocsr()
        self.lu = splu(K[self.free][:, self.free].tocsc()); self.E = E

    def solve(self, eps):
        F = np.zeros(self.ndof); np.add.at(F, self.edof.ravel(), ((self.E * eps)[:, None] * self.f0).ravel())
        u = np.zeros(self.ndof); u[self.free] = self.lu.solve(F[self.free])
        strain = u[self.edof] @ self.Bc.T
        sig = self.E[:, None] * ((strain - eps[:, None] * [1, 1, 0]) @ self.D.T)
        sx, sy, txy = sig.T
        s1 = (sx + sy) / 2 + np.hypot((sx - sy) / 2, txy); ang = 0.5 * np.arctan2(2 * txy, sx - sy)
        return dict(thick=u[self.top_y].mean() / (self.ny * self.h), s1=s1, ang=ang, vol=strain[:, 0] + strain[:, 1], u=u)


class LiFD:
    """Constant-current Li in Si. Each step the current (sized so that all active Si would fill in T) is shared by
    the surface faces that are not yet full (charge) or empty (discharge); Li diffuses inside each particle
    (implicit Euler, no flux to other phases). Clipping at 0/1 acts as the voltage cut-off."""
    def __init__(self, act, T):
        H, W = act.shape; self.act = act; self.N = N = int(act.sum())
        if N == 0: return
        idx = -np.ones(act.shape, int); idx[act] = np.arange(N)
        pa = np.pad(act, 1); pi = np.pad(idx, 1, constant_values=-1); r, c, f = [], [], np.zeros(act.shape, int)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            na = pa[1 + di:1 + di + H, 1 + dj:1 + dj + W]; ni = pi[1 + di:1 + di + H, 1 + dj:1 + dj + W]
            m = act & na; r.append(idx[m]); c.append(ni[m]); f += act & ~na
        A = sp.coo_matrix((np.ones(sum(map(len, r))), (np.concatenate(r), np.concatenate(c))), shape=(N, N)).tocsr()
        Lp = A - sp.diags(np.asarray(A.sum(1)).ravel()); a = PAR["D_si"] * T / NSUB / PX ** 2
        self.lu = splu((sp.identity(N, format="csc") - a * Lp).tocsc()); self.f = f[act].astype(float)

    def run(self, c0, charge, steps):
        if self.N == 0: return {k: np.zeros(self.act.shape) for k in steps}
        c = c0[self.act].astype(float); q = self.N / NSUB * (1 if charge else -1); out = {}
        for k in range(1, max(steps) + 1):
            op = (self.f > 0) & ((c < 1) if charge else (c > 0)); F = self.f[op].sum()
            src = np.zeros(self.N)
            if F > 0: src[op] = self.f[op] / F * q
            c = np.clip(self.lu.solve(c + src), 0, 1)
            if k in steps: o = np.zeros(self.act.shape); o[self.act] = c; out[k] = o
        return out


def blocks(a):
    H, W = a.shape; return a.reshape(H // MB, MB, W // MB, MB).mean((1, 3)).ravel()


def up(a, shape):
    return np.kron(a.reshape(shape[0] // MB, shape[1] // MB), np.ones((MB, MB)))


# ---------------- loading ----------------
def load_labels(spec, nm_per_px=25.0):
    if os.path.exists(spec):                       # any BSE image
        import imagecodecs, tifffile  # noqa: F401
        im = tifffile.imread(spec); im = im[..., 0] if im.ndim == 3 else im
        im = (im / im.max() * 255).astype(np.float32) if im.dtype != np.uint8 else im.astype(np.float32)
        f = 50.0 / nm_per_px                       # segment at 50 nm/px like the QC pipeline
        if abs(f - round(f)) < 1e-6 and f >= 1:
            k = int(round(f)); H, W = im.shape[0] // k * k, im.shape[1] // k * k
            im = im[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3))
        else:
            from skimage.transform import rescale; im = rescale(im, 1 / f, preserve_range=True)
        lab, _ = segment(np.clip(im, 0, 255).astype(np.uint8)); name = os.path.splitext(os.path.basename(spec))[0]
    else:
        from pmdb.io import load_site
        b, s = spec.split("/"); lab, _ = segment(load_site(b, s, resolution="half", normalise="none").image[..., 0]); name = s
    return lab[::2, ::2].copy(), name               # 50 nm -> 0.1 µm


def crop_box(shape, width_um, x0_um=None):
    H = shape[0] // MB * MB; w = min(int(width_um / PX) // MB * MB, shape[1] // MB * MB)
    return H, ((shape[1] - w) // 2 if x0_um is None else int(x0_um / PX)), w


def crop(lab, width_um, x0_um=None):
    H, x0, w = crop_box(lab.shape, width_um, x0_um); return lab[:H, x0:x0 + w].copy()


def crack_mask(pr, pc, th, L, rng, shape, step=3):
    """Slightly meandering crack through (pr, pc) along angle th (image x right, y up), 0.2 µm wide."""
    H, W = shape; m = np.zeros(shape, bool)
    for a0 in (th, th + np.pi):
        y, x, dev = float(pr), float(pc), 0.0
        for _ in range(int(L / step)):
            dev = 0.8 * dev + rng.normal(0, 0.2); a = a0 + dev
            y1, x1 = y - step * np.sin(a), x + step * np.cos(a)
            rr, cc = draw.line(int(round(y)), int(round(x)), int(round(y1)), int(round(x1)))
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W); m[rr[ok], cc[ok]] = True; y, x = y1, x1
    return ndi.binary_dilation(m, np.ones((2, 2), bool))


# ---------------- simulation ----------------
def kpis_of(lab):
    L2 = np.where(lab == 0, 0, np.where(np.isin(lab, (2, 4)), 2, 1)).astype(np.uint8)
    kp, _ = particles(L2, PX); area = lab.size * PX * PX
    comp = measure.label(np.isin(lab, (2, 4))); n_all = comp.max()
    return dict(porosity=(lab == 0).mean(), si_frac=np.isin(lab, (2, 4)).mean(), si_active_frac=(lab == 2).mean(),
                sei_frac=(lab == 3).mean(), si_d50_um=kp["si_d50_um"], si_d90_um=kp["si_d90_um"],
                si_count_per_1000um2=kp["si_count_per_1000um2"], si_solidity=kp["si_solidity"],
                si_circularity=kp["si_circularity"], si_cracked_frac=kp["si_cracked_frac"], si_clark_evans=kp["si_clark_evans"],
                si_fragments_per_1000um2=(n_all - kp["si_count_per_1000um2"] * area / 1000) / area * 1000)


def half_cycle(mesh, lab, li, c0, T, charge, rec=None, n=0, maps=None, n_render=0, frames=None, mid=None):
    """Li in Si at the end, elementwise max principal stress (over DMG steps) and its angle, and the last FE result."""
    act = lab == 2; gr = lab == 1
    rnd = set(range(NSUB // n_render, NSUB + 1, NSUB // n_render)) if n_render else set()
    steps = sorted(set(DMG) | rnd | ({NSUB // 2} if mid is not None else set()))
    if PAR["li"] == "erfc":
        d_si = np.maximum(ndi.distance_transform_edt(act) * PX - PX / 2, 0); cs = {}
        for k in steps:
            z = d_si / (2 * np.sqrt(PAR["D_si"] * T * k / NSUB)); cs[k] = np.where(act, erfc(z) if charge else c0 * erf(z), 0.0)
    else:
        cs = li.run(c0, charge, set(steps))
    s1max = np.full(mesh.E.shape, -np.inf); amax = np.zeros_like(s1max); por = blocks((lab == 0).astype(float)) > 0.5
    sim = blocks(act.astype(float)) > 0.5
    for k in steps:
        t = T * k / NSUB; c = cs[k]; cg = t / T if charge else 1 - t / T
        if mid is not None and k == NSUB // 2: mid["c"] = np.where(act, c, np.nan)
        if k not in DMG and k not in rnd: continue
        eps = blocks(np.where(act, PAR["eps_si"] * c, 0) + np.where(gr, PAR["eps_gr"] * cg, 0))
        r = mesh.solve(eps)
        if k in DMG:
            m = r["s1"] > s1max; s1max[m] = r["s1"][m]; amax[m] = r["ang"][m]
            if maps is not None: maps.append((n, int(charge), t, c.astype(np.float16), r["s1"].astype(np.float16)))
        if rec is not None:
            rec.append(dict(cycle=n, phase="charge" if charge else "discharge", t_min=t / 60 + (0 if charge else T / 60),
                            thickness_pct=100 * r["thick"], si_li=c[act].mean() if act.any() else np.nan,
                            sigma1_p95_si=np.percentile(r["s1"][sim], 95) if sim.any() else np.nan,
                            pore_closure_pct=-100 * (r["vol"][por].mean() if por.any() else 0)))
        if frames is not None and k in rnd:
            frames.append((n, int(charge), t, c.astype(np.float16), r["u"].astype(np.float16), r["vol"].astype(np.float16)))
    return c, s1max, amax, r


def simulate(lab0, cycles=50, crate=1.0, seed=0, snap=True, render=None):
    """render: {cycle: frames per half-cycle} -> per-step Li/displacement frames + end-of-discharge state every cycle."""
    rng = np.random.default_rng(seed); lab = lab0.copy(); H, W = lab.shape; area = lab.size
    mesh = Mesh(H // MB, W // MB, PX * MB, PAR["nu"]); T = 3600 / crate
    born = np.full(lab.shape, -1, int); rows, within, snaps, maps = [], [], [(0, lab.copy())], []
    li_lost = sei_acc = 0.0; inv0 = None; n_cracks = 0; crack_px = 0; extra = {}; c_prev = np.zeros(lab.shape)
    frames, tl = ([], []) if render is not None else (None, None)
    for n in range(1, cycles + 1):
        mesh.factor(blocks(np.vectorize(PAR["E"].get)(lab).astype(float)))
        act = lab == 2; li = LiFD(act, T) if PAR["li"] == "cc" else None; nr = (render or {}).get(n, 0)
        rec = within if n in (1, cycles) or nr else None; mp = maps if (snap and n in (1, cycles)) else None
        mid = {} if n == 1 else None
        c_end, s1c, ac, full = half_cycle(mesh, lab, li, np.where(act, c_prev, 0), T, True, rec, n, mp, nr, frames, mid)
        c_dis, s1d, ad, rd = half_cycle(mesh, lab, li, c_end, T, False, rec, n, mp, nr, frames)
        c_prev = c_dis
        if tl is not None: tl.append((n, c_dis.astype(np.float16), rd["u"].astype(np.float16), rd["vol"].astype(np.float16)))
        s1 = np.maximum(s1c, s1d); ang = np.where(s1d >= s1c, ad, ac)
        rev = ((c_end - c_dis)[act].sum() * PAR["Q_si"] + (lab == 1).sum() * PAR["Q_gr"]) / area
        if n == 1:
            extra = dict(li_mid=mid["c"], s1_map=up(s1, lab.shape), lab1=lab.copy())
        # SEI on all active Si surfaces (electrolyte also reaches Si through the binder/carbon); it locks Li
        # and fills adjacent pores. Crack faces are new surfaces with fresh, fast-growing SEI.
        bnd = act & ndi.binary_dilation(lab != 2, S4)
        born[bnd & (born < 0)] = n - 1
        dlt = np.where(bnd, PAR["k_sei"] * np.sqrt(np.maximum(n - born, 0)), 0)
        inc = (dlt - np.where(bnd, PAR["k_sei"] * np.sqrt(np.maximum(n - 1 - born, 0)), 0)).sum() * PX
        sei_acc += inc; li_lost += inc / (area * PX * PX) * PAR["q_sei"]
        if bnd.any():
            dist, (ii, jj) = ndi.distance_transform_edt(~bnd, return_indices=True)
            lab[(lab == 0) & (dist * PX - PX / 2 <= dlt[ii, jj])] = 3
        inv0 = rev if inv0 is None else inv0
        cap = min(rev, inv0 - li_lost)
        # cracking: Weibull on each particle's peak tensile stress, crack plane normal to sigma1
        L = measure.label(lab == 2); ids = np.arange(1, L.max() + 1)
        if len(ids):
            S = up(s1, lab.shape); A = up(ang, lab.shape)
            smax = ndi.maximum(S, L, ids); pos = ndi.maximum_position(S, L, ids)
            P = 1 - np.exp(-(np.clip(smax, 0, None) / PAR["sigma0"]) ** PAR["m"])
            for k in np.where(rng.random(len(ids)) < P)[0]:
                (pr, pc), th = pos[k], A[pos[k]] + np.pi / 2; Lk = int(np.ceil(np.hypot(*lab.shape) / 4))
                m = crack_mask(pr, pc, th, min(Lk, 80), rng, lab.shape) & (L == ids[k])
                lab[m] = 0; n_cracks += 1; crack_px += m.sum()
        # inactive: tiny fragments or no contact with the graphite/binder matrix
        L = measure.label(lab == 2); ids = np.arange(1, L.max() + 1)
        if len(ids):
            sz = ndi.sum(np.ones_like(L), L, ids) * PX * PX
            touch = ndi.maximum(ndi.binary_dilation(lab == 1, S4).astype(int), L, ids) > 0
            dead = ids[(sz < PAR["a_min"]) | ~touch]; lab[np.isin(L, dead)] = 4
        sim = blocks((lab == 2).astype(float)) > 0.5
        rows.append(dict(cycle=n, capacity_mAh_cm3=cap, reversible_mAh_cm3=rev, li_lost_sei_mAh_cm3=li_lost,
                         si_utilisation=float((c_end - c_dis)[act].mean()) if act.any() else np.nan,
                         thickness_charged_pct=100 * full["thick"], thickness_discharged_pct=100 * rd["thick"],
                         thickness_irrev_pct=100 * crack_px / W / H,
                         sigma1_p95_si=float(np.percentile(s1[sim], 95)) if sim.any() else np.nan,
                         n_cracks=n_cracks, **kpis_of(lab)))
        if snap: snaps.append((n, lab.copy()))
    t = pd.DataFrame(rows); t["retention_pct"] = 100 * t.capacity_mAh_cm3 / t.capacity_mAh_cm3.iloc[0]
    k0 = kpis_of(lab0); t0 = {**{k: np.nan for k in t.columns}, **k0, "cycle": 0}
    t = pd.concat([pd.DataFrame([t0]), t], ignore_index=True)
    return dict(traj=t, within=pd.DataFrame(within), snaps=snaps, maps=maps, frames=frames, tl=tl, **extra)


def _job(a):
    spec, seed, cycles, crate, width, x0, nm, render = a
    lab, name = load_labels(spec, nm); box = crop_box(lab.shape, width, x0); lab = crop(lab, width, x0)
    rd = {1: 24, cycles: 24, **{max(2, round(cycles * f)): 12 for f in (0.25, 0.5, 0.75)}} if render and seed == 0 else None
    r = simulate(lab, cycles, crate, seed, snap=seed == 0, render=rd); r["traj"]["seed"] = seed; r["box"] = box
    return spec, name, seed, r


# ---------------- outputs ----------------
TRAJ = [("retention_pct", "Capacity retention (%)"), ("thickness_charged_pct", "Thickness swelling when charged (%)"),
        ("thickness_irrev_pct", "Irreversible thickening from cracks (%)"), ("porosity", "Porosity"),
        ("sei_frac", "SEI area fraction"), ("si_active_frac", "Active Si area fraction"), ("si_d50_um", "Si d50 (µm)"),
        ("si_d90_um", "Si d90 (µm)"), ("si_solidity", "Si solidity"), ("si_fragments_per_1000um2", "Si fragments <1 µm per 1000 µm²"),
        ("n_cracks", "Cracks (cumulative)"), ("sigma1_p95_si", "Peak tensile stress in Si, p95 (GPa, elastic)")]


def rgb(lab): return CMAP[lab]


def site_figure(name, runs, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r0 = runs[0]; T = pd.concat([r["traj"] for r in runs]); last = r0["snaps"][-1]
    fig = plt.figure(figsize=(20, 16)); gs = fig.add_gridspec(4, 4)
    ext = [0, r0["lab1"].shape[1] * PX, r0["lab1"].shape[0] * PX, 0]
    for j, (im, title, kw) in enumerate([
            (rgb(r0["snaps"][0][1]), "Start (blue pore, grey graphite/binder, orange Si)", {}),
            (rgb(last[1]), f"After cycle {last[0]} (green SEI, red inactive Si, blue lines cracks)", {}),
            (r0["li_mid"], "Cycle 1, half-way through charge: Li in Si (0-1)", dict(cmap="viridis", vmin=0, vmax=1)),
            (np.where(np.isin(r0["lab1"], (2,)), r0["s1_map"], np.nan), "Cycle 1: peak tensile stress in Si (GPa, elastic)", dict(cmap="inferno"))]):
        a = fig.add_subplot(gs[0, j]); h = a.imshow(im, extent=ext, interpolation="nearest", **kw); a.set_title(title, fontsize=9); a.set_xlabel("µm")
        if kw: plt.colorbar(h, ax=a, fraction=.04)
    w = r0["within"]
    for j, (k, lab) in enumerate([("thickness_pct", "Thickness change (%)"), ("si_li", "Li in Si (mean, 0-1)"),
                                  ("sigma1_p95_si", "Tensile stress in Si, p95 (GPa)"), ("pore_closure_pct", "Pore closure (%)")]):
        a = fig.add_subplot(gs[1, j])
        for cyc, g in w.groupby("cycle"): a.plot(g.t_min, g[k], "o-", label=f"cycle {cyc}")
        a.axvline(g.t_min.max() / 2, color="grey", ls=":"); a.set_xlabel("time (min): charge | discharge"); a.set_title(lab, fontsize=9); a.grid(alpha=.3); a.legend(fontsize=8)
    for j, (k, lab) in enumerate(TRAJ[:8]):
        a = fig.add_subplot(gs[2 + j // 4, j % 4]); g = T.groupby("cycle")[k]
        a.plot(g.mean().index, g.mean(), color="C3"); a.fill_between(g.mean().index, g.min(), g.max(), color="C3", alpha=.25)
        a.set_title(lab, fontsize=9); a.set_xlabel("cycle"); a.grid(alpha=.3)
    fig.suptitle(f"{name}: simulated evolution (illustrative; band = range over {len(runs)} random seeds)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/fig_sim_{name}.png", dpi=65); plt.close(fig)
    from PIL import Image
    fr = [Image.fromarray(rgb(l)[::2, ::2]) for _, l in r0["snaps"]]
    fr[0].save(f"{out}/anim_{name}.gif", save_all=True, append_images=fr[1:], duration=150, loop=0)


def compare_figure(res, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 4, figsize=(20, 11))
    for a, (k, lab) in zip(ax.ravel(), TRAJ):
        for i, (name, runs) in enumerate(res.items()):
            g = pd.concat([r["traj"] for r in runs]).groupby("cycle")[k]
            a.plot(g.mean().index, g.mean(), color=f"C{i}", label=name); a.fill_between(g.mean().index, g.min(), g.max(), color=f"C{i}", alpha=.2)
        a.set_title(lab, fontsize=9); a.set_xlabel("cycle"); a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Simulated KPI trajectories per site (mean and range over seeds). Illustrative model, literature parameters.", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/fig_sim_compare.png", dpi=65); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="Batch/site or path to a BSE .tif (default: 2 flagged + 2 typical sites)")
    ap.add_argument("--image", action="append", default=[], help="BSE .tif (same as a path in inputs)")
    ap.add_argument("--nm-per-px", type=float, default=25.0, help="pixel size of --image files")
    ap.add_argument("--cycles", type=int, default=50); ap.add_argument("--crate", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--width", type=float, default=58.0, help="crop width, µm")
    ap.add_argument("--x0", type=float, default=None, help="crop start, µm (default: centre)")
    ap.add_argument("--out", default="sim"); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--li", choices=("cc", "erfc"), default="cc", help="Li model (constant current, or older instant surface)")
    ap.add_argument("--render", action="store_true", help="also save render_<name>.npz for evolve_video.py")
    a = ap.parse_args(); PAR["li"] = a.li; specs = (a.inputs + a.image) or ["Batch_1/5n1q8atc", "Batch_1/4ih2ggld", "Batch_2/epqdaau9", "Batch_3/x77cy643"]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(s, k, a.cycles, a.crate, a.width, a.x0, a.nm_per_px, a.render) for s in specs for k in range(a.seeds)]
    with Pool(min(a.jobs, len(jobs))) as p: done = p.map(_job, jobs)
    res = {}
    for spec, name, seed, r in done: res.setdefault(name, []).append(r)
    summ = {}
    for name, runs in res.items():
        runs.sort(key=lambda r: r["traj"].seed.iloc[0])
        pd.concat([r["traj"] for r in runs]).to_csv(f"{a.out}/traj_{name}.csv", index=False)
        runs[0]["within"].to_csv(f"{a.out}/within_{name}.csv", index=False)
        site_figure(name, runs, a.out)
        r0 = runs[0]; mp = r0["maps"]   # data for viewer.py: label image per cycle, Li/stress per time step
        np.savez_compressed(f"{a.out}/frames_{name}.npz", labs=np.stack([l for _, l in r0["snaps"]]),
                            m_cycle=[m[0] for m in mp], m_charge=[m[1] for m in mp], m_t=[m[2] for m in mp],
                            li=np.stack([m[3] for m in mp]), s1=np.stack([m[4] for m in mp]), px=PX, mb=MB)
        if r0["frames"]:
            fr, tl = r0["frames"], r0["tl"]; H, x0, w = r0["box"]
            src = next(s for s in specs if (os.path.splitext(os.path.basename(s))[0] if os.path.exists(s) else s.split("/")[-1]) == name)
            np.savez_compressed(f"{a.out}/render_{name}.npz", r_cycle=[f[0] for f in fr], r_charge=[f[1] for f in fr],
                                r_t=[f[2] for f in fr], r_c=np.stack([f[3] for f in fr]), r_u=np.stack([f[4] for f in fr]),
                                r_vol=np.stack([f[5] for f in fr]), tl_cycle=[f[0] for f in tl], tl_c=np.stack([f[1] for f in tl]),
                                tl_u=np.stack([f[2] for f in tl]), tl_vol=np.stack([f[3] for f in tl]), x0=x0, H=H, W=w,
                                px=PX, mb=MB, crate=a.crate, src=src, nm_per_px=a.nm_per_px)
        L = pd.concat([r["traj"] for r in runs]); end = L[L.cycle == a.cycles]; c1 = L[L.cycle == 1]
        summ[name] = {k: dict(cycle1=float(c1[k].mean()), end=float(end[k].mean()), end_min=float(end[k].min()), end_max=float(end[k].max()))
                      for k, _ in TRAJ + [("capacity_mAh_cm3", ""), ("si_utilisation", ""), ("si_frac", "")]}
        print(name, "  ".join(f"{k}: {v['cycle1']:.3g} -> {v['end']:.3g}" for k, v in summ[name].items()))
    compare_figure(res, a.out)
    json.dump(dict(params={k: (v if not isinstance(v, dict) else {str(i): e for i, e in v.items()}) for k, v in PAR.items()},
                   cycles=a.cycles, crate=a.crate, seeds=a.seeds, width_um=a.width, summary=summ), open(f"{a.out}/sim.json", "w"), indent=1)
