# rrt_space_cbf_planner.py
import numpy as np
import matplotlib.pyplot as plt

from trajectory import Trajectory
from rrt import RRT
import os
from concurrent.futures import ProcessPoolExecutor


# -----------------------------
# Utilities
# -----------------------------
class NullEnv:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

def inside_mask_xy(pos_xy, xmin, ymin, h, Nx, Ny, mask):
    ix = int(round((pos_xy[0] - xmin) / h))
    iy = int(round((pos_xy[1] - ymin) / h))
    if ix < 0 or ix >= Ny or iy < 0 or iy >= Nx:
        return True  # outside bounds treated as invalid
    return bool(mask[iy, ix])

def dist_to_charge_set_circle(pos_xy, c, rC) -> float:
    d = float(np.linalg.norm(pos_xy - c))
    return max(0.0, d - rC)

# -----------------------------
# Interpolator for point robot
# -----------------------------
def point2d_interpolator(s0, s1, *, u_max: float, segment_len: float, dt:float):
    """
    Return a Trajectory object (q, qd, qdd, duration) as expected by rrt.py.
    - state is [x,y]
    - q(t) is position
    - qd(t) is constant control u
    - qdd(t) is zero
    """
    s0 = np.asarray(s0, float).ravel()
    s1 = np.asarray(s1, float).ravel()
    p0 = s0[:2]
    p1 = s1[:2]

    v = p1 - p0
    dist = float(np.linalg.norm(v))
    if dist < 1e-12:
        # dummy trajectory of zero duration
        return Trajectory(
            q=lambda t: p0.copy(),
            qd=lambda t: np.zeros(2),
            qdd=lambda t: np.zeros(2),
            duration=0.0
        )

    # only go up to segment_len (local extension)
    pT = p0 + (min(segment_len, dist) / dist) * v

    # constant u to go from p0 to pT in duration T
    # choose T so ||u|| <= u_max, with u = (pT-p0)/T
    dp = pT - p0

    u = dp / dt
    un = float(np.linalg.norm(u))
    if un > u_max:
        u = (u_max / un) * u

    def q(t):
        tt = min(max(t, 0.0), dt)
        return p0 + u * tt

    def qd(t):
        return u

    def qdd(t):
        return np.zeros(2)

    return Trajectory(q=q, qd=qd, qdd=qdd, duration=dt)


# =============================
# Shared Space-CBF (FD5) module
# =============================

_SPACE_CBF_CACHE = {}   # key -> SpaceCBF_FD5 instance


def _periodic_delta(coord, center, L):
    d = np.abs(coord - center)
    return np.minimum(d, L - d)


class SpaceCBF_FD5:
    """
    Shared fast Space-CBF evaluator using FD5 stencil.

    Constraint you requested (masked version):
      alpha_h*(epsilon - h^2*sum( (sum rho)^2 )) - 2*h^2*sum( (sum rho)*(sum rho_dot) ) >= 0

    where:
      sum rho = Σ_i rho_i
      sum rho_dot = Σ_i rho_i_dot
      rho_i_dot = -(gx_i*u_xi + gy_i*u_yi) + b_i
      b_i = lap_coeff * Lap5(rho_i)

    Notes:
      - Mask is shared, not per-robot.
      - Grid neighbor indices are shared, precomputed once.
    """
    def __init__(self, *, X, Y, Nx, Ny, L, h, sigma, lap_coeff, Mask_bool, epsilon, alpha_h):
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.L = float(L)
        self.h = float(h)
        self.sigma = float(sigma)
        self.lap_coeff = float(lap_coeff)
        self.epsilon = float(epsilon)
        self.alpha_h = float(alpha_h)

        self.Xf = X.ravel().astype(float)
        self.Yf = Y.ravel().astype(float)
        self.Mask_f = Mask_bool.ravel().astype(float)

        N = self.Nx * self.Ny
        inds = np.arange(N, dtype=np.int32).reshape(self.Nx, self.Ny)
        self.iR = np.roll(inds, -1, axis=1).ravel()
        self.iL = np.roll(inds,  1, axis=1).ravel()
        self.iU = np.roll(inds, -1, axis=0).ravel()
        self.iD = np.roll(inds,  1, axis=0).ravel()

    def _rho_all(self, positions_all):
        positions_all = np.asarray(positions_all, float)
        cx = positions_all[:, 0][:, None]   # (J,1)
        cy = positions_all[:, 1][:, None]

        Xr = self.Xf[None, :]
        Yr = self.Yf[None, :]

        dx = _periodic_delta(Xr, cx, self.L)
        dy = _periodic_delta(Yr, cy, self.L)
        return np.exp(-0.5 * (dx*dx + dy*dy) / (self.sigma*self.sigma))  # (J,N)

    def cbf_lhs(self, *, positions_all, u_all):
        """
        positions_all: (J,2)
        u_all: (J,2) controls for all robots (use zeros for robots you are not moving)
        returns scalar LHS
        """
        positions_all = np.asarray(positions_all, float)
        u_all = np.asarray(u_all, float)
        J = positions_all.shape[0]

        rhoC = self._rho_all(positions_all)  # (J,N)
        rhoR = rhoC[:, self.iR]
        rhoL = rhoC[:, self.iL]
        rhoU = rhoC[:, self.iU]
        rhoD = rhoC[:, self.iD]

        gx = (rhoR - rhoL) / (2.0 * self.h)
        gy = (rhoU - rhoD) / (2.0 * self.h)
        gx *= -1.0
        gy *= -1.0

        lap = (rhoR + rhoL + rhoU + rhoD - 4.0 * rhoC) / (self.h*self.h)
        b   = self.lap_coeff * lap

        sum_rho = np.sum(rhoC, axis=0)  # (N,)

        # sum rho_dot = sum b  - sum_i (gx_i*ux_i + gy_i*uy_i)
        sum_rho_dot = np.sum(b, axis=0)
        sum_rho_dot += np.sum(gx * u_all[:, 0:1], axis=0)
        sum_rho_dot += np.sum(gy * u_all[:, 1:2], axis=0)

        # masked signals
        s = sum_rho * self.Mask_f
        s_dot = sum_rho_dot * self.Mask_f

        hh = self.h * self.h
        lhs = (self.alpha_h * (self.epsilon - hh * float(np.sum(s*s)))
               - 2.0 * hh * float(np.sum(s * s_dot)))
        return lhs

    def feasible(self, *, positions_all, u_all):
        flag = self.cbf_lhs(positions_all=positions_all, u_all=u_all) >= 0.0
        return flag


def get_shared_space_cbf(
    *,
    X, Y, invariance_mask,
    L, sigma, lap_coeff, epsilon, alpha_h
):
    """
    Shared builder: returns the cached SpaceCBF_FD5 for this exact grid+mask config.
    No per-robot masks. Built once.
    """
    Nx, Ny = X.shape[0], X.shape[1]
    h = float(L) / float(Nx - 1)

    key = (
        int(Nx), int(Ny), float(L), float(h),
        float(sigma), float(lap_coeff), float(epsilon), float(alpha_h),
        id(X), id(Y), id(invariance_mask)
    )
    obj = _SPACE_CBF_CACHE.get(key, None)
    if obj is None:
        obj = SpaceCBF_FD5(
            X=X, Y=Y, Nx=Nx, Ny=Ny, L=L, h=h,
            sigma=sigma, lap_coeff=lap_coeff,
            Mask_bool=invariance_mask,
            epsilon=epsilon, alpha_h=alpha_h
        )
        _SPACE_CBF_CACHE[key] = obj
    return obj



# -----------------------------
# Robot wrapper for their RRT
# -----------------------------
class Robot2D_SpaceCBF:
    """
    Implements the interface rrt.py expects:
      - sample_state()
      - state_dist()
      - check_torque_limits(q, qd, qdd)
      - env context
    """

    def __init__(
        self,
        *,
        bounds,
        X, Y,
        invariance_mask,      # boolean mask for forbidden region (or "must avoid")
        invariance_mask_inflated, # inflated mask for faster computation
        charge_mask,          # boolean mask for charging set OR define via charge_center/radius
        u_max: float,
        use_charge_mask: bool = True,
        charge_center=None,
        charge_radius=None,
        space_cbf=None,
        robot_index,
        positions_all_fn=None,
        use_space_cbf: bool = True,
        sigma,
        lap_coeff,
        epsilon,
        alpha_h,
    ):
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        self.X = X
        self.Y = Y
        self.inv_mask = invariance_mask
        self.inv_mask_inflated = invariance_mask_inflated
        self.charge_mask = charge_mask
        # Precompute charge points once (shared mask)
        if self.charge_mask is not None:
            ys, xs = np.where(self.charge_mask)
            # X, Y are meshgrid; pick coordinates of True cells
            self.charge_pts = np.column_stack((self.X[ys, xs], self.Y[ys, xs])).astype(float)
        else:
            self.charge_pts = None
        self.u_max = float(u_max)
        self.env = NullEnv()

        self.use_charge_mask = use_charge_mask
        self.charge_center = None if charge_center is None else np.asarray(charge_center, float).reshape(2)
        self.charge_radius = float(charge_radius) if charge_radius is not None else None
        ys, xs = np.where(self.charge_mask)
        self.charge_pts = np.column_stack((self.X[ys, xs], self.Y[ys, xs])).astype(float)

        self.space_cbf = space_cbf
        self.robot_index = int(robot_index)
        self.positions_all_fn = positions_all_fn
        self.use_space_cbf = bool(use_space_cbf)
        self.robot_index = int(robot_index)
        self.positions_all_fn = positions_all_fn
        xmin, xmax, ymin, ymax = self.bounds
        self._xmin = xmin
        self._ymin = ymin
        self._h = float(self.X[0, 1] - self.X[0, 0])  # or L/(Nx-1)
        self._Nx, self._Ny = self.X.shape  # careful: X is (Ny,Nx) or (Nx,Ny) depending on your meshgrid

        self.space_cbf = None
        if self.use_space_cbf:
            self.space_cbf = get_shared_space_cbf(
                X=self.X, Y=self.Y,
                invariance_mask=self.inv_mask,
                L=(self.bounds[1] - self.bounds[0]),  # assumes square domain [-L/2,L/2]
                sigma=float(sigma),
                lap_coeff=float(lap_coeff),
                epsilon=float(epsilon),
                alpha_h=float(alpha_h),
            )

    def sample_state(self):
        xmin, xmax, ymin, ymax = self.bounds
        xy = np.array([np.random.uniform(xmin, xmax),
                       np.random.uniform(ymin, ymax)], float)
        return np.hstack([xy, np.zeros(2)])  # [x,y,vx,vy]

    def _dist_to_charge_pts(self, q):
        if self.charge_pts is None or self.charge_pts.size == 0:
            return float("inf")
        # vectorized distance to all charge points
        d = self.charge_pts - q[None, :]
        return float(np.min(np.sqrt(np.sum(d * d, axis=1))))

    def state_dist(self, s0, s1):
        s0 = np.asarray(s0, float).ravel()
        s1 = np.asarray(s1, float).ravel()
        p0 = s0[:2]
        p1 = s1[:2]

        # --- Goal-distance mode (only when comparing to the specific goal_state) ---
        # We detect goal by comparing p1 to charge_center (since we set goal_state that way).
        if self.charge_center is not None and np.allclose(p1, self.charge_center, atol=1e-12):
            # success if inside charge_mask
            if inside_mask_xy(p0, self._xmin, self._ymin, self._h, self._Nx, self._Ny, self.charge_mask):
                return 0.0
            # distance-to-charge-set (approx): distance to nearest "True" pixel center
            q = np.asarray(p0, float).reshape(2)

            if self.use_charge_mask:
                return self._dist_to_charge_pts(q)
            else:
                return float(np.linalg.norm(q - self.charge_center))

        # --- Normal metric mode for NN search ---
        # Use position-only Euclidean (simple, works well)
        return float(np.linalg.norm(p0 - p1))

    def check_torque_limits(self, q, qd, qdd):
        """
        rrt.py calls this along the candidate trajectory.
        We repurpose it as: "CBF feasibility check at this point".
        """
        p = np.asarray(q, float).reshape(2)
        u = np.asarray(qd, float).reshape(2)

        # bounds
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
            return False

        # limit on control magnitude
        if float(np.linalg.norm(u)) > self.u_max + 1e-9:
            return False

        # do not step into invariance forbidden inflated set (hard check)
        if inside_mask_xy(p, self._xmin, self._ymin, self._h, self._Nx, self._Ny, self.inv_mask_inflated):

            # Default Space-CBF check (shared + fast)
            if self.use_space_cbf and (self.space_cbf is not None) and (self.positions_all_fn is not None):
                pos_all = np.asarray(self.positions_all_fn(), float).copy()  # (J,2)
                pos_all[self.robot_index, :] = p  # <-- overwrite with current edge sample

                # Build u_all: only this robot uses u, others 0 (unless you later want something else)
                u_all = np.zeros((pos_all.shape[0], 2), dtype=float)
                u_all[self.robot_index, :] = u

                passflag = bool(self.space_cbf.feasible(positions_all=pos_all, u_all=u_all))
                #print("pos: ", pos_all, "u: ", u_all, "check: ", passflag)
                return passflag

        return True


# -----------------------------
# Planner callable
# -----------------------------
def plan_one_robot_rrt(
    *,
    init_xy,
    robot: Robot2D_SpaceCBF,
    u_max: float,
    segment_len: float,
    max_extend_steps: int,     # how many rrt.step(...) calls
    steer_to_goal_every: int,
    k: int,
    rng_seed: int = 0,
    dt: float,
):
    """
    Returns:
      - path_points: (N,2) array of waypoints (including init and final)
      - rrt_obj: the RRT instance (optional for debugging)
    """
    np.random.seed(rng_seed)

    def interpolator(s0, s1):
        return point2d_interpolator(s0, s1, u_max=u_max, segment_len=segment_len, dt=dt)

    rrt = RRT(
        robot,
        init_state=np.hstack([np.asarray(init_xy, float).reshape(2), np.zeros(2)]),
        goal_state = np.hstack([robot.charge_center, np.zeros(2)]),
        state_interpolator=interpolator,
        steer_to_goal_every=steer_to_goal_every,
        k=k,
    )

    # one "iteration" here means one call to rrt.step(sample)
    for _ in range(max_extend_steps):
        x_rand = robot.sample_state()
        rrt.step(x_rand)
        if rrt.solution is not None:
            break

    if rrt.solution is None:
        return None, rrt

    # rrt.solution is a list of Trajectory segments
    pts = [init_xy[:2]]
    for seg in rrt.solution:
        pts.append(seg.q(seg.duration))  # q is already (2,)
    path_points = np.vstack(pts)

    return path_points, rrt
# -----------------------------
# Path metrics + DV helpers
# -----------------------------
def path_length(path_points):
    if path_points is None or len(path_points) < 2:
        return 0.0
    dif = np.diff(path_points, axis=0)
    seg_len = np.linalg.norm(dif, axis=1)
    return float(np.sum(seg_len))


def first_step_direction(path_points, eps=1e-12):
    """
    Returns:
      dv_unit: (2,) or zeros if unavailable
      first_step_vec: (2,) raw displacement
    """
    if path_points is None or len(path_points) < 2:
        z = np.zeros(2, dtype=float)
        return z, z
    step = np.asarray(path_points[1] - path_points[0], float)
    n = float(np.linalg.norm(step))
    if n < eps:
        return np.zeros(2, dtype=float), step
    return step / n, step

def dv_from_path_lookahead(
    path_points,
    k_steps=5,
    beta=0,
    min_norm=1e-12
):
    """
    Weighted average of first k segment vectors.
    Returns unit DV and raw averaged vector.
    """
    if path_points is None:
        z = np.zeros(2, dtype=float)
        return z, z

    p = np.asarray(path_points, float)
    if p.shape[0] < 2:
        z = np.zeros(2, dtype=float)
        return z, z

    segs = p[1:] - p[:-1]               # (M,2)
    m = segs.shape[0]
    k = int(max(1, min(k_steps, m)))
    s = segs[:k]

    idx = np.arange(k, dtype=float)
    w = np.exp(-beta * idx)             # (k,)
    v = (w[:, None] * s).sum(axis=0)    # raw averaged vector

    n = float(np.linalg.norm(v))
    if n < min_norm:
        z = np.zeros(2, dtype=float)
        return z, v
    return v / n, v


def ema_unit_vector(prev_unit, new_unit, alpha=0.3, min_norm=1e-12):
    """
    Exponential moving average for direction vectors, then renormalize.
    """
    prev = np.asarray(prev_unit, float).reshape(2)
    new = np.asarray(new_unit, float).reshape(2)
    v = (1.0 - alpha) * prev + alpha * new
    n = float(np.linalg.norm(v))
    if n < min_norm:
        return np.zeros(2, dtype=float)
    return v / n


# -----------------------------
# NEW: multi-robot planner
# -----------------------------
def plan_all_robots_rrt(
    *,
    init_xy_all,                 # (J,2)
    bounds,
    X, Y,
    invariance_mask,
    invariance_mask_inflated,
    charge_mask,
    charge_center,
    u_max: float,
    dt: float,
    segment_len: float,
    max_extend_steps,
    steer_to_goal_every,
    k,
    base_rng_seed,
    use_charge_mask: bool = True,
    use_space_cbf: bool = True,
    sigma,
    lap_coeff,
    epsilon,
    alpha_h,
):
    """
    Plans independent paths for all robots (others treated static in CBF eval).

    Returns dict:
      {
        "paths": list of (Ni,2) arrays or None,
        "path_lengths": (J,) float array,
        "dv_first": (J,2) unit first-step directions,
        "first_step_vecs": (J,2) raw first-step vectors,
        "rrt_objects": list of rrt instances,
      }
    """
    init_xy_all = np.asarray(init_xy_all, float)
    if init_xy_all.ndim != 2 or init_xy_all.shape[1] != 2:
        raise ValueError("init_xy_all must have shape (J,2)")
    J = init_xy_all.shape[0]

    if segment_len is None:
        segment_len = float(u_max) * float(dt)

    # shared static positions for "other robots are static"
    pos_holder = {"p": init_xy_all.copy()}

    def positions_all_fn():
        return pos_holder["p"]

    # build shared CBF once (speedup)
    shared_space_cbf = None
    if use_space_cbf:
        shared_space_cbf = get_shared_space_cbf(
            X=X, Y=Y,
            invariance_mask=invariance_mask,
            L=(bounds[1] - bounds[0]),
            sigma=float(sigma),
            lap_coeff=float(lap_coeff),
            epsilon=float(epsilon),
            alpha_h=float(alpha_h),
        )

    paths = []
    rrt_objects = []
    path_lengths = np.zeros(J, dtype=float)
    dv_first = np.zeros((J, 2), dtype=float)
    first_step_vecs = np.zeros((J, 2), dtype=float)

    for i in range(J):
        robot_i = Robot2D_SpaceCBF(
            bounds=bounds,
            X=X, Y=Y,
            invariance_mask=invariance_mask,
            invariance_mask_inflated=invariance_mask_inflated,
            charge_mask=charge_mask,
            u_max=u_max,
            use_charge_mask=use_charge_mask,
            charge_center=np.asarray(charge_center, float).reshape(2),
            use_space_cbf=use_space_cbf,
            sigma=sigma,
            lap_coeff=lap_coeff,
            epsilon=epsilon,
            alpha_h=alpha_h,
            positions_all_fn=positions_all_fn,
            robot_index=i,
            space_cbf=shared_space_cbf,    # explicit shared object
        )

        path_i, rrt_i = plan_one_robot_rrt(
            init_xy=init_xy_all[i],
            robot=robot_i,
            u_max=u_max,
            segment_len=segment_len,
            max_extend_steps=max_extend_steps,
            steer_to_goal_every=steer_to_goal_every,
            k=k,
            rng_seed=int(base_rng_seed + i),  # deterministic but decorrelated
            dt=dt,
        )

        paths.append(path_i)
        rrt_objects.append(rrt_i)

        path_lengths[i] = path_length(path_i)
        #dv_i, step_i = first_step_direction(path_i)
        dv_i, step_i = dv_from_path_lookahead(path_i,k_steps=10,  beta=0.35)
        dv_first[i, :] = dv_i
        first_step_vecs[i, :] = step_i

    return {
        "paths": paths,
        "path_lengths": path_lengths,
        "dv_first": dv_first,
        "first_step_vecs": first_step_vecs,
        "rrt_objects": rrt_objects,
    }


# -----------------------------
# Plot helper
# -----------------------------
def plot_cbf_style_with_path(
    *,
    X, Y,
    invariance_mask=None,
    charge_mask=None,
    robot_pos=None,
    path_points=None,
    title="RRT path (CBF-feasible)",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect("equal", adjustable="box")

    if invariance_mask is not None:
        ax.contour(X, Y, invariance_mask.astype(float), levels=[0.5], linewidths=1.0)

    if charge_mask is not None:
        ax.contour(X, Y, charge_mask.astype(float), levels=[0.5], linewidths=2.0)

    if robot_pos is not None:
        ax.scatter([robot_pos[0]], [robot_pos[1]], s=80, marker="o", label="start")

    if path_points is not None:
        ax.plot(path_points[:, 0], path_points[:, 1], linewidth=2.5, label="RRT path")
        ax.scatter([path_points[-1, 0]], [path_points[-1, 1]], s=90, marker="*", label="end")

    ax.legend()
    plt.show()


def _plan_one_robot_worker(args):
    """
    Process-safe worker for one robot.
    args: tuple with all needed immutable data.
    """
    (
        i,
        init_xy_i,
        init_xy_all,
        bounds, X, Y,
        invariance_mask, invariance_mask_inflated,
        charge_mask, charge_center,
        u_max, dt, segment_len, max_extend_steps,
        steer_to_goal_every, k, base_rng_seed,
        use_charge_mask, use_space_cbf,
        sigma, lap_coeff, epsilon, alpha_h
    ) = args

    # frozen positions for this robot's planning context
    pos_holder = {"p": np.asarray(init_xy_all, float).copy()}

    def positions_all_fn():
        return pos_holder["p"]

    robot_i = Robot2D_SpaceCBF(
        bounds=bounds,
        X=X, Y=Y,
        invariance_mask=invariance_mask,
        invariance_mask_inflated=invariance_mask_inflated,
        charge_mask=charge_mask,
        u_max=u_max,
        use_charge_mask=use_charge_mask,
        charge_center=np.asarray(charge_center, float).reshape(2),
        use_space_cbf=use_space_cbf,
        sigma=sigma,
        lap_coeff=lap_coeff,
        epsilon=epsilon,
        alpha_h=alpha_h,
        positions_all_fn=positions_all_fn,
        robot_index=int(i),
        space_cbf=None,  # each process will use its own cache anyway
    )

    path_i, _ = plan_one_robot_rrt(
        init_xy=np.asarray(init_xy_i, float),
        robot=robot_i,
        u_max=u_max,
        segment_len=segment_len,
        max_extend_steps=max_extend_steps,
        steer_to_goal_every=steer_to_goal_every,
        k=k,
        rng_seed=int(base_rng_seed + i),
        dt=dt,
    )

    L_i = path_length(path_i)
    dv_i, step_i = dv_from_path_lookahead(path_i, k_steps=15, beta=0.0)

    return i, path_i, L_i, dv_i, step_i

def plan_all_robots_rrt_parallel(
    *,
    init_xy_all,                 # (J,2)
    bounds,
    X, Y,
    invariance_mask,
    invariance_mask_inflated,
    charge_mask,
    charge_center,
    u_max: float,
    dt: float,
    segment_len: float,
    max_extend_steps,
    steer_to_goal_every,
    k,
    base_rng_seed,
    use_charge_mask: bool = True,
    use_space_cbf: bool = True,
    sigma=None,
    lap_coeff=None,
    epsilon=None,
    alpha_h=None,
    parallel: bool = True,
    n_workers: int | None = None,
):
    init_xy_all = np.asarray(init_xy_all, float)
    if init_xy_all.ndim != 2 or init_xy_all.shape[1] != 2:
        raise ValueError("init_xy_all must have shape (J,2)")
    J = init_xy_all.shape[0]

    if segment_len is None:
        segment_len = float(u_max) * float(dt)

    paths = [None] * J
    rrt_objects = [None] * J  # kept for API compatibility
    path_lengths = np.zeros(J, dtype=float)
    dv_first = np.zeros((J, 2), dtype=float)
    first_step_vecs = np.zeros((J, 2), dtype=float)

    # Build job list
    jobs = []
    for i in range(J):
        jobs.append((
            i,
            init_xy_all[i].copy(),
            init_xy_all.copy(),
            bounds, X, Y,
            invariance_mask, invariance_mask_inflated,
            charge_mask, np.asarray(charge_center, float).reshape(2),
            float(u_max), float(dt), float(segment_len), int(max_extend_steps),
            int(steer_to_goal_every), int(k), int(base_rng_seed),
            bool(use_charge_mask), bool(use_space_cbf),
            float(sigma), float(lap_coeff), float(epsilon), float(alpha_h),
        ))

    do_parallel = bool(parallel) and (J > 1)
    if n_workers is None:
        n_workers = min(J, max(1, (os.cpu_count() or 2) - 1))

    if do_parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, p, L_i, dv_i, step_i in ex.map(_plan_one_robot_worker, jobs):
                paths[i] = p
                path_lengths[i] = L_i
                dv_first[i, :] = dv_i
                first_step_vecs[i, :] = step_i
    else:
        for jb in jobs:
            i, p, L_i, dv_i, step_i = _plan_one_robot_worker(jb)
            paths[i] = p
            path_lengths[i] = L_i
            dv_first[i, :] = dv_i
            first_step_vecs[i, :] = step_i

    return {
        "paths": paths,
        "path_lengths": path_lengths,
        "dv_first": dv_first,
        "first_step_vecs": first_step_vecs,
        "rrt_objects": rrt_objects,
    }

def plan_selected_robots_rrt_parallel(
    *,
    robot_indices,               # list[int] to plan
    init_xy_all,                 # full (J,2), used for shared CBF context
    bounds,
    X, Y,
    invariance_mask,
    invariance_mask_inflated,
    charge_mask,
    charge_center,
    u_max: float,
    dt: float,
    segment_len: float,
    max_extend_steps,
    steer_to_goal_every,
    k,
    base_rng_seed,
    use_charge_mask: bool = True,
    use_space_cbf: bool = True,
    sigma=None,
    lap_coeff=None,
    epsilon=None,
    alpha_h=None,
    parallel: bool = True,
    n_workers: int | None = None,
):
    init_xy_all = np.asarray(init_xy_all, float)
    robot_indices = list(map(int, robot_indices))
    J = init_xy_all.shape[0]

    out_paths = [None] * J
    out_L = np.full(J, np.inf, dtype=float)
    out_dv = np.zeros((J, 2), dtype=float)
    out_step = np.zeros((J, 2), dtype=float)

    if len(robot_indices) == 0:
        return {
            "paths": out_paths,
            "path_lengths": out_L,
            "dv_first": out_dv,
            "first_step_vecs": out_step,
            "rrt_objects": [None] * J,
        }

    if segment_len is None:
        segment_len = float(u_max) * float(dt)

    jobs = []
    for i in robot_indices:
        jobs.append((
            i,
            init_xy_all[i].copy(),
            init_xy_all.copy(),
            bounds, X, Y,
            invariance_mask, invariance_mask_inflated,
            charge_mask, np.asarray(charge_center, float).reshape(2),
            float(u_max), float(dt), float(segment_len), int(max_extend_steps),
            int(steer_to_goal_every), int(k), int(base_rng_seed),
            bool(use_charge_mask), bool(use_space_cbf),
            float(sigma), float(lap_coeff), float(epsilon), float(alpha_h),
        ))

    do_parallel = bool(parallel) and (len(jobs) > 1)
    if n_workers is None:
        n_workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1))

    if do_parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, p, L_i, dv_i, step_i in ex.map(_plan_one_robot_worker, jobs):
                out_paths[i] = p
                out_L[i] = L_i
                out_dv[i, :] = dv_i
                out_step[i, :] = step_i
    else:
        for jb in jobs:
            i, p, L_i, dv_i, step_i = _plan_one_robot_worker(jb)
            out_paths[i] = p
            out_L[i] = L_i
            out_dv[i, :] = dv_i
            out_step[i, :] = step_i

    return {
        "paths": out_paths,
        "path_lengths": out_L,
        "dv_first": out_dv,
        "first_step_vecs": out_step,
        "rrt_objects": [None] * J,
    }


def plot_all_robots_paths(
    *,
    X, Y,
    invariance_mask=None,
    charge_mask=None,
    init_xy_all=None,        # (J,2)
    paths=None,              # list of (Ni,2) or None
    dv_first=None,           # (J,2) optional
    title="All robot paths (independent RRT)",
    show_legend=True,
):
    """
    Visualize all robots and their planned paths on one figure.
    - start points: circles
    - end points: stars
    - optional first-step DV arrows
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect("equal", adjustable="box")

    if invariance_mask is not None:
        ax.contour(X, Y, invariance_mask.astype(float), levels=[0.5], linewidths=1.2)

    if charge_mask is not None:
        ax.contour(X, Y, charge_mask.astype(float), levels=[0.5], linewidths=2.0)

    if init_xy_all is None:
        init_xy_all = np.zeros((0, 2), dtype=float)
    else:
        init_xy_all = np.asarray(init_xy_all, float)

    if paths is None:
        paths = []

    J = max(len(paths), len(init_xy_all))

    # default cycle
    cmap = plt.get_cmap("tab10")

    for i in range(J):
        c = cmap(i % 10)

        # start
        if i < len(init_xy_all):
            s = init_xy_all[i]
            ax.scatter([s[0]], [s[1]], s=70, marker="o", color=c,
                       label=f"robot {i} start" if show_legend else None)

        # path + end
        if i < len(paths) and (paths[i] is not None) and (len(paths[i]) > 0):
            p = np.asarray(paths[i], float)
            ax.plot(p[:, 0], p[:, 1], linewidth=2.0, color=c,
                    label=f"robot {i} path" if show_legend else None)
            ax.scatter([p[-1, 0]], [p[-1, 1]], s=90, marker="*", color=c,
                       label=f"robot {i} end" if show_legend else None)

        # DV arrow from start
        if dv_first is not None and i < len(init_xy_all) and i < len(dv_first):
            dvi = np.asarray(dv_first[i], float)
            n = np.linalg.norm(dvi)
            if n > 1e-12:
                s = init_xy_all[i]
                # fixed arrow length for readability
                Larr = 0.20 * (X.max() - X.min())
                v = (dvi / n) * Larr * 0.08
                ax.arrow(
                    s[0], s[1], v[0], v[1],
                    head_width=0.035 * (X.max() - X.min()),
                    head_length=0.05 * (X.max() - X.min()),
                    fc=c, ec=c, length_includes_head=True, alpha=0.9
                )

    if show_legend:
        # avoid giant duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        h2, l2 = [], []
        for h, l in zip(handles, labels):
            if l not in seen and l is not None:
                seen.add(l)
                h2.append(h); l2.append(l)
        ax.legend(h2, l2, loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()
