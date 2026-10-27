import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mosek.fusion as mf
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import time
import numpy as np
from rrt_path_cbf import plan_all_robots_rrt_parallel
from scipy.ndimage import binary_dilation, gaussian_filter, distance_transform_edt
from concurrent.futures import ProcessPoolExecutor
import os

total_time = 0.0
total_time_count = 0

# ----------------------- Parameters -----------------------
L = 3.5
Nx, Ny = 36,36
Ngrid = Nx * Ny
h = L / (Nx - 1)
dt = 0.05

alpha_v = 1e3
lap_coeff = 0.045
gemma = 1e10
epsilon = 0.001
alpha_h = 0.1
sigma_energy = 0.14
SPRAY_REG = 0.1

MEASURE_NOISE = True
MOTION_NOISE = True

#E_INIT = np.array([0.25, 0.9, 0.7, 0.5], dtype=float)
#E_INIT = np.array([0.25, 0.9, 0.7, 0.5, 0.6,0.5,0.9,0.8,1.0,0.85], dtype=float)
#E_INIT = np.array([0.25, 0.9, 0.7, 0.5, 0.6,0.7], dtype=float)
#E_INIT = np.array([0.25, 0.9, 0.7, 0.5, 0.6,0.5,0.9,0.8], dtype=float)
E_INIT = np.array([0.25, 0.9], dtype=float)
#E_INIT = np.array([0.25, 0.9,0.7, 0.5, 0.4], dtype=float)

E = E_INIT.copy()

# ----------------------- Feature Switches -----------------------
USE_ENERGY_PLANNING = True  # master switch for RRT-based energy planning / charger behavior
USE_BATTERY_CBF = True       # battery CBF in controller; only active if USE_ENERGY_PLANNING is True
SHOW_RRT_PATHS = True        # animation overlay for RRT paths; only active if USE_ENERGY_PLANNING is True
# ----------------------- Async RRT replanning -----------------------
RRT_REPLAN_EVERY_FRAMES = 5    # submit a new RRT plan every this many frames
RRT_ASYNC_ENABLED = True         # run planner in background so animation loop does not wait
RRT_ASYNC_WORKERS = 1            # one persistent background process
# ----------------------- Grid Setup -----------------------
x_grid = np.linspace(-L / 2, L / 2, Nx)
y_grid = np.linspace(-L / 2, L / 2, Ny)
X, Y = np.meshgrid(x_grid, y_grid)

env = np.array([
    [-2.0, -2.0],
    [ 2.0, -2.0],
    [ 2.0,  2.0],
    [-2.0,  2.0],
    [-2.0, -2.0],
], float)

# ----------------------- Swarm target initialization -----------------------
# Charger region
charger_x, charger_y = -1.2, -1.2
charger_radius = 0.5
Mask_G = (((X - charger_x) ** 2 + (Y - charger_y) ** 2) <= charger_radius ** 2)



# ----------------------- Load Initial Data -----------------------
#with open("initial_robots.json", "r") as file:
with open("initial_particles2.json", "r") as file:
    particle_data = json.load(file)


class PDF:
    def __init__(self, position, sigma):
        self.position = np.array(position)
        self.sigma = sigma
        self.field = self.compute_pdf(X, Y)
        self.fake_position = []

    def compute_pdf(self, x, y):
        px, py = self.position
        dx = np.abs(x - px)
        dy = np.abs(y - py)
        dx = np.minimum(dx, L - dx)
        dy = np.minimum(dy, L - dy)
        return np.exp(-((dx) ** 2 + (dy) ** 2) / (2 * self.sigma ** 2))

    def update(self, u_opt_i, dt, x_grid, y_grid, charging_flag):
        local_u = np.asarray(u_opt_i, dtype=float)
        if MOTION_NOISE and (not charging_flag):
            unit_noise = np.random.multivariate_normal([0, 0], [[0.3 ** 2, 0], [0, 0.3 ** 2]])
            while np.linalg.norm(unit_noise) >= 1.0:
                unit_noise = np.random.multivariate_normal([0, 0], [[0.3 ** 2, 0], [0, 0.3 ** 2]])
            px, py = unit_noise * np.linalg.norm(local_u) * 0.5
            self.position = self.position + dt * (local_u + np.array([px, py]))
        else:
            self.position = self.position + dt * local_u

        self.position[0] = np.clip(self.position[0], x_grid[0], x_grid[-1])
        self.position[1] = np.clip(self.position[1], y_grid[0], y_grid[-1])
        self.field = self.compute_pdf(X, Y)
        return local_u

    def compute_pdf_random(self, x=X, y=Y):
        n_samples = 50
        if MEASURE_NOISE:
            cov = np.array([[self.sigma ** 2, 0.0], [0.0, self.sigma ** 2]], dtype=float)
            samples = np.random.multivariate_normal(self.position, cov, size=n_samples)
            samples[:, 0] = np.clip(samples[:, 0], x_grid[0], x_grid[-1])
            samples[:, 1] = np.clip(samples[:, 1], y_grid[0], y_grid[-1])
            px, py = samples.mean(axis=0)
        else:
            px, py = self.position
            px = np.clip(px, x_grid[0], x_grid[-1])
            py = np.clip(py, y_grid[0], y_grid[-1])

        self.fake_position = [px, py]
        dx = np.abs(x - px)
        dy = np.abs(y - py)
        dx = np.minimum(dx, L - dx)
        dy = np.minimum(dy, L - dy)
        return np.exp(-((dx) ** 2 + (dy) ** 2) / (2 * self.sigma ** 2)), self.fake_position


pdfs = [PDF(p["position"], sigma_energy) for p in particle_data]
num_pdfs = len(pdfs)

# ----------------------- Fire Simulation Parameters -----------------------
eta_fire = 0.001
A_fire = 0.3
C_cooling = 0.1
C_s = 0.4
T_ambient = 0.2
T_ignition = 0.7
T_high = 1.3
T_low = 0.5
FIRE_USE_WIND = False
FIRE_SHOW_TEMPERATURE = True

FIRE_COLORBAR_MIN = 0.0
FIRE_COLORBAR_MAX = 3.0

gamma_fire = 0.2

def combustion_rate(T):
    r = np.zeros_like(T, dtype=float)
    mask = T > T_ambient
    r[mask] = np.exp(-gamma_fire / np.maximum(T[mask] - T_ambient, 1e-12))
    return r


def periodic_laplacian(Z, h):
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
        - 4.0 * Z
    ) / (h * h)


def periodic_gradient(Z, h):
    dZdx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2.0 * h)
    dZdy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2.0 * h)
    return dZdx, dZdy


def build_wind_field(X, Y):
    if not FIRE_USE_WIND:
        return np.zeros_like(X), np.zeros_like(Y)
    vx = 0.18 + 0.05 * np.sin(np.pi * Y / L)
    vy = 0.04 * np.cos(np.pi * X / L)
    return vx, vy


fire_wind_x, fire_wind_y = build_wind_field(X, Y)


def initialize_fire_fields(X, Y):
    hot1 = np.exp(-(((X + 0.3) ** 2) + ((Y - 0.4) ** 2)) / (2.0 * 0.3 ** 2))
    hot2 = np.exp(-(((X - 0.5) ** 2) + ((Y + 0.4) ** 2)) / (2.0 * 0.3 ** 2))
    hot3 = np.exp(-(((X + 0.1) ** 2) + ((Y + 0.0) ** 2)) / (2.0 * 0.3 ** 2))

    T0 = T_ambient +  1.2*hot1 + 1.2*hot2 + 1.3 *hot3
    # fuel decays linearly to 0 within 0.5 m of the boundary
    edge_decay_width = 0.7

    dist_left = X - (-L / 2)
    dist_right = (L / 2) - X
    dist_bottom = Y - (-L / 2)
    dist_top = (L / 2) - Y

    dist_to_edge = np.minimum.reduce([dist_left, dist_right, dist_bottom, dist_top])
    S0 = np.clip(dist_to_edge / edge_decay_width, 0.0, 1.0)
    return T0, S0



SUPPRESSION_SIGMA = 0.1   # paper uses sigma_s = 0.1
SUPPRESSION_FLOW = -1.0  * 2  # negative = cooling / extinguishing
WATER_MAX = 100
SPRAY_RATE_MAX = 1.0


def build_periodic_spray_kernel():
    dx = np.minimum(np.abs(X), L - np.abs(X))
    dy = np.minimum(np.abs(Y), L - np.abs(Y))
    K = np.exp(-((dx ** 2 + dy ** 2) / (2.0 * SUPPRESSION_SIGMA ** 2)))
    return np.fft.ifftshift(K)


SPRAY_KERNEL = build_periodic_spray_kernel()
SPRAY_KERNEL_FFT = np.fft.fft2(SPRAY_KERNEL)


def periodic_convolve_field(Z):
    return np.real(np.fft.ifft2(np.fft.fft2(Z) * SPRAY_KERNEL_FFT)) * (h ** 2)


def fire_suppression_input(T, robot_positions=None, spray_rates=None, water_levels=None):
    if robot_positions is None or len(robot_positions) == 0:
        return np.zeros_like(T)

    if spray_rates is None:
        spray_rates = np.zeros(len(robot_positions), dtype=float)

    if water_levels is None:
        water_levels = np.full(len(robot_positions), WATER_MAX, dtype=float)

    u_s = np.zeros_like(T, dtype=float)

    for pos, wdot_i, water_i in zip(robot_positions, spray_rates, water_levels):
        if water_i <= 0.0 or wdot_i <= 0.0:
            continue
        px, py = pos
        shaping = np.exp(-(((X - px) ** 2) + ((Y - py) ** 2)) / (2.0 * SUPPRESSION_SIGMA ** 2))
        u_s += SUPPRESSION_FLOW * wdot_i * shaping

    return u_s

def step_fire_dynamics(T, S, u_s, dt, h):
    lap_T = periodic_laplacian(T, h)
    Tx, Ty = periodic_gradient(T, h)
    reaction = combustion_rate(T)
    adv_term = fire_wind_x * Tx + fire_wind_y * Ty

    dTdt = eta_fire * lap_T - adv_term + A_fire * (S * reaction - C_cooling * (T - T_ambient)) + u_s
    dSdt = -C_s * S * reaction

    T_next = T + dt * dTdt
    S_next = S + dt * dSdt
    T_next = np.maximum(T_next, T_ambient)
    S_next = np.maximum(S_next, 0.0)
    return T_next, S_next

FIRE_TARGET_SMOOTH_SIGMA = 0.2

def compute_target_pdf_from_temperature_tail(T):
    # Tunable tail parameters
    FIRE_TARGET_TAIL_SIGMA = 0.75   # spatial decay length (same units as h, meters)
    FIRE_TARGET_TAIL_WEIGHT = 0.08  # tail strength relative to peak core value

    # Target region: warm enough to matter, but not too hot
    valid_band = (T > T_low) & (T < T_high)

    # Core target inside the band
    core = np.clip(T - T_low, 0.0, max(T_high - T_ignition, 1e-6))
    core[~valid_band] = 0.0

    # Smooth the core so the target is not sharp
    core = gaussian_filter(core, sigma=FIRE_TARGET_SMOOTH_SIGMA)

    # Re-mask the forbidden hot region after smoothing so blur does not leak into it
    core[T >= T_high] = 0.0

    # If no valid target exists, return zero
    core_peak = np.max(core)
    if core_peak <= 1e-10:
        return np.zeros_like(T)

    # Distance from every point outside the band to the nearest point in the band
    # distance_transform_edt(~valid_band) = 0 inside valid_band, >0 outside
    dist_to_band = distance_transform_edt(~valid_band) * h

    # Exponential/Gaussian tail outside the target band
    tail = np.exp(-(dist_to_band ** 2) / (2.0 * FIRE_TARGET_TAIL_SIGMA ** 2))

    # Keep the tail only outside the core band and outside the forbidden hot region
    tail[valid_band] = 0.0
    tail[T >= T_high] = 0.0

    # Scale tail so it stays low compared to the core
    tail *= FIRE_TARGET_TAIL_WEIGHT * core_peak

    # Combine core + tail
    target = core + tail

    # Normalize to total mass = 5 drones
    total = np.sum(target) * h ** 2
    if total <= 1e-10:
        return np.zeros_like(T)

    return target / total * num_pdfs


def central_compute_fire_zones(T):
    unsafe_mask = T >= T_high
    fire_mask = T >= T_ignition
    target_zone_mask = (T >= T_low) & (T < T_high)
    '''unsafe_mask_inflated = binary_dilation(
        unsafe_mask.astype(bool),
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        iterations=5,
    )'''
    rho_target = compute_target_pdf_from_temperature_tail(T)
    return unsafe_mask.astype(bool), fire_mask.astype(bool), target_zone_mask.astype(bool), rho_target

T_fire, S_fire = initialize_fire_fields(X, Y)
Mask_A, Mask_Fire, Mask_target_zone, rho_d = central_compute_fire_zones(T_fire)
Mask = Mask_A
# ----------------------- Energy State -----------------------
u_max = 0.2
alpha_e = 0.01
E_min = 0.1
c_e = 0.09
c_0 = 0.0008
gain = 0.2

Charging_flag = np.full(num_pdfs, False, dtype=bool)

energy_history = [E.copy()]
energy_time = [0.0]
energy_min_history = [float(E.min())]
energy_avg_history = [float(E.mean())]

# ----------------------- Water State -----------------------
Water = np.full(num_pdfs, WATER_MAX, dtype=float)
water_history = [Water.copy()]
water_time = [0.0]
water_min_history = [float(Water.min())]
water_avg_history = [float(Water.mean())]

# ----------------------- RRT planner config -----------------------
RRT_MAX_EXTEND_STEPS = 1000
RRT_STEER_TO_GOAL_EVERY = 1
RRT_K = 40
RRT_BASE_SEED = 11
RRT_SEGMENT_LEN = u_max * dt * 10 * 2
RRT_USE_SPACE_CBF = True

rrt_paths = None
rrt_path_lengths = None
rrt_dv_first = None
rrt_first_step_vecs = None

rrt_paths = None
rrt_path_lengths = None
rrt_dv_first = None
rrt_first_step_vecs = None

planner_executor = None
planner_future = None
planner_latest_plan = None
planner_last_submit_frame = None
planner_last_ready_frame = None


def setup_background_planner():
    global planner_executor
    if not (USE_ENERGY_PLANNING and RRT_ASYNC_ENABLED):
        return
    if planner_executor is None:
        planner_executor = ProcessPoolExecutor(max_workers=RRT_ASYNC_WORKERS)


def shutdown_background_planner():
    global planner_executor, planner_future
    if planner_future is not None:
        planner_future.cancel()
        planner_future = None
    if planner_executor is not None:
        planner_executor.shutdown(wait=False, cancel_futures=True)
        planner_executor = None


def fallback_charge_plan(robot_positions):
    robot_positions = np.asarray(robot_positions, float)
    goal = np.array([charger_x, charger_y], dtype=float)
    vec = goal[None, :] - robot_positions
    dist_to_center = np.linalg.norm(vec, axis=1)
    dv_first = np.divide(
        vec,
        dist_to_center[:, None],
        out=np.zeros_like(vec),
        where=dist_to_center[:, None] > 1e-12,
    )
    path_lengths = np.maximum(0.0, dist_to_center - charger_radius)
    first_step_mag = np.minimum(path_lengths, RRT_SEGMENT_LEN)
    first_step_vecs = dv_first * first_step_mag[:, None]
    return {
        "paths": [None] * len(robot_positions),
        "path_lengths": path_lengths,
        "dv_first": dv_first,
        "first_step_vecs": first_step_vecs,
    }

def submit_background_rrt_plan(frame, robot_positions, mask_a, Mask_A_Inflate):
    global planner_future, planner_last_submit_frame, planner_last_runtime

    if not USE_ENERGY_PLANNING:
        return
    if planner_future is not None and not planner_future.done():
        return

    if RRT_ASYNC_ENABLED:
        setup_background_planner()
        planner_future = planner_executor.submit(
            timed_plan_all_robots_rrt_parallel,
            init_xy_all=np.asarray(robot_positions, float),
            bounds=(-L / 2, L / 2, -L / 2, L / 2),
            X=X,
            Y=Y,
            invariance_mask=np.asarray(mask_a, bool),
            invariance_mask_inflated=np.asarray(Mask_A_Inflate, bool),
            charge_mask=Mask_G,
            charge_center=np.array([charger_x, charger_y], dtype=float),
            u_max=u_max * 5,
            dt=4 * dt,
            segment_len=RRT_SEGMENT_LEN,
            max_extend_steps=RRT_MAX_EXTEND_STEPS,
            steer_to_goal_every=RRT_STEER_TO_GOAL_EVERY,
            k=RRT_K,
            base_rng_seed=RRT_BASE_SEED,
            use_charge_mask=True,
            use_space_cbf=RRT_USE_SPACE_CBF,
            sigma=sigma_energy,
            lap_coeff=lap_coeff,
            epsilon=epsilon,
            alpha_h=alpha_h,
            parallel=False,
            n_workers=1,
        )
    else:
        planner_future = None
        out = timed_plan_all_robots_rrt_parallel(
            init_xy_all=np.asarray(robot_positions, float),
            bounds=(-L / 2, L / 2, -L / 2, L / 2),
            X=X,
            Y=Y,
            invariance_mask=np.asarray(mask_a, bool),
            invariance_mask_inflated=np.asarray(Mask_A_Inflate, bool),
            charge_mask=Mask_G,
            charge_center=np.array([charger_x, charger_y], dtype=float),
            u_max=u_max * 5,
            dt=4 * dt,
            segment_len=RRT_SEGMENT_LEN,
            max_extend_steps=RRT_MAX_EXTEND_STEPS,
            steer_to_goal_every=RRT_STEER_TO_GOAL_EVERY,
            k=RRT_K,
            base_rng_seed=RRT_BASE_SEED,
            use_charge_mask=True,
            use_space_cbf=RRT_USE_SPACE_CBF,
            sigma=sigma_energy,
            lap_coeff=lap_coeff,
            epsilon=epsilon,
            alpha_h=alpha_h,
            parallel=False,
            n_workers=1,
        )
        set_latest_rrt_plan(out["plan"])
        planner_last_runtime = float(out["planner_runtime"])

    planner_last_submit_frame = frame

def poll_background_rrt_plan():
    global planner_future, planner_last_runtime
    if planner_future is None or not planner_future.done():
        return
    try:
        out = planner_future.result()
        set_latest_rrt_plan(out["plan"])
        planner_last_runtime = float(out["planner_runtime"])
    except Exception as exc:
        print(f"Background RRT failed: {exc}")
    finally:
        planner_future = None

def compute_initial_rrt_plan():
    robot_positions = np.array([pdf.position for pdf in pdfs], dtype=float)

    return plan_all_robots_rrt_parallel(
        init_xy_all=robot_positions,
        bounds=(-L / 2, L / 2, -L / 2, L / 2),
        X=X,
        Y=Y,
        invariance_mask=np.asarray(Mask_A, bool),
        invariance_mask_inflated=np.asarray(Mask_Fire, bool),
        charge_mask=Mask_G,
        charge_center=np.array([charger_x, charger_y], dtype=float),
        u_max=u_max * 5,
        dt=4 * dt,
        segment_len=RRT_SEGMENT_LEN,
        max_extend_steps=RRT_MAX_EXTEND_STEPS,
        steer_to_goal_every=RRT_STEER_TO_GOAL_EVERY,
        k=RRT_K,
        base_rng_seed=RRT_BASE_SEED,
        use_charge_mask=True,
        use_space_cbf=RRT_USE_SPACE_CBF,
        sigma=sigma_energy,
        lap_coeff=lap_coeff,
        epsilon=epsilon,
        alpha_h=alpha_h,
        parallel=False,
        n_workers=1,
    )

def set_latest_rrt_plan(plan_out):
    global planner_latest_plan, planner_last_ready_frame
    global rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs

    planner_latest_plan = plan_out
    planner_last_ready_frame = planner_last_submit_frame
    rrt_paths = plan_out["paths"]
    rrt_path_lengths = plan_out["path_lengths"]
    rrt_dv_first = plan_out["dv_first"]
    rrt_first_step_vecs = plan_out["first_step_vecs"]


def get_active_rrt_plan(robot_positions):
    return planner_latest_plan

def should_submit_rrt_plan(frame):
    if planner_last_submit_frame is None:
        return True
    return (frame - planner_last_submit_frame) >= RRT_REPLAN_EVERY_FRAMES

def planner_outputs_to_energy_terms(robot_positions, dv_first, path_lengths, *, u_max, c_e, c_0):
    robot_positions = np.asarray(robot_positions, float)
    dv_first = np.asarray(dv_first, float)
    path_lengths = np.asarray(path_lengths, float)

    Ve_dist = path_lengths.copy()
    scaleE = (c_e * u_max + c_0) / u_max
    Ve_energy = Ve_dist * scaleE
    dVex_energy = -dv_first[:, 0] * gain
    dVey_energy = -dv_first[:, 1] * gain

    bad = ~np.isfinite(Ve_energy)
    Ve_energy[bad] = 1e6
    dVex_energy[~np.isfinite(dVex_energy)] = 0.0
    dVey_energy[~np.isfinite(dVey_energy)] = 0.0
    return Ve_energy, dVex_energy, dVey_energy

planner_last_runtime = None

def timed_plan_all_robots_rrt_parallel(**kwargs):
    t0 = time.perf_counter()
    plan = plan_all_robots_rrt_parallel(**kwargs)
    t_plan = time.perf_counter() - t0
    return {"plan": plan, "planner_runtime": t_plan}


class InvarianceControllerSmall:
    def __init__(self, use_battery_cbf=True):
        J = int(num_pdfs)
        self.use_battery_cbf = bool(use_battery_cbf)
        M = self.model = mf.Model("Main")
        self.u = M.variable("u", [J, 2], mf.Domain.inRange(-u_max, u_max))
        self.w_dot = M.variable("w_dot", J, mf.Domain.inRange(0.0, SPRAY_RATE_MAX))
        self.s = M.variable("s", 1, mf.Domain.greaterThan(0.0))
        self.t = M.variable("t", 1, mf.Domain.greaterThan(0.0))

        M.constraint(
            "rotated_norm_sq",
            mf.Expr.vstack(self.t, mf.Expr.reshape(self.u, [2 * J])),
            mf.Domain.inQCone(),
        )

        obj = mf.Expr.add(self.t, mf.Expr.mul(self.s, float(gemma)))
        obj = mf.Expr.add(obj, mf.Expr.mul(float(SPRAY_REG), mf.Expr.sum(self.w_dot)))

        self.a_vx = M.parameter("a_vx", J)
        self.a_vy = M.parameter("a_vy", J)
        self.a_vw = M.parameter("a_vw", J)
        self.a_hx = M.parameter("a_hx", J)
        self.a_hy = M.parameter("a_hy", J)
        self.c_v = M.parameter("c_v", 1)
        self.c_h = M.parameter("c_h", 1)
        self.alpha_V = M.parameter("alpha_V", 1)
        self.alpha_h_psi = M.parameter("alpha_h_psi", 1)
        self.w_cap = M.parameter("w_cap", J)

        ux = self.u.slice([0, 0], [J, 1]).reshape([J])
        uy = self.u.slice([0, 1], [J, 2]).reshape([J])
        self.z = M.variable("z", J, mf.Domain.greaterThan(0.0))

        M.constraint("w_cap_con", mf.Expr.sub(self.w_dot, self.w_cap), mf.Domain.lessThan(0.0))

        for i in range(J):
            M.constraint(
                f"u_sq_{i}",
                mf.Expr.vstack([self.z.index(i), ux.index(i), uy.index(i)]),
                mf.Domain.inQCone(),
            )

        if self.use_battery_cbf:
            self.energy_slack = M.variable("s_e", 1, mf.Domain.greaterThan(0.0))
            self.Ve = M.parameter("Ve", J)
            self.dVex = M.parameter("dVex", J)
            self.dVey = M.parameter("dVey", J)
            self.Ecur = M.parameter("Ecur", J)
            self.Emin = M.parameter("Emin", 1)
            self.alpha_e = M.parameter("alpha_e", 1)
            self.c_e = M.parameter("c_e", 1)
        else:
            self.energy_slack = None
            self.Ve = None
            self.dVex = None
            self.dVey = None
            self.Ecur = None
            self.Emin = None
            self.alpha_e = None
            self.c_e = None

        M.objective("obj", mf.ObjectiveSense.Minimize, obj)

        v_dot = mf.Expr.add(
            mf.Expr.add(
                mf.Expr.add(mf.Expr.dot(self.a_vx, ux), mf.Expr.dot(self.a_vy, uy)),
                mf.Expr.dot(self.a_vw, self.w_dot),
            ),
            self.c_v.index(0),
        )
        h_dot = mf.Expr.add(
            mf.Expr.add(mf.Expr.dot(self.a_hx, ux), mf.Expr.dot(self.a_hy, uy)),
            self.c_h.index(0),
        )

        M.constraint(
            "CLF",
            mf.Expr.sub(mf.Expr.add(self.alpha_V.index(0), v_dot), self.s.index(0)),
            mf.Domain.lessThan(0.0),
        )
        M.constraint(
            "CBF",
            mf.Expr.add(self.alpha_h_psi.index(0), h_dot),
            mf.Domain.greaterThan(0.0),
        )

        if self.use_battery_cbf:
            for i in range(J):
                lhs = mf.Expr.add(
                    mf.Expr.add(mf.Expr.mul(self.c_e.index(0), self.z.index(i)), c_0),
                    mf.Expr.mul(
                        0.5,
                        mf.Expr.add(
                            mf.Expr.mul(self.dVex.index(i), ux.index(i)),
                            mf.Expr.mul(self.dVey.index(i), uy.index(i)),
                        ),
                    ),
                )
                rhs = mf.Expr.mul(
                    self.alpha_e.index(0),
                    mf.Expr.sub(
                        mf.Expr.sub(self.Ecur.index(i), self.Emin.index(0)),
                        self.Ve.index(i),
                    ),
                )
                M.constraint(
                    f"EnergyCBF_{i}",
                    mf.Expr.sub(rhs, lhs),
                    mf.Domain.greaterThan(0.0),
                )

        M.setLogHandler(None)

    def set_params_from_coeffs(self, coeffs, alpha_V_value, alpha_h_psi_value, w_cap):
        self.a_vx.setValue(np.asarray(coeffs["A_vx"], float))
        self.a_vy.setValue(np.asarray(coeffs["A_vy"], float))
        self.a_vw.setValue(np.asarray(coeffs["A_vw"], float))
        self.a_hx.setValue(np.asarray(coeffs["A_hx"], float))
        self.a_hy.setValue(np.asarray(coeffs["A_hy"], float))
        self.c_v.setValue([float(coeffs["c_v"])])
        self.c_h.setValue([float(coeffs["c_h"])])
        self.alpha_V.setValue([float(alpha_V_value)])
        self.alpha_h_psi.setValue([float(alpha_h_psi_value)])
        self.w_cap.setValue(np.asarray(w_cap, float))

    def set_energy_params(self, Ve, dVex, dVey, Ecur, Emin, alpha_e, c_e):
        if not self.use_battery_cbf:
            return
        self.Ve.setValue(np.asarray(Ve, float))
        self.dVex.setValue(np.asarray(dVex, float))
        self.dVey.setValue(np.asarray(dVey, float))
        self.Ecur.setValue(np.asarray(Ecur, float))
        self.Emin.setValue([float(Emin)])
        self.alpha_e.setValue([float(alpha_e)])
        self.c_e.setValue([float(c_e)])

    def solve(self):
        J = int(num_pdfs)
        try:
            self.model.solve()
            return (
                self.u.level().reshape(J, 2),
                self.w_dot.level().reshape(J),
                self.s.level()[0],
                self.t.level()[0],
            )
        except Exception:
            return np.zeros((J, 2), dtype=float), np.zeros(J, dtype=float), 0.0, 0.0


# ----------------------- Plot/History State -----------------------
time_history = []
psi_history = []
error_history = []
slack_history = []
fire_area_history = []

rho_raw = np.sum([pdf.field for pdf in pdfs], axis=0)
rho_eff = np.sum(
    [(Water[i] / WATER_MAX) * periodic_convolve_field(pdf.field) for i, pdf in enumerate(pdfs)],
    axis=0,
)
error_norm = h ** 2 * np.sum((rho_d - rho_eff) ** 2)
error_history.append(error_norm)
slack_history.append(0.0)
psi_val = epsilon - np.sum(rho_raw[Mask] ** 2) * h ** 2
psi_history.append(psi_val)
fire_area_history.append(h ** 2 * np.sum(T_fire > T_ignition))
time_history.append(0.0)

font_size = 22
fig, ax_scatter = plt.subplots(figsize=(8, 6))
ax_scatter.set_title(f"Swarm + Fire Field at t={0.00:.2f}", fontsize=font_size)
ax_scatter.set_xlabel("X", fontsize=font_size)
ax_scatter.set_ylabel("Y", fontsize=font_size)
ax_scatter.tick_params(axis='both', which='major', labelsize=font_size)
ax_scatter.set_xlim(-L / 2, L / 2)
ax_scatter.set_ylim(-L / 2, L / 2)
scat = ax_scatter.scatter([pdf.position[0] for pdf in pdfs], [pdf.position[1] for pdf in pdfs], color='blue', s=50, label='Robot')
fire_img = ax_scatter.imshow(
    T_fire if FIRE_SHOW_TEMPERATURE else rho_d,
    extent=[-L / 2, L / 2, -L / 2, L / 2],
    origin='lower',
    cmap='hot' if FIRE_SHOW_TEMPERATURE else 'YlGn',
    alpha=0.65,
    vmin=(FIRE_COLORBAR_MIN if FIRE_SHOW_TEMPERATURE else None),
    vmax=(FIRE_COLORBAR_MAX if FIRE_SHOW_TEMPERATURE else None),
)
unsafe_contour = ax_scatter.contour(X, Y, Mask_A.astype(float), colors="red", levels=[0.5], linewidths=1.5)
fire_contour = ax_scatter.contour(X, Y, Mask_Fire.astype(float), colors="pink", levels=[0.5], linewidths=1.0)
target_zone_contour = ax_scatter.contour(X, Y, Mask_target_zone.astype(float), colors="cyan", levels=[0.5], linewidths=1.2, linestyles='dotted')
ax_scatter.add_patch(Circle((charger_x, charger_y), charger_radius, edgecolor='yellow', facecolor='yellow', alpha=0.25, linewidth=2))

fire_cbar = None
if FIRE_SHOW_TEMPERATURE:
    fire_cbar = fig.colorbar(fire_img, ax=ax_scatter, fraction=0.046, pad=0.04)
    fire_cbar.set_label('Temperature', fontsize=12)
    fire_cbar.ax.axhline(T_ambient, color='white', linewidth=1.2, linestyle='-')
    fire_cbar.ax.axhline(T_ignition, color='cyan', linewidth=1.2, linestyle='--')
    fire_cbar.ax.axhline(T_high, color='red', linewidth=1.2, linestyle='--')

legend_handles = [
    Line2D([0], [0], color='red', lw=1.5, label='Unsafe zone: T ≥ T_high'),
    Line2D([0], [0], color='pink', lw=1.0, label='Fire zone: T ≥ T_ignite'),
    Line2D([0], [0], color='cyan', lw=1.2, linestyle='dotted', label='Target zone: T_low ≤ T < T_high'),
    Line2D([0], [0], color='black', lw=1.5, linestyle='dashed', label='Swarm density contour'),
]
ax_scatter.legend(handles=legend_handles, loc='lower right', fontsize=7)

energy_texts = []
for i in range(num_pdfs):
    txt = ax_scatter.text(pdfs[i].position[0], pdfs[i].position[1] + 0.2, f"{E[i]:.2f}", fontsize=10, color="black", ha="center", va="center")
    energy_texts.append(txt)

water_texts = []
for i in range(num_pdfs):
    txt = ax_scatter.text(pdfs[i].position[0], pdfs[i].position[1] - 0.2, f"{Water[i]:.1f}", fontsize=10, color="blue", ha="center", va="center")
    water_texts.append(txt)

traj_x_list = [[] for _ in range(num_pdfs)]
traj_y_list = [[] for _ in range(num_pdfs)]
traj_lines = []
for _ in range(num_pdfs):
    line, = ax_scatter.plot([], [], 'b-', linewidth=1.5)
    traj_lines.append(line)

rho_test = np.sum([pdf.field for pdf in pdfs], axis=0)
contour_levels = [0.05, 0.25, 0.5, 0.7, 0.9]
contour = ax_scatter.contour(X, Y, rho_test, levels=contour_levels, colors='black', linewidths=1.5, linestyles='dashed', alpha=0.5)

path_lines = []
path_ends = []
for _ in range(num_pdfs):
    ln, = ax_scatter.plot([], [], linewidth=1.8, alpha=0.9)
    ed = ax_scatter.scatter([], [], s=35, marker='x')
    path_lines.append(ln)
    path_ends.append(ed)


def update_rrt_path_overlay(paths, path_lines, path_ends):
    J = len(path_lines)
    for i in range(J):
        p = None if (paths is None or i >= len(paths)) else paths[i]
        if p is None or len(p) == 0:
            path_lines[i].set_data([], [])
            path_ends[i].set_offsets(np.empty((0, 2)))
            continue
        p = np.asarray(p, float)
        path_lines[i].set_data(p[:, 0], p[:, 1])
        path_ends[i].set_offsets(p[-1:, :2])


def clear_rrt_path_overlay(path_lines, path_ends):
    for i in range(len(path_lines)):
        path_lines[i].set_data([], [])
        path_ends[i].set_offsets(np.empty((0, 2)))


def remove_contour_set(contour_set):
    if contour_set is None:
        return None
    if hasattr(contour_set, "collections"):
        for coll in list(contour_set.collections):
            try:
                coll.remove()
            except ValueError:
                pass
    else:
        try:
            contour_set.remove()
        except (ValueError, AttributeError):
            pass
    return None


controller = InvarianceControllerSmall(use_battery_cbf=(USE_ENERGY_PLANNING and USE_BATTERY_CBF))
print("Controller built")


def _periodic_delta(coord, center, L):
    d = np.abs(coord - center)
    return np.minimum(d, L - d)


N = Nx * Ny
inds = np.arange(N, dtype=np.int32).reshape(Nx, Ny)
iR = np.roll(inds, -1, axis=1).ravel()
iL = np.roll(inds, 1, axis=1).ravel()
iU = np.roll(inds, -1, axis=0).ravel()
iD = np.roll(inds, 1, axis=0).ravel()

Xf = X.ravel().astype(float)
Yf = Y.ravel().astype(float)
Mask_f = Mask.ravel().astype(float)
rho_d_f = rho_d.ravel().astype(float)


def sumfield_coeffs_fd5(positions_meas, Water, *, Xf, Yf, iR, iL, iU, iD, Nx, Ny, L, h, sigma, lap_coeff, rho_d_f, Mask_f, alpha_v, alpha_h, epsilon):
    J = positions_meas.shape[0]
    cx = positions_meas[:, 0][:, None]
    cy = positions_meas[:, 1][:, None]
    Xr = Xf[None, :]
    Yr = Yf[None, :]

    dx = _periodic_delta(Xr, cx, L)
    dy = _periodic_delta(Yr, cy, L)
    rhoC = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))

    rhoR = rhoC[:, iR]
    rhoL = rhoC[:, iL]
    rhoU = rhoC[:, iU]
    rhoD = rhoC[:, iD]

    gx = (rhoR - rhoL) / (2.0 * h)
    gy = (rhoU - rhoD) / (2.0 * h)
    gx *= -1.0
    gy *= -1.0

    lap = (rhoR + rhoL + rhoU + rhoD - 4.0 * rhoC) / (h * h)
    b = lap_coeff * lap

    # CBF remains on the raw summed swarm density
    sum_rho = np.sum(rhoC, axis=0)
    a_vec = sum_rho * Mask_f
    scale = -2.0 * h * h

    # CLF uses normalized-water weighted spray-shaped density
    rhoC_2d = rhoC.reshape(J, Ny, Nx)
    gx_2d = gx.reshape(J, Ny, Nx)
    gy_2d = gy.reshape(J, Ny, Nx)
    b_2d = b.reshape(J, Ny, Nx)

    kC = np.array([periodic_convolve_field(rhoC_2d[j]).ravel() for j in range(J)])
    kgx = np.array([periodic_convolve_field(gx_2d[j]).ravel() for j in range(J)])
    kgy = np.array([periodic_convolve_field(gy_2d[j]).ravel() for j in range(J)])
    kb = np.array([periodic_convolve_field(b_2d[j]).ravel() for j in range(J)])

    W = (np.asarray(Water, float) / WATER_MAX)[:, None]
    rho_eff = np.sum(W * kC, axis=0)
    d_vec = rho_d_f - rho_eff

    A_vx = scale * ((W * kgx) @ d_vec)
    A_vy = scale * ((W * kgy) @ d_vec)
    c_v = float(scale * np.sum((W * kb) * d_vec[None, :]))

    # Water_dot_i = -w_dot_i, and W_i = Water_i / WATER_MAX
    # so d/dt(W_i K*rho_i) contributes -(w_dot_i/WATER_MAX) K*rho_i
    A_vw = -(scale) * (kC @ d_vec)
    temp = np.ones_like(d_vec)
    A_vw = -(kC @ temp)

    A_hx = scale * (gx @ a_vec)
    A_hy = scale * (gy @ a_vec)
    c_h = float(scale * np.sum(b * a_vec[None, :]))

    coeffs = dict(A_vx=A_vx, A_vy=A_vy, A_vw=A_vw, A_hx=A_hx, A_hy=A_hy, c_v=c_v, c_h=c_h)
    alpha_v_value = float(alpha_v * h * h * (np.sum(d_vec * d_vec) + np.sum(rho_d_f * rho_d_f)))
    alpha_h_value = float(alpha_h * (epsilon - h * h * np.sum(a_vec * a_vec)))
    return coeffs, alpha_v_value, alpha_h_value


FRAMES = 10000
INTERVAL_MS = 50

def reset_run_state(run_idx: int = 0):
    global total_time, total_time_count
    global pdfs, controller
    global E, Water, Charging_flag
    global T_fire, S_fire, Mask_A, Mask_Fire, Mask_target_zone, Mask, rho_d, Mask_f, rho_d_f
    global time_history, psi_history, error_history, slack_history, fire_area_history
    global energy_history, energy_time, energy_min_history, energy_avg_history
    global water_history, water_time, water_min_history, water_avg_history
    global rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs
    global planner_future, planner_latest_plan, planner_last_submit_frame, planner_last_ready_frame, planner_last_runtime
    global RRT_BASE_SEED
    planner_last_runtime = None
    shutdown_background_planner()

    # change seed across runs so runs are not identical
    RRT_BASE_SEED = 11 + 1000 * run_idx

    pdfs = [PDF(p["position"], sigma_energy) for p in particle_data]
    controller = InvarianceControllerSmall(use_battery_cbf=(USE_ENERGY_PLANNING and USE_BATTERY_CBF))

    E = E_INIT.copy()
    Water = np.full(num_pdfs, WATER_MAX, dtype=float)
    Charging_flag = np.full(num_pdfs, False, dtype=bool)

    T_fire, S_fire = initialize_fire_fields(X, Y)
    Mask_A, Mask_Fire, Mask_target_zone, rho_d = central_compute_fire_zones(T_fire)
    Mask = Mask_A
    Mask_f = Mask_A.ravel().astype(float)
    rho_d_f = rho_d.ravel().astype(float)

    rho_raw0 = np.sum([pdf.field for pdf in pdfs], axis=0)
    rho_eff0 = np.sum(
        [(Water[i] / WATER_MAX) * periodic_convolve_field(pdf.field) for i, pdf in enumerate(pdfs)],
        axis=0,
    )

    err0 = h**2 * np.sum((rho_d - rho_eff0) ** 2)
    psi0 = epsilon - np.sum(rho_raw0[Mask] ** 2) * h**2
    fire0 = h**2 * np.sum(T_fire > T_ignition)

    time_history = [0.0]
    psi_history = [psi0]
    error_history = [err0]
    slack_history = [0.0]
    fire_area_history = [fire0]

    energy_history = [E.copy()]
    energy_time = [0.0]
    energy_min_history = [float(E.min())]
    energy_avg_history = [float(E.mean())]

    water_history = [Water.copy()]
    water_time = [0.0]
    water_min_history = [float(Water.min())]
    water_avg_history = [float(Water.mean())]

    rrt_paths = None
    rrt_path_lengths = None
    rrt_dv_first = None
    rrt_first_step_vecs = None

    planner_future = None
    planner_latest_plan = None
    planner_last_submit_frame = None
    planner_last_ready_frame = None

    total_time = 0.0
    total_time_count = 0


def simulate_one_step_headless(frame):
    global time_history, psi_history, error_history, slack_history, fire_area_history
    global total_time, total_time_count
    global E, Water, Charging_flag, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs, planner_last_runtime
    global T_fire, S_fire, Mask_A, Mask_Fire, Mask_target_zone, Mask, rho_d, Mask_f, rho_d_f

    current_time = (frame + 1) * dt
    time_history.append(current_time)

    tmp = [pdf.compute_pdf_random() for pdf in pdfs]
    robot_positions = np.array([t[1] for t in tmp], dtype=float)

    start = time.perf_counter()
    planner_runtime_this_step = np.nan
    Mask_A, Mask_Fire, Mask_target_zone, rho_d = central_compute_fire_zones(T_fire)
    Mask = Mask_A
    Mask_f = Mask_A.ravel().astype(float)
    rho_d_f = rho_d.ravel().astype(float)

    if USE_ENERGY_PLANNING:
        poll_background_rrt_plan()

        if should_submit_rrt_plan(frame):
            submit_background_rrt_plan(frame, robot_positions, Mask_A, Mask_Fire)

        active_plan = get_active_rrt_plan(robot_positions)

        if active_plan is not None:
            rrt_paths = active_plan["paths"]
            rrt_path_lengths = active_plan["path_lengths"]
            rrt_dv_first = active_plan["dv_first"]
            rrt_first_step_vecs = active_plan["first_step_vecs"]
        else:
            rrt_paths = None
            rrt_path_lengths = np.zeros(num_pdfs, dtype=float)
            rrt_dv_first = np.zeros((num_pdfs, 2), dtype=float)
            rrt_first_step_vecs = np.zeros((num_pdfs, 2), dtype=float)

        if USE_BATTERY_CBF:
            Ve_energy, dVex_energy, dVey_energy = planner_outputs_to_energy_terms(
                robot_positions=robot_positions,
                dv_first=rrt_dv_first,
                path_lengths=rrt_path_lengths,
                u_max=u_max,
                c_e=c_e,
                c_0=c_0,
            )
            hE = E - E_min - Ve_energy
        else:
            Ve_energy = np.zeros(num_pdfs, dtype=float)
            dVex_energy = np.zeros(num_pdfs, dtype=float)
            dVey_energy = np.zeros(num_pdfs, dtype=float)
            hE = E - E_min
    else:
        rrt_paths = None
        rrt_path_lengths = None
        rrt_dv_first = None
        rrt_first_step_vecs = None
        Ve_energy = np.zeros(num_pdfs, dtype=float)
        dVex_energy = np.zeros(num_pdfs, dtype=float)
        dVey_energy = np.zeros(num_pdfs, dtype=float)
        hE = E - E_min

    if planner_last_runtime is not None:
        planner_runtime_this_step = float(planner_last_runtime)
        planner_last_runtime = None

    coeffs, alpha_v_value, alpha_h_value = sumfield_coeffs_fd5(
        robot_positions,
        Water,
        Xf=Xf, Yf=Yf, iR=iR, iL=iL, iU=iU, iD=iD,
        Nx=Nx, Ny=Ny, L=L, h=h, sigma=pdfs[0].sigma, lap_coeff=lap_coeff,
        rho_d_f=rho_d_f, Mask_f=Mask_f,
        alpha_v=alpha_v, alpha_h=alpha_h, epsilon=epsilon
    )

    w_cap = np.minimum(SPRAY_RATE_MAX, Water / max(dt, 1e-12))
    controller.set_params_from_coeffs(coeffs, alpha_v_value, alpha_h_value, w_cap=w_cap)

    if USE_ENERGY_PLANNING and USE_BATTERY_CBF:
        controller.set_energy_params(
            Ve=Ve_energy,
            dVex=dVex_energy,
            dVey=dVey_energy,
            Ecur=E,
            Emin=E_min,
            alpha_e=alpha_e,
            c_e=c_e,
        )

    u_opt, wdot_opt, s_val, t_val = controller.solve()

    #if alpha_v_value <= 700:
    #    wdot_opt = np.ones_like(wdot_opt)

    u_s_field = fire_suppression_input(
        T_fire,
        robot_positions=robot_positions,
        spray_rates=wdot_opt,
        water_levels=Water,
    )
    T_fire, S_fire = step_fire_dynamics(T_fire, S_fire, u_s_field, dt, h)

    Mask_A, Mask_Fire, Mask_target_zone, rho_d = central_compute_fire_zones(T_fire)
    Mask = Mask_A
    Mask_f = Mask_A.ravel().astype(float)
    rho_d_f = rho_d.ravel().astype(float)

    curr_time = time.perf_counter() - start
    total_time += curr_time
    total_time_count += 1

    if USE_ENERGY_PLANNING:
        r_goal_switch = charger_radius + 0.35
        charger_center = np.array([charger_x, charger_y], dtype=float)
        for i in range(num_pdfs):
            d = np.linalg.norm(robot_positions[i] - charger_center)
            if (hE[i] <= 0.05) or Charging_flag[i] or d <= charger_radius:
                if d <= r_goal_switch:
                    if E[i] <= 0.91:
                        Charging_flag[i] = True
                        if d <= charger_radius:
                            u_opt[i] = [0.0, 0.0]
                            E[i] = min(1.0, E[i] + 0.5 * dt)
                        else:
                            v = charger_center - robot_positions[i]
                            n = np.linalg.norm(v)
                            u_opt[i] = (u_max * v / n) if n > 1e-12 else np.zeros(2, dtype=float)
                    else:
                        Charging_flag[i] = False
    else:
        Charging_flag[:] = False

    u_norm = np.sqrt(u_opt[:, 0] ** 2 + u_opt[:, 1] ** 2)
    E[:] = E - (c_e * u_norm + c_0) * dt
    E[:] = np.clip(E, 0.0, 1.0)

    energy_time.append(current_time)
    energy_history.append(E.copy())
    energy_min_history.append(float(E.min()))
    energy_avg_history.append(float(E.mean()))

    Water[:] = np.clip(Water - wdot_opt * dt, 0.0, WATER_MAX)
    water_time.append(current_time)
    water_history.append(Water.copy())
    water_min_history.append(float(Water.min()))
    water_avg_history.append(float(Water.mean()))

    for i, pdf in enumerate(pdfs):
        pdf.update(u_opt[i], dt, x_grid, y_grid, Charging_flag[i])

    rho_raw = np.sum([pdf.field for pdf in pdfs], axis=0)
    rho_eff = np.sum(
        [(Water[i] / WATER_MAX) * periodic_convolve_field(pdf.field) for i, pdf in enumerate(pdfs)],
        axis=0,
    )

    psi_val = epsilon - np.sum(rho_raw[Mask] ** 2) * h ** 2
    error_norm = h ** 2 * np.sum((rho_d - rho_eff) ** 2)
    fire_area = h ** 2 * np.sum(T_fire > T_ignition)

    psi_history.append(psi_val)
    error_history.append(error_norm)
    fire_area_history.append(fire_area)
    slack_history.append(float(s_val))

    return {
        "time": current_time,
        "density_error": float(error_norm),
        "psi": float(psi_val),
        "energy": E.copy(),
        "fire_area": float(fire_area),
        "water": Water.copy(),
        "step_runtime": float(curr_time),
        "avg_runtime_per_robot": float(curr_time / num_pdfs),
        "planner_runtime": float(planner_runtime_this_step) if np.isfinite(planner_runtime_this_step) else np.nan,
    }


def run_ensemble_timeseries_fire(
    n_runs: int = 100,
    sim_time: float = 50.0,
    *,
    save_dir: str = "fire_ensemble_runs",
    verbose: bool = False,
    force_sync_rrt: bool = False,
):
    """
    Headless ensemble version for the centralized colorbar fire script.
    Saves one .npz per run.
    """
    global RRT_ASYNC_ENABLED, planner_last_submit_frame, planner_last_ready_frame

    n_steps = int(round(sim_time / dt))
    time_vec = (np.arange(n_steps, dtype=float) + 1.0) * dt
    os.makedirs(save_dir, exist_ok=True)

    old_async = RRT_ASYNC_ENABLED
    if force_sync_rrt:
        RRT_ASYNC_ENABLED = False

    try:
        overall_runtime_sum = 0.0
        overall_runtime_count = 0

        for run_idx in range(n_runs):
            reset_run_state(run_idx=run_idx)

            if USE_ENERGY_PLANNING:
                init_plan = compute_initial_rrt_plan()
                planner_last_submit_frame = 0
                set_latest_rrt_plan(init_plan)
                planner_last_ready_frame = 0

            density_error_ts = np.zeros(n_steps, dtype=float)
            psi_ts = np.zeros(n_steps, dtype=float)
            fire_area_ts = np.zeros(n_steps, dtype=float)
            step_runtime_ts = np.zeros(n_steps, dtype=float)
            avg_runtime_per_robot_ts = np.zeros(n_steps, dtype=float)
            planner_runtime_ts = np.full(n_steps, np.nan, dtype=float)
            energy_ts = np.zeros((n_steps, num_pdfs), dtype=float)
            water_ts = np.zeros((n_steps, num_pdfs), dtype=float)

            for step in range(n_steps):
                out = simulate_one_step_headless(step)

                density_error_ts[step] = out["density_error"]
                psi_ts[step] = out["psi"]
                energy_ts[step, :] = out["energy"]
                fire_area_ts[step] = out["fire_area"]
                water_ts[step, :] = out["water"]
                step_runtime_ts[step] = out["step_runtime"]
                avg_runtime_per_robot_ts[step] = out["avg_runtime_per_robot"]
                planner_runtime_ts[step] = out["planner_runtime"]

            np.savez_compressed(
                os.path.join(save_dir, f"run_{run_idx:03d}.npz"),
                time=time_vec,
                density_error=density_error_ts,
                psi=psi_ts,
                energy=energy_ts,
                fire_area=fire_area_ts,
                water=water_ts,
                step_runtime=step_runtime_ts,
                avg_runtime_per_robot=avg_runtime_per_robot_ts,
                planner_runtime=planner_runtime_ts,
            )

            if verbose:
                print(
                    f"run {run_idx:03d} saved | "
                    f"mean psi = {psi_ts.mean():.6f} | "
                    f"mean fire area = {fire_area_ts.mean():.6f} | "
                    f"mean runtime/robot = {avg_runtime_per_robot_ts.mean():.6f}s"
                )

            overall_runtime_sum += avg_runtime_per_robot_ts.mean()
            overall_runtime_count += 1

        overall_avg_runtime_per_step_per_robot = overall_runtime_sum / max(1, overall_runtime_count)

        print(f"Overall avg runtime per step per robot over all runs: {overall_avg_runtime_per_step_per_robot:.6f}s")
        print(f"Saved {n_runs} runs to: {save_dir}")

        return {
            "save_dir": save_dir,
            "n_runs": n_runs,
            "n_steps": n_steps,
            "time": time_vec,
            "overall_avg_runtime_per_step_per_robot": overall_avg_runtime_per_step_per_robot,
        }

    finally:
        shutdown_background_planner()
        RRT_ASYNC_ENABLED = old_async

def animate(frame):
    global time_history, psi_history, error_history, slack_history, fire_area_history, contour
    global traj_x_list, traj_y_list, total_time, total_time_count
    global E, Water, Charging_flag, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs
    global T_fire, S_fire, Mask_A, Mask_Fire, Mask_target_zone, Mask, rho_d, Mask_f, rho_d_f
    global unsafe_contour, target_zone_contour, fire_img, fire_contour

    if hasattr(animate, "primed") or frame != 0:
        current_time = (frame + 1) * dt
        time_history.append(current_time)
        tmp = [pdf.compute_pdf_random() for pdf in pdfs]
        robot_positions = np.array([t[1] for t in tmp])

        start = time.perf_counter()

        Mask_A, Mask_A_Inflate, Mask_Fire, Mask_target_zone, rho_d = central_compute_fire_zones(T_fire)
        Mask = Mask_A
        Mask_f = Mask_A.ravel().astype(float)
        rho_d_f = rho_d.ravel().astype(float)

        coeffs, alpha_v_value, alpha_h_value = sumfield_coeffs_fd5(
            robot_positions,
            Water,
            Xf=Xf, Yf=Yf, iR=iR, iL=iL, iU=iU, iD=iD,
            Nx=Nx, Ny=Ny, L=L, h=h, sigma=pdfs[0].sigma, lap_coeff=lap_coeff,
            rho_d_f=rho_d_f, Mask_f=Mask_f,
            alpha_v=alpha_v, alpha_h=alpha_h, epsilon=epsilon,
        )
        w_cap = np.minimum(SPRAY_RATE_MAX, Water / max(dt, 1e-12))
        controller.set_params_from_coeffs(coeffs, alpha_v_value, alpha_h_value, w_cap=w_cap)

        if USE_ENERGY_PLANNING:
            poll_background_rrt_plan()

            if should_submit_rrt_plan(frame):
                submit_background_rrt_plan(frame, robot_positions, Mask_A, Mask_A_Inflate)

            active_plan = get_active_rrt_plan(robot_positions)

            if active_plan is not None:
                rrt_paths = active_plan["paths"]
                rrt_path_lengths = active_plan["path_lengths"]
                rrt_dv_first = active_plan["dv_first"]
                rrt_first_step_vecs = active_plan["first_step_vecs"]
            else:
                rrt_paths = None
                rrt_path_lengths = np.zeros(num_pdfs, dtype=float)
                rrt_dv_first = np.zeros((num_pdfs, 2), dtype=float)
                rrt_first_step_vecs = np.zeros((num_pdfs, 2), dtype=float)

            if SHOW_RRT_PATHS:
                update_rrt_path_overlay(rrt_paths, path_lines, path_ends)
            else:
                clear_rrt_path_overlay(path_lines, path_ends)

            if USE_BATTERY_CBF:
                Ve_energy, dVex_energy, dVey_energy = planner_outputs_to_energy_terms(
                    robot_positions=robot_positions,
                    dv_first=rrt_dv_first,
                    path_lengths=rrt_path_lengths,
                    u_max=u_max,
                    c_e=c_e,
                    c_0=c_0,
                )
                controller.set_energy_params(
                    Ve=Ve_energy,
                    dVex=dVex_energy,
                    dVey=dVey_energy,
                    Ecur=E,
                    Emin=E_min,
                    alpha_e=alpha_e,
                    c_e=c_e,
                )
                hE = E - E_min - Ve_energy
            else:
                hE = E - E_min
        else:
            rrt_paths = None
            rrt_path_lengths = None
            rrt_dv_first = None
            rrt_first_step_vecs = None
            hE = E - E_min
            clear_rrt_path_overlay(path_lines, path_ends)

        u_opt, wdot_opt, s_val, t_val = controller.solve()
        print(wdot_opt)
        u_s_field = fire_suppression_input(
            T_fire,
            robot_positions=robot_positions,
            spray_rates=wdot_opt,
            water_levels=Water,
        )
        T_fire, S_fire = step_fire_dynamics(T_fire, S_fire, u_s_field, dt, h)
        Mask_A, Mask_A_Inflate, Mask_Fire, Mask_target_zone, rho_d = central_compute_fire_zones(T_fire)
        Mask = Mask_A
        Mask_f = Mask_A.ravel().astype(float)
        rho_d_f = rho_d.ravel().astype(float)

        curr_time = time.perf_counter() - start
        total_time += curr_time
        total_time_count += 1

        for patch in list(ax_scatter.patches):
            patch.remove()
        unsafe_contour = remove_contour_set(unsafe_contour)
        fire_contour = remove_contour_set(fire_contour)
        target_zone_contour = remove_contour_set(target_zone_contour)
        contour = remove_contour_set(contour)

        fire_img.set_data(T_fire if FIRE_SHOW_TEMPERATURE else rho_d)
        if FIRE_SHOW_TEMPERATURE:
            fire_img.set_clim(FIRE_COLORBAR_MIN, FIRE_COLORBAR_MAX)
            if fire_cbar is not None:
                fire_cbar.update_normal(fire_img)
        else:
            fire_img.set_clim(
                vmin=float(np.min(rho_d)),
                vmax=float(np.max(rho_d)) + 1e-12,
            )

        unsafe_contour = ax_scatter.contour(X, Y, Mask_A.astype(float), colors="red", levels=[0.5], linewidths=1.5)
        fire_contour = ax_scatter.contour(X, Y, Mask_Fire.astype(float), colors="pink", levels=[0.5], linewidths=1.0)
        target_zone_contour = ax_scatter.contour(X, Y, Mask_target_zone.astype(float), colors="cyan", levels=[0.5], linewidths=1.2, linestyles='dotted')
        ax_scatter.add_patch(Circle((charger_x, charger_y), charger_radius, edgecolor='yellow', facecolor='yellow', alpha=0.25, linewidth=2))

        def direct_to_goal_dv(x, c):
            v = np.asarray(c, float) - np.asarray(x, float)
            n = np.linalg.norm(v)
            return v / n if n > 1e-12 else np.zeros(2)

        if USE_ENERGY_PLANNING:
            r_goal_switch = charger_radius + 0.35
            charger_center = np.array([charger_x, charger_y], dtype=float)
            for i in range(num_pdfs):
                d = np.linalg.norm(robot_positions[i] - charger_center)
                if (hE[i] <= 0.05) or Charging_flag[i] or d <= charger_radius:
                    if d <= r_goal_switch:
                        if E[i] <= 0.91:
                            Charging_flag[i] = True
                            if d <= charger_radius:
                                u_opt[i] = [0, 0]
                                E[i] = min(1.0, E[i] + 0.5 * dt)
                            else:
                                v = charger_center - robot_positions[i]
                                n = np.linalg.norm(v)
                                u_des = (u_max * v / n) if n > 1e-12 else np.zeros(2, dtype=float)
                                u_opt[i] = u_des
                        else:
                            Charging_flag[i] = False
        else:
            Charging_flag[:] = False

        u_norm = np.sqrt(u_opt[:, 0] ** 2 + u_opt[:, 1] ** 2)
        E[:] = E - (c_e * u_norm + c_0) * dt
        E[:] = np.clip(E, 0.0, 1.0)

        energy_time.append(current_time)
        energy_history.append(E.copy())
        energy_min_history.append(float(E.min()))
        energy_avg_history.append(float(E.mean()))

        Water[:] = np.clip(Water - wdot_opt * dt, 0.0, WATER_MAX)
        water_time.append(current_time)
        water_history.append(Water.copy())
        water_min_history.append(float(Water.min()))
        water_avg_history.append(float(Water.mean()))

        for i, pdf in enumerate(pdfs):
            pdf.update(u_opt[i], dt, x_grid, y_grid, Charging_flag[i])

        rho_now = np.sum([pdf.field for pdf in pdfs], axis=0)

        contour = ax_scatter.contour(
            X, Y, rho_now,
            levels=contour_levels,
            colors='black',
            linewidths=1.5,
            linestyles='dashed',
            alpha=0.5
        )

        robot_positions = np.array([pdf.position for pdf in pdfs], dtype=float)
        scat.set_offsets(robot_positions)
        for i, txt in enumerate(energy_texts):
            txt.set_position((robot_positions[i, 0], robot_positions[i, 1] + 0.2))
            txt.set_text(f"{E[i]:.2f}")
        for i, txt in enumerate(water_texts):
            txt.set_position((robot_positions[i, 0], robot_positions[i, 1] - 0.2))
            txt.set_text(f"{Water[i]:.1f}")

        for i in range(num_pdfs):
            traj_x_list[i].append(pdfs[i].position[0])
            traj_y_list[i].append(pdfs[i].position[1])
            traj_lines[i].set_data(traj_x_list[i], traj_y_list[i])

        ax_scatter.set_title(f"Swarm + Fire Field at t={frame * dt:.2f}", fontsize=font_size)

        rho_raw = np.sum([pdf.field for pdf in pdfs], axis=0)
        rho_eff = np.sum(
            [(Water[i] / WATER_MAX) * periodic_convolve_field(pdf.field) for i, pdf in enumerate(pdfs)],
            axis=0,
        )
        psi_val = epsilon - np.sum(rho_raw[Mask] ** 2) * h ** 2
        psi_history.append(psi_val)
        error_norm = h ** 2 * np.sum((rho_d - rho_eff) ** 2)
        error_history.append(error_norm)
        fire_area_history.append(h ** 2 * np.sum(T_fire > T_ignition))
        slack_history.append(s_val)

        if USE_ENERGY_PLANNING and USE_BATTERY_CBF:
            min_hE_str = f"{np.min(hE):.4f}"
        else:
            min_hE_str = "N/A"
        unsafe_frac = float(np.mean(Mask_A))
        max_temp = float(np.max(T_fire))
        min_temp = float(np.min(T_fire))
        min_temp = float(np.min(T_fire))
        print(f"Iteration {frame} | t={current_time:.2f} | slack={s_val:.4f} | solver_t={t_val:.4f} | min_hE={min_hE_str} | min_W={float(Water.min()):.2f} | Tmax={max_temp:.3f} | unsafe_frac={unsafe_frac:.3f} | step_time={curr_time:.4f}s")
        print(np.min(S_fire))
    animate.primed = True
    return [fire_img, scat, *traj_lines, *path_lines, *path_ends]


def plot_summary_histories():
    fig_hist, (ax_err, ax_psi, ax_fire) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax_err.plot(time_history, error_history, linewidth=2)
    ax_err.set_ylabel(r"$\|\rho_d - \rho_{\mathrm{eff}}\|_{L^2}^2$")
    ax_err.set_title("Swarm Tracking, Safety, and Fire Area History")
    ax_err.grid(True, alpha=0.3)

    ax_psi.plot(time_history, psi_history, linewidth=2)
    ax_psi.axhline(0.0, linestyle='--', linewidth=1.5)
    ax_psi.set_ylabel(r"$\psi(\rho)$")
    ax_psi.grid(True, alpha=0.3)

    ax_fire.plot(time_history, fire_area_history, linewidth=2, label="With drones")
    baseline = np.loadtxt("fire_area_baseline.csv", delimiter=",", skiprows=1)
    baseline = baseline[baseline[:, 0] <= time_history[-1]]
    ax_fire.plot(baseline[:, 0], baseline[:, 1], linewidth=2, linestyle="--", label="No drones baseline")
    ax_fire.legend()
    ax_fire.set_xlabel("Time")
    ax_fire.set_ylabel(r"Area$(T > T_{ign})$")
    ax_fire.grid(True, alpha=0.3)

    fig_hist.tight_layout()
    plt.show()


def main():
    global planner_last_submit_frame, planner_last_ready_frame

    setup_background_planner()

    if USE_ENERGY_PLANNING:
        print("Computing initial RRT plan...")
        init_plan = compute_initial_rrt_plan()
        planner_last_submit_frame = 0
        set_latest_rrt_plan(init_plan)
        planner_last_ready_frame = 0

    try:
        ani = animation.FuncAnimation(
            fig, animate, frames=FRAMES, interval=INTERVAL_MS,
            blit=False, repeat=False
        )
        plt.show()
        plot_summary_histories()
        if total_time_count > 0:
            print(f"Average step time: {total_time / total_time_count:.5f}s")
    finally:
        shutdown_background_planner()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    # main()   # interactive animation

    results = run_ensemble_timeseries_fire(
        n_runs=100,
        sim_time=50.0,
        save_dir="fire_ensemble_100runs_50s_central_2_test",
        verbose=True,
        force_sync_rrt=False,
    )
