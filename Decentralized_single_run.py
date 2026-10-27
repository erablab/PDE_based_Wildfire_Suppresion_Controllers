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
import scipy.sparse as sp
from rrt_path_cbf import plan_all_robots_rrt_parallel
from scipy.ndimage import binary_dilation, gaussian_filter, distance_transform_edt
from concurrent.futures import ProcessPoolExecutor

total_time = 0.0
total_time_count = 0

# ----------------------- Parameters -----------------------
L = 3.5
Nx, Ny = 36,36
Ngrid = Nx * Ny
h = L / (Nx - 1)
dt = 0.05

alpha_v = 1
lap_coeff = 0.045
gemma = 10000
epsilon = 0.001
alpha_h = 0.1
sigma_energy = 0.14
gemma_E = 100000
SPRAY_REG = 1

MEASURE_NOISE = True
MOTION_NOISE = True

# ----------------------- Feature Switches -----------------------
USE_ENERGY_PLANNING = True  # master switch for RRT-based energy planning / charger behavior
USE_BATTERY_CBF = True       # battery CBF in controller; only active if USE_ENERGY_PLANNING is True
SHOW_RRT_PATHS = True        # animation overlay for RRT paths; only active if USE_ENERGY_PLANNING is True
# ----------------------- Async RRT replanning -----------------------
RRT_REPLAN_EVERY_FRAMES = 5     # submit a new RRT plan every this many frames
RRT_ASYNC_ENABLED = True         # run planner in background so animation loop does not wait
RRT_ASYNC_WORKERS = 1            # one persistent background process

# ----------------------- Local sensing / decentralized control -----------------------
DETECTION_RADIUS = 1.1         # each robot senses temperature and neighbor positions inside this radius
MEMORY_HORIZON_SEC = 1.0         # past sensed temperature persists for 1 second with linear decay
SHOW_DECENTRALIZED_TEMP_MAP = True  # False: show true global temperature, True: show summed decentralized sensed map

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

E = np.array([0.25, 0.9, 0.7, 0.5], dtype=float)

SUPPRESSION_SIGMA = 0.10   # paper uses sigma_s = 0.1
SUPPRESSION_FLOW = -1.0  * 2  # negative = cooling / extinguishing
WATER_MAX = 100.0
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

def compute_target_pdf_from_temperature_tail(T):
    # Tunable tail parameters
    FIRE_TARGET_TAIL_SIGMA = 0.75   # spatial decay length (same units as h, likely meters)
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

    return target / total * 5

T_fire, S_fire = initialize_fire_fields(X, Y)
Mask_A, Mask_Fire, Mask_target_zone, rho_d = central_compute_fire_zones(T_fire)
Mask = Mask_A

# ----------------------- Load Initial Data -----------------------
with open("initial_robots.json", "r") as file:
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

    def update(self, u_opt_i, dt, x_grid, y_grid):
        local_u = np.asarray(u_opt_i, dtype=float)
        if MOTION_NOISE:
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

# ----------------------- Local memory / neighbor state -----------------------
temp_memory_base = np.zeros((num_pdfs, Ny, Nx), dtype=float)
temp_memory_weight = np.zeros((num_pdfs, Ny, Nx), dtype=float)
neighbor_velocity_guess = np.zeros((num_pdfs, 2), dtype=float)  # placeholder for later scripts

# ----------------------- Energy State -----------------------
u_max = 0.2
alpha_e = 0.5
E0 = 0.4
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
RRT_K = 20
RRT_BASE_SEED = 11
RRT_SEGMENT_LEN = u_max * dt * 10 * 2
RRT_USE_SPACE_CBF = True

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


def submit_background_rrt_plan(frame, robot_positions, mask_a, mask_fire):
    global planner_future, planner_last_submit_frame

    if not USE_ENERGY_PLANNING:
        return
    if planner_future is not None and not planner_future.done():
        return

    if RRT_ASYNC_ENABLED:
        setup_background_planner()
        planner_future = planner_executor.submit(
            plan_all_robots_rrt_parallel,
            init_xy_all=np.asarray(robot_positions, float),
            bounds=(-L / 2, L / 2, -L / 2, L / 2),
            X=X,
            Y=Y,
            invariance_mask=np.asarray(mask_a, bool),
            invariance_mask_inflated=np.asarray(mask_fire, bool),
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
            epsilon=epsilon / num_pdfs,
            alpha_h=alpha_h,
            parallel=False,
            n_workers=1,
        )
    else:
        planner_future = None
        set_latest_rrt_plan(
            plan_all_robots_rrt_parallel(
                init_xy_all=np.asarray(robot_positions, float),
                bounds=(-L / 2, L / 2, -L / 2, L / 2),
                X=X,
                Y=Y,
                invariance_mask=np.asarray(mask_a, bool),
                invariance_mask_inflated=np.asarray(mask_fire, bool),
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
                epsilon=epsilon/ num_pdfs,
                alpha_h=alpha_h,
                parallel=False,
                n_workers=1,
            )
        )

    planner_last_submit_frame = frame


def poll_background_rrt_plan():
    global planner_future
    if planner_future is None or not planner_future.done():
        return
    try:
        set_latest_rrt_plan(planner_future.result())
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
        epsilon=epsilon/ num_pdfs,
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

class LocalInvarianceController:
    def __init__(self, use_battery_cbf=True):
        self.use_battery_cbf = bool(use_battery_cbf)
        M = self.model = mf.Model("LocalController")

        self.u = M.variable("u", 2, mf.Domain.inRange(-u_max, u_max))
        self.w_dot = M.variable("w_dot", 1, mf.Domain.inRange(0.0, SPRAY_RATE_MAX))
        self.s = M.variable("s", 1, mf.Domain.greaterThan(0.0))
        self.t = M.variable("t", 1, mf.Domain.greaterThan(0.0))
        #self.z = M.variable("z", 1, mf.Domain.greaterThan(0.0))

        ux = self.u.index(0)
        uy = self.u.index(1)
        w = self.w_dot.index(0)

        M.constraint("u_norm", mf.Expr.vstack(self.t, self.u), mf.Domain.inQCone())
        #M.constraint("z_norm", mf.Expr.vstack(self.z, self.u), mf.Domain.inQCone())

        self.a_vx = M.parameter("a_vx", 1)
        self.a_vy = M.parameter("a_vy", 1)
        self.a_vw = M.parameter("a_vw", 1)
        self.c_v = M.parameter("c_v", 1)
        self.alpha_V = M.parameter("alpha_V", 1)
        self.w_cap = M.parameter("w_cap", 1)

        self.f_all = M.parameter("f_all", [Ngrid, 1])
        self.rho_i = M.parameter("rho_i", [Ngrid, 1])
        self.b_i = M.parameter("b_i", [Ngrid, 1])
        self.lambda_i = M.parameter("lambda_i", 1)
        self.delta_i = M.parameter("delta_i", 1)

        obj = mf.Expr.add(self.t.index(0), mf.Expr.mul(self.s.index(0), float(gemma)))
        obj = mf.Expr.add(obj, mf.Expr.mul(float(SPRAY_REG), w))

        M.constraint("w_cap_con", mf.Expr.sub(w, self.w_cap.index(0)), mf.Domain.lessThan(0.0))

        v_dot = mf.Expr.add(
            mf.Expr.add(
                mf.Expr.add(mf.Expr.mul(self.a_vx.index(0), ux), mf.Expr.mul(self.a_vy.index(0), uy)),
                mf.Expr.mul(self.a_vw.index(0), w),
            ),
            self.c_v.index(0),
        )

        ux_rho = mf.Expr.mul(D_x_const, mf.Expr.mul(self.rho_i, ux))
        uy_rho = mf.Expr.mul(D_y_const, mf.Expr.mul(self.rho_i, uy))
        rho_dot_self = mf.Expr.add(mf.Expr.add(ux_rho, uy_rho), self.b_i)
        h_dot_self = mf.Expr.mul(-2.0, mf.Expr.dot(self.f_all, rho_dot_self))

        M.constraint(
            "CLF",
            mf.Expr.sub(mf.Expr.add(self.alpha_V.index(0), v_dot), self.s.index(0)),
            mf.Domain.lessThan(0.0),
        )
        M.constraint(
            "CBF",
            mf.Expr.sub(mf.Expr.add(self.lambda_i.index(0), h_dot_self), self.delta_i.index(0)),
            mf.Domain.greaterThan(0.0),
        )


        if self.use_battery_cbf:
            self.energy_slack = M.variable("s_e", 1, mf.Domain.greaterThan(0.0))
            self.Ve = M.parameter("Ve", 1)
            self.dVex = M.parameter("dVex", 1)
            self.dVey = M.parameter("dVey", 1)
            self.Ecur = M.parameter("Ecur", 1)
            self.Emin = M.parameter("Emin", 1)
            self.alpha_e = M.parameter("alpha_e", 1)
            self.c_e = M.parameter("c_e", 1)

            lhs = mf.Expr.add(
                mf.Expr.add(mf.Expr.mul(self.c_e.index(0), self.t.index(0)), c_0),
                mf.Expr.mul(
                    0.5,
                    mf.Expr.add(
                        mf.Expr.mul(self.dVex.index(0), ux),
                        mf.Expr.mul(self.dVey.index(0), uy),
                    ),
                ),
            )
            rhs = mf.Expr.mul(
                self.alpha_e.index(0),
                mf.Expr.sub(
                    mf.Expr.sub(self.Ecur.index(0), self.Emin.index(0)),
                    self.Ve.index(0),
                ),
            )
            M.constraint(
                "EnergyCBF",
                mf.Expr.add(mf.Expr.sub(rhs, lhs), self.energy_slack.index(0)),
                mf.Domain.greaterThan(0.0),
            )
            obj = mf.Expr.add(obj, mf.Expr.mul(self.energy_slack.index(0), float(gemma_E)))
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
        M.setLogHandler(None)

    def set_params_from_coeffs(self, coeffs, alpha_V_value, w_cap):
        self.a_vx.setValue([float(coeffs["A_vx"])])
        self.a_vy.setValue([float(coeffs["A_vy"])])
        self.a_vw.setValue([float(coeffs["A_vw"])])
        self.c_v.setValue([float(coeffs["c_v"])])
        self.alpha_V.setValue([float(alpha_V_value)])
        self.w_cap.setValue([float(w_cap)])

    def set_local_cbf_params(self, f_all, rho_i, b_i, lambda_i, delta_i):
        self.f_all.setValue(np.asarray(f_all, float).reshape(Ngrid, 1))
        self.rho_i.setValue(np.asarray(rho_i, float).reshape(Ngrid, 1))
        self.b_i.setValue(np.asarray(b_i, float).reshape(Ngrid, 1))
        self.lambda_i.setValue([float(lambda_i)])
        self.delta_i.setValue([float(delta_i)])

    def set_energy_params(self, Ve, dVex, dVey, Ecur, Emin, alpha_e, c_e):
        if not self.use_battery_cbf:
            return
        self.Ve.setValue([float(Ve)])
        self.dVex.setValue([float(dVex)])
        self.dVey.setValue([float(dVey)])
        self.Ecur.setValue([float(Ecur)])
        self.Emin.setValue([float(Emin)])
        self.alpha_e.setValue([float(alpha_e)])
        self.c_e.setValue([float(c_e)])

    def solve(self):
        try:
            self.model.solve()
            return (
                np.asarray(self.u.level(), dtype=float),
                float(self.w_dot.level()[0]),
                float(self.s.level()[0]),
                float(self.t.level()[0]),
            )
        except Exception:
            return np.zeros(2, dtype=float), 0.0, 0.0, 0.0


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
    (np.max(temp_memory_base * temp_memory_weight, axis=0) if SHOW_DECENTRALIZED_TEMP_MAP else T_fire) if FIRE_SHOW_TEMPERATURE else rho_d,
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
    Line2D([0], [0], color='limegreen', lw=1.2, alpha=0.8, label='Neighbor link (within D)'),
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

neighbor_pair_indices = [(i, j) for i in range(num_pdfs) for j in range(i + 1, num_pdfs)]
neighbor_lines = []
for _ in neighbor_pair_indices:
    ln, = ax_scatter.plot([], [], color='limegreen', linewidth=1.2, alpha=0.8, zorder=2)
    neighbor_lines.append(ln)


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


def update_neighbor_overlay(robot_positions, neighbor_sets, neighbor_lines):
    connected_pairs = set()
    for i, nbrs in enumerate(neighbor_sets):
        for j in nbrs:
            if i < j:
                connected_pairs.add((i, j))
            elif j < i:
                connected_pairs.add((j, i))

    for line, (i, j) in zip(neighbor_lines, neighbor_pair_indices):
        if (i, j) in connected_pairs:
            line.set_data([robot_positions[i, 0], robot_positions[j, 0]],
                          [robot_positions[i, 1], robot_positions[j, 1]])
        else:
            line.set_data([], [])


def compute_display_temperature_map():
    if SHOW_DECENTRALIZED_TEMP_MAP:
        return np.max(temp_memory_base * temp_memory_weight, axis=0)
    return T_fire


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


def build_Dx_Dy_sparse(Nx, Ny, dx, dy):
    N = Nx * Ny
    def idx(i, j):
        return i * Ny + j

    rows, cols, vals = [], [], []
    for i in range(Nx):
        for j in range(Ny):
            r = idx(i, j)
            jp = (j + 1) % Ny
            jm = (j - 1) % Ny
            rows += [r, r]
            cols += [idx(i, jp), idx(i, jm)]
            vals += [1.0 / (2 * dx), -1.0 / (2 * dx)]
    Dx = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    rows, cols, vals = [], [], []
    for i in range(Nx):
        for j in range(Ny):
            r = idx(i, j)
            ip = (i + 1) % Nx
            im = (i - 1) % Nx
            rows += [r, r]
            cols += [idx(ip, j), idx(im, j)]
            vals += [1.0 / (2 * dy), -1.0 / (2 * dy)]
    Dy = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return -Dx, -Dy


def to_fusion_sparse(A):
    A = A.tocoo()
    return mf.Matrix.sparse(
        A.shape[0], A.shape[1],
        A.row.astype(np.int32).tolist(),
        A.col.astype(np.int32).tolist(),
        A.data.astype(float).tolist(),
    )


def build_L(Nx, Ny, h):
    N = Nx * Ny
    idx = lambda i, j: i * Ny + j
    rows, cols, data = [], [], []
    for i in range(Nx):
        for j in range(Ny):
            c = idx(i, j)
            rows += [c] * 5
            cols += [
                c,
                idx((i + 1) % Nx, j), idx((i - 1) % Nx, j),
                idx(i, (j + 1) % Ny), idx(i, (j - 1) % Ny),
            ]
            data += [-4.0 / h**2] + [1.0 / h**2] * 4
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


Dx_sp, Dy_sp = build_Dx_Dy_sparse(Nx, Ny, h, h)
D_x_const = to_fusion_sparse(Dx_sp)
D_y_const = to_fusion_sparse(Dy_sp)
Lap_sp = build_L(Nx, Ny, h)


def lap_sparse(r):
    return lap_coeff * (Lap_sp @ np.asarray(r, float).reshape(-1)).reshape(Ngrid, 1)


class WorstCaseNeighborOpt:
    def __init__(self, Kmax, eps_reg=1e-8):
        M = self.model = mf.Model("WorstCaseNeighbor")
        self.u = M.variable("u", 2 * Kmax, mf.Domain.inRange(-u_max, u_max))
        self.f_all = M.parameter("f_all", [Ngrid, 1])
        self.alpha_h = M.parameter("alpha_h", 1)
        self.b_vec = M.parameter("b_vec", [Ngrid, 1])
        self.rho_neigh = [M.parameter(f"rho_neigh_{j}", [Ngrid, 1]) for j in range(Kmax)]

        terms = []
        for j in range(Kmax):
            ux_term = mf.Expr.mul(D_x_const, mf.Expr.mul(self.rho_neigh[j], self.u.index(2 * j)))
            uy_term = mf.Expr.mul(D_y_const, mf.Expr.mul(self.rho_neigh[j], self.u.index(2 * j + 1)))
            terms.append(mf.Expr.add(ux_term, uy_term))
        divergence = mf.Expr.add(terms) if terms else mf.Expr.constTerm(Ngrid, 1, 0.0)
        hdot = mf.Expr.mul(-2.0, mf.Expr.dot(self.f_all, mf.Expr.add(divergence, self.b_vec)))

        t_reg = M.variable("t_reg", 1, mf.Domain.greaterThan(0.0))
        M.constraint(
            "reg_norm",
            mf.Expr.vstack(mf.Expr.constTerm(0.5), t_reg, self.u),
            mf.Domain.inRotatedQCone(),
        )
        M.objective(
            mf.ObjectiveSense.Minimize,
            mf.Expr.add(self.alpha_h.index(0), mf.Expr.add(hdot, mf.Expr.mul(eps_reg, t_reg.index(0)))),
        )
        M.setLogHandler(None)

    def set_params(self, f_all, alpha_h, rho_list, b_vec):
        self.f_all.setValue(np.asarray(f_all, float).reshape(Ngrid, 1))
        self.alpha_h.setValue([float(alpha_h)])
        self.b_vec.setValue(np.asarray(b_vec, float).reshape(Ngrid, 1))
        zeros = np.zeros((Ngrid, 1), dtype=float)
        for j, prm in enumerate(self.rho_neigh):
            if j < len(rho_list):
                prm.setValue(np.asarray(rho_list[j], float).reshape(Ngrid, 1))
            else:
                prm.setValue(zeros)

    def solve(self):
        self.model.solve()
        return float(self.model.primalObjValue())


class MaxSelfSafetyOpt:
    def __init__(self):
        M = self.model = mf.Model("MaxSelfSafety")
        self.u = M.variable("u", 2, mf.Domain.inRange(-u_max, u_max))
        self.f_all = M.parameter("f_all", [Ngrid, 1])
        self.rho_i = M.parameter("rho_i", [Ngrid, 1])
        self.b_i = M.parameter("b_i", [Ngrid, 1])

        ux_term = mf.Expr.mul(D_x_const, mf.Expr.mul(self.rho_i, self.u.index(0)))
        uy_term = mf.Expr.mul(D_y_const, mf.Expr.mul(self.rho_i, self.u.index(1)))
        rho_dot = mf.Expr.add(mf.Expr.add(ux_term, uy_term), self.b_i)
        hdot = mf.Expr.mul(-2.0, mf.Expr.dot(self.f_all, rho_dot))
        M.objective(mf.ObjectiveSense.Maximize, hdot)
        M.setLogHandler(None)

    def set_params(self, f_all, rho_i, b_i):
        self.f_all.setValue(np.asarray(f_all, float).reshape(Ngrid, 1))
        self.rho_i.setValue(np.asarray(rho_i, float).reshape(Ngrid, 1))
        self.b_i.setValue(np.asarray(b_i, float).reshape(Ngrid, 1))

    def solve(self):
        self.model.solve()
        return float(self.model.primalObjValue())


K_LOCAL_MAX = max(0, num_pdfs - 1)
worst_case_neighbor_opt = WorstCaseNeighborOpt(K_LOCAL_MAX)
max_self_safety_opt = MaxSelfSafetyOpt()
controllers = [LocalInvarianceController(use_battery_cbf=(USE_ENERGY_PLANNING and USE_BATTERY_CBF)) for _ in range(num_pdfs)]
print("Controllers built")


def compute_local_relaxation_terms(cbf_data):
    rho_neigh = cbf_data["rho_neigh"]
    alpha_h_local = float(cbf_data["alpha_h_local"])
    if len(rho_neigh) > 0:
        worst_case_neighbor_opt.set_params(
            f_all=cbf_data["f_all"],
            alpha_h=alpha_h_local,
            rho_list=rho_neigh,
            b_vec=cbf_data["b_neigh"],
        )
        lambda_i = worst_case_neighbor_opt.solve()
    else:
        lambda_i = alpha_h_local

    max_self_safety_opt.set_params(
        f_all=cbf_data["f_all"],
        rho_i=cbf_data["rho_i"],
        b_i=cbf_data["b_i"],
    )
    h_i_best = max_self_safety_opt.solve()
    delta_i = min(0.0, lambda_i + h_i_best)
    return lambda_i, h_i_best, delta_i


def euclidean_disk_mask(pos_xy, radius):
    return (((X - pos_xy[0]) ** 2 + (Y - pos_xy[1]) ** 2) <= radius ** 2)


def compute_neighbor_sets(robot_positions, radius):
    robot_positions = np.asarray(robot_positions, float)
    neigh = []
    for i in range(len(robot_positions)):
        d = np.linalg.norm(robot_positions - robot_positions[i], axis=1)
        ids = np.where((d <= radius) & (np.arange(len(robot_positions)) != i))[0]
        neigh.append(ids.tolist())
    return neigh


def update_local_temperature_memory(T_field, robot_positions):
    global temp_memory_base, temp_memory_weight
    decay_step = dt / max(MEMORY_HORIZON_SEC, 1e-12)
    temp_memory_weight = np.maximum(temp_memory_weight - decay_step, 0.0)

    for i, pos in enumerate(robot_positions):
        detect_mask = euclidean_disk_mask(pos, DETECTION_RADIUS)
        temp_memory_base[i, detect_mask] = T_field[detect_mask]
        temp_memory_weight[i, detect_mask] = 1.0
        stale = temp_memory_weight[i] <= 0.0
        temp_memory_base[i, stale] = 0.0


def get_shared_temperature_estimate(robot_idx, neighbor_sets):
    source_ids = [robot_idx] + list(neighbor_sets[robot_idx])
    shared = [temp_memory_base[j] * temp_memory_weight[j] for j in source_ids]
    if len(shared) == 1:
        return np.asarray(shared[0], float).copy()
    return np.maximum.reduce(shared)


def local_problem_data_fd5(
    *,
    positions_local,
    velocities_local,
    water_local,
    Xf, Yf, iR, iL, iU, iD, Nx, Ny, L, h, sigma, lap_coeff,
    rho_d_f_local, Mask_f_local, alpha_v, alpha_h, epsilon,
):
    M_local = positions_local.shape[0]
    cx = positions_local[:, 0][:, None]
    cy = positions_local[:, 1][:, None]
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

    rhoC_2d = rhoC.reshape(M_local, Ny, Nx)
    gx_2d = gx.reshape(M_local, Ny, Nx)
    gy_2d = gy.reshape(M_local, Ny, Nx)
    b_2d = b.reshape(M_local, Ny, Nx)

    kC = np.array([periodic_convolve_field(rhoC_2d[j]).ravel() for j in range(M_local)])
    kgx = np.array([periodic_convolve_field(gx_2d[j]).ravel() for j in range(M_local)])
    kgy = np.array([periodic_convolve_field(gy_2d[j]).ravel() for j in range(M_local)])
    kb = np.array([periodic_convolve_field(b_2d[j]).ravel() for j in range(M_local)])

    W_local = (np.asarray(water_local, float) / WATER_MAX)[:, None]
    rho_eff = np.sum(W_local * kC, axis=0)
    d_vec = rho_d_f_local - rho_eff

    scale = -2.0 * h * h

    A_vx = float(scale * W_local[0, 0] * np.dot(kgx[0], d_vec))
    A_vy = float(scale * W_local[0, 0] * np.dot(kgy[0], d_vec))
    A_vw = float(scale * np.dot(kC[0], d_vec))

    fixed_rho_eff_dot = np.sum(
        W_local * (kgx * velocities_local[:, 0:1] + kgy * velocities_local[:, 1:2] + kb),
        axis=0,
    )
    c_v = float(scale * np.sum(fixed_rho_eff_dot * d_vec))

    coeffs = dict(A_vx=A_vx, A_vy=A_vy, A_vw=A_vw, c_v=c_v)
    alpha_v_value = float(alpha_v * h * h * np.sum(d_vec * d_vec))

    sum_rho = np.sum(rhoC, axis=0)
    a_vec = sum_rho * Mask_f_local
    alpha_h_local = float(alpha_h * (epsilon / num_pdfs * M_local - h * h * np.sum(a_vec * a_vec)))

    cbf_data = {
        "f_all": (sum_rho * Mask_f_local * (h * h)).reshape(Ngrid, 1),
        "rho_i": rhoC[0].reshape(Ngrid, 1),
        "b_i": b[0].reshape(Ngrid, 1),
        "rho_neigh": [rhoC[j].reshape(Ngrid, 1) for j in range(1, M_local)],
        "b_neigh": (np.sum(b[1:], axis=0).reshape(Ngrid, 1) if M_local > 1 else np.zeros((Ngrid, 1), dtype=float)),
        "alpha_h_local": alpha_h_local,
    }
    return coeffs, alpha_v_value, cbf_data


FRAMES = 10000
INTERVAL_MS = 50

def animate(frame):
    global time_history, psi_history, error_history, slack_history, fire_area_history, contour
    global traj_x_list, traj_y_list, total_time, total_time_count
    global E, Water, Charging_flag, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs
    global T_fire, S_fire, Mask_A, Mask_Fire, Mask_target_zone, Mask, rho_d, Mask_f, rho_d_f
    global unsafe_contour, target_zone_contour, fire_img, fire_contour, neighbor_velocity_guess

    if hasattr(animate, "primed") or frame != 0:
        current_time = (frame + 1) * dt
        time_history.append(current_time)
        tmp = [pdf.compute_pdf_random() for pdf in pdfs]
        robot_positions = np.array([t[1] for t in tmp], dtype=float)

        start = time.perf_counter()

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
            clear_rrt_path_overlay(path_lines, path_ends)

        neighbor_sets = compute_neighbor_sets(robot_positions, DETECTION_RADIUS)
        update_neighbor_overlay(robot_positions, neighbor_sets, neighbor_lines)
        update_local_temperature_memory(T_fire, robot_positions)

        w_cap = np.minimum(SPRAY_RATE_MAX, Water / max(dt, 1e-12))
        u_opt = np.zeros((num_pdfs, 2), dtype=float)
        wdot_opt = np.zeros(num_pdfs, dtype=float)
        local_slacks = []
        local_solver_terms = []

        for i in range(num_pdfs):
            shared_temp_i = get_shared_temperature_estimate(i, neighbor_sets)
            local_mask_A_i, _, _, rho_d_local_i = central_compute_fire_zones(shared_temp_i)

            local_ids = [i] + list(neighbor_sets[i])
            positions_local = robot_positions[local_ids]
            velocities_local = neighbor_velocity_guess[local_ids].copy()
            velocities_local[0, :] = 0.0  # self velocity is the decision variable, not a fixed neighbor input
            water_local = Water[local_ids]

            coeffs_i, alpha_v_i, cbf_data_i = local_problem_data_fd5(
                positions_local=positions_local,
                velocities_local=velocities_local,
                water_local=water_local,
                Xf=Xf, Yf=Yf, iR=iR, iL=iL, iU=iU, iD=iD,
                Nx=Nx, Ny=Ny, L=L, h=h, sigma=pdfs[0].sigma, lap_coeff=lap_coeff,
                rho_d_f_local=rho_d_local_i.ravel().astype(float),
                Mask_f_local=local_mask_A_i.ravel().astype(float),
                alpha_v=alpha_v, alpha_h=alpha_h, epsilon=epsilon,
            )
            lambda_i, h_i_best, delta_i = compute_local_relaxation_terms(cbf_data_i)

            controllers[i].set_params_from_coeffs(coeffs_i, alpha_v_i, w_cap=float(w_cap[i]))
            controllers[i].set_local_cbf_params(
                f_all=cbf_data_i["f_all"],
                rho_i=cbf_data_i["rho_i"],
                b_i=cbf_data_i["b_i"],
                lambda_i=lambda_i,
                delta_i=delta_i,
            )

            if USE_ENERGY_PLANNING and USE_BATTERY_CBF:
                controllers[i].set_energy_params(
                    Ve=Ve_energy[i],
                    dVex=dVex_energy[i],
                    dVey=dVey_energy[i],
                    Ecur=E[i],
                    Emin=E_min + 0.05,
                    alpha_e=alpha_e,
                    c_e=c_e,
                )

            u_i, w_i, s_i, t_i = controllers[i].solve()
            u_opt[i] = u_i
            wdot_opt[i] = w_i
            local_slacks.append(s_i)
            local_solver_terms.append(t_i)

        s_val = float(np.max(local_slacks)) if local_slacks else 0.0
        t_val = float(np.max(local_solver_terms)) if local_solver_terms else 0.0

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

        for patch in list(ax_scatter.patches):
            patch.remove()
        unsafe_contour = remove_contour_set(unsafe_contour)
        fire_contour = remove_contour_set(fire_contour)
        target_zone_contour = remove_contour_set(target_zone_contour)
        contour = remove_contour_set(contour)

        display_temp_map = compute_display_temperature_map()
        fire_img.set_data(display_temp_map if FIRE_SHOW_TEMPERATURE else rho_d)
        if FIRE_SHOW_TEMPERATURE:
            display_vmax = max(FIRE_COLORBAR_MAX, float(np.max(display_temp_map)) + 1e-12)
            fire_img.set_clim(FIRE_COLORBAR_MIN, display_vmax)
            if fire_cbar is not None:
                fire_cbar.update_normal(fire_img)
                fire_cbar.set_label('Temperature (decentralized sum)' if SHOW_DECENTRALIZED_TEMP_MAP else 'Temperature', fontsize=12)
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
            neighbor_cmd_guess = u_opt.copy()
            for i in range(num_pdfs):
                d = np.linalg.norm(robot_positions[i] - charger_center)
                if (hE[i] <= 0.05) or Charging_flag[i] or d <= charger_radius:
                    if d <= r_goal_switch:
                        if E[i] <= 0.91:
                            Charging_flag[i] = True
                            if d <= charger_radius:
                                u_opt[i] = [0,0]
                                neighbor_cmd_guess[i] = u_opt[i]
                                E[i] = min(1.0, E[i] + 0.5 * dt)
                            else:
                                v = charger_center - robot_positions[i]
                                n = np.linalg.norm(v)
                                u_des = (u_max * v / n) if n > 1e-12 else np.zeros(2, dtype=float)
                                u_opt[i] = u_des
                                neighbor_cmd_guess[i] = u_opt[i]
                        else:
                            Charging_flag[i] = False
        else:
            Charging_flag[:] = False

        neighbor_velocity_guess[:] = u_opt
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
            pdf.update(u_opt[i], dt, x_grid, y_grid)

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

        map_tag = 'decentralized sensed sum' if SHOW_DECENTRALIZED_TEMP_MAP else 'global'
        ax_scatter.set_title(f"Swarm + Fire Field at t={frame * dt:.2f} [{map_tag}]", fontsize=font_size)

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
        print(f"Iteration {frame} | t={current_time:.2f} | local_slack_max={s_val:.4f} | local_solver_t_max={t_val:.4f} | min_hE={min_hE_str} | min_W={float(Water.min()):.2f} | Tmax={max_temp:.3f} | unsafe_frac={unsafe_frac:.3f} | step_time={curr_time:.4f}s")

    animate.primed = True
    return [fire_img, scat, *traj_lines, *path_lines, *path_ends, *neighbor_lines]


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

    init_positions = np.array([pdf.position for pdf in pdfs], dtype=float)
    update_local_temperature_memory(T_fire, init_positions)
    update_neighbor_overlay(init_positions, compute_neighbor_sets(init_positions, DETECTION_RADIUS), neighbor_lines)
    if FIRE_SHOW_TEMPERATURE:
        fire_img.set_data(compute_display_temperature_map())
        if fire_cbar is not None:
            fire_cbar.set_label('Temperature (decentralized sum)' if SHOW_DECENTRALIZED_TEMP_MAP else 'Temperature', fontsize=12)

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
    main()
