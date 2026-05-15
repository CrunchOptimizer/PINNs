# -*- coding: utf-8 -*-
"""
Stokes Wedge Solver: Multi-Stage Training (Resumable)
-----------------------------------------------------
Phase 1: 2000 Iterations (Damping 1e-11)
Phase 2: 1000 Iterations (Damping 5e-9, Early Stop at 5e-16)

Feature:
- Checks for 'stokes_params_02000.pkl'.
- If found, SKIPS Phase 1 and starts immediately at Phase 2 (Iter 2001).
- If not found, runs Phase 1 first.
- Enforces Pressure Anchor p=0 at (0,2) to fix floating constant.

Outputs:
- checkpoints/stokes_params_xxxx.pkl
- diagnostics_xxxx.png
- history.pkl
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.flatten_util import ravel_pytree
from functools import partial
import numpy as np
import scipy.linalg
import csv
import os
import timeit
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import lineax as lx
import sys

# --- CONFIGURATION ---
jax.config.update("jax_enable_x64", True)

SEED = 1234
DIM = 2
LAYER_SIZES = [DIM, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 3]

# --- PHASE SETTINGS ---
# Phase 1 RANDOM RESAMPLING
P1_ITERS = 500
P1_DAMPING = 1e-12
P1_BATCH_INT = 1000
P1_BATCH_BND = 1000

# Phase 2 APEX HEAVY SAMPLING (RAISED DAMPING)
P2_ITERS = 1000
P2_DAMPING = 5e-9
P2_TOLERANCE = 5e-19 
P2_BATCH_INT = 1000
P2_BATCH_BND = 1000

# Common Settings
CHECKPOINT_FREQ = 100
CHECKPOINT_DIR = "."
HISTORY_FILE = "training_history.pkl"
REF_FILE = "../st_flow.csv"
# Domain (Wedge)
V1 = jnp.array([0.5, 0.0])  # Apex
V2 = jnp.array([0.0, 2.0])  # Top Left
V3 = jnp.array([1.0, 2.0])  # Top Right

# ANCHOR POINT (Matches stokes_pinn.py: x=0, y=2)
V_ANCHOR = jnp.array([0.0, 2.0])

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
# --- PHYSICS & MODEL CORE ---

def act(t): return jnp.tanh(t)
def act_p(t): return 1.0 - jnp.square(jnp.tanh(t))
def act_pp(t):
    tanh_t = jnp.tanh(t)
    return -2.0 * tanh_t * (1.0 - jnp.square(tanh_t))
@jit
def derivative_propagation(params, x):
    z = x
    dz_dx = jnp.eye(len(x))
    d2z_dxx = jnp.zeros((len(x), len(x), len(x)))

    for w, b in params[:-1]:
        z_pre = jnp.dot(w, z) + b
        dz_pre = jnp.dot(w, dz_dx)
        d2z_pre = jnp.einsum("ij,jkl->ikl", w, d2z_dxx)

        z = act(z_pre)
        s_p = act_p(z_pre)
        s_pp = act_pp(z_pre)

        dz_dx = s_p[:, None] * dz_pre
        term1 = s_pp[:, None, None] * jnp.einsum("ij,ik->ijk", dz_pre, dz_pre)
        term2 = s_p[:, None, None] * d2z_pre
        d2z_dxx = term1 + term2

    final_w, final_b = params[-1]
    z = jnp.dot(final_w, z) + final_b
    dz_dx = jnp.dot(final_w, dz_dx)
    d2z_dxx = jnp.einsum("ij,jkl->ikl", final_w, d2z_dxx)
    return z, dz_dx, d2z_dxx


@jit
def interior_res(params, x):
    _, jac, hess = derivative_propagation(params, x)
    u_x, v_y = jac[0, 0], jac[1, 1]
    p_x, p_y = jac[2, 0], jac[2, 1]
    lap_u = hess[0, 0, 0] + hess[0, 1, 1]
    lap_v = hess[1, 0, 0] + hess[1, 1, 1]
    return jnp.stack([p_x - lap_u, p_y - lap_v, u_x + v_y])

@jit
def boundary_res(params, x):
    pred, _, _ = derivative_propagation(params, x)
    u_pred, v_pred = pred[0], pred[1]
    is_top = x[1] > 1.999
    u_lid = 16.0 * (x[0] ** 2) * ((1.0 - x[0]) ** 2)
    u_target = jnp.where(is_top, u_lid, 0.0)
    return jnp.stack([u_pred - u_target, v_pred - 0.0])

@jit
def anchor_res(params, x):
    # Fixes pressure to 0 at the anchor point
    pred, _, _ = derivative_propagation(params, x)
    p_pred = pred[2]
    return jnp.stack([p_pred - 0.0])


# --- SAMPLING UTILS ---
@partial(jit, static_argnums=(1, 2))
def sample_wedge_random(key, n_int, n_bnd):
    k1, k2, k3 = random.split(key, 3)
    r1 = random.uniform(k1, (n_int, 1))
    r2 = random.uniform(k2, (n_int, 1))
    sqrt_r1 = jnp.sqrt(r1)
    w1, w2, w3 = 1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2
    x_int = w1 * V1 + w2 * V2 + w3 * V3

    if n_bnd > 0:
        n_edge = n_bnd // 3
        t = random.uniform(k3, (n_bnd, 1))
        xb_top = V2 + t * (V3 - V2)
        xb_left = V2 + t * (V1 - V2)
        xb_right = V3 + t * (V1 - V3)
        idx = jnp.arange(n_bnd)
        xb = jnp.where((idx < n_edge)[:, None], xb_top, xb_left)
        xb = jnp.where((idx >= 2*n_edge)[:, None], xb_right, xb)
    else:
        xb = jnp.zeros((0, 2))
    return x_int, xb

@partial(jit, static_argnums=(1,))
def sample_apex_heavy(key, n_points):
    k1, k2, k3 = random.split(key, 3)
    n_uni = n_points // 10
    n_apex = n_points - n_uni
    
    x_uni, _ = sample_wedge_random(k1, n_uni, 0)
    
    log_y = random.uniform(k2, (n_apex, 1), minval=jnp.log(1e-7), maxval=jnp.log(2))
    y_apex = jnp.exp(log_y)
    width_at_y = 0.5 * y_apex 
    x_off = random.uniform(k3, (n_apex, 1), minval=-0.5, maxval=0.5) * width_at_y
    x_apex = jnp.hstack([0.5 + x_off, y_apex])
    
    return jnp.concatenate([x_uni, x_apex], axis=0)


# ==============================================================================
# JACOBI PRECONDITIONED GAUSS-NEWTON SOLVER
# ==============================================================================
class JacobiGNSolver:
    def __init__(self, ls_steps):
        self.ls_steps = ls_steps

    def build_J(self, f_params, x_int, x_bnd, unravel_fn):
        def get_int_row(x):
            def scalar_res_fn(fp, i): return interior_res(unravel_fn(fp), x)[i]
            grads = [jax.grad(lambda fp: scalar_res_fn(fp, i))(f_params) for i in range(3)]
            return interior_res(unravel_fn(f_params), x), jnp.stack(grads)
        
        def get_bnd_row(x):
            def scalar_res_fn(fp, i): return boundary_res(unravel_fn(fp), x)[i]
            grads = [jax.grad(lambda fp: scalar_res_fn(fp, i))(f_params) for i in range(2)]
            return boundary_res(unravel_fn(f_params), x), jnp.stack(grads)

        # New Anchor Row
        def get_anchor_row(x):
            def scalar_res_fn(fp, i): return anchor_res(unravel_fn(fp), x)[i]
            grads = [jax.grad(lambda fp: scalar_res_fn(fp, 0))(f_params)]
            return anchor_res(unravel_fn(f_params), x), jnp.stack(grads)

        r_i, J_i = vmap(get_int_row)(x_int)
        r_b, J_b = vmap(get_bnd_row)(x_bnd)
        r_a, J_a = get_anchor_row(V_ANCHOR) # Single point

        # Flatten
        J_i_flat = J_i.reshape(-1, J_i.shape[-1])
        J_b_flat = J_b.reshape(-1, J_b.shape[-1])
        J_a_flat = J_a.reshape(-1, J_a.shape[-1])
        
        r_i_flat = r_i.reshape(-1)
        r_b_flat = r_b.reshape(-1)
        r_a_flat = r_a.reshape(-1)

        # Weighting: Standard Mean(r^2) implies 1/N. 
        # A single anchor point would be drowned out.
        # We scale the anchor by sqrt(N_total) so its contribution to the mean 
        # matches the magnitude of a full loss term (like in the PINN script).
        N_total = r_i_flat.shape[0] + r_b_flat.shape[0] + 1
        w_anchor = jnp.sqrt(N_total)
        
        r_a_flat = r_a_flat * w_anchor
        J_a_flat = J_a_flat * w_anchor

        J = jnp.concatenate([J_i_flat, J_b_flat, J_a_flat])
        r = jnp.concatenate([r_i_flat, r_b_flat, r_a_flat])
        #jax.debug.print("Shape of J: {}", jnp.shape(J))
        #jax.debug.print("Shape of r: {}", jnp.shape(r))
        
        return J, r

    @partial(jit, static_argnums=(0, 4, 5, 6))
    def step(self, params, key, damping, use_heavy_sampling, n_int, n_bnd):
        f_params, unravel_fn = ravel_pytree(params)
        k1, k2 = random.split(key)

        if use_heavy_sampling:
            x_i = sample_apex_heavy(k1, n_int)
            _, x_b = sample_wedge_random(k2, n_int, n_bnd)
        else:
            x_i, x_b = sample_wedge_random(k2, n_int, n_bnd)
        
        J, r = self.build_J(f_params, x_i, x_b, unravel_fn)
        loss = 0.5 * jnp.mean(r**2)
        
        K = jnp.dot(J, J.T, precision=jax.lax.Precision.HIGHEST)
        K = 0.5 * (K + K.T)

        diag_K = jnp.diag(K)
        scale = 1.0 / (jnp.sqrt(diag_K) + 1e-16)
        K_tilde = K * scale[:, None] * scale[None, :]
        r_tilde = r * scale
        K_reg = K_tilde + damping * jnp.eye(K_tilde.shape[0])
        solver = lx.AutoLinearSolver(well_posed=None)
        A_op = lx.MatrixLinearOperator(K_reg)
        solution = lx.linear_solve(A_op, r_tilde, solver)
        #solution = lx.linear_solve(A_op, r_tilde, solver=lx.Cholesky())
        y = solution.value

        L = jnp.linalg.cholesky(K_reg)
        y = jax.scipy.linalg.cho_solve((L, True), r_tilde)
        w_dual = y * scale
        w_primal = J.T @ w_dual
        
        # Line search evaluation needs to include anchor too for consistency
        def evaluate(p_flat):
            p = unravel_fn(p_flat)
            ri = vmap(lambda x: interior_res(p, x))(x_i).reshape(-1)
            rb = vmap(lambda x: boundary_res(p, x))(x_b).reshape(-1)
            ra = anchor_res(p, V_ANCHOR).reshape(-1)
            
            # Apply same weighting
            N_total = ri.shape[0] + rb.shape[0] + 1
            w_anchor = jnp.sqrt(N_total)
            ra = ra * w_anchor

            return 0.5 * jnp.mean(jnp.concatenate([ri, rb, ra])**2)

        def check_step(s): return evaluate(f_params - s * w_primal)
        losses = vmap(check_step)(self.ls_steps)
        best_idx = jnp.argmin(losses)
        new_params = unravel_fn(f_params - self.ls_steps[best_idx] * w_primal)
        return losses[best_idx], new_params


# --- UTILS ---
def read_csv(fileName):
    if not os.path.exists(fileName): 
        return None, None, None, None, None
    with open(fileName, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        data = np.array([list(map(float, row)) for row in reader])
    return data[:, 3:4], data[:, 4:5], data[:, 0:1], data[:, 1:2], data[:, 2:3]

def init_params(sizes, key):
    keys = random.split(key, len(sizes))
    def glorot(m, n, k): return (random.normal(k, (n, m)) * jnp.sqrt(2/(m+n)), jnp.zeros(n))
    return [glorot(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def save_checkpoint(params, filename):
    with open(filename, "wb") as f: pickle.dump(params, f)

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f: return pickle.load(f)
    return None

def plot_results(params, filename="stokes_multistage_result.png"):
    x, y = np.linspace(0, 1.05, 200), np.linspace(0, 2.05, 400)
    X, Y = np.meshgrid(x, y)
    xy = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    preds = vmap(lambda x: derivative_propagation(params, x)[0])(xy)
    u, v = preds[:, 0].reshape(X.shape), preds[:, 1].reshape(X.shape)
    vel = np.sqrt(u**2 + v**2)
    mask = (Y <= 2.0) & (4*X + Y >= 2.0) & (4*X - Y <= 2.0)
    
    plt.figure(figsize=(6, 8))
    plt.contourf(X, Y, np.where(mask, np.log10(vel + 1e-16), np.nan), levels=50, cmap="magma")
    plt.colorbar(label="Log10 Velocity")
    plt.streamplot(X, Y, np.where(mask, u, np.nan), np.where(mask, v, np.nan), color="w", density=1.5, linewidth=0.5)
    plt.title("Stokes Flow (Multistage Training)")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {filename}")

def get_reference_on_centerline():
    x_ref, y_ref, u_ref, _, _ = read_csv(REF_FILE)
    if x_ref is None: return None, None

    points = np.column_stack((x_ref.flatten(), y_ref.flatten()))
    values = u_ref.flatten()

    y_target = np.logspace(np.log10(1e-4), np.log10(2.0), 500)
    x_target = np.full_like(y_target, 0.5)
    u_interp = griddata(points, values, (x_target, y_target), method='linear')
    
    y_probes = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 1.8])
    x_probes = np.full(y_probes.shape, 0.5)
    u_probes_ref = griddata(points, values, (x_probes, y_probes), method='linear')
    return (y_target, u_interp), (y_probes, u_probes_ref)

def run_diagnostics(params, iter_idx):
    ckpt_name = os.path.join(CHECKPOINT_DIR, f"checkpoint_{iter_idx:05d}.pkl")
    save_checkpoint(params, ckpt_name)

    ref_line_data, ref_probe_data = get_reference_on_centerline()
    has_ref = ref_line_data is not None

    y_probes = jnp.array([0.01, 0.1, 0.5, 1.0, 1.5, 1.8])
    x_probes = jnp.full_like(y_probes, 0.5)
    xy_probes = jnp.stack([x_probes, y_probes], axis=1)

    preds = vmap(lambda x: derivative_propagation(params, x)[0])(xy_probes)
    u_p, v_p = preds[:, 0], preds[:, 1]
    vel_mag = jnp.sqrt(u_p**2 + v_p**2)

    print(f"\n   --- Vortex Strength (Centerline x=0.5) at Iter {iter_idx} ---")
    if has_ref:
        _, u_ref_probes = ref_probe_data
        print(f"   {'y':<8} | {'|V| Pred':<12} | {'|V| Ref':<12} | {'Diff':<12}")
        print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        for i in range(len(y_probes)):
            yv = y_probes[i]
            vm = vel_mag[i]
            vr = np.abs(u_ref_probes[i]) if not np.isnan(u_ref_probes[i]) else 0.0
            diff = abs(vm - vr)
            print(f"   {yv:<8.2f} | {vm:<12.5e} | {vr:<12.5e} | {diff:<12.5e}")
    else:
        print(f"   {'y':<10} | {'|Velocity|':<20} (NO REFERENCE FOUND)")
        for y_val, mag in zip(y_probes, vel_mag):
            print(f"   {y_val:<10.2f} | {mag:<20.5e}")
    print(f"   {'-'*50}\n")

    Nx, Ny = 200, 300
    x = np.linspace(0, 1.0, Nx)
    y = np.linspace(0, 2.0, Ny)
    X, Y = np.meshgrid(x, y)
    xy_grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    
    predict_fn = jit(vmap(lambda x: derivative_propagation(params, x)[0]))
    preds_grid = predict_fn(xy_grid)
    u_grid = preds_grid[:, 0].reshape(Ny, Nx)
    v_grid = preds_grid[:, 1].reshape(Ny, Nx)
    vel_grid = np.sqrt(u_grid**2 + v_grid**2)

    y_line = np.logspace(np.log10(1e-4), np.log10(2.0), 500)
    x_line = np.full_like(y_line, 0.5)
    xy_line = jnp.stack([x_line, y_line], axis=1)
    preds_line = predict_fn(xy_line)
    vel_line = np.sqrt(preds_line[:, 0]**2 + preds_line[:, 1]**2)
    mask = (Y <= 2.0) & (4*X + Y >= 2.0) & (4*X - Y <= 2.0)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title(f"Streamlines (Iter {iter_idx})")
    cont = ax1.contourf(X, Y, np.where(mask, np.log10(vel_grid + 1e-16), np.nan), levels=50, cmap="magma")
    plt.colorbar(cont, ax=ax1, label="Log10 Velocity")
    ax1.streamplot(X, Y, np.where(mask, u_grid, 0), np.where(mask, v_grid, 0), 
                   color="w", density=1.2, linewidth=0.6, arrowsize=0.6)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 2.05)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"Centerline Profile (x=0.5) vs Reference")
    if has_ref:
        y_ref_line, u_ref_line = ref_line_data
        ax2.semilogy(y_ref_line, np.abs(u_ref_line) + 1e-16, 'k-', lw=2.5, alpha=0.6, label="Reference")
    
    ax2.semilogy(y_line, vel_line + 1e-16, 'r--', lw=2.0, label="Prediction")
    ax2.set_xlabel("y (Distance from bottom)")
    ax2.set_ylabel("|Velocity| (Log Scale)")
    ax2.set_ylim(1e-16, 10.0) 
    ax2.set_xlim(0, 2.0)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left")
    
    plot_name = os.path.join(CHECKPOINT_DIR, f"diagnostics_{iter_idx:05d}.png")
    plt.tight_layout()
    plt.savefig(plot_name, dpi=150)
    plt.close()
    print(f"   >>> Plot saved to {plot_name}")


# --- MAIN ORCHESTRATOR ---
if __name__ == "__main__":
    print(f"--- STOKES SOLVER (MULTISTAGE RESUMABLE) ---")
    print(f"Phase 1: {P1_ITERS} iters (N={P1_BATCH_INT}+{P1_BATCH_BND})")
    print(f"Phase 2: {P2_ITERS} iters (N={P2_BATCH_INT}+{P2_BATCH_BND})")
    
    # LOAD FULL REFERENCE DATA (u, v, p)
    x_s, y_s, u_s, v_s, p_s = read_csv(REF_FILE)
    has_val = x_s is not None
    X_val, U_val, V_val, P_val = None, None, None, None
    if has_val:
        X_val = jnp.hstack([x_s, y_s])
        U_val = jnp.array(u_s)
        V_val = jnp.array(v_s)
        P_val = jnp.array(p_s)

    key = random.PRNGKey(SEED)
    solver = JacobiGNSolver(ls_steps=0.5**jnp.linspace(0, 15, 16))
    history = [] 

    params = init_params(LAYER_SIZES, key)
    start_iter = 1
        
    print("Compiling Phase 1 Kernel...")
    _, _ = solver.step(params, key, P1_DAMPING, False, P1_BATCH_INT, P1_BATCH_BND)
    print("Compiling Phase 2 Kernel...")
    _, _ = solver.step(params, key, P2_DAMPING, True, P2_BATCH_INT, P2_BATCH_BND)

    print("Solver Compiled/Ready.")
    print("\n>>> STARTING TRAINING...")
    start_t = timeit.default_timer()

    total_iters = P1_ITERS + P2_ITERS
    phase = 1
    
    for i in range(start_iter, total_iters + 1):
        if i <= P1_ITERS:
            current_damping = P1_DAMPING
            use_heavy = False
            current_n_int = P1_BATCH_INT
            current_n_bnd = P1_BATCH_BND
            phase = 1
        else:
            current_damping = P2_DAMPING
            use_heavy = True
            current_n_int = P2_BATCH_INT
            current_n_bnd = P2_BATCH_BND
            phase = 2

        key, sk = random.split(key)
        
        loss, params = solver.step(params, sk, current_damping, use_heavy, current_n_int, current_n_bnd)
        
        err_u, err_v, err_p = 0.0, 0.0, 0.0
        if has_val and i % CHECKPOINT_FREQ == 0:
            preds_val = vmap(lambda x: derivative_propagation(params, x)[0])(X_val)
            u_p_val, v_p_val, p_p_val = preds_val[:, 0:1], preds_val[:, 1:2], preds_val[:, 2:3]
            
            err_u = jnp.linalg.norm(u_p_val - U_val) / jnp.linalg.norm(U_val)
            err_v = jnp.linalg.norm(v_p_val - V_val) / jnp.linalg.norm(V_val)
            err_p = jnp.linalg.norm(p_p_val - P_val) / jnp.linalg.norm(P_val)
        
        history.append([i, float(loss), float(err_u), float(err_v), float(err_p)])

        if i % CHECKPOINT_FREQ == 0:
            elapsed = timeit.default_timer() - start_t
            # Print u, v, AND p error
            print(f"P{phase} | Iter {i:4d} | Loss: {loss:.4e} | Time: {elapsed:.1f}s | Err U: {err_u:.2e} | Err V: {err_v:.2e} | Err P: {err_p:.2e}")
            
        if phase == 2 and loss < P2_TOLERANCE:
            print(f"\n>>> EARLY STOPPING TRIGGERED AT ITER {i} (Loss < {P2_TOLERANCE})")
            save_checkpoint(params, os.path.join(CHECKPOINT_DIR, f"early_stop_{i:05d}.pkl"))
            break

    run_diagnostics(params, i)
    save_checkpoint(params, "stokes_final_params.pkl")
    with open(HISTORY_FILE, "wb") as f: pickle.dump(history, f)
    print(f"Training history saved to {HISTORY_FILE}")
    plot_results(params)
