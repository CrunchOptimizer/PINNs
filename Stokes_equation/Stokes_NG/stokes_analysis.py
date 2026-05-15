import jax
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import csv

jax.config.update("jax_enable_x64", True)

CHECKPOINT_FILE = "checkpoint_03000.pkl"
REF_FILE = "../st_flow.csv"
OUTPUT_NPY = "stokes_fields.npy"
OUTPUT_PLOT = "stokes_analysis_plot.png"

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

def predict_flow(params, xy):
    preds = vmap(lambda x: derivative_propagation(params, x)[0])(xy)
    return preds[:, 0], preds[:, 1], preds[:, 2]

def load_params(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint {filename} not found.")
    with open(filename, "rb") as f:
        return pickle.load(f)

def read_ref_csv(fileName):
    if not os.path.exists(fileName):
        return None
    try:
        with open(fileName, newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                float(header[0])
                f.seek(0)
            except ValueError:
                pass
            data = np.array([list(map(float, row)) for row in reader])
        
        x = data[:, 3]
        y = data[:, 4]
        u = data[:, 0]
        v = data[:, 1]
        p = data[:, 2]
        return x, y, u, v, p
    except Exception:
        return None

def compute_relative_error(pred, ref):
    norm_diff = np.linalg.norm(pred - ref)
    norm_ref = np.linalg.norm(ref)
    if norm_ref < 1e-12: return 0.0
    return norm_diff / norm_ref

def main():
    params = load_params(CHECKPOINT_FILE)
    print(f"Loaded {CHECKPOINT_FILE}")

    ref_data = read_ref_csv(REF_FILE)
    if ref_data:
        x_ref, y_ref, u_ref, v_ref, p_ref = ref_data
        
        xy_ref = jnp.stack([x_ref, y_ref], axis=1)
        u_pred, v_pred, p_pred = predict_flow(params, xy_ref)
        u_pred, v_pred, p_pred = np.array(u_pred), np.array(v_pred), np.array(p_pred)
        
        err_u = compute_relative_error(u_pred, u_ref)
        err_v = compute_relative_error(v_pred, v_ref)
        
        err_p_raw = compute_relative_error(p_pred, p_ref)
        p_shift = np.mean(p_ref) - np.mean(p_pred)
        p_pred_corrected = p_pred + p_shift
        err_p_corrected = compute_relative_error(p_pred_corrected, p_ref)
        
        print(f"Rel L2 Error u:          {err_u:.6e}")
        print(f"Rel L2 Error v:          {err_v:.6e}")
        print(f"Rel L2 Error p (Raw):    {err_p_raw:.6e}")
        print(f"Rel L2 Error p (Corr):   {err_p_corrected:.6e}")
    
    nx, ny = 400, 800
    x_line = np.linspace(0, 1.0, nx)
    y_line = np.linspace(0, 2.0, ny)
    X, Y = np.meshgrid(x_line, y_line)
    
    mask = (Y <= 2.0) & (4*X + Y >= 2.0) & (4*X - Y <= 2.0)
    
    xy_grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    u_flat, v_flat, p_flat = predict_flow(params, xy_grid)
    
    U = np.array(u_flat).reshape(ny, nx)
    V = np.array(v_flat).reshape(ny, nx)
    P = np.array(p_flat).reshape(ny, nx)
    Vel = np.sqrt(U**2 + V**2)
    
    U_masked = np.where(mask, U, np.nan)
    V_masked = np.where(mask, V, np.nan)
    P_masked = np.where(mask, P, np.nan)
    Vel_masked = np.where(mask, Vel, np.nan)

    np.save(OUTPUT_NPY, {
        'x': X, 'y': Y, 'mask': mask,
        'u': U_masked, 'v': V_masked, 'p': P_masked, 'vel_mag': Vel_masked
    })

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([0.0, 1.0], [2.0, 2.0], 'k-', lw=2.5)
    ax1.plot([0.0, 0.5], [2.0, 0.0], 'k-', lw=2.5)
    ax1.plot([1.0, 0.5], [2.0, 0.0], 'k-', lw=2.5)
    
    ax1.streamplot(X, Y, U_masked, V_masked, 
                   color='blue', linewidth=1.0, density=2.5, arrowsize=1.0)
    
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 2.0)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("Streamlines")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(1, 2, 2)
    
    y_center = np.linspace(0, 2.0, 1000)
    x_center = np.full_like(y_center, 0.5)
    xy_center = jnp.stack([x_center, y_center], axis=1)
    
    u_c, v_c, _ = predict_flow(params, xy_center)
    vel_transverse_abs = np.abs(u_c)
    np.save("y_center.npy", y_center)
    np.save("vel_center.npy", vel_transverse_abs)
    
    ax2.semilogy(y_center, vel_transverse_abs + 1e-20, 'b-', lw=1.5)
    
    ax2.set_xlabel("y (Distance along centerline)")
    ax2.set_ylabel("|u| (Transverse Velocity)")
    ax2.set_title("Moffatt Eddy Strength")
    
    ax2.set_xlim(0, 2.0)
    ax2.set_ylim(1e-16, 1.0)
    ax2.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
