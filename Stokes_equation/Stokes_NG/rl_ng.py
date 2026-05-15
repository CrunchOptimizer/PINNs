import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

jax.config.update("jax_enable_x64", True)

from jax import random, vmap, jit
from jax.flatten_util import ravel_pytree
from functools import partial
import lineax as lx

# ══════════════════════════════════════════════════════
# ── Problem sizes (exact — no guessing) ──
# ══════════════════════════════════════════════════════
N_INT    = 1000
N_BND    = 1000
N_ANC    = 1
N_RES    = N_INT * 3 + N_BND * 2 + N_ANC * 1   # 5001
N_PARAMS = 64*2 + 64 + 64*64*9 + 64*9 + 64*3 + 3  # exact param count
BYTES    = 8   # FP64

# ── Only use FLOPs we can compute exactly ──
# J@Jt:     (N_RES x N_PARAMS) @ (N_PARAMS x N_RES) = standard matmul
# Cholesky: standard formula N^3/3
jjt_flops      = 2 * N_RES**2 * N_PARAMS
cholesky_flops = N_RES**3 // 3
reliable_flops = jjt_flops + cholesky_flops

# ── Memory for those same operations ──
J_bytes        = N_RES * N_PARAMS * BYTES * 3   # write J + read twice
K_bytes        = N_RES * N_RES    * BYTES * 2   # write K + read for Chol
res_bytes      = N_RES            * BYTES * 2
reliable_bytes = J_bytes + K_bytes + res_bytes

AI    = reliable_flops / reliable_bytes
RIDGE = 66.9e12 / 3.35e12   # H100 FP64 ridge point

print(f"── Problem sizes ────────────────────────────")
print(f"  N_RES    : {N_RES:,}  ({N_INT} int x3 + {N_BND} bnd x2 + {N_ANC} anc x1)")
print(f"  N_PARAMS : {N_PARAMS:,}")
print(f"  Jacobian : ({N_RES} x {N_PARAMS})")
print(f"\n── Reliable FLOPs (exact formulas only) ─────")
print(f"  J @ Jᵀ  (2N²P)  : {jjt_flops/1e9:.3f} GFLOPs")
print(f"  Cholesky (N³/3) : {cholesky_flops/1e9:.3f} GFLOPs")
print(f"  Total           : {reliable_flops/1e9:.3f} GFLOPs")
print(f"\n── Memory (exact) ───────────────────────────")
print(f"  J matrix        : {J_bytes/1e6:.1f} MB")
print(f"  K = J@Jᵀ        : {K_bytes/1e6:.1f} MB")
print(f"  Residuals       : {res_bytes/1e6:.2f} MB")
print(f"  Total           : {reliable_bytes/1e6:.1f} MB")
print(f"\n── Arithmetic Intensity ─────────────────────")
print(f"  AI              : {AI:.2f} FLOP/byte")
print(f"  Ridge point     : {RIDGE:.2f} FLOP/byte")
print(f"  Bound           : {'COMPUTE-BOUND' if AI >= RIDGE else 'MEMORY-BOUND'}")

# ══════════════════════════════════════════════════════
# ── Solver (inline) ──
# ══════════════════════════════════════════════════════
SEED         = 1234
LAYER_SIZES  = [2, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 3]
P1_DAMPING   = 1e-12
P2_DAMPING   = 5e-9
P1_BATCH_INT = 1000
P1_BATCH_BND = 1000
P2_BATCH_INT = 1000
P2_BATCH_BND = 1000

V1       = jnp.array([0.5, 0.0])
V2       = jnp.array([0.0, 2.0])
V3       = jnp.array([1.0, 2.0])
V_ANCHOR = jnp.array([0.0, 2.0])

def act(t):   return jnp.tanh(t)
def act_p(t): return 1.0 - jnp.square(jnp.tanh(t))
def act_pp(t):
    tanh_t = jnp.tanh(t)
    return -2.0 * tanh_t * (1.0 - jnp.square(tanh_t))

@jit
def derivative_propagation(params, x):
    z = x
    dz_dx   = jnp.eye(len(x))
    d2z_dxx = jnp.zeros((len(x), len(x), len(x)))
    for w, b in params[:-1]:
        z_pre   = jnp.dot(w, z) + b
        dz_pre  = jnp.dot(w, dz_dx)
        d2z_pre = jnp.einsum("ij,jkl->ikl", w, d2z_dxx)
        z       = act(z_pre)
        s_p     = act_p(z_pre)
        s_pp    = act_pp(z_pre)
        dz_dx   = s_p[:, None] * dz_pre
        term1   = s_pp[:, None, None] * jnp.einsum("ij,ik->ijk", dz_pre, dz_pre)
        term2   = s_p[:, None, None] * d2z_pre
        d2z_dxx = term1 + term2
    fw, fb   = params[-1]
    z        = jnp.dot(fw, z) + fb
    dz_dx    = jnp.dot(fw, dz_dx)
    d2z_dxx  = jnp.einsum("ij,jkl->ikl", fw, d2z_dxx)
    return z, dz_dx, d2z_dxx

@jit
def interior_res(params, x):
    _, jac, hess = derivative_propagation(params, x)
    lap_u = hess[0, 0, 0] + hess[0, 1, 1]
    lap_v = hess[1, 0, 0] + hess[1, 1, 1]
    return jnp.stack([jac[2,0] - lap_u, jac[2,1] - lap_v, jac[0,0] + jac[1,1]])

@jit
def boundary_res(params, x):
    pred, _, _ = derivative_propagation(params, x)
    u_lid    = 16.0 * (x[0]**2) * ((1.0 - x[0])**2)
    u_target = jnp.where(x[1] > 1.999, u_lid, 0.0)
    return jnp.stack([pred[0] - u_target, pred[1]])

@jit
def anchor_res(params, x):
    pred, _, _ = derivative_propagation(params, x)
    return jnp.stack([pred[2]])

@partial(jit, static_argnums=(1, 2))
def sample_wedge_random(key, n_int, n_bnd):
    k1, k2, k3 = random.split(key, 3)
    r1  = random.uniform(k1, (n_int, 1))
    r2  = random.uniform(k2, (n_int, 1))
    s   = jnp.sqrt(r1)
    x_int = (1.0-s)*V1 + s*(1.0-r2)*V2 + s*r2*V3
    t   = random.uniform(k3, (n_bnd, 1))
    idx = jnp.arange(n_bnd)
    ne  = n_bnd // 3
    xb  = jnp.where((idx < ne)[:,None],    V2 + t*(V3-V2), V2 + t*(V1-V2))
    xb  = jnp.where((idx >= 2*ne)[:,None], V3 + t*(V1-V3), xb)
    return x_int, xb

@partial(jit, static_argnums=(1,))
def sample_apex_heavy(key, n_points):
    k1, k2, k3 = random.split(key, 3)
    n_uni  = n_points // 10
    n_apex = n_points - n_uni
    x_uni, _ = sample_wedge_random(k1, n_uni, 0)
    log_y  = random.uniform(k2, (n_apex,1), minval=jnp.log(1e-7), maxval=jnp.log(2))
    y_apex = jnp.exp(log_y)
    x_off  = random.uniform(k3, (n_apex,1), minval=-0.5, maxval=0.5) * 0.5 * y_apex
    return jnp.concatenate([x_uni, jnp.hstack([0.5 + x_off, y_apex])])

def init_params(sizes, key):
    keys = random.split(key, len(sizes))
    def glorot(m, n, k):
        return (random.normal(k, (n,m)) * jnp.sqrt(2/(m+n)), jnp.zeros(n))
    return [glorot(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

class JacobiGNSolver:
    def __init__(self, ls_steps):
        self.ls_steps = ls_steps

    def build_J(self, f_params, x_int, x_bnd, unravel_fn):
        def get_int_row(x):
            grads = [jax.grad(lambda fp, i=i:
                     interior_res(unravel_fn(fp), x)[i])(f_params)
                     for i in range(3)]
            return interior_res(unravel_fn(f_params), x), jnp.stack(grads)
        def get_bnd_row(x):
            grads = [jax.grad(lambda fp, i=i:
                     boundary_res(unravel_fn(fp), x)[i])(f_params)
                     for i in range(2)]
            return boundary_res(unravel_fn(f_params), x), jnp.stack(grads)
        def get_anchor_row(x):
            grads = [jax.grad(lambda fp:
                     anchor_res(unravel_fn(fp), x)[0])(f_params)]
            return anchor_res(unravel_fn(f_params), x), jnp.stack(grads)

        r_i, J_i = vmap(get_int_row)(x_int)
        r_b, J_b = vmap(get_bnd_row)(x_bnd)
        r_a, J_a = get_anchor_row(V_ANCHOR)
        J_i = J_i.reshape(-1, J_i.shape[-1])
        J_b = J_b.reshape(-1, J_b.shape[-1])
        J_a = J_a.reshape(-1, J_a.shape[-1])
        r_i = r_i.reshape(-1)
        r_b = r_b.reshape(-1)
        r_a = r_a.reshape(-1)
        N_total  = r_i.shape[0] + r_b.shape[0] + 1
        w_anchor = jnp.sqrt(N_total)
        r_a = r_a * w_anchor
        J_a = J_a * w_anchor
        return jnp.concatenate([J_i, J_b, J_a]), jnp.concatenate([r_i, r_b, r_a])

    @partial(jit, static_argnums=(0, 4, 5, 6))
    def step(self, params, key, damping, use_heavy, n_int, n_bnd):
        f_params, unravel_fn = ravel_pytree(params)
        k1, k2 = random.split(key)
        if use_heavy:
            x_i = sample_apex_heavy(k1, n_int)
            _, x_b = sample_wedge_random(k2, n_int, n_bnd)
        else:
            x_i, x_b = sample_wedge_random(k2, n_int, n_bnd)
        J, r     = self.build_J(f_params, x_i, x_b, unravel_fn)
        loss     = 0.5 * jnp.mean(r**2)
        K        = jnp.dot(J, J.T, precision=jax.lax.Precision.HIGHEST)
        K        = 0.5 * (K + K.T)
        diag_K   = jnp.diag(K)
        scale    = 1.0 / (jnp.sqrt(diag_K) + 1e-16)
        K_tilde  = K * scale[:, None] * scale[None, :]
        r_tilde  = r * scale
        K_reg    = K_tilde + damping * jnp.eye(K_tilde.shape[0])
        L        = jnp.linalg.cholesky(K_reg)
        y        = jax.scipy.linalg.cho_solve((L, True), r_tilde)
        w_dual   = y * scale
        w_primal = J.T @ w_dual
        def evaluate(p_flat):
            p  = unravel_fn(p_flat)
            ri = vmap(lambda x: interior_res(p, x))(x_i).reshape(-1)
            rb = vmap(lambda x: boundary_res(p, x))(x_b).reshape(-1)
            ra = anchor_res(p, V_ANCHOR).reshape(-1) * jnp.sqrt(ri.shape[0]+rb.shape[0]+1)
            return 0.5 * jnp.mean(jnp.concatenate([ri, rb, ra])**2)
        losses   = vmap(lambda s: evaluate(f_params - s * w_primal))(self.ls_steps)
        best_idx = jnp.argmin(losses)
        return losses[best_idx], unravel_fn(f_params - self.ls_steps[best_idx] * w_primal)

# ══════════════════════════════════════════════════════
# ── Warmup + Timing ──
# ══════════════════════════════════════════════════════
key    = random.PRNGKey(SEED)
params = init_params(LAYER_SIZES, key)
solver = JacobiGNSolver(ls_steps=0.5**jnp.linspace(0, 15, 16))

print(f"\n[warmup] compiling Phase 1...")
_, _ = solver.step(params, key, P1_DAMPING, False, P1_BATCH_INT, P1_BATCH_BND)
jax.effects_barrier()
print(f"[warmup] compiling Phase 2...")
_, _ = solver.step(params, key, P2_DAMPING, True, P2_BATCH_INT, P2_BATCH_BND)
jax.effects_barrier()
print(f"[warmup] done")

N_TIMING = 5
times_p1, times_p2 = [], []

print(f"\n[profile] timing {N_TIMING} Phase 1 steps...")
for i in range(N_TIMING):
    key, sk = random.split(key)
    t0 = time.perf_counter()
    loss, params = solver.step(params, sk, P1_DAMPING, False, P1_BATCH_INT, P1_BATCH_BND)
    jax.effects_barrier()
    t1 = time.perf_counter()
    times_p1.append(t1 - t0)
    print(f"  P1 step {i+1}: {(t1-t0)*1000:.1f} ms   loss={float(loss):.4e}")

print(f"\n[profile] timing {N_TIMING} Phase 2 steps...")
for i in range(N_TIMING):
    key, sk = random.split(key)
    t0 = time.perf_counter()
    loss, params = solver.step(params, sk, P2_DAMPING, True, P2_BATCH_INT, P2_BATCH_BND)
    jax.effects_barrier()
    t1 = time.perf_counter()
    times_p2.append(t1 - t0)
    print(f"  P2 step {i+1}: {(t1-t0)*1000:.1f} ms   loss={float(loss):.4e}")

avg_p1   = np.mean(times_p1)
avg_p2   = np.mean(times_p2)
avg_all  = (avg_p1 + avg_p2) / 2.0
perf_avg = reliable_flops / avg_all

PEAK_FLOPS = 66.9e12
PEAK_BW    = 3.35e12

print(f"\n── Final Results ─────────────────────────────")
print(f"  Phase 1 avg  : {avg_p1*1000:.1f} ms   {reliable_flops/avg_p1/1e12:.4f} TFLOP/s")
print(f"  Phase 2 avg  : {avg_p2*1000:.1f} ms   {reliable_flops/avg_p2/1e12:.4f} TFLOP/s")
print(f"  Overall avg  : {avg_all*1000:.1f} ms   {perf_avg/1e12:.4f} TFLOP/s")
print(f"  H100 peak    : {PEAK_FLOPS/1e12:.1f} TFLOP/s")
print(f"  Utilization  : {100*perf_avg/PEAK_FLOPS:.2f}%  (J@Jt + Cholesky only)")
print(f"  AI           : {AI:.2f} FLOP/byte")
print(f"  Ridge        : {RIDGE:.2f} FLOP/byte")
print(f"  Bound        : {'COMPUTE-BOUND' if AI >= RIDGE else 'MEMORY-BOUND'}")

# ══════════════════════════════════════════════════════
# ── Roofline Plot ──
# ══════════════════════════════════════════════════════
ai_line = np.logspace(-2, 4, 1000)
roof    = np.minimum(PEAK_FLOPS, PEAK_BW * ai_line)

plt.figure(figsize=(10, 6))
plt.loglog(ai_line, roof, "k-", lw=2.5, label="H100 roofline (FP64)")
plt.axvline(RIDGE, color="gray", ls="--", lw=1.2, alpha=0.6,
            label=f"Ridge = {RIDGE:.1f} FLOP/byte")

plt.scatter([AI], [perf_avg],
            color="crimson", s=200, zorder=6,
            edgecolors="k", lw=1.5,
            label=f"Stokes GN (Phase 1+2 avg)\n"
                  f"AI = {AI:.1f} FLOP/byte\n"
                  f"{perf_avg/1e12:.4f} TFLOP/s\n"
                  f"{100*perf_avg/PEAK_FLOPS:.2f}% of H100 FP64 peak")

plt.xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
plt.ylabel("Performance (FLOP/s)",             fontsize=12)
plt.title("Roofline Model — Stokes PINN Gauss-Newton\n"
          "H100 SXM5 FP64  |  FLOPs = J@Jᵀ + Cholesky (exact)  |  Time = wall-clock",
          fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("roofline_stokes_gn.png", dpi=220)
plt.close()
print("\nSaved: roofline_stokes_gn.png")