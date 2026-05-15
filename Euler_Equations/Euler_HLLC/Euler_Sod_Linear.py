# Python Language Realted Modules
import functools as ft
from functools import partial
from collections.abc import Callable                                                                                        
from typing import Any, TypeVar, NamedTuple 
import sys, os
import time                                                                                             
# Numpy and Scipy Related packages
import numpy as np                                                                                                          
from pyDOE import lhs                                                                                                       
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.interpolate import griddata
from soap_jax import soap
from jax.nn import softplus
import jax
import jax.numpy as jnp



# JAX and JAX backends related  and differentiable packages
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree, Scalar                                                                                 
import lineax as lx
import equinox as eqx                                                                                                       
import equinox.internal as eqxi
import optax
import optimistix as optx  

## Create Directory for saving Model
dir_path = "./checkpoints"
os.makedirs(dir_path, exist_ok=True)

# Global setup and variables:
np.random.seed(1234)
jax.config.update("jax_enable_x64", True)
MODEL_FILE_NAME = "./checkpoints/euler_sod.eqx" 
Y = TypeVar("Y") 
Out = TypeVar("Out") 
Aux = TypeVar("Aux") 

## Gas Constant
gamma = 1.4



def plot_solution(x, y_true, y_pred, label, filename):
    """Generic plotting function for exact vs predicted."""
    plt.figure()
    plt.plot(x, y_true, "-g", lw=2.0, label=f"{label} exact")
    plt.plot(x, y_pred, "--k", lw=2.0, label=f"{label} pred")
    plt.xlabel("x")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class BFGSTrustRegion(optx.AbstractBFGS):                                                                                 
    rtol: float 
    atol: float 
    norm: Callable = optx.max_norm 
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()




class SSBFGSTrustRegion(optx.AbstractSSBFGS):                                                                                 
    rtol: float 
    atol: float 
    norm: Callable = optx.max_norm 
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()
    
class SSBFGSBacktracking(optx.AbstractSSBFGS):
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.BacktrackingArmijo() 
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()



class BroydenTrustRegion(optx.AbstractSSBroyden):                                                                                 
    rtol: float 
    atol: float 
    norm: Callable = optx.max_norm 
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset() 

class BFGSBacktrackingWolfe(optx.AbstractSSBFGS):
    """BFGS Hessian + indirect Levenberg Marquardt update."""
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.BacktrackingStrongWolfe()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()

class BFGSBacktrackingZoom(optx.AbstractSSBFGS):
    """BFGS Hessian + indirect Levenberg Marquardt update."""
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.Zoom = optx.Zoom(initial_guess_strategy="keep")
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()


class BroydenBacktrackingZoom(optx.AbstractSSBroyden):
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.Zoom = optx.Zoom(initial_guess_strategy="keep")
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()

class BroydenBacktrackingWolfe(optx.AbstractSSBroyden):
    """BFGS Hessian + indirect Levenberg Marquardt update."""
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.BacktrackingStrongWolfe()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()
 


initializer = jax.nn.initializers.glorot_normal()                                                                                     
                                                                                                                                      
                                                                                                                                                                                                                  
def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:                                                              
    out, in_ = weight.shape                                                                                                           
    return initializer(key, shape=(out, in_))                                                                                         
                                                  

def init_linear_weight(model, init_fn, key):
    is_linear = lambda x_loc: isinstance(x_loc, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]
    get_bias = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)                                             
                          if is_linear(x)]
    weights = get_weights(model)                                                                                                      
    biases = get_bias(model)
    sz_w = [x.size for x in weights]
    sz_b = [x.size for x in biases]                                                                                                   
                                                                                                                                      
    print("w", sz_w)                                                                                                                  
    print("b", sz_b)                                                                                                                  
                                                                                                                                      
    print(f"Total params: {sum(sz_w) + sum(sz_b)}")                                                                                   
                                                                                                                                      
    new_biases = jax.tree_util.tree_map(lambda p_loc: 0.0 * jnp.abs(p_loc), biases)  

    new_weights = [init_fn(weight, subkey)                                                                                            
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]                                           
    new_model = eqx.tree_at(get_weights, model, new_weights)                                                                          
    new_model = eqx.tree_at(get_bias, new_model, new_biases)                                                                          
                                                                                                                                      
    return new_model                                                                                                                  
                      
class ParamTanh(eqx.Module):
    alpha: jnp.ndarray
    
    def __call__(self, x):
        return jnp.tanh(self.alpha*x)

class RationalActivationStable(eqx.Module):
    a: jnp.ndarray
    b: jnp.ndarray
    eps: float = 1e-6

    def __init__(self, key):
        key_a, key_b = jax.random.split(key)
        self.a = jax.random.normal(key_a, (3,))
        self.b = jax.random.normal(key_b, (2,))

    def __call__(self, x):
        num = self.a[0] + self.a[1]*x + self.a[2]*x**2
        den = 1.0 + self.b[0]*x + self.b[1]*x**2
        return num / (den + self.eps)

class FourierFeatures(eqx.Module):
    B: jnp.ndarray  
    
    def __init__(self, in_dim, mapping_size, key):
        key, subkey = jax.random.split(key)
        self.B = jax.random.normal(subkey, shape=(mapping_size, in_dim)) * 10.0

    def __call__(self, x):
        x_proj = x @ self.B.T 
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class EulerPINN(eqx.Module):
    layers: list

    def __init__(self, in_dim, out_dim, key):
        keys = jax.random.split(key, 24)
        tn = 20
        mapping_size = 6

        self.layers = [
            #FourierFeatures(in_dim, mapping_size, keys[0]),
            eqx.nn.Linear(in_dim, tn, key=keys[1]),
            ParamTanh(jnp.array(0.9))]

        for i in range(6):
            self.layers.append(eqx.nn.Linear(tn, tn, key=keys[i+2]))
            self.layers.append(ParamTanh(jnp.array(0.9)))
            #self.layers.append(RationalActivationStable(keys[i+2+1]))
        self.layers.append(eqx.nn.Linear(tn, out_dim, key=keys[-1]))
    
    def __call__(self, x, t):
        xt = jnp.stack([x, t], axis=-1)
        for layer in self.layers[:-1]:
            xt = layer(xt)
        return self.layers[-1](xt)
    




def prim_vars(net, x, t):
    """ρ, u, p"""
    out = net[0](x, t)

    return out[0], out[1], out[2], out[3]


def safe_prim_vars(net, x, t):
    rho, u, p, nui = prim_vars(net, x, t)

    rho = jnp.maximum(rho, 1e-6)
    p   = jnp.maximum(p,   1e-6)

    return rho, u, p, nui



# def flux_vars(net, x, t):
#     """ρu, ρu²+p, u(E+p)"""
#     out = net[1](x, t)
#     return out[0], out[1], out[2]

def flux_vars(net, x, t):
    out = net[0](x, t)
    f1 = out[0]*out[1]
    f2 = out[0]*out[1]**2 + out[2]
    E = out[2] / (gamma - 1.0) + 0.5 * out[0] * out[1]**2
    f3 = out[1]*(E + out[2])
    # out = net[1](x, t)
    # f3 = out[0]
    return f1, f2, f3

def energy(net, x, t):
    rho, u, p, _ = prim_vars(net, x, t)
    return p / (gamma - 1.0) + 0.5 * rho * u**2

def momentum(net, x, t):
    rho, u, _, _ = prim_vars(net, x, t)
    return rho * u

def viscosity(p, p_x):
    eps0 = 1e-4
    C = 5e-3
    return eps0 + C * jnp.abs(p_x) / (jnp.abs(p) + 1e-6)

gamma = 1.4

def entropy_residual(model, x, t):
    # Predict conservative variables
    rho, _, _, _ = prim_vars(model, x, t)
    rho_u = momentum(model, x, t)
    E = energy(model, x, t)
    

    # Primitive variables
    u = rho_u / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)

    # Entropy per unit mass
    s = jnp.log(jnp.maximum(p / rho**gamma, 1e-08))

    # Define rho*s for grad computations
    def rho_s_func(xx, tt):
        rho_, _, _, _ = prim_vars(model, xx, tt)
        rho_u_ = momentum(model, xx, tt)
        E_ = energy(model, xx, tt)
        u_ = rho_u_ / rho_
        p_ = (gamma - 1.0) * (E_ - 0.5 * rho_ * u_**2)
        s_ = jnp.log(p_ / rho_**gamma)
        return rho_ * s_, rho_ * u_ * s_

    # Gradients
    rho_s_val, rho_u_s_val = rho_s_func(x, t)

    # Time derivative of rho*s
    rho_s_t = jax.grad(lambda tt: rho_s_func(x, tt)[0])(t)

    # Space derivative of rho*u*s
    rho_u_s_x = jax.grad(lambda xx: rho_s_func(xx, t)[1])(x)

    # Entropy residual
    R_s = rho_s_t + rho_u_s_x
    return R_s

def grad_x(f):
    return jax.grad(f, argnums=0)

def grad_t(f):
    return jax.grad(f, argnums=1)



@eqx.filter_jit
def euler_residual(net, x, t):

    def rho_fn(xx, tt):
        return prim_vars(net, xx, tt)[0]
    
    def u_fn(xx, tt):
        return prim_vars(net, xx, tt)[1]
    
    def p_fn(xx, tt):
        return prim_vars(net, xx, tt)[2]

    def mom_fn(xx, tt):
        return momentum(net, xx, tt)

    def E_fn(xx, tt):
        return energy(net, xx, tt)

    def F1_fn(xx, tt):
        return flux_vars(net, xx, tt)[0]

    def F2_fn(xx, tt):
        return flux_vars(net, xx, tt)[1]

    def F3_fn(xx, tt):
        return flux_vars(net, xx, tt)[2]

    rho_t  = grad_t(rho_fn)(x, t)
    u_t  = grad_t(u_fn)(x, t)
    p_t  = grad_t(p_fn)(x, t)


    rhou_t = grad_t(mom_fn)(x, t)
    E_t    = grad_t(E_fn)(x, t)


    ####
    rho_x = grad_x(rho_fn)(x, t)
    u_x = grad_x(u_fn)(x, t)
    p_x = grad_x(p_fn)(x, t)



    F1_x = grad_x(F1_fn)(x, t)
    F2_x = grad_x(F2_fn)(x, t)
    F3_x = grad_x(F3_fn)(x, t)

    # second derivatives (viscosity)
    rho_xx  = grad_x(lambda xx, tt: grad_x(rho_fn)(xx, tt))(x, t)
    rhou_xx = grad_x(lambda xx, tt: grad_x(mom_fn)(xx, tt))(x, t)
    E_xx    = grad_x(lambda xx, tt: grad_x(E_fn)(xx, tt))(x, t)

    u_xx  = grad_x(lambda xx, tt: grad_x(u_fn)(xx, tt))(x, t)
    p_xx  = grad_x(lambda xx, tt: grad_x(p_fn)(xx, tt))(x, t)

    rho, u, p, nui = prim_vars(net, x, t)
    nu = nui*nui
    R0 = rho_t + u*rho_x + rho*u_x - nu*rho_xx
    R1 = rho*u_t + rho*u*u_x + p_x - nu*(rho*u_xx + 2*rho_x * u_x)
    R2 = p_t + u*p_x + gamma* p*u_x - nu*(p_xx + rho*(gamma-1)*u_x*u_x)
    R3 = nu



    

    # rho, u, p, nui = prim_vars(net, x, t)

    # nu = nui*nui

    # R0 = rho_t  + F1_x - nu * rho_xx
    # R1 = rhou_t + F2_x - nu * rhou_xx
    # R2 = E_t    + F3_x - nu * E_xx
    # R3 = nu  # positivity constraint

    return R0, R1, R2, R3


@eqx.filter_jit
def euler_residual_hllc(net, x, t, dx):

    

    # -----------------------------------
    # Primitive variables
    # -----------------------------------
    def prim_fn(xx, tt):
        return safe_prim_vars(net, xx, tt)  # (rho, u, p, nui)

    
    
    rho, u, p, nui = prim_fn(x, t)
    nu = nui * nui

    #u_fn = lambda x, t: prim_fn(x, t)[1]
    #u_x = grad_x(u_fn)(x, t)

    #lam = 1/(0.08*(jnp.abs(u_x) +  1))

    # -----------------------------------
    # Conservative variables
    # -----------------------------------
    def cons_vars(xx, tt):
        rho, u, p, _ = prim_fn(xx, tt)
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        return jnp.array([rho, rho * u, E])

    # -----------------------------------
    # Time derivatives
    # -----------------------------------
    U_t = jax.jacobian(cons_vars, argnums=1)(x, t)
    rho_t, rhou_t, E_t = U_t

    # -----------------------------------
    # HLLC fluxes (finite-volume style)
    # -----------------------------------
    #F_plus  = hllc_flux_safe(net, x, x + dx, t)
    #F_minus = hllc_flux_safe(net, x - dx, x, t)


    F_plus  = hllc_flux(net, x, x + dx, t)
    F_minus = hllc_flux(net, x - dx, x, t)

    Fx = (F_plus - F_minus) / dx

    # -----------------------------------
    # Viscous terms
    # -----------------------------------
    rho_xx  = grad_x(lambda xx, tt: grad_x(lambda x2, t2: prim_fn(x2, t2)[0])(xx, tt))(x, t)
    rhou_xx = grad_x(lambda xx, tt: grad_x(lambda x2, t2: cons_vars(x2, t2)[1])(xx, tt))(x, t)
    E_xx    = grad_x(lambda xx, tt: grad_x(lambda x2, t2: cons_vars(x2, t2)[2])(xx, tt))(x, t)

    # -----------------------------------
    # Residuals
    # -----------------------------------
    R0 = rho_t  + Fx[0] #- nu * rho_xx
    R1 = rhou_t + Fx[1] #- nu * rhou_xx
    R2 = E_t    + Fx[2] #- nu * E_xx
    R3 = nu     # positivity constraint

    return R0, R1, R2, R3





def hllc_flux(net, xl, xr, t, gamma=1.4):
    eps = 1e-8
    rho_floor = 1e-6
    p_floor   = 1e-6

    # --------------------------------------------------
    # Primitive variables (SAFE)
    # --------------------------------------------------
    def prim_vars(x):
        rho, u, p, _ = net[0](x, t)
        rho = jnp.maximum(rho, rho_floor)
        p   = jnp.maximum(p,   p_floor)
        return rho, u, p

    # --------------------------------------------------
    # Conservative variables
    # --------------------------------------------------
    def cons_vars(rho, u, p):
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        return jnp.array([rho, rho * u, E])

    # --------------------------------------------------
    # Euler flux
    # --------------------------------------------------
    def euler_flux(rho, u, p):
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        return jnp.array([
            rho * u,
            rho * u**2 + p,
            u * (E + p)
        ])

    # --------------------------------------------------
    # Sound speed (SAFE)
    # --------------------------------------------------
    def sound_speed(rho, p):
        return jnp.sqrt(jnp.maximum(gamma * p / rho, eps))

    # --------------------------------------------------
    # States
    # --------------------------------------------------
    rhoL, uL, pL = prim_vars(xl)
    rhoR, uR, pR = prim_vars(xr)

    UL = cons_vars(rhoL, uL, pL)
    UR = cons_vars(rhoR, uR, pR)

    FL = euler_flux(rhoL, uL, pL)
    FR = euler_flux(rhoR, uR, pR)

    cL = sound_speed(rhoL, pL)
    cR = sound_speed(rhoR, pR)

    # --------------------------------------------------
    # Wave speeds
    # --------------------------------------------------
    SL = jnp.minimum(uL - cL, uR - cR)
    SR = jnp.maximum(uL + cL, uR + cR)

    # --------------------------------------------------
    # Contact wave speed S*
    # --------------------------------------------------
    denom = rhoL * (SL - uL) - rhoR * (SR - uR)
    denom = jnp.where(jnp.abs(denom) < eps, eps, denom)

    Sstar = (
        pR - pL
        + rhoL * uL * (SL - uL)
        - rhoR * uR * (SR - uR)
    ) / denom

    # --------------------------------------------------
    # Star states (SAFE)
    # --------------------------------------------------
    def star_state(U, rho, u, p, S):
        denom = S - Sstar
        denom = jnp.where(jnp.abs(denom) < eps, eps, denom)

        rho_star = rho * (S - u) / denom
        rho_star = jnp.maximum(rho_star, rho_floor)

        m_star = rho_star * Sstar

        E = U[2]
        E_star = (
            (S - u) * E - p * u + p * Sstar
        ) / denom

        return jnp.array([rho_star, m_star, E_star])

    UL_star = star_state(UL, rhoL, uL, pL, SL)
    UR_star = star_state(UR, rhoR, uR, pR, SR)

    # ------------------
    # Flux selection 
    # ------------------
    F = jnp.where(
        SL >= 0.0,
        FL,
        jnp.where(
            Sstar >= 0.0,
            FL + SL * (UL_star - UL),
            jnp.where(
                SR > 0.0,
                FR + SR * (UR_star - UR),
                FR
            )
        )
    )

    return F


def hllc_flux_safe(net, xl, xr, t):

    rhoL, uL, pL, _ = safe_prim_vars(net, xl, t)
    rhoR, uR, pR, _ = safe_prim_vars(net, xr, t)

    cL = jnp.sqrt(gamma * pL / rhoL)
    cR = jnp.sqrt(gamma * pR / rhoR)

    SL = jnp.minimum(uL - cL, uR - cR)
    SR = jnp.maximum(uL + cL, uR + cR)

    den = rhoL * (SL - uL) - rhoR * (SR - uR)
    den = jnp.where(jnp.abs(den) < 1e-6, 1e-6, den)

    Sstar = (
        pR - pL
        + rhoL * uL * (SL - uL)
        - rhoR * uR * (SR - uR)
    ) / den

    # Conservative states
    def cons(rho, u, p):
        E = p / (gamma - 1) + 0.5 * rho * u**2
        return jnp.array([rho, rho * u, E])

    UL = cons(rhoL, uL, pL)
    UR = cons(rhoR, uR, pR)

    FL = jnp.array([rhoL*uL,
                     rhoL*uL**2 + pL,
                     uL*(UL[2] + pL)])

    FR = jnp.array([rhoR*uR,
                     rhoR*uR**2 + pR,
                     uR*(UR[2] + pR)])

    cmax = jnp.maximum(jnp.abs(uL) + cL, jnp.abs(uR) + cR)

    # Rusanov fallback
    FRus = 0.5*(FL + FR) - 0.5*cmax*(UR - UL)

    return FRus


@eqx.filter_jit
def neumann_residual(net, x, t):
    rho_x = jax.grad(lambda xx: prim_vars(net, xx, t)[0])(x)
    u_x   = jax.grad(lambda xx: prim_vars(net, xx, t)[1])(x)
    p_x   = jax.grad(lambda xx: prim_vars(net, xx, t)[2])(x)
    return rho_x, u_x, p_x



@eqx.filter_jit
def loss_fn(net, w_d, w_f, xy_f, xt_u, u_data, xt_bc):

    # data loss
    pred = jax.vmap(net[0])(xt_u[:, 0], xt_u[:, 1])
    rho_p, u_p, p_p = pred[:, 0], pred[:, 1], pred[:, 2]
    rho_d, u_d, p_d = u_data[:, 0], u_data[:, 1], u_data[:, 2]

    ## It was: [10, 1, 10]
    loss_data = (
        10*jnp.mean((rho_p - rho_d) ** 2)
        + jnp.mean((u_p - u_d) ** 2)  
        + 10*jnp.mean((p_p - p_d) ** 2)
    )

    dx = 5e-05
    # PDE residual
    #R = jax.vmap(euler_residual, in_axes=(None, 0, 0))(net, xy_f[:, 0], xy_f[:, 1])
    R = jax.vmap(euler_residual_hllc, in_axes=(None, 0, 0, None))(net, xy_f[:, 0], xy_f[:, 1], dx)

    R = jnp.stack(R, axis=-1)  # shape (N,7)
    loss_f = jnp.mean(R[:, :-1]**2) + jnp.mean(R[:, -1]**2) 
    #R_s = jax.vmap(lambda xt: entropy_residual(net, xt[0], xt[1]))(xy_f)
    #loss_entropy = jnp.mean(jnp.maximum(-R_s, 0.0)**2)
    #R = loss_f #+ loss_entropy

    # boundary condition
    bc = jax.vmap(neumann_residual, in_axes=(None, 0, 0))(
         net, xt_bc[:, 0], xt_bc[:, 1]
    )
    loss_bc = sum(jnp.mean(b**2) for b in bc)

    return w_d * (loss_data + loss_bc) + w_f * loss_f


def initial_condition_sod(x):
    rho = jnp.where(x <= 0.5, 1.0, 0.125)
    p   = jnp.where(x <= 0.5, 1.0, 0.1)
    u   = jnp.zeros_like(x)
    return rho, u, p



def load(filename, model):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model) 
   

def load_sol(filename):
    d = np.load(filename, allow_pickle=True)
    x = d[:, 0].reshape(-1, 1)
    r = d[:, 1].reshape(-1, 1)
    u = d[:, 2].reshape(-1, 1)
    p = d[:, 3].reshape(-1, 1)

    return jnp.array(x), jnp.array(r), jnp.array(u), jnp.array(p)
    


if __name__=="__main__":   
    x_min=0
    x_max=1
    t_min = 0.0
    t_max=0.15
    Nx=1001 #1001
    Nt=301 #301
    lambda_d = 1.0
    lambda_f = 1.0
    niters = 5001

    x = np.linspace(x_min, x_max, Nx)                                   
    t = np.linspace(0, t_max, Nt)                                     
    t_grid, x_grid = np.meshgrid(t, x)                                 
    T = t_grid.flatten()[:, None]                                      
    X = x_grid.flatten()[:, None]                                      
    x_int = X[:, 0][:, None]                                        
    t_int = T[:, 0][:, None]                                        
    x_f_train = np.hstack((x_int, t_int))

    
    # Initial Condition
    x = np.linspace(x_min, x_max, Nx)                                   
    t = np.linspace(0, t_max, Nt)                                       
    t_grid, x_grid = np.meshgrid(t, x)                                 
    T = t_grid.flatten()[:, None]                                      
    X = x_grid.flatten()[:, None]                                      
    x_ic = x_grid[:, 0][:, None]                                   
    t_ic = t_grid[:, 0][:, None]                                   
    xt_ic = np.hstack((x_ic, t_ic))

    x_lb = x_grid[0, :][:, None]   # first x row
    t_lb = t_grid[0, :][:, None]
    xt_lb = np.hstack((x_lb, t_lb))

    x_rb = x_grid[-1, :][:, None]   # first x row
    t_rb = t_grid[0, :][:, None]
    xt_rb = np.hstack((x_rb, t_rb))

    xt_bc = np.vstack((xt_lb, xt_rb))


    plt.scatter(xt_bc[:, 0], xt_bc[:, 1])
    plt.savefig("bc.png")

    
    rho_ic, u_ic, p_ic = initial_condition_sod(x_ic)
    rho_ic = rho_ic.reshape(-1, 1)
    u_ic = u_ic.reshape(-1, 1)
    p_ic = p_ic.reshape(-1, 1)
    u_ic = np.concatenate((rho_ic, u_ic, p_ic), axis=1)
    

  
    # Residual points

    # Convert np to jnp array: Train Data
    xy_f = jnp.array(x_f_train, dtype=jnp.float64)
    x_u_train = jnp.array(xt_ic, dtype=jnp.float64)
    u_train = jnp.array(u_ic, dtype=jnp.float64)

    # Convert np to jnp array: Test Data
    x_star, r_star, u_star, p_star = load_sol("xrup.npy")
    t_star = t_max*np.ones_like(x_star)
    xy_star = jnp.concatenate((x_star, t_star), axis=1)

    # Initiate PINN model                                                                                                   
    key = jr.PRNGKey(1234)
    key, init_key = jr.split(key)
    pinn = EulerPINN(2, 4, init_key)
    pinn_1 = init_linear_weight(pinn, trunc_init, init_key)

    key = jr.PRNGKey(5678)
    key, init_key = jr.split(key)
    pinn = EulerPINN(2, 1, init_key)
    pinn_2 = init_linear_weight(pinn, trunc_init, init_key)

    pinn = (pinn_1, )
    ## To load the Equinox model 
    #pinn = load(MODEL_FILE_NAME, pinn) 

    schedule = optax.linear_schedule(init_value=0.0001, end_value=0.00001,transition_steps=1000)
    lr = 1e-04
    optimizer = optax.adamw(learning_rate=lr)
    #optimizer = soap(learning_rate=3e-3, b1=0.95, b2=0.95, weight_decay=0.01, precondition_frequency=5)
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_inexact_array))
    
    @eqx.filter_jit
    def train_step_opt(network, state):
        l, grads = eqx.filter_value_and_grad(loss_fn)(network, lambda_d, lambda_f, xy_f, x_u_train, u_train, xt_bc)

        updates, new_state = optimizer.update(grads, state, network)
        new_network = eqx.apply_updates(network, updates) 
        return new_network, new_state, l    
    
    loss_history = []
    error_l2_r_list =[]
    error_l2_u_list =[] 
    error_l2_p_list =[] 
 
    print("Training started!")
    t0 = time.time()
    

    
    for epoch in range(0, niters):
        pinn, opt_state, loss = train_step_opt(pinn, opt_state)
        if epoch % 100 == 0:
            uvp_pred = jax.vmap(pinn[0], in_axes=(0, 0))(xy_star[:, 0], xy_star[:, 1])       
            ##### Slice the predicted value in vectors
            r_pred = uvp_pred[:, 0].reshape(-1, 1) 
            u_pred = uvp_pred[:, 1].reshape(-1, 1) 
            p_pred = uvp_pred[:, 2].reshape(-1, 1)

            
            # Add entry to loss
            loss_history.append(loss) 


            # Compute Relative Error
            r_err_l2 = jnp.linalg.norm((r_pred - r_star), 2) / jnp.linalg.norm(r_star, 2) 
            u_err_l2 = jnp.linalg.norm((u_pred - u_star), 2) / jnp.linalg.norm(u_star, 2) 
            p_err_l2 = jnp.linalg.norm((p_pred - p_star), 2) / jnp.linalg.norm(p_star, 2)  
            
            # Store history
            error_l2_r_list.append(r_err_l2)
            error_l2_u_list.append(u_err_l2)
            error_l2_p_list.append(p_err_l2) 

            # Round off the error
            r_err_l2 = np.round(r_err_l2, 5)
            u_err_l2 = np.round(u_err_l2, 5)
            p_err_l2 = np.round(p_err_l2, 5)


            # Print Loss
            print("===================================") 
            print(f"Epoch: {epoch}, loss: {loss:.7f}")  
            print("===================================")

            # Print L2-Relative Error
            print("========================================================================") 
            print(f"Epoch: {epoch}, L2-Error of (r, ru, e): ({r_err_l2*100:.4f}%, {u_err_l2*100:.4f}%, {p_err_l2*100:.4f}%)")  
            print("========================================================================")


            plot_solution(xy_star[:, 0], r_star, r_pred, "rho", f"rho_epoch_{epoch}.png")
            plot_solution(xy_star[:, 0], u_star, u_pred, "u", f"u_epoch_{epoch}.png")
            plot_solution(xy_star[:, 0], p_star, p_pred, "p", f"p_epoch_{epoch}.png")
 

    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)
    t1 = time.time() 
    print(f"Elapsed Time for Adam: {t1-t0} sec") 

    # # Quasi-Nwton Part
    params, static = eqx.partition(pinn, eqx.is_inexact_array)

    static_args = {"w_d": lambda_d,
                   "w_f": lambda_f,
                   "xt_u": x_u_train,
                   "u_data": u_train,
                   "xt_bc": xt_bc}



    def make_loss(xy_f, params):
        return partial(loss_fn, w_d=static_args["w_d"],
                       w_f=static_args["w_f"],
                       xy_f=xy_f,
                       xt_u = static_args["xt_u"],
                       u_data = static_args["u_data"],
                       xt_bc = static_args["xt_bc"]
                       )

    def loss_newton(dynamic_model, static_model, loss_local_fn):
        model = eqx.combine(dynamic_model, static_model)
        return loss_local_fn(model) 
    
    # # Solvers and Lineae Search for Quasi-newton   
    #solver = SSBFGSTrustRegion(rtol=1e-14, atol=1e-14, verbose=["loss"]) 
    
    #  # Run quasi-Newton Optimizer
    #solver = SSBFGSTrustRegion(rtol=1e-14, atol=1e-14, verbose=["loss"])
    solver = BroydenTrustRegion(rtol=1e-14, atol=1e-14, verbose=["loss"])

    #solver = BroydenBacktrackingZoom(rtol=1e-14, atol=1e-14, verbose=["loss"])
    solver = optx.BestSoFarMinimiser(solver)

 
    t3 = time.time()
    max_outer_iter = 1
    steps_per_iter = 20000
    
    for itr in range(max_outer_iter):
        #print(f"In outer loop: {itr}")
        #lb = jnp.array([[x_min, t_min]])
        #ub = jnp.array([[x_max, t_max]])
        #x_f_train = lb + (ub-lb)*lhs(2, 1000)
        #xy_f = jnp.asarray(x_f_train)
        loss_local_fn = make_loss(xy_f, params)
        sol = optx.minimise(lambda w, _: loss_newton(w, static, loss_local_fn), solver, params, max_steps=steps_per_iter, throw=False) 
        params = sol.value 

    
    t4 = time.time()
    print(f"Elapsed time for BFGS Stage 1: {t4-t3} sec")
       
    # # Print Elapsed Time
    print(f"Elapsed time for BFGS: {t4-t3} sec")
    bfgs_steps =  sol.stats["num_steps"]
    print(f"BFGS Steps: {bfgs_steps}") 

    # #### Infer stahge after Quasinewton loop
    pinn = eqx.combine(sol.value, static) 
    uvp_pred = jax.vmap(pinn[0], in_axes=(0, 0))(xy_star[:, 0], xy_star[:, 1])  

    ##### Slice the predicted value in vectors
    r_pred = uvp_pred[:, 0].reshape(-1, 1) 
    u_pred = uvp_pred[:, 1].reshape(-1, 1) 
    p_pred = uvp_pred[:, 2].reshape(-1, 1)


        
    # Compute Relative Error
    r_err_l2 = jnp.linalg.norm((r_pred - r_star), 2) / jnp.linalg.norm(r_star, 2) 
    u_err_l2 = jnp.linalg.norm((u_pred - u_star), 2) / jnp.linalg.norm(u_star, 2) 
    p_err_l2 = jnp.linalg.norm((p_pred - p_star), 2) / jnp.linalg.norm(p_star, 2)  
    
    # Store history
    error_l2_r_list.append(r_err_l2)
    error_l2_u_list.append(u_err_l2)
    error_l2_p_list.append(p_err_l2) 
    
    # Round off the error
    r_err_l2 = np.round(r_err_l2, 5)
    u_err_l2 = np.round(u_err_l2, 5)
    p_err_l2 = np.round(p_err_l2, 5)

    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)


    epoch_quasi=max_outer_iter * steps_per_iter


    plot_solution(xy_star[:, 0], r_star, r_pred, "rho", f"rho_epoch_{epoch_quasi}.png")
    plot_solution(xy_star[:, 0], u_star, u_pred, "u", f"u_epoch_{epoch_quasi}.png")
    plot_solution(xy_star[:, 0], p_star, p_pred, "p", f"p_epoch_{epoch_quasi}.png")
 

    plt.figure()
    plt.plot(xy_star[:, 0], r_pred, "--k")
    plt.plot(xy_star[:, 0], u_pred, "--b")
    plt.plot(xy_star[:, 0], p_pred, "--r")
    plt.xlabel("x")
    plt.ylabel("rho, u, p")
    plt.savefig("rup.png")
    plt.close()
    
    
    plt.figure()
    plt.plot(xy_star[:, 0], r_pred, "--k")
    plt.xlabel("x")
    plt.ylabel("rho")
    plt.savefig("r.png")
    plt.close()

    plt.figure()
    plt.plot(xy_star[:, 0], u_star, "-g", lw=2.0, label="ru")
    plt.plot(xy_star[:, 0], u_pred, "--k")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.savefig("u.png")
    plt.close()



    plt.figure()
    plt.plot(xy_star[:, 0], p_star, "-g", lw=2.0, label="e")
    plt.plot(xy_star[:, 0], p_pred, "--k")
    plt.xlabel("x")
    plt.ylabel("p")
    plt.savefig("p.png")
    plt.close()


     # Print L2-Relative Error
    print("========================================================================") 
    print(f"After Quasi-Newton Optimization L2-Error of r: {r_err_l2*100:.4f}")  
    print("========================================================================")

    print("========================================================================") 
    print(f"After Quasi-Newton Optimization L2-Error of ru: {u_err_l2*100:.4f}")  
    print("========================================================================")

    print("========================================================================") 
    print(f"After Quasi-Newton Optimization L2-Error of e: {p_err_l2*100:.4f}")  
    print("========================================================================")

    np.save("xy_star.npy", xy_star, allow_pickle=True)
    np.save("uvp_pred.npy", uvp_pred, allow_pickle=True)


    
