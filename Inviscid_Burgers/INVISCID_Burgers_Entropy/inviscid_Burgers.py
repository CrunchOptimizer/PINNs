import functools as ft
import lineax as lx
from collections.abc import Callable
from typing import Any, TypeVar
import equinox.internal as eqxi
from jaxtyping import Array, PyTree, Scalar
from typing import NamedTuple
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import scipy
import optax
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import scipy.io as sio
import math
import json
import chex
import optimistix as optx
from scipy.interpolate import griddata

import time


### PINN Training ####
np.random.seed(1234)
jax.config.update("jax_enable_x64", True)
MODEL_FILE_NAME = "./checkpoints/viscous_burgers.eqx"
###################################################################
Y = TypeVar("Y")
Out = TypeVar("Out")
Aux = TypeVar("Aux")


class DoglegMax(optx.AbstractGaussNewton[Y, Out, Aux]):
    """Dogleg with trust region shape given by the max norm instead of the two norm."""
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: optx.DoglegDescent[Y]
    search: optx.ClassicalTrustRegion[Y]
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = optx.max_norm
        self.descent = optx.DoglegDescent(
            linear_solver=lx.AutoLinearSolver(well_posed=False),
            root_finder=optx.Bisection(rtol=0.001, atol=0.001),
            trust_region_norm=optx.max_norm,
        )
        self.search = optx.ClassicalTrustRegion()
        self.verbose = frozenset()

        
class BFGSTrustRegion(optx.AbstractSSBFGS):
    """Standard BFGS + classical trust region update."""
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()


class BFGSDogleg(optx.AbstractSSBFGS):
    """BFGS Hessian + dogleg update."""
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.DoglegDescent(linear_solver=lx.SVD())
    verbose: frozenset[str] = frozenset()
    
class BFGSBacktracking(optx.AbstractSSBFGS):
    """Standard BFGS + backtracking line search."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.BacktrackingArmijo()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()


class BFGSDampedNewton(optx.AbstractSSBFGS):
    """BFGS Hessian + direct Levenberg Marquardt update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.DampedNewtonDescent()
    verbose: frozenset[str] = frozenset()


class BFGSIndirectDampedNewton(optx.AbstractSSBFGS):
    """BFGS Hessian + indirect Levenberg Marquardt update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.IndirectDampedNewtonDescent()
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

def load(filename, model):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)

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

    new_biases = jax.tree.map(lambda p_loc: 0.0 * jnp.abs(p_loc), biases)

    new_weights = [init_fn(weight, subkey)
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_bias, new_model, new_biases)

    return new_model



def steep_tanh(x, k=5.0):
    return jnp.tanh(k * x)


class NeuralNetwork(eqx.Module):
    layers: list
    units: int = 50
    n_in: int = 2
    n_out: int = 1
    n_layers:int = 4
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    activation: callable = jax.nn.tanh

    
    def __init__(self, key, x_min, x_max, y_min, y_max, activation=jax.nn.tanh):
        key_list = jax.random.split(key, 20)
 
        self.layers = [eqx.nn.Linear(self.n_in, self.units, key=key_list[0])]
        self.activation = activation

        for i in range(0, self.n_layers):
            self.layers.append(eqx.nn.Linear(self.units, self.units, key=key_list[i])) 

        self.layers.append(eqx.nn.Linear(self.units, self.n_out, key=key_list[10]))
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = y_min
        self.t_max = y_max
        

    def __call__(self, x, t):
        xt = jnp.hstack((x, t))
        for layer in self.layers[:-1]:
            #xt = jax.nn.tanh(layer(xt))
            xt = self.activation(layer(xt))
        return self.layers[-1](xt)

@eqx.filter_jit
def pde_residual(network, xx, tt):
    u_fn = lambda _xx, _tt: network[0](_xx, _tt)[0]
    v_fn = lambda _xx, _tt: network[1](_xx, _tt)[0]    
    
    v_x = jax.grad(v_fn, argnums=0)(xx, tt)
    u_x = jax.grad(u_fn, argnums=0)(xx, tt)
    u_t = jax.grad(u_fn, argnums=1)(xx, tt)
    u = u_fn(xx, tt)
    v = v_fn(xx, tt)


    # Residual-based viscosity
    R_u = u_t + v_x
    R_v = v - 0.5 * u**2

    eta_t = u * u_t
    q_x = u**2 * u_x
    E = jax.nn.relu(eta_t + q_x)
    return E, R_u, R_v



@eqx.filter_jit
def loss_fn(network, weight_d, weight_f, xy_r, xt, u):
    u_pred = jax.vmap(network[0], in_axes=(0, 0))(xt[:, 0], xt[:, 1])
    E, R_u, R_v = jax.vmap(pde_residual, in_axes=(None, 0, 0))(network, xy_r[:, 0], xy_r[:, 1])
    u_pred = u_pred.reshape(-1,1)
    u_loss = jnp.mean(jnp.square(u_pred - u))/jnp.mean(jnp.square(u))    
    loss_f = jnp.mean(jnp.square(R_u))  + jnp.mean(jnp.square(E)) + jnp.mean(jnp.square(R_v))
    total_loss = weight_d * (u_loss + u_loss)  + weight_f * loss_f
    return total_loss


if __name__ == "__main__":

    ## Data Parse:
    data = scipy.io.loadmat('Data/burgers_shock.mat')
    N_f = 50000
    N_u = 300
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
   

    print(f"X Star Shape: {X.shape} and {T.shape}")
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # Conver Numpy array to Jax Numpy
    x_u_train = jnp.array(X_u_train)
    u_train = jnp.array(u_train)
    x_f_train = jnp.array(X_f_train)
  

    

    # Training parameters
    niters = 1001 #101 #11 #16001
    lambda_d = 1.0
    lambda_f = 0.01 #1.0
    

    
    # Initiate PINN mode
    key = jr.PRNGKey(42)
    key, init_key = jr.split(key)    
    pinn = NeuralNetwork(init_key, lb[0], lb[1], ub[0], ub[1])
    pinn1 = init_linear_weight(pinn, trunc_init, init_key)


    key = jr.PRNGKey(1234)
    key, init_key = jr.split(key)    
    pinn = NeuralNetwork(init_key, lb[0], lb[1], ub[0], ub[1])
    pinn2 = init_linear_weight(pinn, trunc_init, init_key)

    pinn = (pinn1, pinn2)
    
    ## To load the Equinox model
    #pinn = load(MODEL_FILE_NAME, pinn)


    #schedule = optax.piecewise_constant_schedule(
    #    init_value=lr,
    #    boundaries_and_scales={
    #        int(1e4): 0.5,
    #        int(2e4): 0.2,
    #    }
    #)


    schedule = optax.linear_schedule(init_value=0.0001, end_value=0.00001,transition_steps=1000)
    #optimizer = optax.adam(learning_rate=schedule)

    lr = 1e-04
    optimizer = optax.adamw(learning_rate=lr)
    
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step_opt(network, state):
        l, grad = eqx.filter_value_and_grad(loss_fn)(network, lambda_d, lambda_f, x_f_train, x_u_train, u_train)
        updates, new_state = optimizer.update(grad, state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_state, l

    loss_history = []
    error_l2_u_list =[]

    #counter = tqdm(np.arange(N_EPOCHS))
    t0 = time.time()
    for epoch in range(0, niters):
        pinn, opt_state, loss = train_step_opt(pinn, opt_state)
        if epoch % 100 == 0:
            u_pred = jax.vmap(pinn[0], in_axes=(0, 0))(X_star[:, 0], X_star[:, 1])
            u_pred = u_pred.reshape(-1, 1)
            u_star = u_star.reshape(-1, 1)

            u_err_l2 = jnp.linalg.norm((u_pred - u_star), 2) / jnp.linalg.norm(u_star, 2)
            loss_history.append(loss)
            error_l2_u_list.append(u_err_l2)
            u_err_l2 = np.round(u_err_l2, 5)
            
            print("========================================================================")
            print(f"Epoch: {epoch}, loss: {loss:.7f}, L2-Error of u: {u_err_l2*100:.4f}")
            print("========================================================================")

    t1 = time.time()
    print(f"Elapsed Time for Adam: {t1-t0} sec")
    params, static = eqx.partition(pinn, eqx.is_inexact_array)

    loss_local_fn = lambda w: loss_fn(w, lambda_d, lambda_f, x_f_train, x_u_train, u_train)

    def loss_newton(dynamic_model, static_model):
        model = eqx.combine(dynamic_model, static_model)
        return loss_local_fn(model)

    

    solver = BroydenBacktrackingZoom(rtol=1e-14, atol=1e-14, verbose=["loss"])
    #solver = BFGSTrustRegion(rtol=1e-10, atol=1e-10, verbose=["loss"])
    solver = optx.BestSoFarMinimiser(solver)
    t3 = time.time()
    sol = optx.minimise(loss_newton, solver, params, args=static, max_steps=10000, throw=False)
    t4 = time.time()
    print(f"Elapsed time for BFGS: {t4-t3} sec")
    
    bfgs_steps =  sol.stats["num_steps"]
    print(f"BFGS Steps: {bfgs_steps}")
    pinn = eqx.combine(sol.value, static)
    u_pred = jax.vmap(pinn[0], in_axes=(0, 0))(X_star[:, 0], X_star[:, 1])
    u_star = u_star.reshape(-1, 1)
    u_err_l2 = jnp.linalg.norm((u_pred - u_star), 2) / jnp.linalg.norm(u_star, 2)
    print("========================================================================")
    print(f"L2-Error of u: {u_err_l2*100:.4f}")
    print("========================================================================")

    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)

    plt.figure()
    plt.scatter(X_star[:,0], X_star[:, 1], c=u_pred)
    plt.savefig("pinn.png")

    plt.figure()
    plt.scatter(X_star[:,0], X_star[:, 1], c=u_star)
    plt.savefig("act.png")

    u_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    np.save("u_pred.npy", u_pred, allow_pickle=True)
    np.save("u_act.npy", u_star, allow_pickle=True)
    np.save("xy_star.npy", X_star, allow_pickle=True )

    np.save("loss_history.npy", np.array(loss_history), allow_pickle=True)
    np.save("error_l2.npy", np.array(error_l2_u_list), allow_pickle=True)


    

    
    
