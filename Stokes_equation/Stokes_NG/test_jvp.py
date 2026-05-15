import jax
import jax.numpy as jnp



def f(x):
    return jnp.array([
        x[0]**2,
        x[0]*x[1],
        jnp.sin(x[0] + x[1] + x[2])
    ])


def hessian_single(fi, x):
    d = x.shape[0]
    I = jnp.eye(d)
    grad_f = jax.grad(fi)
    H = jnp.stack([jax.jvp(grad_f, (x, ), (v, ))[1] for v in I], axis=1)
    return H


def hessian(f, x):

    m = f(x).shape[0]

    H = jnp.stack([hessian_single(lambda x: f(x)[i], x) for i in range(m)], axis=0)


    return H

                

    
    




x = jnp.array([1.0, 2.0, 3.0])

y = f(x)

print(f"x: {x}")
print(f"y: {y}")
print(f"x shape: {x.shape}")
print(f" f shape: {y.shape}")


d = len(x)

I = jnp.eye(d)

print([e for e in I])


jac = jnp.vstack([jax.jvp(f, (x,), (e, ))[1] for e in I]).T

print(jac)


print(hessian(f, x))

