import pdb
import numpy as onp
from tqdm import tqdm
import matplotlib.pyplot as plt

import jax
from jax import vmap, jit, flatten_util, grad, value_and_grad
import jax.numpy as jnp
import optax




@jit
def f1(args):
    x1 = args["xs"][0]
    x2 = args["xs"][1]
    y1 = args["ys"][0]
    y2 = args["ys"][1]
    z = args["z"]

    return x1*x2**2 + (y1-y2)*z
grad_f1 = jit(value_and_grad(f1))


@jit
def f2(args):
    x1 = args["xs"][0]
    x2 = args["xs"][1]
    y1 = args["ys"][0]
    y2 = args["ys"][1]
    z = args["z"]

    return (x1+x2)*(-z) + y1*y2
grad_f2 = jit(value_and_grad(f2))




my_args = {
    "xs": jnp.array([1.5, -0.5]),
    "ys": jnp.array([-2.5, 0.25]),
    "z": 1.0
}

# Note: ravel_fn is a callable for unflattening a 1D vector of the same length back to a pytree of the same structure as my_args
my_args_flattened, ravel_fn = flatten_util.ravel_pytree(my_args)

val1, g1 = grad_f1(my_args)
val2, g2 = grad_f2(my_args)



@jit
def f1_flat(args_flat):
    args = ravel_fn(args_flat)
    return f1(args)
grad_f1_flat = jit(value_and_grad(f1_flat))


@jit
def f2_flat(args_flat):
    args = ravel_fn(args_flat)
    return f2(args)
grad_f2_flat = jit(value_and_grad(f2_flat))


val1_flat, g1_flat = grad_f1_flat(my_args_flattened)
val2_flat, g2_flat = grad_f2_flat(my_args_flattened)

all_grads = jnp.array([g1_flat, g2_flat])
m = all_grads.shape[0]

def normalize(g):
    return g / jnp.linalg.norm(g)
all_grads_norm = vmap(normalize)(all_grads)

assert(m == 2)
all_grads_norm_pinv = jnp.linalg.pinv(all_grads_norm)
gu_unnorm = all_grads_norm_pinv @ jnp.ones(m)
gu_norm = normalize(gu_unnorm)


proj_dists = vmap(lambda gi: jnp.dot(gi, gu_norm))(all_grads) # Note we use the unnormalized gradients
g_config = proj_dists.sum() * gu_norm


# Compare g_config with the following!
my_fn = lambda args_flat: f1_flat(args_flat) + f2_flat(args_flat)
grad(my_fn)(my_args_flattened)




# NEXT: try to do an optimization with both vanilla and config grads! Also see if we can come up with some more conflicting functions to try this out on.

lr = 1e-3

def combined_fn_flat(args_flat):
    args = ravel_fn(args_flat)
    f1_val = f1(args)
    f2_val = f2(args)
    return f1_val + f2_val, (f1_val, f2_val)
vanilla_grad_fn = value_and_grad(combined_fn_flat, has_aux=True)

optimizer = optax.adam(learning_rate=lr)
params = my_args_flattened
opt_state = optimizer.init(params)

n_epochs = 100
f1_vals = list()
f2_vals = list()
loss_vals = list()
for i in tqdm(range(n_epochs)):
    (loss, (f1_val, f2_val)), vanilla_grads = vanilla_grad_fn(params)
    f1_vals.append(f1_val)
    f2_vals.append(f2_val)
    loss_vals.append(loss)

    updates, opt_state = optimizer.update(vanilla_grads, opt_state, params)
    params = optax.apply_updates(params, updates)

plt.plot(f1_vals, label="F1", linestyle="--", color="red")
plt.plot(f2_vals, label="F2", linestyle="--", color="blue")
plt.plot(loss_vals, label="total", linestyle="--", color="black")

plt.ylabel("Loss Value")
plt.xlabel("Iteration")
plt.legend()
plt.show()
