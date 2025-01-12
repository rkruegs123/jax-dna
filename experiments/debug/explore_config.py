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
    # args = ravel_fn(args_flat)
    # val = f2(args)
    val = jnp.linalg.norm(args_flat)
    return val

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
vanilla_grads = grad(my_fn)(my_args_flattened)

@jit
def get_config_grad(all_grads):
    m = all_grads.shape[0]

    all_grads_norm = vmap(normalize)(all_grads)

    all_grads_norm_pinv = jnp.linalg.pinv(all_grads_norm)
    gu_unnorm = all_grads_norm_pinv @ jnp.ones(m)
    gu_norm = normalize(gu_unnorm)


    proj_dists = vmap(lambda gi: jnp.dot(gi, gu_norm))(all_grads) # Note we use the unnormalized gradients
    g_config = proj_dists.sum() * gu_norm
    return g_config





# NEXT: try to do an optimization with both vanilla and config grads! Also see if we can come up with some more conflicting functions to try this out on.

lr = 1e-3

def combined_fn_flat(args_flat):
    f1_val = f1_flat(args_flat)
    f2_val = f2_flat(args_flat)
    return f1_val + f2_val, (f1_val, f2_val)
vanilla_grad_fn = jit(value_and_grad(combined_fn_flat, has_aux=True))

optimizer_vanilla = optax.adam(learning_rate=lr)
params_vanilla = my_args_flattened
opt_state_vanilla = optimizer_vanilla.init(params_vanilla)

optimizer_cfg = optax.adam(learning_rate=lr)
params_cfg = my_args_flattened
opt_state_cfg = optimizer_cfg.init(params_cfg)

n_epochs = 1000

f1_vals_vanilla = list()
f2_vals_vanilla = list()
loss_vals_vanilla = list()

f1_vals_cfg = list()
f2_vals_cfg = list()
loss_vals_cfg = list()

for i in tqdm(range(n_epochs)):
    (loss, (f1_val, f2_val)), vanilla_grads = vanilla_grad_fn(params_vanilla)
    f1_vals_vanilla.append(f1_val)
    f2_vals_vanilla.append(f2_val)
    loss_vals_vanilla.append(loss)

    updates, opt_state_vanilla = optimizer_vanilla.update(vanilla_grads, opt_state_vanilla, params_vanilla)
    params_vanilla = optax.apply_updates(params_vanilla, updates)

    val1_cfg, g1_cfg = grad_f1_flat(params_cfg)
    val2_cfg, g2_cfg = grad_f2_flat(params_cfg)
    all_grads_cfg = jnp.array([g1_cfg, g2_cfg])
    cfg_grads = get_config_grad(all_grads_cfg)

    f1_vals_cfg.append(val1_cfg)
    f2_vals_cfg.append(val2_cfg)
    loss_vals_cfg.append(val1_cfg + val2_cfg)

    updates, opt_state_cfg = optimizer_cfg.update(cfg_grads, opt_state_cfg, params_cfg)
    params_cfg = optax.apply_updates(params_cfg, updates)


plt.plot(f1_vals_vanilla, label="F1", linestyle="--", color="red")
plt.plot(f2_vals_vanilla, label="F2", linestyle="--", color="blue")
plt.plot(loss_vals_vanilla, label="total", linestyle="--", color="black")

plt.plot(f1_vals_cfg, label="F1", linestyle="-", color="red")
plt.plot(f2_vals_cfg, label="F2", linestyle="-", color="blue")
plt.plot(loss_vals_cfg, label="total", linestyle="-", color="black")

plt.ylabel("Loss Value")
plt.xlabel("Iteration")
plt.legend()
plt.show()
