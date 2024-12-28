import jax
from jax import lax, custom_vjp
import jax.numpy as jnp


def pytree_where(condition, x, y):
    """Generalized `jnp.where` that can select between pytrees."""
    def where_leaf(leaf_x, leaf_y):
        if isinstance(leaf_x, (jnp.ndarray, jax.Array)):
            return jnp.where(condition, leaf_x, leaf_y)
        return leaf_x if leaf_condition else leaf_y

    return jax.tree_map(where_leaf, x, y)


def clip_pytree(pytree, min_val, max_val):
    """Generalized `jnp.clip` that clips float-valued leaves of pytrees."""
    def clip_if_float(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.clip(x, min_val, max_val)
        return x

    return jax.tree_map(clip_if_float, pytree)

def is_float_array(x):
    """Checks if an element is a float-valued jax array."""
    return isinstance(x, (jnp.ndarray, jax.Array)) and jnp.issubdtype(x.dtype, jnp.floating)

def sum_squares(x):
    """Computes the sum of squares of an array of floats, otherwise returns 0.0"""
    if is_float_array(x):
        # Handle scalar arrays
        if x.ndim == 0:
            return x * x
        return jnp.sum(jnp.square(x))
    return 0.0

def compute_pytree_norm(grads):
    """Computes the norm of a pytree, only including float-valued leaves"""
    global_norm = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda acc, x: acc + sum_squares(x),
        grads,
        0.0
    ))
    return global_norm

def clip_pytree_norm(pytree, max_norm):
    """Clip pytree to a maximum global norm, ignoring integer arrays and handling scalar arrays."""

    global_norm = compute_pytree_norm(pytree)

    factor = jnp.minimum(max_norm / (global_norm + 1e-6), 1.0)

    def scale_if_float(g):
        if is_float_array(g):
            return g * factor
        return g

    return jax.tree_map(scale_if_float, pytree)


def get_clip_grad_fn(mode, x1, x2=None):
    """Constructs an identity function that clips the gradient.

    Mode determines the method of clipping the gradient. 'norm' will rescale
    via a maximum norm, and `raw` will simply clip based on a min/max value.

    If mode == 'norm', x1 is the max norm and x2 is unused.

    If mode == 'raw', x1 and x2 are the min and max values, respectively.
    """

    if mode not in ['norm', 'raw']:
        raise ValueError(f"Invalid mode: {mode}")

    @custom_vjp
    def clip_gradient(should_clip, x):
        return x  # identity function

    def clip_gradient_fwd(should_clip, x):
        return x, (should_clip)  # save bounds as residuals

    def clip_gradient_bwd(res, g):
        should_clip = res
        if mode == "raw":
            clipped = clip_pytree(g, x1, x2)
        elif mode == "norm":
            clipped = clip_pytree_norm(g, x1)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        maybe_clipped = pytree_where(should_clip, clipped, g)
        return (None, maybe_clipped)  # use None to indicate zero cotangent for should_clip

    clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

    return clip_gradient




if __name__ == "__main__":
    import functools
    from tqdm import tqdm
    import pdb
    from jax_md import simulate, space, quantity


    clip_gradient = get_clip_grad_fn("norm", 0.1)
    # clip_gradient = get_clip_grad_fn("raw", -0.1, 0.1)

    kT_fn = lambda p, m: quantity.temperature(momentum=p, mass=m)

    key = jax.random.PRNGKey(0)
    spatial_dimension = 3
    LANGEVIN_PARTICLE_COUNT = 10
    LANGEVIN_DYNAMICS_STEPS = 100000
    dtype = jnp.float32

    key, R_key, R0_key, T_key, masses_key = jax.random.split(key, 5)
    mass = jax.random.uniform(
        masses_key, (LANGEVIN_PARTICLE_COUNT,),
        minval=0.1, maxval=10.0, dtype=dtype)

    R_init = jax.random.normal(
        R_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    _, shift = space.free()
    T =  1.0

    clip_every = 1

    def sim_fn(R0):
        E = functools.partial(
            lambda R, R0, **kwargs: jnp.sum((R - R0) ** 2), R0=R0)

        init_fn, apply_fn = simulate.nvt_langevin(
            E, shift, dt=1e-2, kT=T, gamma=0.3)
        apply_fn = jax.jit(apply_fn)

        def step_fn(state, i):
            state = clip_gradient(i % clip_every == 0, state)
            state = apply_fn(state)
            return state, None

        state = init_fn(key, R_init, mass=mass, T_initial=dtype(1.0))
        fin_state, _ = lax.scan(step_fn, state, jnp.arange(LANGEVIN_DYNAMICS_STEPS))

        return fin_state

    R0 = jax.random.normal(
        R0_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    fin_state = sim_fn(R0)

    def loss_fn(R0):
        fin_state = sim_fn(R0)
        return fin_state.position.sum(), fin_state

    (loss, fin_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(R0)
    print(f"Loss: {loss}")
    print(f"Grads: {grads}")
    print(f"Grads norm: {compute_pytree_norm(grads)}")
