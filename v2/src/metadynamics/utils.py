import pdb

import jax.numpy as jnp
from jax import vmap


"""
# Returns gaussian with height `a`, center `b`, and width `c`
def get_gaussian(a, b, c):
    def gaussian(x):
        return a * jnp.exp(-(x-b)**2/(2*c**2))
    return gaussian
"""

def gaussian(height, center, width, x):
    return height * jnp.exp(-(x-center)**2/(2*width**2))
gaussian_mixture = vmap(gaussian, (0, 0, 0, None)) # takes 3 lists: heights, centers, and widths

sum_of_gaussians = lambda heights, centers, widths, x: jnp.sum(gaussian_mixture(heights, centers, widths, x))
# sum_of_gaussians = lambda heights, centers, widths, x: jnp.sum(gaussian_mixture(heights, centers, widths, x)) + 1*jnp.sum(gaussian_mixture(heights, -centers, widths, x))

def get_height_fn(height_0, well_tempered=False, delta_T=1.0, kt=None):
    if well_tempered:
        if kt is None:
            raise RuntimeError(f"Value of kT required for well-tempered metadynamics")

        def height_fn(curr_bias):
            return height_0 * jnp.exp(-curr_bias / (kt*delta_T))
        return height_fn

    else:
        return lambda curr_bias: height_0 # FIXME: May have to take more than the body at some point


if __name__ == "__main__":
    n_gauss = 4
    heights = jnp.full(n_gauss, 1)
    centers = jnp.array([0., 1., 2., 3.])
    widths = jnp.full(n_gauss, 1)


    x = 1.5

    vmapped_mixture = gaussian_mixture(heights, centers, widths, x)

    vmapped_sum = sum_of_gaussians(heights, centers, widths, x)
    iterative_sum = jnp.sum(jnp.array([gaussian(a, b, c, x) for a, b, c in zip(heights, centers, widths)]))

    pdb.set_trace()

    print("done")
