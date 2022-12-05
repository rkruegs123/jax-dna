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

def get_height_fn(height_0, well_tempered=False):
    if well_tempered:
        """
        # From PySAGES -- have to read papers to figure out if V is just the bias potential or the total potential
        def next_height(pstate):
            V = evaluate_potential(pstate)
            return height_0 * np.exp(-V / (deltaT * kB))
        """
        raise NotImplementedError
    else:
        return lambda body: height_0 # FIXME: May have to take more than the body at some point


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
