import pdb

import jax.numpy as jnp
from jax import vmap
import jax


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


def gaussian_2d(height, center, width, cv1, cv2):
    width1, width2 = width
    center1, center2 = center

    term1 = (cv1 - center1)**2 / (2*width1**2)
    term2 = (cv2 - center2)**2 / (2*width2**2)

    return height * jnp.exp(-(term1 + term2))
gaussian_mixture_2d = vmap(gaussian_2d, (0, 0, 0, None, None)) # takes 3 lists: heights, centers, and widths
sum_of_gaussians_2d = lambda heights, centers, widths, cv1, cv2: jnp.sum(gaussian_mixture_2d(heights, centers, widths, cv1, cv2))

steep_sigmoid = lambda x: 1 / (1 + jnp.exp(-5*x))

# cv1 is our n_bp
# cv2 is our interstrand distnace
def get_repulsive_wall_fn(d_critical, wall_strength):
    def repulsive_wall_fn(heights, centers, widths, cv1, cv2):
        g_sum = sum_of_gaussians_2d(heights, centers, widths, cv1, cv2)
        # wall_dg = jax.nn.sigmoid(cv2 - d_critical) * wall_strength
        wall_dg = steep_sigmoid(cv2 - d_critical) * wall_strength
        return g_sum + wall_dg
    return repulsive_wall_fn




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

    # Test utilities for single CV
    n_gauss = 4
    heights = jnp.full(n_gauss, 1)
    centers = jnp.array([0., 1., 2., 3.])
    widths = jnp.full(n_gauss, 1)


    x = 1.5

    vmapped_mixture = gaussian_mixture(heights, centers, widths, x)

    vmapped_sum = sum_of_gaussians(heights, centers, widths, x)
    iterative_sum = jnp.sum(jnp.array([gaussian(a, b, c, x) for a, b, c in zip(heights, centers, widths)]))

    pdb.set_trace()


    # Test utilities for two CVs
    n_guass = 4
    heights = jnp.full(n_gauss, 1.0)
    centers = jnp.array([[0.0, 0.0],
                         [1.0, 1.0],
                         [2.0, 2.0],
                         [3.0, 3.0]])
    widths = jnp.full((n_gauss, 2), 1.0)
    cv1 = 1.0
    cv2 = 1.0
    vmapped_mixture = gaussian_mixture_2d(heights, centers, widths, cv1, cv2)
    vmapped_sum = sum_of_gaussians_2d(heights, centers, widths, cv1, cv2)
    iterative_sum = jnp.sum(jnp.array([gaussian_2d(a, b, c, cv1, cv2) for a, b, c in zip(heights, centers, widths)]))

    d_critical = 2.0
    wall_strength = 1e3
    wall_fn = get_repulsive_wall_fn(d_critical, wall_strength)
    before_critical = wall_fn(heights, centers, widths, cv1, cv2)
    cv2 = 2.0
    after_critical = wall_fn(heights, centers, widths, cv1, cv2)

    pdb.set_trace()

    print("done")
