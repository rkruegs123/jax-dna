import jax.numpy as jnp

def get_f1_smoothing_params(eps, x0, a, xc, x_low, x_high):

    def solve_f1_b(x):
        return a**2*(-jnp.exp(a*(3*x0 + 2*xc)) + 2*jnp.exp(a*(x + 2*x0 + 2*xc)) \
                     - jnp.exp(a*(2*x + x0 + 2*xc)))*jnp.exp(-2*a*x) \
                     / (2*jnp.exp(a*(x + 2*xc)) \
                        + jnp.exp(a*(2*x + x0)) \
                        - 2*jnp.exp(a*(2*x + xc)) \
                        - jnp.exp(a*(x0 + 2*xc)))
    solved_b_low = solve_f1_b(x_low)
    solved_b_high = solve_f1_b(x_high)

    def solve_f1_xc_star(x):
        return (a*x*jnp.exp(a*(x + 2*xc)) - a*x*jnp.exp(a*(x0 + 2*xc))
                + 2*jnp.exp(a*(x + 2*xc)) + jnp.exp(a*(2*x + x0)) - 2*jnp.exp(a*(2*x + xc))
                - jnp.exp(a*(x0 + 2*xc)))*jnp.exp(-2*a*xc)/(a*(jnp.exp(a*x) - jnp.exp(a*x0)))

    solved_xc_low = solve_f1_xc_star(x_low)
    solved_xc_high = solve_f1_xc_star(x_high)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def get_f2_smoothing_params(k, x0, xc, x_low, x_high):
    def solve_f2_b(x):
        return (x - x0)**2/(2*(x - xc)*(x - 2*x0 + xc))

    def solve_f2_xc_star(x):
        return (x*x0 - 2*x0*xc + xc**2)/(x - x0)

    solved_b_low = solve_f2_b(x_low)
    solved_b_high = solve_f2_b(x_high)

    solved_xc_low = solve_f2_xc_star(x_low)
    solved_xc_high = solve_f2_xc_star(x_high)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def get_f3_smoothing_params(r_star, eps, sigma):
    def solve_f3_b(x):
        return -36*sigma**6*(-2*sigma**6 + x**6)**2 \
            / (x**14*(-sigma + x)*(sigma + x)*(sigma**2 - sigma*x + x**2) \
               * (sigma**2 + sigma*x + x**2))

    def solve_f3_xc(x):
        return x*(-7*sigma**6 + 4*x**6)/(3*(-2*sigma**6 + x**6))

    solved_b = solve_f3_b(r_star)
    solved_xc = solve_f3_xc(r_star)

    return solved_b, solved_xc


def get_f4_smoothing_params(a, x0, delta_x_star):
    def solve_f4_b(x):
        return -a**2*(-x + x0)**2/(a*x**2 - 2*a*x*x0 + a*x0**2 - 1)

    solved_b_plus = solve_f4_b(x0 + delta_x_star)
    solved_b_minus = solve_f4_b(x0 - delta_x_star)

    def solve_f4_xc(x):
        return (-a*x*x0 + a*x0**2 - 1)/(a*(-x + x0))

    solved_xc_plus = solve_f4_xc(x0 + delta_x_star)
    solved_delta_xc_plus = solved_xc_plus - x0

    solved_xc_minus = solve_f4_xc(x0 - delta_x_star)
    solved_delta_xc_minus = x0 - solved_xc_minus

    return solved_b_plus, solved_delta_xc_plus


def get_f5_smoothing_params(a, x_star):
    def solve_f5_b(x, x0):
        return -a**2*(x - x0)**2/(a*x**2 - 2*a*x*x0 + a*x0**2 - 1)

    solved_b = solve_f5_b(x=x_star, x0=0.0)

    def solve_f5_xc(x, x0):
        return (a*x*x0 - a*x0**2 + 1)/(a*(x - x0))

    solved_xc = solve_f5_xc(x=x_star, x0=0.0)

    return solved_b, solved_xc