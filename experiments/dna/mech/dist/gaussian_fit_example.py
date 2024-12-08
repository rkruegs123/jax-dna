import pdb
import numpy as onp
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jaxopt import OptaxSolver
import optax
from jax import vmap


# https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
def compute_weighted_avg_and_var(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = jnp.average(values, weights=weights)
    variance = jnp.average((values-average)**2, weights=weights)
    return (average, variance)


# Negative log-likelihood function that uses additional metadata
def neg_log_likelihood(params, data, weights):
    mu, sigma = params
    n = data.shape[0]
    # Incorporating metadata in the log likelihood (just an illustrative example)
    weighted_data = data * weights
    # log_likelihood = -0.5 * n * jnp.log(2 * jnp.pi * sigma**2) - jnp.sum((weighted_data - mu) ** 2) / (2 * sigma**2)

    term1 = -0.5 * n * jnp.log(2 * jnp.pi * sigma**2)
    term2_denom = 2 * sigma**2

    sqr_res_fn = lambda idx: weights[idx]*n * (data[idx] - mu)**2
    all_sqr_res = vmap(sqr_res_fn)(jnp.arange(n))
    term2_num = jnp.sum(all_sqr_res)

    log_likelihood = term1 - term2_num / term2_denom
    return -log_likelihood



# Example data
# data = jnp.array([2.3, 2.5, 2.8, 3.0, 3.2, 3.3, 3.7, 3.8, 4.0, 4.2])
true_mean = 0     # Mean of the distribution
true_std_dev = 1  # Standard deviation of the distribution
size = 10000  # Number of samples

# Generate random samples
data = jnp.array(onp.random.normal(loc=true_mean, scale=true_std_dev, size=size))
# pdb.set_trace()
scipy_mu, scipy_sigma = norm.fit(data)

uniform_prob = 1 / data.shape[0]
weights = jnp.full((data.shape[0],), uniform_prob)
n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

# Example metadata (for illustration, could be any relevant additional information)
# weights = jnp.array([1.0, 0.9, 1.1, 1.2, 1.0, 0.95, 1.05, 1.1, 0.98, 1.02])


# Initial guess for the parameters (mean and standard deviation)
init_params = jnp.array([jnp.mean(data), jnp.std(data)])

# Define the Adam optimizer with a specific learning rate
optimizer = optax.adam(learning_rate=0.0001)

# Create the OptaxSolver using the Adam optimizer
solver = OptaxSolver(fun=neg_log_likelihood, opt=optimizer, maxiter=5000, implicit_diff=True)

# Run the optimization using solver.run, passing data and metadata as a tuple
opt_params = solver.run(init_params, data=data, weights=weights).params

# Extract the optimized mean and standard deviation
mu_opt, sigma_opt = opt_params

print(f"Optimized Mean (JAX): {mu_opt}")
print(f"Optimized Standard Deviation (JAX): {sigma_opt}")

print(f"Optimized Mean (Scipy): {scipy_mu}")
print(f"Optimized Standard Deviation (Scipy): {scipy_sigma}")

exp_mean, exp_var = compute_weighted_avg_and_var(data, weights)

print(f"Computed Mean: {exp_mean}")
print(f"Computed Standard Deviation (Scipy): {onp.sqrt(exp_var)}")


sns.histplot(data, label="Samples", color="red", stat="density")

xs = onp.linspace(mu_opt-4*sigma_opt, mu_opt+4*sigma_opt, 1000)

ys = norm.pdf(xs, exp_mean, onp.sqrt(exp_var))
plt.plot(xs, ys, label=f"Computed fit (mu={onp.round(float(exp_mean), 3)}, sigma={onp.round(float(onp.sqrt(exp_var)), 3)})")

ys = norm.pdf(xs, mu_opt, sigma_opt)
plt.plot(xs, ys, label=f"JAXopt fit (mu={onp.round(float(mu_opt), 3)}, sigma={onp.round(float(sigma_opt), 3)})")

ys_scipy = norm.pdf(xs, scipy_mu, scipy_sigma)
plt.plot(xs, ys_scipy, label=f"Scipy fit (mu={onp.round(float(scipy_mu), 3)}, sigma={onp.round(float(scipy_sigma), 3)})")

plt.legend()
plt.show()
