# jaxmd-oxdna

A playground for implementing oxDNA in JAX-MD. Apologies for the naming -- I only wanted one hyphen

To install jax-md rigid body check in locally, do FIXME. Must have this to run code.

## Misc. Notes

July 19, 2022
- Renderer not working. Can't debug without it
  - Note that we'll have to appropriately treat body/space frame for rendering properly when adding additional interaction sites
  - Note that this could be a good example of using a Shape/PointUnion
- Unclear to me how some things are differentiable when they're written with normal python conditionals and loops. Is it because they are fixed at compile time?
  - Upshot: as we write, will want to check that gradients propagate. Can't do this until 3D rigid body gradients work. Does Sam think this is something we should be mindful of, or can we assume that we'll be able to make whatever we write differentiable?
- The typical custom potential acts on a *single* metric or displacement between pairs of objects (i.e. isotropic spheres or rigid bodies). `smap.pair` is built for such a case. However, in the oxDNA case, the potential is a function of *eight* metrics between a pair of rigid bodies. So, there is no obvious way to implement a function like `oxdna_potential(oxdna_metric_fn)` because `oxdna_metric_fn` can only return one metric and `oxdna_potential` won't have access to the pair of rigid bodies to compute the other metric. To me, this seems to leave two options: (1) compute the entire potential in `oxdna_metric_fn`, or (2) in some way allow `metric_fn` to pass >1 values to the potential.
  - Does it matter in terms of efficiency? I have no reason to think so, as the metric function is vmapped.
- `smap.pair` (or the neighbor version assumes that `potential(A, B) = potential(B, A)`. This can be seen via the use of `normalization`, where we just sum the entire symmetric matrix and divide by 2. However, in oxDNA, order *does* matter. What's the best way to handle this? Mask half the matrix and don't normalize? Perhaps we can prevent double counting by passing a custom mask function, but then we still need to prevent the division by 2. Also, we need to ensure that the custom mask function doesn't apply the mask symmetrically.
  - NOTE: Note that we could also just not use `smap.pair`... Will be made a bit harder because we'll have to do neighbor lists...
- The oxDNA potential is two sums. In JAX-MD lingo, this can be thought of as two potentials, one with a static neighbor list and another with a dynamic neighbor list. What is the best way to implemement this?
  - See Sam's response.

- Other, non-unique questions for Sam
  - Can you return multiple things from your displacement/metric function in the usual formulation of an `smap.pair` potential? If so, we could just return all 8 variables used by the oxDNA potential. However, we think not because of the diagonal masking
  - Is there any reason not to depart from the philosophy of going from: displacement/metric function -> function that acts on this -> elevating this to act on the whole systme? We see no special property of this, just that its general for many isotropic potentials. Can you confirm?

- Another question for Sam: why not implement thresholded potentials with `lax.cond` instead of that trick that Carl does?
  - `multiplicative_isotropic_cutoff`
  - even if you do it with `jnp.where`, it will be discontinuous. Not ethat LAMMPS does this with `lj.cut`, but this can make simulations less stable
  - you can use `jnp.where`, but it will just have a cutoff with a discontinuity