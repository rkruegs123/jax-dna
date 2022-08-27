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

July 22, 2022
- Need initial configuration so that losses don't go to `nan`
- Need to solve for those additional parameters that aren't given but are determined
  - Can eventually throw them out with Carl's trick
- Then, can proceed implementing. Will have to be careful when to vmap vs. just go element-wise. Will have to fiddle with flattening or not based on how neighbors are handled

July 24, 2022
- One game of Catan
- reimplement some potentials using Carl's trick
- Joel Polos

Next Questions for Sam
- Why is if-statement throwing error when not even jit-ing it?
  - e.g. for f3

July 29, 2022
- I *think* when we add the real sites w.r.t. the COM, we get a NAN from the config for a 10bp polyA. This is probably because the config is for oxDNA2, not 1
  - TODO: check by reverting back to old site relative positions
- Can visualize either using cogli2 or just oxView
  - TODO: F/u with Lorenzo about +/- for toggling configurations in cogli2
- Should really just add a rough sketch of stacking and dynamic neighbors...
- For first neighbor function, can probably just use partition.neighbor_list
  - Will have to see if smap elevates the neighbor function at all, hoewver. I think not -- i.e. `smap.pair_neighbor_list` only elevates `fn` to *use* a neighbor list. The neighbor function produced by `partition.neighbor_list` should be all we need...
  - The one negative is that `partition.neighbor_list` appears to only be built for defining a radius w.r.t to some metric. I imagine the neighbor list in oxDNA is more complicated than that...

July 30, 2022
- So, `partition.neighbor_list` is exactly what we want, but it doesn't support `RigidBody` yet. Emailed Sam.
  - Note that by fixing the format as `partition.OrderedSparse`, we can assure that `i < j` for a pair `(i, j)` in the neighbor list. So, we can enforce the direction that gets passed to the potentials!
- In the meantime, we should just implement all the potentials. Note the above ability to constrain the direction when doing so.

Aug. 2, 2022: The state of things as it stands, in no particular order:
- Should really learn to use oxDNA by myself -- running small-scale simulations, generating configuration files, etc
- Need to figure out 5'-3' for reading and config files (and setting neighbors) and for potentials
- Dynamic neighbors
  - `partition.neighbor_list` should be exactly what we want for an initial implementation (only parameterized by distance, dr) but isn't compatible with RigidBodies
  - We should use `OrderedSparse` to enforce directionality. Just have to be sure to have `i < j` for the directionality that we want to always pass in
  - Have to mask out bonded pairs using a custom mask function, or something like that. Not sure the best way to do this.
  - To begin, we were just going to set a massive radius to effectively include everything. Even simpler (and to relieve any blockage) would be to just treat is as a static neighbor list for now, as long as we are in keeping with `SparseOrdereed` semantics
- ~~Need to update parameters with `kT`~~
- Need to debug understanding of interaction site distances from COM
- Can visualize easily with either `cogli2` (using `]` or `/`) and `oxView`
- Eventually, will want to use `multiplicative_isotropic_cutoff`. Not a major thing to worry about for now
  - Can do 1 - the normal function or multiply two together (one of which is 1 - ) to get a cutoff on one or both sides
  - Should check how this is implemented in the backend with `jnp.where`. Should also understand that expression it's based off of
- If we want to optimize over the original parameters, have to be careful about hings like taking logs of negatives, like with the FENE. Can always have a cutoff for this.
- Could replace `sympy` expressions with normal python functions to save time, and also to have differentiability should we later want that
- Langevin integrator with Sam
  - Have Megan explain stuff to me
- Talk to people about exactly what thye want to do with oxDNA 3 post-doc advertisement
  - As easy as oxDNA 1.5?
- Potentials have to take in the probabilistic sequence. Again, likely indexed by `nbs_i` and `nbs_j`
- Should really start collecting the data we'll want to use for a benchmark
- Should check differences in parameters between 1.0, 1.5, and 2.0
- Have to normalize dot products by magnitudes every now and then. Might be some other straggling FIXMEs
- Better way to take matrix-wise dot products?
- Review quaternions
- Watch those MD lectures
- For future: convert trajectory to all-atom trajectory for cooler visualization? Would it be too unrealistic?
- Sam also recommended the `fresnel` visualizer. I think from Chrisy's Sharon's lab. Might want to look into.
- Note that we get trace errors when using normal `if` statements without any `jit`ing because JAX-MD traces to determine force vs. energy
- Non-bonded excluded volume
- Go over all code with Megan
- Need to ask sam: how is the moment of inertia passed to the rigid body integrators? Is it calculated from the mass somehow? If so, how does it generalize beyond isotropic spheres (i.e. take the shape of the rigid body into account)?


Sam RigidBody notes:
- can set moment of inertia explicitly as RigidBody(center of mass mass, moment of inertia). E.g. RigidBody(float, [float, float, float])
  - Something about the moment of inertia having to be idaognal?
- neighbor_fn.update(state.position.center, nbrs) instead of neighbor_fn.update(state.position, nbrs)
- custom mask function to omit bonded things from neighbor calculation?
- Psatbine has one function in particular that takes you from a sparse mask function a dense mask function. Note just sparse, not ordered sparse. Should be easy to go from one to the other
  - https://pastebin.com/57Hn10d1
  - So, allows you to go from mask function that acts on tuples to work with custom_mask_function
- For loops and jitting:
  - for i in range(CONSTANT) IS jittable because it can be unrolled
  - For i in range(VARIABLE) is not jittable becaues it cannot be unrolled -- you are trying to do a pythonic thing on a trace object
  - So, in stuff with max, e.g. for b in range(2) can be done PYTHONICALLY.



#### Aug 17. 2022
State of things
- we have a 3p to 5p energy function that we think works. The only thing we haven't been able to test is excluded volume -- we probably could if we tried harder. We've been comparing with oxDNA utilites that give the subterms of the energy function (note: should submit an issue saying that it should b emade explicit that it's the averages. Could probably make a GitHub issue on this documentation)
- we've just been using "static" dynamic neighbors for now. in `test_neighbors.py`, we have a method for taking a set of pairs to mask out and oin ht eneighbor function. Also note that we can just call `neighbor_fn` on `body.position` instead of `body`, so neighbor lists in general are no problem for rigid bodies
  - Have to talk to Sam about the ordering issue, backwards compatability, and what is best to use (i.e. sparse vs. dense)
- we try to simulat ewith Nose Hoover right now but at the oxDNa dt, the energy diverges. Hopefully this isn't a problem wtih Langevin and/or MC/VMMC. to be determined.
  - note: htink we've tried smaller dts and it doesn't diverge. Coul doduble check...


Sam questions/notes
- tried your neighbor list recomendation for sparse -> dense
  - got it working with some modifications. However, for a disallowed set of pairs, isn't it just easier to mask them on the dense representation?
    - barring any other suggestion from you, I'd just like to do this. However, how will this work with backwards compatability?
  - Why does the sparse list appear to be [y, x] instead of [x, y]? In that, for i < j, the 2xN matrix looks like [j, i] instead of [j, ]
- Nose Hoover NVT diverging/unstable. Could just be that we need langevin or brownian. We will see and keep you posted...
- since JAX-MD automatically traces, debugging can be hard
  - can't debug past the first iteration
  - what do you recommend?
- return log prob from integrator
- external forces
- umbrella sampling for VMMC


Next stepss:
- talk to Sam about above
- ryan implements 5p to 3p utiliites
  1. 5pTO3p <-> 3pTO5p
  2. (topology, trajectoroy/configuration pair, 5pTO3p) <-> (same pair 3pTO5p)
- don't forget to updat ehtings like `read_config` and the papropriat neighbors...
- ryan then updates and tests energy function appropriately
- Megan gets started on observables (see `PARAMETERIZATION.md`)
- minor ryan note: for neighbor_testing.py, have to update j,i as well as i,j

#### August 22, 2022

TODO (been working on `topology.py` and `trajectory.py`)
- test everything. e.g...
  - (i) read and write, no simulation. note that this is just through traj_df
    - first a config, then an entire trajectory
  - then, maybe some utilities?
- update the energy function accordingly
  - some things should be the same, e.g. FENE
  - test the new energy function
    - (ii) read, simulate, and write. This way, we go between traj_df and states. Couldn't test this until we could actually simulate...
- clean (e.g. comments, todos, fixmes)
  - could add every_n to TrajectoryInfo.write()
  - can probably remove jax_traj_to_oxdna_traj
  - update simulate.py and others
  - remove unecessary stuff from utils.py
- implement `read_config` using `read_trajectory`?

#### August 24, 2022

TODO:
- draft skeleton for parameter optimization
- think about optimizing for speed. Need to benchmark first

Some next steps:
- dynamic neighbors
  - have to figure out max interaction radius. How to deal with when we have random params? Take one big minimum? Probably just the maximum of some set of parameters that we can fix...
- understand smooth fene
- debug langevin
  - hpoefully not as slow as nose-hoover. hopefully nose-hoover is particularly slow because of the chaining method
  - then we could actually test the grad stuff better. with nose hoover, we get the warning that the compiltaion is too slow, etc. this really shouldn't be the case...
- debug the first pass at parameter optimization
  - to not have to worry about exploding gradients or anything, our first pass can just be optimizing using the *existing* parameters as a starting point
- little things
  - profile `forward` -- is it the `init_fn` that takes so olong? Is this just because of Nose-Hoover?
  - optimize a subset of things
  - maybe reimplement scan for the rollout once we've debugged the first parameter optimization
