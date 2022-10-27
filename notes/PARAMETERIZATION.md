# parameterization

Here are notes for how we are going to parameterize the model.

We start with the ideal goal of starting from completely random parameters. We imagine a single training loop (for now -- maybe we will have separate ones to iterate on som efixed parameters) hwere each test case is a pair of (simulation information, loss function) where simulation information defines things like input, configuration, topology file, and th eloss function is defined on the trajectory, or set of trajectories from simulation information.

The simplest case of suc ha pair is that for a small duplex. The loss function should optimize for 3-4 things -- propeller twist, helix radius (which is a function of the FENE and stacking heights), angle betwee backone bases b/w complementary bases (should be antiparaellel), and optionally the FENE and stacking heights explicitly. Note how we have to optimize for all of these things.

The most general philosophy for each test case would be "optimize for everything htat has to be true in this simulation." For example, in a melting curve, in addition to optimizing for the meltin gtemperature, ew could enforce a particular persistence length, alignment of base normals for stacking, etc.
However, our initial htinking is that this is unecessary, and that we can use individual test cases to enforce different parts of the model. In this way, various parts of the model that *always* have to be true (e.g. helix radius) won't dominate the loss function. Also, it will be more efficient.
Note that weighting of different loss functions will be an issue/concern.
If this doesn't work, then we'll likely have to do some iterative procedure, where we first optimize for structure parameters, and then do a separate loop where we optimize for e.g. thermodynamic parameters

Note that there is an open question about how the log adjustment to gradients hsould be taken into account in oyur pairs. To be continued...

Ah -- one other thing that I forgot. Note that Tom used a lot of MC/VMMC to do his parameterization. One first thought, one would think that this is just for speed (and myabe it is, because Tom fixed the backbone adn stacking lengths). However, with random parameters, MC will also be important -- this is because we could have exploding MD for very wrong parameters. This is another reason we might have to split up our optimization -- it mayb e the case that some simulations require MD, so therefore need some reasonable starting parameters.


Update: note that for stochastic simulations, the gradient over a batch of trajectories isn't the same as the gradient of the loss function. Instead, we have an estimate of the gradient that includes the log probability. So, for each test case, we actually want three bitrs of informatoin: (i) simulation info, (ii) grad estimate, and (iii) the actual loss temr. What we really want is functions that will give all of these


## Misc
- Need to understand the smoothed FENE spring with a max backbone force. In Ben's paper?
- note that we can use th eoxDNa observables as a reference
  - e.g th epitch observable to compute angles between two bonded nucldoetides. Note the 3->5 vs 5->3. Then sum over all bonded pairs, divide by 360 to get pitch (e.g. bp per turn)

## STRUCTURAL
Required simulations:
- 1e5-1e7 MD steps (use MD instead of VMMC because you don't need to accelerate sampling of rare configurations)  (can be done in parallel, e.g. 100-1000 batches of 1e4 steps)

Examples:
- pitch
- helix radius
- avg base-base distance
- avg backbone-backbone distance


## MECHANICAL

## THERMODYNAMIC
