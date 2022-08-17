# parameterization

Here are notes for how we are going to parameterize the model.

We start with the ideal goal of starting from completely random parameters. We imagine a single training loop (for now -- maybe we will have separate ones to iterate on som efixed parameters) hwere each test case is a pair of (simulation information, loss function) where simulation information defines things like input, configuration, topology file, and th eloss function is defined on the trajectory, or set of trajectories from simulation information.

The simplest case of suc ha pair is that for a small duplex. The loss function should optimize for 3-4 things -- propeller twist, helix radius (which is a function of the FENE and stacking heights), angle betwee backone bases b/w complementary bases (should be antiparaellel), and optionally the FENE and stacking heights explicitly. Note how we have to optimize for all of these things.

The most general philosophy for each test case would be "optimize for everything htat has to be true in this simulation." For example, in a melting curve, in addition to optimizing for the meltin gtemperature, ew could enforce a particular persistence length, alignment of base normals for stacking, etc.
However, our initial htinking is that this is unecessary, and that we can use individual test cases to enforce different parts of the model. In this way, various parts of the model that *always* have to be true (e.g. helix radius) won't dominate the loss function. Also, it will be more efficient.
Note that weighting of different loss functions will be an issue/concern.
If this doesn't work, then we'll likely have to do some iterative procedure, where we first optimize for structure parameters, and then do a separate loop where we optimize for e.g. thermodynamic parameters

Note that there is an open question about how the log adjustment to gradients hsould be taken into account in oyur pairs. To be continued...


## Misc
- Need to understand the smoothed FENE spring with a max backbone force. In Ben's paper?
- note that we can use th eoxDNa observables as a reference
  - e.g th epitch observable to compute angles between two bonded nucldoetides. Note the 3->5 vs 5->3. Then sum over all bonded pairs, divide by 360 to get pitch (e.g. bp per turn)
