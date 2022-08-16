# parameterization

Here we have notes for what we need to implement for parameterization
We first catalog a set of utility functions that we need to implement for parameter optimization. Then, we catalog a set of simulations and their associated losses

## Utility functions

Note that naming is up for debate

- `get_pitch`
  - computes the pitch of a dpulex. Requires that you have a simulation of a single udplex.
  - How to check your strands rae bonded?
  - Can use the pitch observable that computes angle bewteen two bonded nucleotides. Sum all pitches, divide by 360, call it a day
