# DNA28 Conference

## Outstanding Questions

- Is it inappropriate to visualize oxDNA output at an atomistic level?
- Need to talk about oxDNA 3
- Ask about moment of inertia in Peter's thesis
- Ask Matt Patitz and Erik Winfree what type of search they have done over the space of circuits
  . e.g. rather than explicit design, do they ever search the space of solutions to a particular algorithm
  - Might be some interesting "compressibility" notions here as well
- Note: lunch tomorrow with Tom
- Really need to get a sense of how long simulations take to see if backprop is feasible
  - E.g. some of Petr's simulations for thingsa as simpleas a generalized 3D aTAM take ~2 weeks
- Should get source for Anderson thermosat -- is it described in a paper somewhere?
- Where is the Langevin thermostat on the oxDNA documentation?
- is oxDNA CUDA implementation vectorized at all?
- have people done profiling? what is the slow part?
- What about different forms of DNA other than B-DNA? Efforts to do this? Is there any purpose/application in the world of DNA nanotechnology?
- Efforts to model interaction with proteins and/or small molecules
- dynamic neighbor calculation
- single strand dynamics iwth coaxial and cross stacking and hydrogen bonding (e.g. for a polyA strand)


## Notes

### Misc notes
- oxView started because cogli2 wasn't working on someone on Peter Sulc's group's machine
- I mentioned Nose Hoover to Erik Poppleton and he mentioned I "have to get on the Anderson thermostat." Though, I think he mentioned for speed reasons...
  - Also, don't need to update the rotational component (the quaternion)
- There are now python bindings for oxDNA
- Erik and (other post doc) mentioned some recent errors found in the energy function calculations in the oxDNA code
  - Think it was mostly in oxRNA, and maybe the electrostsatics. Also some errors in the energy output. Should remember to frequently pull new version, as well as ask for details when the time is appropriate
- Mitra Basu at NSF is a big funder of the community
- Petr's group strongly advocates using the Andersen thermostat
  - Also, could potentially be faster
- Petr's group is applying for a grant with an October deadline: NSF Pathway to Open Source Ecosystem
  - but only for old things, not new things
  - could use for things like new servers, hackathons, grants, etc
- Petr emphasizes that there is a non-zero moment of inertia along the x-axis (otherwise it'd be 2D). Need to understand why all set to 1.0. He says because it is "effective"
- Petr offers help for JAX implementation
- Note that if we suffer from speed, the emulator will be a great next step...
- Note that Petr is confident in the eficiency of the CUDA implementation. Much faster than the LAMMPS implementation
- Note that Petr's name is pronounced "pet-er" not "peet-er"
- https://www.youtube.com/channel/UC4ClrQ8xMypSbDZG2QCehpA
  - "Create DNA designs in oxView for oxDNA"
  - a blender tutorial! from Joakim. Apparently he has done a very good job on this
  - to use Brady Johston's, export to PDB from oxDNA web page
    - tacoxdna.sissa.it. oxDNA -> PDB. For larger things, have to download the code
    - when oyu put it in blender, it translates it to atomistic
- can send Petr email to get on oxDNA developers Slack
- https://github.com/lorenzo-rovigatti/oxDNA
  - very good/literate ontebooks
    - servers to run these on are not given, but oxdna.org has free servers for running simulations
    - https://github.com/lorenzo-rovigatti/oxDNA/tree/master/examples/OXPY_Jupyter
    - e.g. `literate_sim.ipynb`
    - note that this is a central repository for C++ stuff and python bindings stuff
- Some Snodin 2014 paper has the original discussion of the John/Brownian/Andersen-like (all different names for the same thing) thermostat, as well as the firs disucssion of the smoothed FENE part (according to Petr)
- the FENE spring is tyupically used for modelling actual covalent bonds BECAUSE it is undefined beyond a particular distance, sa it should be, because that would mean that the covalent bond is breaking
  - Tom originally modelle with a normal spring, but th ereviewers wanted this
- The only difference between relaxation and ismulation code is that relaxation code (i) allows a max force (for undefined FENE spring) and (ii) someitme suses smaller time steps. Also in cases wher eyou hav eoverlapping things, in which case your excluded volume term would blow up, you use MC for a little bit
- Petr also is in discussion with a student about trying to build an emulator for oxDNA using GNNs. Would be interested in collaborating on this
- Note that diagnostics is the easiest application to commercialize because you don't need to purify things to such a degree that they are safe for humans
- thye almost never use MC for actual simulatoins becaues it is slower because they say it can't be parallelized (should ask Sam about this)
  - they alaso use rarely use VMMC because you have to provide some extra weighting parameters for transitions or something
- oxDNA CUDA implementation is for a single GPU, doesn't scale to multiple GPUs
  - in theory, maybe this isn't a massive bottleneck because you can just run N simultaions across N GPUs, and maybe you'd want to run N simulations anyway
- i get the sense that Petr thinks a JAX-MD implementation would be useful for parameter fitting, but not for inverse design because simulations are just too slow
  - Note: emulator would be a reaosnable next step if this is the case


### Models of self-assembling systems: varying levels of abstraction and objectives (Matt Patitz)
- Intro: Matt is quite theoretical, resonsible for putting field on map in itheoretical CS field
- survey of different scale models and how they fit together/when they're useful
e.g. oxDNA-viewer, scadnano, abstract tile method thing from winfree group
- abstract tile assembly model (aTAM, Winfree 98)
  - fundamental units are 2D square tiles
  - binding domains ("glues") with types and strengths (integer-values)
  - tiles can't be rotated
  - infinite supply of each type
  - sides of tiles match if glue strengths and values match
  - assembly begins with a "seed" tile and grows one tile at a time stochastically and asynchronously
    - Have an infinte tile set, and a single seed
  - IDEA: this could easyily be implemented in JAX-MD. But, how do you model the "infinite supply of each type"?
  - Q: Why model with a single seed? Why not just have a bunch floating around in solution?
  - Temperature is interpreted as the minimum binding threshold
  - Becomes interested with "cooperative binding" -- i.e. when you need two tiles or more to be in place to get a particular one to come in and bind
  - IDEA: could also extend to 3D in JAX-MD
  - E.g. we can incorporate counting logic!
    - A binary counter. Each row is one more than the previous row. Could we reproduce this in JAX-MD? Get a tile set/seed that produces a binary counter?
    - They validated this experintally -- how do they constraaint expeirments such that there is a single seed?
  - Another example: binary addition, XOR (gets us the Sierpinski Triangle -- an infinite shape called a discrete self-similar fractal)
    - Winfree 2007
  - IDEA: in the same way that XOR gets the Sierpinski Triangle, could target other self-similar fractals! aTAM is very good for this
  - Has already been extended to 3D cubes, squares + dominoes, polyominoes
    - e.g. extended ot Shih grou''s work with criss-cross slates. "Robust nucleation control via crisscross polymerization of highly coordinated DNA slats"
    - IDEA: could definitely do in JAX-MD...
  - aTAM
    - good for desgning algorithmic rather than fully-addressed (i.e. hard-coded) systems. high-level algorithmic design, determining which info needs to be where, designig paths to propagate it
    - universal computation is possible but geometry imposes restirctions
    - cooperation is fundamentally required for algorithhmic behavior
    - limitation: doesn't account for erroneous attachments, assumes perfectly flat tiles, glue strengths are integer values, allows infintely sized assemblies, no restriction on tile or glue types, only growth from dedicated seed assembly assumed
  - Ok...
  - kinetic tile sasembly model (kTAM). Winfree 98.
    - Supposed to be closer to reality
    - Rather than just using a temperature parametre to decide if things will bind, describes dynamics of sasembly according to set of reversible chemical reactions. Have two rates: an association rate and dissociation rate.
      - in the original aTAM, there was no dissociation
      - Also, concentration dependent. Nice!
    - Note: definitely closer to JAX-MD. Would want to do this over aTAM
    - one idea for JAX-MD design on this: could set a max number of types, and regularize to minimize numer of types
    - there workflow: design algorithms in aTAM, then move over to kTAM. Sometimes thing that work in aTAM don't work in kTAM.Have developed various proofreading techniques, etc.
    - pros: testing atam designs, incorporating error-prevention mechanisms, estimate growth tempereatures, estimate growth reates
    - cons: basic version of model assumes equal and constant tile ocncentratoins. doesn't account for depletion, assumes integer stregnth for binding, etc
  - Ok...
  - Next level: atomic r molecular level modelling
  - High accuracy, but very computaionally expensive
  - coarse-grained models like oxDNA are alternative
  - They used MD simulations to test the feasibility of tile attachments in different scenarios
  - note that for these, you must have completed a full design in advance
    - Q: how is this different than aTAM or kTAM?
  - Note: self-assemlby of large structures is too computationally expensive! Need to start with e.g. staples already attached. Can't just start with a soup...
  - Ok...
  - Thermodynamic Binding NEtworks (TBNs)
  - No geometry or reaction pathways. Very abstract...
  - assemblies form entirely due to the thermodynamic tradeoff b/w enthalpty (# of bonds formed) and entropy (count of distinct complexes)
    - enthalpy is maximized first, then entropy
  - Prevents a pathway of series of models:
    - (1) aTAM -> scadnano python api -> scadnano -> oxDNA viewer or CanDo structural modeling
    - (2) aTAM -> kTAM
  - Questions...
  - Peter Sulc asks great question: wlil there be an iterative design algorithm? Or is this all hard work and intuition?
    - Answer: most of them are intuition

### Covered DNA core tiles for rubust tuning of spuiious nucleation (Trent Rogers)
- Trying to tune the spuirous nucleation rate of DNA nanotech
- motivation: want to make structures
- two methods for assembling nanoscale shapes
  - DNA origami. Scaffold strand, and a bunch of staple strands
  - tile-based approach: use as atomic components. More intuitive -- e.g. can assign a tile to a pixel in an image. Also, not fundamentally limited in size. Called SST?
- tile approach suffers from low yield
  - why?
  - one hypothesis: get profileferation of small structures. if we don't control the number of these structures, you get a bunch of partially formed structures and no free monomers to allow them to form into target structure
- goal of their project: develop a temperature-robust mechanism for controlling the rate of nucleation
  - to address that hypothesis above
  - temperature robust is good so you don't have to play around with thte tempeature to get the right growth temeprature at which it will actually work
  - previous attempts that try, e.g. hihgly-cooperative slats to make an arbitrarily large kinetic barrier to nucleation
- shows a motif that allows the tunability of nucleation while maintaining a sufficiently high growth rate
  - basically a cover to block certain interactions
- Note from Erik Winfree: tubes and microtubules have extemely controlled nucleation in boilogy. Lessons can be learned in comparison to e.g. DNA nanotech where the length distribution f e.eg tubes can relaly change

### SAT-assembly: Designing minimum complexity building blocks to assemble arbitrary shapes (Petr Sulc)
- Goal construct multi-component assemblies from individual building blocks
- inspiration from things like viral capsids or ribosomes, as wella s material designs like terastack (pyrochlore) lattice or cubic diamond lattice
  - e.g. doing computation with light with cubic diamond lattice
- patchy particles as DNA origamis
  - DNA origams are limited in size by size of scaffold. E.g. 50 nm is as big as you can get for a big circle like thing
- no explicit solvent in oxDNA
- computational models aren't efficient enough to see how the oirgamis assemble. can't just make a soup and wait to see how things diffuse around
  - WANT as physical as possible of an assembly simulation as possible
  - where patchy particles come in -- represent th eentire origami as a patchy particle with interaction sites (patches) -- interaction sites correspond to single strand overrhangs of the origamis
    - note: these are origimas that assemble with each other into one much bigger thing
  - NOTE: PATCHY PARTICLES
  - Need to be in the right temperature for nucleation and growth
    - no seed. Have to grow by homogenous nucleation
  - Question: why use oxDNA for patchy particle simulation?
- they were always getting kinetic traps with their patchy partilce simulation
  - INVERSE DESIGN PROBLEM: WANT INTERACTION DESIGN MATRIX SUCH THAT IT FORMS THE DESIRED CRYSTAL
  - CANT DO BRUTE FORCE. THIS IS PERFECt
  - solution: SAT-assembly for inverse design.
    - formulating the inverse design problem as a SAT problem because then you can throw SAT solvers at it.
- question: how do they actually do these experiments? Do they first make the origams, then put them all in one big soup?
- question: how do you define the desired crystal?
  - need to understand what the loss function would be...
- questoin: do you fix the numbero finteractions ites? Can you have multiple copies of an interactoin site type?
- solution of the SAT problem guarantees an energy minimum. Takes a fraction of a millisecond to generate a possible solution
- can also prove unsat!
- ok... also want to make finite-size multi componet systems...
  - (after in the talk where he shows he can do this for crystal lattices)
  - "fully addressiable" -- each component in the self-sasembled system is its own unique species. Not eeconomical and tedious
  - "minimal kit" -- identical species in the self-assembled system
- so they basically developed aTAN system but in 3D and with rotations
  - give target topology. give to SAT solver, then iterate to get number of species,e tc
  - once gets a solution, export to simulation and run for 1-2 weeks... wow that's long..
- Ok...
- Then, wanted to characterize the differnet solutions (i.e. more or less colors/species) in temrms of physics
- there's all different type sof caveats he found with the potential -- e.g. whether or not bonds can rotate, etc
- remmber that the whole motivation for this is to get crystals out of DNA origami to do computation with light
  - question: why want to do this out of DNA? Aren't there simpler systems to make crystals out of, like actual patchy particles or magnetic handshake things?
- oxDNA.org is a GPU surver with 8 GPUs to run jobs
- note that they can target more than one assmelby (i.e. convert such a problem to a SAT problem. Like if you say you want to make shape X ~and~ Y in equilibrium)
- Note: one thing they don't know how to incorporate is how to incorpoirate structures that they watn to AVOID
  - to do this in their world, they'd have to reduce to quantifivable SAT (Q-SAT), which is veyr hard
  - this would be great for JAX-MD
- Note htat Petr says that the special high-presure case in which carbon forms the diamond lattice hasn't been reproduced in simulation...
- I asked Petr: if you want to assemble crystal lattices of a pariticular shape to do, e.g. light computation, why both doing with DNA? Seems very complicate.
  - His answer: because you need a sufficiently small crystal to get a particular wavelength, and existing alternatives (e.g. oclloids) are just too big. MAYBE vinny manoharan can make things small enough such that colloids would work. That would probably be better...


### On Turedo Hierarchies and Intrinsic Universality (Samuel Nalin)
- RNA folding is a subset of Oritatami Systems which is asubset of Turedos
- skipping notes on next couple because they are quite theoretical and irrelevant

### Single-pass transmembrane double-stranded DNA with functional toeholds for non-destructive intra- and extravesicular nucleic acid target recognition
