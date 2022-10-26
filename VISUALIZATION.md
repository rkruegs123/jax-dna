# cogli2

[`cogli2`](https://sourceforge.net/projects/cogli1/) is a straightforward visualization tool for oxDNA trajectories.
Once installed, all you have to do is provide a topology and trajectory file:
`./cogli2/build/bin/cogli2 -t <path-to-topology> <path-to-trajectory>`

**Relevant Cogli Flags**:
  --com-centre, -v                Centre the first configuration's centre of mass
  --drums                         Color-code DNA bases depending on chemical identity, according to the DRuMS color scheme: adenine is blue, guanine is green, cytosine is red, thymine is yellow, uracil is orange.
  --bcab                          Give DNA bases the same color as their respective backbones (bcab == Base Color As Backbone)
  --drums-full                    Color-code DNA nucleotides (both backbone and base) depending on chemical identity, according to the DRuMS color scheme: adenine is blue, guanine is green, cytosine is red, thymine is yellow, uracil is orange.
  --end-handling, -e <keyword>    Set how the appearance of the strand direction (5' -> 3') of DNA and RNA strands is handled. The supported keywords are 'normal' (default, no change in appearance), 'size' (the 5' end backbone is bigger, the 3' end backbone is smaller) and 'opacity' (the opacity of the strand decreases 5' -> 3')


**Relevant Cogli Keyboards Bindings**:
  ]    next frame
  /    previous frame
  q    zoom in
  Q    zoom out
  b    hide/show box
  v    centre the configuration's centre of mass
