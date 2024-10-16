# Notes on RNA optimization from Munich Visit

This will all be with average sequence for now. There are *no* wobble base pairs in the averages equence case for oxRNA.

Also, all of this is in Python 2.7

## Thermodynamics

Should do **normal duplexes** of length 5, 6, 8, 10, 12. Use the files `Ravg_melt.py` or `Ravg_width.py` to get the corresponding Tms or widths for a given length. Note that we could try fitting the width.

Should do **double overhangs** for paramaeterizing cross stacking. Use the script `Ravg_melt_double_overhang.py`. Note that these take the followign form:

___________
    | | | |
    ___________

Should do do **mismatches** for parameterizing HB, CX, and stacking. Use the scripts named `Rmelt_INT[11, 22]mismatch_from_file_AVG.py`. Again, for now, there is no wobble base pairs (e.g. GU) for average sequenceing. These take the following form:

___________________
|  |  |  \  \  |  |
___________________

where \ denotes an invalid base pair.

Should also do **bulges** using the file `Rmelt_BULGEX_from_file_AVG.py`. E.g. do X=1. These take the following form:

       _____
______/  \  \______
|  |  |     |  |  |
___________________

where \ denotes a bulged nucleotide

For **hairpins** should do stem lengths of 6 and 10 for loop lengths of 5 and 10 (i.e. the entire 2x2 matrix of these vaalues). Use files `Rmelt_hpin_from_file_AVG.py` and `Rmelt_hpin_from_file_avg_width.py`. Again, we could fit to the width.

## Structural

Unlike DNA, the center of a base pair in RNA does not align with the helical axis. The angle here is called the "inclination."

Petr shared with me a file, `geom.py`, that computes some structural quantities -- pitch, rise per base pair, and inclination. Note that rise per base pair is not just a function of geometry -- it's also a function of e.g. stackiong. `geom.py` depends on `readers.py`.
