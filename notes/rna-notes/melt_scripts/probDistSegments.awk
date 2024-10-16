# usage: gawk -v  binSize=$1 -v col=$2 -f probDistSegments.awk file > outfile
# Normalized probability distribution for real random variables
# with binning such that one measures the distribution of
# the segment [0,binSize[ and translations for x>0
# and ]-binSize,0] and translations for x<0 but in fact x=0 is not included
# integral P(x) dx = 1
# means Sum_i P(x_i) = 1 / binSize
# NB: bins are thus effectively centered on   (1/2 integer) * binSize
# DO not use when random variables are not real! The point x=0 is special...
# assumes data is collumn "col" (passed)
# binSize must be set in passing argument
BEGIN {
      start=0;
      count = 0;
}

!/^#/ {
            if ( $col >= 0 )  x=   int(($col)/binSize);
            if ( $col < 0 )   x= - int(-($col)/binSize) - 1;
            P[x] ++;
            count ++;
            if(start==0) {
								start=1;
								xmin = x;
								xmax = x;
            }
            else { if (x < xmin) xmin = x;
                   if (x > xmax) xmax = x;
						}
}

END{ 
      print "# Probability distribution of ", \
             count, " events with binsize=", binSize;
      print "# middle of bin   P(x_i)   error_bar";
for ( x=xmin ; x<=xmax; x++) 
     print (0.5+x)*binSize , " ", P[x]/(count*binSize), " ", sqrt(P[x])/(count*binSize);
} 







