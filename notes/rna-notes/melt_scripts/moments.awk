# usage: gawk -f moments.awk -v column=2 (to do column number 2)
# Determines the statistics (min, max, moments, cumulants, etc...)
# of the random variable appearing in a given column 
# (passed as an argument) in the file treated
#
BEGIN {
		fail=0;
		if(column==0) {
				print "usage: awk -f moments.awk -v column=2 filename (to do column number 2)";
				fail=1;
				exit;
		}
		counter=0;
  flag=0;
}

!/^#/  {
  counter++;
  datum[counter]=$column;
  med  += datum[counter];
  quad += datum[counter]**2 ;
  cub  += datum[counter]**3 ;
  quar += datum[counter]**4 ;
  if (flag == 1) {
    if ($column > datamax) datamax=$column;
    if ($column < datamin) datamin=$column;
  }
  else {
    flag=1;
    datamax = $column; datamin =$column;
  }
}

END {
  counter_m= counter -1 ;

  med /=  counter;
  quad /= counter;
  cub  /= counter;
  quar /= counter;
  cum2 = cumulant2(med,quad);
  sigma = sqrt(cum2);
  cum3 = cumulant3(med,quad,cub);
  cum4 = cumulant4(med,quad,cub,quar);


  print "# I read ", counter , " data values";
  print "# min = ", datamin, "  max = " , datamax ;
  print "# mean of distribution = ", med, "  +- error estimate = " sqrt(cum2/counter_m);
  print "# unbiased estimator of variance of distribution = ", counter*cum2/counter_m;
  print "# moments of the sample, no corrections for finite size of sample";
  print "# x bar   x^2 bar   x^3 bar   x^4 bar"
  print med, " ", quad, " ", cub, " ", quar
  print "# connected moments of the sample, no corrections for finite size of sample";
  print "# x bar_c  x^2 bar_c  x^3 bar_c  x^4 bar_c"
  print "# " med, "   ", cum2, "    ", cum3, "    ", cum4
  print "# mean     sigma     skewness      kurtosis";
  print "# " med, "  ", sigma, "   ",  skew(sigma,cum3), "   ", kurt(sigma,cum4);

  counter=0;
  med = 0.0;
  quad_c = 0.0;
  cub_c = 0.0;
  quar_c = 0.0;
  for (counter=1; counter <= counter_m+1; counter++) {
    med  += datum[counter];
  }
  med /= (counter_m+1);
  for (counter=1; counter <= counter_m+1; counter++) {
    quad_c += (datum[counter]-med)**2 ;
    cub_c  += (datum[counter]-med)**3 ;
    quar_c += (datum[counter]-med)**4 ;
  }
  quad_c /= (counter_m+1);
  cub_c /= (counter_m+1);
  quar_c /= (counter_m+1);
  quar_c = quar_c - 3.0*quad_c*quad_c;

#  print "connected moments of the sample, no corrections for finite size of sample";
# print "x bar_c  x^2 bar_c  x^3 bar_c  x^4 bar_c"
# print med, "   ", quad_c, "    ", cub_c, "    ", quar_c

}

function cumulant2(m1,m2)
{
 return (m2 - m1**2) ;
}

function cumulant3(m1,m2,m3)
{
 return (m3 -3.0*m2*m1 + 2.0*m1*m1*m1);
}

function cumulant4(m1,m2,m3,m4)
{
 return (m4 -4.0*m3*m1 + 6.0*m2*m1*m1 -3.0*(m1**4)  \
             - 3*(m2-m1*m1)*(m2-m1*m1));
}

function skew(sig,mc3)
{
 if (sig == 0) return (0);
 else return ( mc3 / ( sig*sig*sig ) ) ;
}

function kurt(sig,mc4)
{
 if (sig == 0) return (0);
 else return (  mc4 / ( sig*sig*sig*sig ) ) ;
}




