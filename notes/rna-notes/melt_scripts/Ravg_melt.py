#!/usr/bin/env python


import sys
import math
import random


bases = ['A','C','G','U']
inverted = {'A' : 'U', 'C' : 'G', 'G' : 'C', 'U' : 'A' }
 
def GenRanSeq(length):
	sek = ''
	compsek = ''
	for j in range(length):
		compsek += '*'
	for i in range(length):
		base = random.choice(bases) 
		sek += base 
		idx = length - 1  -i
		compsek = compsek[0:idx] + inverted[base] + compsek[idx+1:]
		#compsek[length - 1 - i] = inverted[base]
	return sek,compsek

if len(sys.argv) != 3:
	print 'Usage: ./program LENGTH box'
	sys.exit(1)

box = float(sys.argv[2])
length = int(sys.argv[1]) - 1


#taken from Turner et al

DHavg = -6.82*2  -9.38 -7.69 -10.48*2 -10.44*2 -11.4*2  - 12.44*2 - 10.64 -13.39*2  -14.88   
DSavg = -19.0*2  -26.7 -20.5  -27.1*2  -26.9*2  -29.5*2   -32.5*2   -26.7  -32.7*2   -36.9   

DHavg /= 16
DSavg /= 16


print DHavg
print DSavg

DHinitterm = 3.61 + 3.72
DSinitterm = -1.5 + 10.5

DH = (DHavg*length + DHinitterm)
DS = (DSavg*length + DSinitterm) 


molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
#saltcorr =  0.368*(length ) * math.log(0.5)

#print 0.5 * molcon * box**3.

RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )

s,c = GenRanSeq(length+1)
print s,c, RealTemp
#print RealTemp - 273.15, RealTemp
