#!/usr/bin/env python


import sys
import math
import random

import numpy as np

#from Turner 95
#3-unpaired (that is 3 overhang)
unpaired3_dHs = [-4.9,-9.0,-7.4,-5.7,-0.9,-4.1,-2.8,-0.7,-5.5,-8.6,-6.4,-5.8,-2.3,-7.5,-3.6,-2.2]
unpaired3_dSs = [-13.2,-23.4,-20.0,-16.4,-1.2,-10.7,-7.9,-1.8,-15.0,-22.2,-16.6,-16.4,-5.4,-20.4,-9.7,-6.8 ]

#5-unpaired
unpaired5_dHs = [1.6,-2.4,-1.6,-0.5,2.2,3.3,0.7,6.9,0.7,0.8,-4.6,0.6,3.1,-1.4,-0.4,0.6]
unpaired5_dSs = [6.1,-6.0,-4.5,-0.7,7.9,11.8,3.1,22.8,3.4,3.4,-14.8,2.7,10.6,-4.3,-1.2,2.7 ]



bases = ['A','C','G','U']
inverted = {'A' : 'U', 'C' : 'G', 'G' : 'C', 'U' : 'A' }


def is_symmetric(seq):
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'U') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'U' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
  
  return symmetric 

 
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
	if sek == compsek:
		print >> sys.stderr, "Attention, generated selfcomplementary sequence"
	return sek,compsek


def GetTm(length,box,overhang_type,index):
	avgDH_3 = unpaired3_dHs[index]
	avgDS_3 = unpaired3_dSs[index]
	avgDH_5 = unpaired5_dHs[index]
	avgDS_5 = unpaired5_dSs[index]


	DHavg = -6.82*2  -9.38 -7.69 -10.48*2 -10.44*2 -11.4*2  - 12.44*2 - 10.64 -13.39*2  -14.88   
	DSavg = -19.0*2  -26.7 -20.5  -27.1*2  -26.9*2  -29.5*2   -32.5*2   -26.7  -32.7*2   -36.9   

	DHavg /= 16
	DSavg /= 16


	#print DHavg
	#print DSavg

	DHinitterm = 3.61 + 3.72
	DSinitterm = -1.5 + 10.5

	#DH = (DHavg*length + DHinitterm)
	#DS = (DSavg*length + DSinitterm) 

	if overhang_type == None:
		DH = (DHavg*length + DHinitterm)
		DS = (DSavg*length + DSinitterm) 
	else:
		#print >> sys.stderr, 'Number of base-steps = ',length-1
		DH = (DHavg*(length-1) + DHinitterm)
		DS = (DSavg*(length-1) + DSinitterm) 

	if overhang_type == '3':
		DH += 2*avgDH_3
		DS += 2*avgDS_3
		#print >> sys.stderr, ' 3 overhang',avgDH_3, avgDS_3 

	elif overhang_type == '5':
		DH += 2*avgDH_5
		DS += 2*avgDS_5
		#print >> sys.stderr, '5 overhang ',avgDH_5,avgDS_5
		

	molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
	#saltcorr =  0.368*(length ) * math.log(0.5)

	#print 0.5 * molcon * box**3.
	#print >> sys.stderr, DH, DS
	RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )
	
	return RealTemp


#unpaired ends

if len(sys.argv) < 3:
	print 'Usage: ./program LENGTH box [3/5]'
	sys.exit(1)

box = float(sys.argv[2])
length = int(sys.argv[1]) 

if len(sys.argv) >= 4:
	overhang_type = sys.argv[3]
else:
	overhang_type = None

	

#taken from Turner et al 99
allTms = []

for index in range(len( unpaired3_dHs)) :
	Tm = GetTm(length,box,overhang_type,index)
	#print index, Tm
	allTms.append(Tm)

print sorted(allTms)
print max(allTms), min(allTms)
