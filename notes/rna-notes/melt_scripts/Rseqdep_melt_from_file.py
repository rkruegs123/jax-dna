#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'stack.dat'
hstackDH = DATAFILEPATH + 'stack.dh'




num_to_base = ['A','C','G','U']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3}

def are_complementary(baseA, baseB):
	numA = base_to_num[baseA]
	numB = base_to_num[baseB]
	if(numA  + numB  == 3 or numA + numB == 5):
		return True
	else:
		return False
def ReadParams(filename):
	fin = open(filename,'r')
	lines = fin.readlines()
	i = 0
	final_en = {}

	while i < (len(lines)):
		line = lines[i]
		if len(line.split() ) == 4 and line.split() == ['Y','Y','Y','Y']:
			inttypesA = lines[i+5]
			inttypesB = lines[i+6]	
			
			itypesA = inttypesA.split()
			itypesB = inttypesB.split()

			vals = {}
			vals[0] = lines[i+8].split()
			vals[1] = lines[i+9].split()
			vals[2] = lines[i+10].split()
			vals[3] = lines[i+11].split()
		

			for j in range(len(itypesA)):	
					basestrtype = itypesA[j][0] + itypesB[j][0]
					for x in range(4):
						for y in range(4):
							strtype = basestrtype + num_to_base[x] + num_to_base[y]			
							energy = vals[x][j*4+y] 
							if energy != '.' and  are_complementary(num_to_base[x], num_to_base[y]):
								#print strtype, ' = ', energy
								final_en[strtype] = float(energy)

		 
			i = i + 12					
			#sys.exit(1)	
		else:
			i += 1	
	return final_en			

def ReadDSandDH(fileDH,fileDG):
	dgT = 37 + 273.15

	
	finalDG = ReadParams(fileDG)
	finalDH = ReadParams(fileDH)
	finalDS = {}
	for key in finalDH.keys():
		finalDS[key] = 1000.0 * (finalDH[key] - finalDG[key]) / dgT


	return finalDS,finalDH


#misses are specified as 5-AX-3 3-UY-5 as AUXY 


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



def get_avg_HS():
	DH_vals = []
	DS_vals = []
	newds, newdh = ReadDSandDH(hstackDH,hstackDG)
	#print len(newds.keys())	
	for key in newds.keys():
		DH_vals.append(newdh[key])
		DS_vals.append(newds[key])
	return np.mean(DH_vals), np.mean(DS_vals)

 
if len(sys.argv) < 3:
	print 'Usage: ./program seqLENGTH  box'
	sys.exit(1)




#sys.exit(1)


stemlength = int(sys.argv[1]) -1
box = 20.


#taken from Turner et al

#DHavg = -6.82*2  -9.38 -7.69 -10.48*2 -10.44*2 -11.4*2  - 12.44*2 - 10.64 -13.39*2  -14.88   
#DSavg = -19.0*2  -26.7 -20.5  -27.1*2  -26.9*2  -29.5*2   -32.5*2   -26.7  -32.7*2   -36.9   
#DHavg /= 16
#DSavg /= 16

import numpy as np
#taken from Turner 95 are the terminal mismatches
#termMis_dH_avg = np.mean(termMis_dH.values())
#termMis_dS_avg = np.mean(termMis_dS.values())

#taken from Turner 2011 are loop termination mismatches

DHavg, DSavg = get_avg_HS() 



#print DHavg
#print DSavg

#init terms are not included for hpins
#DHinitterm = 3.61 + 3.72
#DSinitterm = -1.5 + 10.5

#penalty for AU  or GU ends
DHinitterm = 3.61 + (3.72 * 2/3.)
DSinitterm = -1.5 + (10.5 * 2/3.)


DH = (DHavg*stemlength + DHinitterm)
DS = (DSavg*stemlength + DSinitterm) 


 

molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
#saltcorr =  0.368*(length ) * math.log(0.5)

#print 0.5 * molcon * box**3.

RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )



s,comp = GenRanSeq(stemlength+1)
print s, comp , RealTemp 
#print RealTemp - 273.15, RealTemp
