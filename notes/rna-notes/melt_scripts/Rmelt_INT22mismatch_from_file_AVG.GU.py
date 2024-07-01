#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'int22.dat'
hstackDH = DATAFILEPATH + 'int22.dh'


num_to_base = ['A','C','G','U']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3}

#pairs in the internal loop
intpairs = ['AA','AC', 'AG', 'AU', 'CA','CC', 'CG','CU','GA','GC','GG','GU','UA','UC','UG','UU']

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
	i = 31
	final_en = {}

	while i < (len(lines)):
		line = lines[i]
		if len(line.split() ) == 1 and line.split() == ['Y']:
			inttypesA = lines[i+6]
			inttypesB = lines[i+7]	
			
			itypesA = [inttypesA.split()[0], inttypesA.split()[-1]]
			itypesB = [inttypesB.split()[0], inttypesB.split()[-1]]
			
			#print itypesA

			vals = {}
			for it in range(9,25):
				vals[it-9] = lines[i+it].split()
		
			
			for j in range(len(intpairs)):	
					#print itypesA, itypesB
					basestrtype = '' . join(itypesA+itypesB)
					for x in range( len(intpairs) ):
						for y in range( len(intpairs) ):
							xtype = intpairs[x]
							ytype = intpairs[y]
							strtype = basestrtype + intpairs[x] + intpairs[y]
							#print x, vals[x] 		
							energy = vals[x][y]
							#print strtype, energy
							if energy != '.' and not are_complementary(ytype[0],ytype[1]) and not are_complementary(xtype[0],xtype[1]):
								#print strtype, ' = ', energy
								final_en[strtype] = float(energy)

		 
			i = i + 26					
			#sys.exit(1)	
		else:
			i += 1
	#print final_en	
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

def GenRanMis2(length):
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

	mispos = int(length/2)
	misbase = random.choice(bases)
	idx = length - 1 - mispos
	while are_complementary(sek[mispos],misbase):
		misbase = random.choice(bases)
	compsek = compsek[0:idx] + misbase + compsek[idx+1:]

	mispos = int(length/2) + 1
	misbase = random.choice(bases)
	idx = length - 1 - mispos
	while are_complementary(sek[mispos],misbase):
		misbase = random.choice(bases)
	compsek = compsek[0:idx] + misbase + compsek[idx+1:]

	return sek,compsek

def GenHpinSeq(stem,loop):
	stemA, stemB = GenRanSeq(stem)
	loopA, nic = GenRanSeq(loop)
	return stemA + loopA + stemB


def get_avg_mis_HS():
	DH_vals = []
	DS_vals = []
	newds, newdh = ReadDSandDH(hstackDH,hstackDG)
	
	for key in newds.keys():
		DH_vals.append(newdh[key])
		DS_vals.append(newds[key])
	return np.mean(DH_vals), np.mean(DS_vals)

 
if len(sys.argv) < 2:
	print 'Usage: ./program seqLENGTH box'
	sys.exit(1)



ReadParams(hstackDG)

#sys.exit(1)

stemlength = int(sys.argv[1]) - 3 -1 
box = 20.

if len (sys.argv) >= 3:
	box = float(sys.argv[2])


#Average includes wobble base pairing:
DHavg = -10.1138888889
DSavg = -27.4866999839



import numpy as np
#taken from Turner 95 are the terminal mismatches
#termMis_dH_avg = np.mean(termMis_dH.values())
#termMis_dS_avg = np.mean(termMis_dS.values())

#taken from Turner 2011 are hairpin termination mismatches

IntMis_dH_avg, IntMis_dS_avg = get_avg_mis_HS() 



#print DHavg
#print DSavg

#init terms are not included for hpins
#DHinitterm = 3.61 + 3.72
#DSinitterm = -1.5 + 10.5

#penalty for AU ends
DHinitterm = 3.61 + (3.72 * 2/3.)
DSinitterm = -1.5 + (10.5 * 2/3.)



DH = (DHavg*stemlength + DHinitterm)
DS = (DSavg*stemlength + DSinitterm) 


#this is Turner 95
#IntMis_dH_avg = 0.
#IntMis_dS_avg = 1000.*  (-0.8/(273.15 + 37.))
#print IntMis_dH_avg, IntMis_dS_avg,IntMis_dH_avg - (37+273.15)* IntMis_dS_avg / 1000.0,  
DH += IntMis_dH_avg
DS += IntMis_dS_avg


molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
#saltcorr =  0.368*(length ) * math.log(0.5)

#print 0.5 * molcon * box**3.

RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )




s,comp = GenRanMis2(stemlength+3+1)
print s, comp , RealTemp 
#print RealTemp - 273.15, RealTemp
