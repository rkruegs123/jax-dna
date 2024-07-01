#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'tstacki.dat'
hstackDH = DATAFILEPATH + 'tstacki.dh'



loopDG = {1 : 3.8, 2 : 2.8, 3 : 3.2, 4 : 3.6, 5 : 4.0, 6 : 4.4, 7 : 4.6, 8 : 4.7, 9 : 4.8, 10 : 4.9, 11 : 5.0, 12 : 5.1, 13 : 5.2, 14 : 5.3, 15 : 5.4, 16 : 5.4, 17 : 5.5, 18 : 5.5, 19 : 5.6, 20 : 5.7, 21 : 5.7, 22 : 5.8, 23 : 5.8, 24 : 5.8, 25 : 5.9, 26 : 5.9, 27 : 6.0, 28 : 6.0, 29 : 6.0, 30 : 6.1}

loopDH = {1 : 10.6, 2 : 7.1, 3 : 7.1, 4 : 7.1, 5 : 7.1, 6 : 7.1, 7 : 7.1, 8 : 7.1, 9 : 7.1, 10 : 7.1, 11 : 7.1, 12 : 7.1, 13 : 7.1, 14 : 7.1, 15 : 7.1, 16 : 7.1, 17 : 7.1, 18 : 7.1, 19 : 7.1, 20 : 7.1, 21 : 7.1, 22 : 7.1, 23 : 7.1, 24 : 7.1, 25 : 7.1, 26 : 7.1, 27 : 7.1, 28 : 7.1, 29 : 7.1, 30 : 7.1}

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
							if energy != '.' and not are_complementary(num_to_base[x], num_to_base[y]):
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

def GenRanMMis(length,mis):
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
	for k in range(mis):
		mispos = int(length/2) + k
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
	print len(newds.keys())	
	for key in newds.keys():
		DH_vals.append(newdh[key])
		DS_vals.append(newds[key])
	return np.mean(DH_vals), np.mean(DS_vals)

 
if len(sys.argv) < 3:
	print 'Usage: ./program seqLENGTH [no_of_nucs_in_the_bulge] box'
	sys.exit(1)




#sys.exit(1)
bulge  = int(sys.argv[2])


looplength = bulge

stemlength = int(sys.argv[1]) - 2 
box = 20.

if(bulge == 1):
	stemlength += 1

if len (sys.argv) >= 4:
	box = float(sys.argv[3])

#Average includes wobble base pairing:
DHavg = -10.1138888889
DSavg = -27.4866999839


import numpy as np
#taken from Turner 95 are the terminal mismatches
#termMis_dH_avg = np.mean(termMis_dH.values())
#termMis_dS_avg = np.mean(termMis_dS.values())




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

#each mismatch is counted twice, for initiation and termination
if bulge > 1:
	DH += 3.72  
	DS += 10.5 

DH += loopDH[looplength] 
DS += -( loopDG[looplength] - loopDH[looplength])*1000.0/(273.15 + 37) 
 

molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
#saltcorr =  0.368*(length ) * math.log(0.5)

#print 0.5 * molcon * box**3.

RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )



s,comp = GenRanSeq(stemlength+bulge+1)
print s, comp , RealTemp 
#print RealTemp - 273.15, RealTemp
