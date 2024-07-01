#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'dangle.dat'
hstackDH = DATAFILEPATH + 'dangle.dh'


num_to_base = ['A','C','G','U']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3}

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
	final_en3 = {}
	final_en5 = {}

	while i < (len(lines)):
		line = lines[i]
		if len(line.split() ) == 4 and line.split() == ['X','X','X','X']:
			inttypesA = lines[i+5]
			inttypesB = lines[i+6]	
			
			itypesA = inttypesA.split()
			itypesB = inttypesB.split()

			#vals = {}
			vals = lines[i+8].split()
		

			for j in range(len(itypesA)):	
					basestrtype = itypesA[j][0] + itypesB[j][0]
					if len(itypesA[j]) == 2:
						overtype = '3'
					elif len(itypesB[j] ) == 2:
						overtype = '5'
					else:
						print "Cannot determine overhang type!!!"
						
					for x in range(4):
							strtype = basestrtype + num_to_base[x] 
							#print vals
							#print vals[j*4 +x] 
							energy = vals[j*4+x] 
							if energy != '.': 
								#print strtype, ' = ', energy
								if overtype == '3':
									final_en3[strtype] = float(energy)
								else:
									final_en5[strtype] = float(energy)
								
		 
			i = i + 9					
			#sys.exit(1)	
		else:
			i += 1
	#print 'TROJKY'
	#print final_en3
	#print 'PETKY'
	#print final_en5	
	return final_en3, final_en5			

def ReadDSandDH35(fileDH,fileDG):
	dgT = 37 + 273.15

	
	finalDG3, finalDG5 = ReadParams(fileDG)
	finalDH3, finalDH5 = ReadParams(fileDH)
	finalDS3 = {}
	finalDS5 = {}

	for key in finalDH3.keys():
		finalDS3[key] = 1000.0 * (finalDH3[key] - finalDG3[key]) / dgT
		finalDS5[key] = 1000.0 * (finalDH5[key] - finalDG5[key]) / dgT


	return finalDS3,finalDH3, finalDS5,finalDH5



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


def get_avg_mis_HS(overtype):
	DH_vals3 = []
	DS_vals3 = []
	DH_vals5 = []
	DS_vals5 = []

	newds3, newdh3, newds5, newdh5 = ReadDSandDH35(hstackDH,hstackDG)
	print len(newds3.keys())	
	for key in newds3.keys():
		DH_vals3.append(newdh3[key])
		DS_vals3.append(newds3[key])
		DH_vals5.append(newdh5[key])
		DS_vals5.append(newds5[key])

	if overtype == '3':
		return np.mean(DH_vals3), np.mean(DS_vals3)
	elif overtype == '5':
		return np.mean(DH_vals5), np.mean(DS_vals5)
		
 
if len(sys.argv) < 3:
	print 'Usage: ./program seqLENGTH [overtype=3/5] box'
	sys.exit(1)



#ReadParams(hstackDG)

#sys.exit(1)

mismatch = int(sys.argv[2])



overtype = sys.argv[2]
if(overtype != '3' and overtype != '5'):
	print 'Error, overtype is 3 or 5'
	sys.exit(1)
	 
stemlength = int(sys.argv[1]) -1
box = 20.

if len (sys.argv) >= 4:
	box = float(sys.argv[3])

#taken from Turner et al

#DHavg = -6.82*2  -9.38 -7.69 -10.48*2 -10.44*2 -11.4*2  - 12.44*2 - 10.64 -13.39*2  -14.88   
#DSavg = -19.0*2  -26.7 -20.5  -27.1*2  -26.9*2  -29.5*2   -32.5*2   -26.7  -32.7*2   -36.9   

#DHavg /= 16
#DSavg /= 16

#Average includes wobble base pairing:
DHavg = -10.1138888889
DSavg = -27.4866999839



import numpy as np
#taken from Turner 95 are the terminal mismatches
#termMis_dH_avg = np.mean(termMis_dH.values())
#termMis_dS_avg = np.mean(termMis_dS.values())

#taken from Turner 2011 are loop termination mismatches

IntMis_dH_avg, IntMis_dS_avg = get_avg_mis_HS(overtype) 



#print DHavg
#print DSavg

#init terms are not included for hpins
#DHinitterm = 3.61 + 3.72
#DSinitterm = -1.5 + 10.5

#penalty for AU ends
#DHinitterm = 3.61 + 3.72
#DSinitterm = -1.5 + 10.5
DHinitterm = 3.61 + (3.72 * 2/3.)*2
DSinitterm = -1.5 + (10.5 * 2/3.)*2


DH = (DHavg*stemlength + DHinitterm)
DS = (DSavg*stemlength + DSinitterm) 

#each overhang is counted twice, for as it is at both ends
DH += IntMis_dH_avg*2
DS += IntMis_dS_avg*2

 

molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
#saltcorr =  0.368*(length ) * math.log(0.5)

#print 0.5 * molcon * box**3.

RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )



s,comp = GenRanSeq(stemlength+2)
print s, comp , RealTemp 
#print RealTemp - 273.15, RealTemp
