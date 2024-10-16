#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'dnatstackh.dat'
hstackDH = DATAFILEPATH + 'dnatstackh.dh'


num_to_base = ['A','C','G','T']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}

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
							if energy != '.':
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
termMis_dH =   { 'ATAA' : -3.9, 'ATAC': 2.0, 'ATAG' : -3.5, 'ATCA' : -2.3, 'ATCC' : 6.0, 'ATCT' : -0.3 , 'ATGA' : -3.1, 'ATGG' : -3.5, 'ATTC' : 4.6, 'ATTT' : -1.7, 
		 'CGAA' : -9.1, 'CGAC': -5.6, 'CGAG' : -5.6, 'CGCA' : -5.7, 'CGCC' : -3.4, 'CGCT' : -2.7 , 'CGGA' : -8.2, 'CGGG' : -9.2, 'CGTC' : -5.3, 'CGTT' : -8.6,

		 'TAAA' : -4.0, 'TAAC': -6.3, 'TAAG' : -8.9, 'TACA' : -4.3, 'TACC' : -5.1, 'TACT' : -1.8 , 'TAGA' : -3.8, 'TAGG' : -8.9, 'TATC' : -1.4, 'TATT' : 1.4
		}
termMis_dS =   { 'ATAA' : -10.2, 'ATAC': 9.6, 'ATAG' : -8.7, 'ATCA' : -5.3, 'ATCC' : 21.6, 'ATCT' : 1.5 , 'ATGA' : -7.3, 'ATGG' : -8.7, 'ATTC' : 17.4, 'ATTT' : -2.7, 
		 'CGAA' : -24.5, 'CGAC': -13.5, 'CGAG' : -13.4, 'CGCA' : -15.2, 'CGCC' : -7.6, 'CGCT' : -6.3 , 'CGGA' : -21.8, 'CGGG' : -24.6, 'CGTC' : -12.6, 'CGTT' : -23.9,
		 'GCAA' : -13.2, 'GCAC': -8.2, 'GCAG' : -13.9, 'GCCA' : -19.6, 'GCCC' : 3.9, 'GCCT' : -12.2 , 'GCGA' : -17.8, 'GCGG' : -15.1, 'GCTC' : 2.1, 'GCTT' : -14.0,
		 'TAAA' : -9.7, 'TAAC': -17.7, 'TAAG' : -25.2, 'TACA' : -11.6, 'TACC' : -14.6, 'TACT' : -4.2 , 'TAGA' : -8.5, 'TAGG' : -25.0, 'TATC' : -2.5, 'TATT' : 6.0
		}

loopDG = {3 : 5.4, 4 : 5.6, 5 : 5.7, 6 : 5.4, 7:6.0, 8 : 5.5, 9: 6.4 }
loopDH = {3 : 1.3, 4 : 4.8, 5 : 3.6, 6 : -2.9, 7 : 1.3, 8 : -2.9, 9 : 5.0} 


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

def GenHpinSeq(stem,loop):
	stemA, stemB = GenRanSeq(stem)
	loopA, nic = GenRanSeq(loop)
	return stemA + loopA + stemB


def get_avg_term_mis_HS():
	DH_vals = []
	DS_vals = []
	newds, newdh = ReadDSandDH(hstackDH,hstackDG)
	
	for key in termMis_dH.keys():
		DH_vals.append(newdh[key])
		DS_vals.append(newds[key])
	return np.mean(DH_vals), np.mean(DS_vals)

 
if len(sys.argv) != 3:
	print 'Usage: ./program stemLENGTH loopLENGTH'
	sys.exit(1)



ReadParams(hstackDG)

#sys.exit(1)

stemlength = int(sys.argv[1]) - 1
looplength = int(sys.argv[2])


#taken from Turner et al

DHavg = -6.82*2  -9.38 -7.69 -10.48*2 -10.44*2 -11.4*2  - 12.44*2 - 10.64 -13.39*2  -14.88   
DSavg = -19.0*2  -26.7 -20.5  -27.1*2  -26.9*2  -29.5*2   -32.5*2   -26.7  -32.7*2   -36.9   

DHavg /= 16
DSavg /= 16

import numpy as np
#taken from Turner 95 are the terminal mismatches
#termMis_dH_avg = np.mean(termMis_dH.values())
#termMis_dS_avg = np.mean(termMis_dS.values())

#taken from Turner 2011 are hairpin termination mismatches
termMis_dH_avg, termMis_dS_avg = get_avg_term_mis_HS() 



#print DHavg
#print DSavg

#init terms are not included for hpins
#DHinitterm = 3.61 + 3.72
#DSinitterm = -1.5 + 10.5

#penalty for AU ends
DHinitterm =  3.72
DSinitterm =   10.5

DH = (DHavg*stemlength + DHinitterm)
DS = (DSavg*stemlength + DSinitterm) 


if ( looplength <= 9):
	#DS += -loopDG[looplength]*1000. /(273.15 + 37)  #Turner 99
	DH += loopDH[looplength] 
	DS += -( loopDG[looplength] - loopDH[looplength])*1000.0/(273.15 + 37) 
else:
	#DS += -(loopDG[9] + 1.75 * GAS_CONSTANT * (273.15 + 37) * math.log(looplength/9.)  )*1000. / (273.15 + 37)
	DH  += loopDH[9]
	DS += -( loopDG[9] + 1.75 * GAS_CONSTANT * (273.15 + 37) * math.log(looplength/9.) -  loopDH[9])*1000.0/(273.15 + 37) 

DH += termMis_dH_avg
DS += termMis_dS_avg

print (1000.*termMis_dH_avg - (273.15+37)*termMis_dS_avg)/1000.

#molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
#saltcorr =  0.368*(length ) * math.log(0.5)

#print 0.5 * molcon * box**3.

RealTemp = (DH )*1000 / (DS)
print 'Warning wrong loops penalties'
s = GenHpinSeq(stemlength+1,looplength)
print s , RealTemp 
#print RealTemp - 273.15, RealTemp
