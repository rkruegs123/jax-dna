#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'tstackh.dat'
hstackDH = DATAFILEPATH + 'tstackh.dh'


num_to_base = ['A','C','G','U']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3}

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
termMis_dH =   { 'AUAA' : -3.9, 'AUAC': 2.0, 'AUAG' : -3.5, 'AUCA' : -2.3, 'AUCC' : 6.0, 'AUCU' : -0.3 , 'AUGA' : -3.1, 'AUGG' : -3.5, 'AUUC' : 4.6, 'AUUU' : -1.7, 
		 'CGAA' : -9.1, 'CGAC': -5.6, 'CGAG' : -5.6, 'CGCA' : -5.7, 'CGCC' : -3.4, 'CGCU' : -2.7 , 'CGGA' : -8.2, 'CGGG' : -9.2, 'CGUC' : -5.3, 'CGUU' : -8.6,
		 'GCAA' : -5.2, 'GCAC': -4.0, 'GCAG' : -5.6, 'GCCA' : -7.2, 'GCCC' : 0.5, 'GCCU' : -4.2 , 'GCGA' : -7.1, 'GCGG' : -6.2, 'GCUC' : -0.3, 'GCUU' : -5.0,
		 'UAAA' : -4.0, 'UAAC': -6.3, 'UAAG' : -8.9, 'UACA' : -4.3, 'UACC' : -5.1, 'UACU' : -1.8 , 'UAGA' : -3.8, 'UAGG' : -8.9, 'UAUC' : -1.4, 'UAUU' : 1.4
		}
termMis_dS =   { 'AUAA' : -10.2, 'AUAC': 9.6, 'AUAG' : -8.7, 'AUCA' : -5.3, 'AUCC' : 21.6, 'AUCU' : 1.5 , 'AUGA' : -7.3, 'AUGG' : -8.7, 'AUUC' : 17.4, 'AUUU' : -2.7, 
		 'CGAA' : -24.5, 'CGAC': -13.5, 'CGAG' : -13.4, 'CGCA' : -15.2, 'CGCC' : -7.6, 'CGCU' : -6.3 , 'CGGA' : -21.8, 'CGGG' : -24.6, 'CGUC' : -12.6, 'CGUU' : -23.9,
		 'GCAA' : -13.2, 'GCAC': -8.2, 'GCAG' : -13.9, 'GCCA' : -19.6, 'GCCC' : 3.9, 'GCCU' : -12.2 , 'GCGA' : -17.8, 'GCGG' : -15.1, 'GCUC' : 2.1, 'GCUU' : -14.0,
		 'UAAA' : -9.7, 'UAAC': -17.7, 'UAAG' : -25.2, 'UACA' : -11.6, 'UACC' : -14.6, 'UACU' : -4.2 , 'UAGA' : -8.5, 'UAGG' : -25.0, 'UAUC' : -2.5, 'UAUU' : 6.0
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
DHinitterm =  3.72 * 0.5
DSinitterm =   10.5 * 0.5

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

#molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
#saltcorr =  0.368*(length ) * math.log(0.5)

#print 0.5 * molcon * box**3.

for T_test in range(22,52):
	print T_test, (1000.*DH - (273.15+T_test)*DS)/1000.

RealTemp = (DH )*1000 / (DS)

s = GenHpinSeq(stemlength+1,looplength)
print '#',s , RealTemp 
#print RealTemp - 273.15, RealTemp
