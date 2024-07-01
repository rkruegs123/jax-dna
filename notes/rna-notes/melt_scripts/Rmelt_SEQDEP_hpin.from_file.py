#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'tstackh.dat'
hstackDH = DATAFILEPATH + 'tstackh.dh'

stackDG = DATAFILEPATH + 'stack.dat'
stackDH = DATAFILEPATH + 'stack.dh'


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


def ReadParams_stack(filename):
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
	#print final_en
	return final_en			


def ReadDSandDH_stack(fileDH,fileDG):
	dgT = 37 + 273.15

	
	finalDG = ReadParams_stack(fileDG)
	finalDH = ReadParams_stack(fileDH)
	finalDS = {}
	for key in finalDH.keys():
		finalDS[key] = 1000.0 * (finalDH[key] - finalDG[key]) / dgT


	return finalDS,finalDH



#misses are specified as 5-AX-3 3-UY-5 as AUXY 
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
	
	print len(termMis_dH.keys())
	print len(newdh.keys())
	for key in newdh.keys():
		DH_vals.append(newdh[key])
		DS_vals.append(newds[key])
	return np.mean(DH_vals), np.mean(DS_vals)


def GetTM(sequence,stemlength,looplength,stack_dh,stack_ds,miss_dh,miss_ds):
#takes sequence in 5'-3' direction
	initpair = sequence[0] + sequence[-1]
	DH = 0
	DS = 0
	print 'Init Pair is', initpair
	if initpair in  ['GU','UG','AU','UA']:
		DH += 3.72
		DS += 10.5	
	if ( looplength <= 9):
		#DS += -loopDG[looplength]*1000. /(273.15 + 37)  #Turner 99
		DH += loopDH[looplength] 
		DS += -( loopDG[looplength] - loopDH[looplength])*1000.0/(273.15 + 37) 
	else:
		#DS += -(loopDG[9] + 1.75 * GAS_CONSTANT * (273.15 + 37) * math.log(looplength/9.)  )*1000. / (273.15 + 37)
		DH  += loopDH[9]
		DS += -( loopDG[9] + 1.75 * GAS_CONSTANT * (273.15 + 37) * math.log(looplength/9.) -  loopDH[9])*1000.0/(273.15 + 37) 

	compseq = ''
	length = stemlength
	for i in range(stemlength):
		compseq = sequence[-i-1] + compseq

	for i in range(stemlength-1):
		pair = sequence[i] + compseq[length-1-i] + sequence[i+1] + compseq[length-1-i-1]
		print 'Pair is ',pair
		DH += stack_dh[pair]
		DS += stack_ds[pair]

	termmis_key = sequence[stemlength-1] + compseq[0] + sequence[stemlength]+  sequence[stemlength+looplength-1] 
	termMis_dH = miss_dh[termmis_key] 
	termMis_dS = miss_ds[termmis_key]
	DH += termMis_dH
	DS += termMis_dS


	print termmis_key
	print 'Final DH, DS, DG', DH, DS, (DH * 1000 - (273.15+37) * DS) / 1000.0
	RealTemp = (DH )*1000 / (DS)
	return RealTemp

 
if len(sys.argv) != 4:
	print 'Usage: ./program sequence stemLENGTH loopLENGTH'
	sys.exit(1)



#inpu = ReadParams(hstackDG)
#print inpu
#sys.exit(1)

sequence = sys.argv[1]
stemlength = int(sys.argv[2]) 
looplength = int(sys.argv[3])


import numpy as np

#taken from Turner 2011 are hairpin termination mismatches
newds_t , newdh_t = ReadDSandDH(hstackDH,hstackDG)
newds_stack, newdh_stack = ReadDSandDH_stack(stackDH,stackDG)


#print DHavg
#print DSavg

#init terms are not included for hpins
#DHinitterm = 3.61 + 3.72
#DSinitterm = -1.5 + 10.5

#penalty for AU  or GU ends
#DHinitterm =  3.72
#DSinitterm =   10.5

RealTemp = GetTM(sequence[::-1],stemlength,looplength, newdh_stack, newds_stack, newdh_t, newds_t)

print sequence , RealTemp 
#print RealTemp - 273.15, RealTemp
