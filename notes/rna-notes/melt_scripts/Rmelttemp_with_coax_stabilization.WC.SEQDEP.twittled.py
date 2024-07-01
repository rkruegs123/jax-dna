#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 

hstackDG = DATAFILEPATH + 'stack.dat'
hstackDH = DATAFILEPATH + 'stack.dh'


coaxhstackDG = DATAFILEPATH + 'coaxial.dat'
coaxhstackDH = DATAFILEPATH + 'coaxial.dh'





num_to_base = ['A','C','G','U']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3}

def are_complementary(baseA, baseB):
	numA = base_to_num[baseA]
	numB = base_to_num[baseB]
	#if(numA  + numB  == 3 or numA + numB == 5):
	if(numA  + numB  == 3 ):
		return True
	else:
		return False




def GetTm(seq,compseq,extra_bp_seq,extra_bp_compseq,DH_vals,DS_vals,DH_coax,DS_coax):
	#penalty for AU  or GU ends
	#selfcomplementarity:
	length = len(seq)
	terminus = str(seq[0]) + str(compseq[length-1])
	terminusB = str(seq[length-1]) + str(compseq[0])

	DHinitterm = 3.61
	DSinitterm = -1.5

	if( terminus in ['AU','UA','GU','UG'] ) : 
		DHinitterm += 3.72
		DSinitterm += 10.5
		#print >> sys.stderr, ' weak terminal penalty'

	if( terminusB in ['AU','UA','GU','UG'] ) : 
		DHinitterm += 3.72
		DSinitterm += 10.5
		#print >> sys.stderr, ' weak terminal penalty'
	
	DH = DS = 0

	for i in range(length-1):
		pair = seq[i] + compseq[length-1-i] + seq[i+1] + compseq[length-1-i-1]
		DH += DH_vals[pair]
		DS += DS_vals[pair]
		#print >> sys.stderr, 'Adding',pair, DH_vals[pair], DS_vals[pair]
	DH += ( DHinitterm)
	DS += ( DSinitterm) 
	 

	coax_pair = extra_bp_seq + extra_bp_compseq + seq[0]  + compseq[length-1]
	print coax_pair, 'with DH , Dg', DH_coax[coax_pair], (DH_coax[coax_pair]*1000. -(273.15+37)*DS_coax[coax_pair])/1000.
	#now the coaxial stacking stabilization
	DH  += DH_coax[coax_pair] 
	DS  += DS_coax[coax_pair]
	
	
	print 'Dh,Ds, Dg37: ', DH, DS, (DH*1000. -(273.15+37)*DS)/1000.
	
	molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)

	
	
	RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )

	#print DH,DS,molcon/4	
	return RealTemp	
	
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
							if energy != '.' and are_complementary(num_to_base[x], num_to_base[y]):
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

def GenRanMis(length,mis):
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
	mispos = 0
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

 

if len(sys.argv) < 6:
	print 'Usage: ./program seqLENGTH SEQUENCE COMPSEQUENCE(orWorO) coaxoverA coaxoverB box'
	sys.exit(1)


#ta prvni sekvence je soucasti hairpinu. coaxoverA je na 5' konci hairpinu

#sys.exit(1)


sequence = sys.argv[1]
compsequence = sys.argv[2]

coax_over_A = sys.argv[3]
coax_over_B = sys.argv[4]

box = 20.
if(len(sys.argv) > 6):
	box = float(sys.argv[5])


length = len(sequence)

if compsequence == 'W' or compsequence == 'O': #WC autocomplete
	compsek = ''
	for j in range(length):
		compsek += '*'
	for i in range(length):
		base = sequence[i]  
		idx = length - 1  -i
		if compsequence == 'W':
			inversion = inverted[base]
		else:
			inversion = inverted_wobble[base]

		compsek = compsek[0:idx] + inversion + compsek[idx+1:]

	compsequence = compsek

#GetTM takes sequnce in 5-3 order, so they need to be reversed"
sequence = sequence[::-1]
compsequence = compsequence[::-1]


for j in range(len(sequence)):
	if not are_complementary(sequence[j],compsequence[len(sequence)-1-j]):
		print 'Error, bases are not complementary'
		sys.exit(1)





import numpy as np


newds, newdh = ReadDSandDH(hstackDH,hstackDG)
newdscoax, newdhcoax = ReadDSandDH(coaxhstackDH,coaxhstackDG)


#this is specific to extra sequence of length 5!
# CGTTGCGATGCATCGCAACGCACTG


#uncomment for different sequences
#SIM_TOTAL_LENGTH = 26
#extrabit = (SIM_TOTAL_LENGTH - len(sequence) -1) * ['A']

extrabit = list('CGUUGCGAUGCAUCGCAACG')
extrabit[0] = coax_over_A
extrabit[-1] = coax_over_B

extrabit = "".join(extrabit)
 
RealTemp = GetTm(sequence,compsequence,coax_over_A, coax_over_B ,newdh,newds,newdhcoax, newdscoax)

#sequence je samostatna; coax_over_A sousedi na 3' konci hairpin sequence, immediately adjacent
print sequence[::-1],extrabit + compsequence[::-1] , RealTemp

