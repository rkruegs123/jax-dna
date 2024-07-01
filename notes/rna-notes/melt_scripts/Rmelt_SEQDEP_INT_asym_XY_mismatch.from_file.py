#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 



#hstackDG = DATAFILEPATH + 'tstacki23.dat'
#hstackDH = DATAFILEPATH + 'tstacki23.dh'
hstackDG = DATAFILEPATH + 'tstacki.dat'
hstackDH = DATAFILEPATH + 'tstacki.dh'

stackDG = DATAFILEPATH + 'stack.dat'
stackDH = DATAFILEPATH + 'stack.dh'




loopDG = { 4 : 1.1, 5 : 2.0, 6 : 2.0, 7 : 2.1, 8 : 2.3, 9 : 2.4, 10 : 2.5, 11 : 2.6, 12 : 2.7, 13 : 2.8, 14 : 2.9, 15 : 2.9, 16 : 3.0, 17 : 3.1, 18 : 3.1, 19 : 3.2, 20 : 3.3, 21 : 3.3, 22 : 3.4, 23 : 3.4, 24 : 3.5, 25 : 3.5, 26 : 3.5, 27 : 3.6, 28 : 3.6, 29 : 3.7, 30 : 3.7}


loopDH = {4 : -7.2, 5 : -6.8, 6 : -1.3, 7 : -1.3, 8 : -1.3, 9 : -1.3, 10 : -1.3, 11 : -1.3, 12 : -1.3, 13 : -1.3, 14 : -1.3, 15 : -1.3, 16 : -1.3, 17 : -1.3, 18 : -1.3, 19 : -1.3, 20 : -1.3, 21 : -1.3, 22 : -1.3, 23 : -1.3, 24 : -1.3, 25 : -1.3, 26 : -1.3, 27 : -1.3, 28 : -1.3, 29 : -1.3, 30 : -1.3} 

asym_DH = 0.
#asym_DH = 3.2
#asym_DG = 0.48
asym_DG = 0.6
#asym_DG = 0.3

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

def GetRealTM(sequence,compseq,loop_start,looplengthX,looplengthY,stack_dh,stack_ds,miss_dh,miss_ds,box):
#takes sequence in 5'-3' directioni, loop start is the index of the first element of the sequence that i in the loop
	looplength = looplengthX + looplengthY
	loop_stopX = loop_start + looplengthX
	loop_stopY = loop_start + looplengthY


	initpair = sequence[0] + compseq[-1]
	DH = 0
	DS = 0
	#init_term
	DH += 3.61 
	DS +=-1.5 
	print 'Init Pair is', initpair
	if initpair in  ['GU','UG','AU','UA']:
		DH += 3.72
		DS += 10.5

	initpair = sequence[-1] + compseq[0]
	print 'Init Pair is', initpair
	if initpair in  ['GU','UG','AU','UA']:
		DH += 3.72
		DS += 10.5

	if looplength < 3:
		print 'Error'
		return -10

	if looplengthX > looplengthY:
		print 'Error, first loop needs to be shorter'
		return -10
	'''
	initpair = sequence[loop_start-1] + compseq[-loop_start]
	print 'Init Pair is', initpair
	if initpair in  ['GU','UG','AU','UA']:
		DH += 3.72
		DS += 10.5
	print loop_start + looplength, -1-loop_start-looplength
	initpair = sequence[loop_start+looplength] + compseq[-1-loop_start-looplength]
	print 'Init Pair is', initpair
	if initpair in  ['GU','UG','AU','UA']:
		DH += 3.72
		DS += 10.5
	'''

	#DS += -loopDG[looplength]*1000. /(273.15 + 37)  #Turner 99
	DH += loopDH[looplength] 
	DS += -( loopDG[looplength] - loopDH[looplength])*1000.0/(273.15 + 37) 
	asym_length = int(abs(looplengthX - looplengthY))
	DH += asym_length * asym_DH 
	DS += asym_length * -(asym_DG - asym_DH)*1000./(273.15 + 37)	

	print 'Asym length is',asym_length	
	#length = stemlength
	for i in range(0,loop_start-1):
		pair = sequence[i] + compseq[-1-i] + sequence[i+1] + compseq[-i-1-1]
		print 'Pair is ',pair
		DH += stack_dh[pair]
		DS += stack_ds[pair]

		
	for i in range(loop_start+looplengthX,len(sequence)-1):
		pair = sequence[i] + compseq[-1-i-asym_length] + sequence[i+1] + compseq[-i-1-1-asym_length]
		print 'Pair is ',pair
		DH += stack_dh[pair]
		DS += stack_ds[pair]

	#mismatches 
	i = loop_start - 1
	termmis_key = sequence[i] + compseq[-i-1] + sequence[i+1]+  compseq[-1-i-1] 
	print 'Mismatch key',termmis_key
	termMis_dH = miss_dh[termmis_key] 
	termMis_dS = miss_ds[termmis_key]
	DH += termMis_dH
	DS += termMis_dS
	
	i = loop_start + looplengthX - 1 
	termmis_key = sequence[i+1]+  compseq[-1-i-1-asym_length]  +  sequence[i] + compseq[-i-1-asym_length]  
	print 'Mismatch key',termmis_key 
	termMis_dH = miss_dh[termmis_key] 
	termMis_dS = miss_ds[termmis_key]
	DH += termMis_dH
	DS += termMis_dS



	molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
	RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )

	print 'Final DH, DS,DG37', DH, DS,(DH*1000 - (273.15+37)*DS)/1000.
	#RealTemp = (DH )*1000 / (DS)
	return RealTemp




if len(sys.argv) < 7:
	print 'Usage: ./program sequence compSequence mis_start no_of_nucs_in_the_missX no_of_nucs_in_the_missY box'
	sys.exit(1)


sequence = sys.argv[1]
compsequence = sys.argv[2]

misstart = int(sys.argv[3])
mislength  = int(sys.argv[4])


mislengthB = int(sys.argv[5])

box = float(sys.argv[6])

looplengthX = mislength
looplengthY = mislengthB

if looplengthX + looplengthY < 5:
	print 'Mismatch too short'
	sys.exit(1)

#taken from Turner et al
import numpy as np
#taken from Turner 2011 are loop termination mismatches

newds_stack, newdh_stack = ReadDSandDH_stack(stackDH,stackDG)
newds_mis, newdh_mis = ReadDSandDH(hstackDH,hstackDG)

'''
for j in range(len(sequence)):
        if (j < misstart or j >= misstart + mislength ) and not are_complementary(sequence[j],compsequence[len(sequence)-1-j]):
                print 'Error, bases are not complementary'
                sys.exit(1)
'''

sequence = sequence[::-1]
compsequence = compsequence[::-1]
misstart = len(sequence) - 1- misstart-looplengthX+1

print 'looplengths',looplengthX, looplengthY
RealTemp = GetRealTM(sequence,compsequence,misstart,looplengthX,looplengthY, newdh_stack, newds_stack,newdh_mis,newds_mis,box)

print sequence[::-1] ,compsequence[::-1], RealTemp 


#from here down



