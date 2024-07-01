#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 

hstackDG = DATAFILEPATH + 'int22.dat'
hstackDH = DATAFILEPATH + 'int22.dh'

#hstackDG = DATAFILEPATH + 'tstacki.dat'
#hstackDH = DATAFILEPATH + 'tstacki.dh'

stackDG = DATAFILEPATH + 'stack.dat'
stackDH = DATAFILEPATH + 'stack.dh'




intpairs = ['AA','AC', 'AG', 'AU', 'CA','CC', 'CG','CU','GA','GC','GG','GU','UA','UC','UG','UU']
#loopDG = { 4 : 1.1, 5 : 2.0, 6 : 2.0, 7 : 2.1, 8 : 2.3, 9 : 2.4, 10 : 2.5, 11 : 2.6, 12 : 2.7, 13 : 2.8, 14 : 2.9, 15 : 2.9, 16 : 3.0, 17 : 3.1, 18 : 3.1, 19 : 3.2, 20 : 3.3, 21 : 3.3, 22 : 3.4, 23 : 3.4, 24 : 3.5, 25 : 3.5, 26 : 3.5, 27 : 3.6, 28 : 3.6, 29 : 3.7, 30 : 3.7}


#loopDH = {4 : -7.2, 5 : -6.8, 6 : -1.3, 7 : -1.3, 8 : -1.3, 9 : -1.3, 10 : -1.3, 11 : -1.3, 12 : -1.3, 13 : -1.3, 14 : -1.3, 15 : -1.3, 16 : -1.3, 17 : -1.3, 18 : -1.3, 19 : -1.3, 20 : -1.3, 21 : -1.3, 22 : -1.3, 23 : -1.3, 24 : -1.3, 25 : -1.3, 26 : -1.3, 27 : -1.3, 28 : -1.3, 29 : -1.3, 30 : -1.3} 


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


def Read_INT_22_Params(filename):
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


def Read_INT_22_DSandDH(fileDH,fileDG):
	dgT = 37 + 273.15

	
	finalDG = Read_INT_22_Params(fileDG)
	finalDH = Read_INT_22_Params(fileDH)
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

def GetRealTM(sequence,compseq,loop_start,looplength,stack_dh,stack_ds,miss_dh,miss_ds,box=20):
#takes sequence in 5'-3' directioni, loop start is the index of the first element of the sequence that i in the loop
	loop_stop = loop_start + looplength
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

	if looplength != 2:
		print 'Error, looplength fot this script must be 1'

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
	#DH += loopDH[2*looplength] 
	#DS += -( loopDG[2*looplength] - loopDH[2*looplength])*1000.0/(273.15 + 37) 

	
	#length = stemlength
	for i in range(0,loop_start-1):
		pair = sequence[i] + compseq[-1-i] + sequence[i+1] + compseq[-i-1-1]
		print 'Pair is ',pair
		DH += stack_dh[pair]
		DS += stack_ds[pair]

		
	for i in range(loop_start+looplength,len(sequence)-1):
		pair = sequence[i] + compseq[-1-i] + sequence[i+1] + compseq[-i-1-1]
		print 'Pair is ',pair
		DH += stack_dh[pair]
		DS += stack_ds[pair]

	#mismatches 
	i = loop_start - 1
	termmis_key = sequence[i] + sequence[i+3] + compseq[-1-i] + compseq[-i-1-3] +  sequence[i+1] + compseq[-i-1-1] + sequence[i+2] + compseq[-i-1-1-1]  
	print 'Mismatch 22 key',termmis_key
	#print miss_dh
	termMis_dH = miss_dh[termmis_key] 
	termMis_dS = miss_ds[termmis_key]
	DH += termMis_dH
	DS += termMis_dS
	
	print 'And the mismatch contributions are: dh, ds, dg: ', termMis_dH, termMis_dS, (termMis_dH * 1000. - (273.15+37)*termMis_dS) / 1000.
	#i = loop_start + looplength - 1 
	#termmis_key = sequence[i+1]+  compseq[-1-i-1]  +  sequence[i] + compseq[-i-1]  
	#print 'Mismatch key',termmis_key 
	#termMis_dH = miss_dh[termmis_key] 
	#termMis_dS = miss_ds[termmis_key]
	#DH += termMis_dH
	#DS += termMis_dS


	molcon = (2/(8.4*10.0**(-9)*box)**3)/(6.02214129*10**23)
	print 'Final DH, DS,DG37', DH, DS,(DH*1000 - (273.15+37)*DS)/1000.
	#RealTemp = (DH )*1000 / (DS)
	RealTemp = (DH )*1000 / (DS + 1.9872041* math.log(molcon/4) )



	return RealTemp




if len(sys.argv) < 6:
	print 'Usage: ./program sequence compSequence mis_start no_of_nucs_in_the_miss box'
	sys.exit(1)


sequence = sys.argv[1]
compsequence = sys.argv[2]

misstart = int(sys.argv[3])
mislength  = int(sys.argv[4])

if mislength != 2:
	print 'Mismatch must be 2'
	sys.exit(1)

box = float(sys.argv[5])

looplength = mislength


#taken from Turner et al
import numpy as np
#taken from Turner 2011 are loop termination mismatches

newds_stack, newdh_stack = ReadDSandDH_stack(stackDH,stackDG)
newds_mis, newdh_mis = Read_INT_22_DSandDH(hstackDH,hstackDG)

#print 'Reading from ', hstackDH , ' and obtained ', newdh_mis

for j in range(len(sequence)):
        if (j < misstart or j >= misstart + mislength ) and not are_complementary(sequence[j],compsequence[len(sequence)-1-j]):
                print 'Error, bases are not complementary'
                sys.exit(1)

sequence = sequence[::-1]
compsequence = compsequence[::-1]
misstart = len(sequence) - 1- misstart-looplength+1

RealTemp = GetRealTM(sequence,compsequence,misstart,looplength, newdh_stack, newds_stack,newdh_mis,newds_mis,box)

print sequence[::-1] ,compsequence[::-1], RealTemp 


#from here down



