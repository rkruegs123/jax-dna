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


def is_symmetric(seq):
  symmetric_points = 0
  for i in range(length):
        if  are_complementary( base_to_num[seq[i]], base_to_num[ seq[length - 1 -i]]): 
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1

  return symmetric


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
inverted_wobble = {'A' : 'U', 'C' : 'G', 'G' : 'U', 'U' : 'G' }
 
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

def GenRanSeqGU(length):
	sek = ''
	compsek = ''
	for j in range(length):
		compsek += '*'
	for i in range(length):
		base = random.choice(bases) 
		sek += base 
		idx = length - 1  -i
		if random.random() < 0.5:
			compsek = compsek[0:idx] + inverted_wobble[base] + compsek[idx+1:]
		else:
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

def contains_four_correction(seq,compseq):
	correction = 0
	if 'GGUC' in seq and 'GGUC' in compseq:
		startindex = 0
		while startindex != -1:
			#print 'Looking from', startindex
			sindex = seq.find('GGUC',startindex)
			#print 'Found ', sindex
			if sindex == -1:
				break
			compslice = compseq[len(seq)-sindex-4:len(seq)-sindex]
			print >> sys.stderr, 'Detected complementary slice ',compslice
			if compslice == 'GGUC':
				correction += 1
			startindex = sindex + 2
			
	return correction


def GetTM(seq,compseq,DH_vals,DS_vals):
	#penalty for AU  or GU ends
	#selfcomplementarity:
	length = len(seq)
	terminus = str(seq[0]) + str(compseq[length-1])
	terminusB = str(seq[length-1]) + str(compseq[0])

	DHinitterm = 3.6
	DSinitterm = -1.5

	if( terminus in ['AU','UA','GU','UG'] ) : 
		DHinitterm += 3.7
		DSinitterm += 10.5
		#print >> sys.stderr, ' weak terminal penalty'

	if( terminusB in ['AU','UA','GU','UG'] ) : 
		DHinitterm += 3.7
		DSinitterm += 10.5
		#print >> sys.stderr, ' weak terminal penalty'
	
	DH = DS = 0

	for i in range(length-1):
		pair = seq[i] + compseq[length-1-i] + seq[i+1] + compseq[length-1-i-1]
		DH += DH_vals[pair]
		DS += DS_vals[pair]
		print >> sys.stderr, 'Adding',pair, DH_vals[pair], DS_vals[pair]
	DH += ( DHinitterm)
	DS += ( DSinitterm) 

	multicor = contains_four_correction(seq,compseq)
	if multicor > 0:
			DH += 0 * -30.8 * multicor
			DS += 0 * -86.0 * multicor
			print >> sys.stderr,'Adding correction for GGUC',	 

	molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)

	if seq == compseq: 
	     #print 'Symmetrix!'
	     DS += -1.4;
	     molcon *= 4;

	RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )

	print DH,DS,(DH*1000. - (273.15+37)*DS)/1000. 
	return RealTemp	


def fill_all_steps(filled,seq,compseq):
	length = len(seq)
	for i in range(len(seq)-1):
		pair = seq[i] + compseq[length-1-i] + seq[i+1] + compseq[length-1-i-1]
		filled[pair] += 1
 
if len(sys.argv) < 3:
	print 'Usage: ./program seqLENGTH SEQUENCE COMPSEQUENCE(orWorO) box'
	sys.exit(1)




#sys.exit(1)


sequence = sys.argv[1]
compsequence = sys.argv[2]


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

box = 20.

if(len(sys.argv) > 4):
	box = float(sys.argv[4])




import numpy as np


newds, newdh = ReadDSandDH(hstackDH,hstackDG)


RealTemp = GetTM(sequence,compsequence,newdh,newds)

print sequence[::-1],compsequence[::-1],RealTemp

#val = filled.values()
#print >> sys.stderr,  min(val),max(val)
#for key,val in filled.items():
#	print >> sys.stderr, key, val
