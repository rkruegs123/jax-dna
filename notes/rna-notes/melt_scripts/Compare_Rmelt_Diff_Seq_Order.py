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


def GetTM(seq,compseq,DH_vals,DS_vals):
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
	 

	molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)

	if seq == compseq: 
	     #print 'Symmetrix!'
	     DS += -1.4;
	     molcon *= 4;

	RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )

	#print DH,DS,molcon/4	
	return RealTemp	


def WC_comp(sek):
	length = len(sek)
	compsek = ''
	for j in range(length):
		compsek += '*'
	for i in range(length):
		base = sek[i] 
		idx = length - 1  -i
		compsek = compsek[0:idx] + inverted[base] + compsek[idx+1:]
		#compsek[length - 1 - i] = inverted[base]
	return compsek

def get_ran_diff_st(sek,dh,ds,iters=5000):
	temp = []
    	mytm = GetTM(sek,WC_comp(sek),dh,ds)	
	mysek = list(sek[1:-1])
	seks = []
	avgdiff = 0
	for i in range(iters):
		if random.random() < 0.5:
			random.shuffle(mysek)
		else :
			r =random.randint(0,len(mysek)-1)
			if mysek[r] == 'A':
				mysek[r] = 'U'
			elif mysek[r] == 'U':
				mysek[r] = 'A'
			elif mysek[r] == 'C':
				mysek[r] = 'G'
			elif mysek[r] == 'G':
				mysek[r] = 'C'
		jsek = ''
			
		for m in mysek:
			jsek = jsek + m
		realsek = sek[0] + jsek + sek[-1]
		compsek = WC_comp(realsek)
		#print 'Temperature of ',realsek,
		if not realsek != compsek:
			tm = GetTM(realsek,compsek,dh,ds)
			#print tm
			temp.append(tm)
			seks.append(realsek)
			avgdiff += abs(tm - mytm)
			
	
	
	return max(temp) - min(temp),seks[np.argmin(temp)],seks[np.argmax(temp)],avgdiff 


def fill_all_steps(filled,seq,compseq):
	length = len(seq)
	for i in range(len(seq)-1):
		pair = seq[i] + compseq[length-1-i] + seq[i+1] + compseq[length-1-i-1]
		filled[pair] += 1
 



#sys.exit(1)


box = 20.

if(len(sys.argv) > 3):
	box = float(sys.argv[3])




import numpy as np


newds, newdh = ReadDSandDH(hstackDH,hstackDG)
if len(sys.argv) < 2:
	print "Usage: ./sek [length] cases"
	sys.exit(1)
 
#sek = sys.argv[1]
cases = 500

if len(sys.argv) >= 3:
	cases = int(sys.argv[2])

length = int(sys.argv[1])

import random
import time

seed = int(time.time())



#print get_TM('CAGGTCG',20,1.0)
#print get_TM('TAGAAATGCAAG',20,0.5)

#sys.exit(1)

minT = 500
maxT = 0

genseqs = []
gentemps = []
gendiffs = []
genseqmax = []
genseqmin = []

print WC_comp('AAAGGG')

for it in range(cases):
	sek  = ''
	for i in range(length):
		sek = sek + random.choice(bases)

	print sek,WC_comp(sek)	
	temp = GetTM(sek,WC_comp(sek),newdh,newds)
	if not sek == WC_comp(sek):
		if temp > maxT:
			maxT = temp
		if temp < minT:
			minT = temp
		#maxst,maxsek,minsek = get_rand_diff_st(sek)
		genseqs.append(sek)
		gentemps.append(temp) 
		#gendiffs.append(maxst)
		#genseqmax.append(maxsek)
		#genseqmin.append(minsek)


mymax = 0
canalA = ''
canalB = ''
myavgs = 0
for j in genseqs:		
	canmax,canA,canB,mavg = get_ran_diff_st(j,newdh,newds)
	myavgs += (mavg)
	if(canmax > mymax):
		mymax = canmax
		canalA = canA
		canalB = canB 
		myavgs += mavg

print length,mymax, myavgs / float(cases * 5000), canalA, canalB, maxT, minT

