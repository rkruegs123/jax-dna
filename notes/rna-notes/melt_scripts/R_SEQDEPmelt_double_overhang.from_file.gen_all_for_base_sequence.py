#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
hstackDG = DATAFILEPATH + 'dangle.dat'
hstackDH = DATAFILEPATH + 'dangle.dh'

stackDG = DATAFILEPATH + 'stack.dat'
stackDH = DATAFILEPATH + 'stack.dh'

num_to_base = ['A','C','G','U']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3}

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

def GetTM(seq,compseq,DH_vals,DS_vals,DH_over_vals_3,DS_over_vals_3,DH_over_vals_5, DS_over_vals_5,over_type,overA,overB):
	#the program receives sequences in 5'-3' order
	length = len(seq)

	terminus =  str(compseq[length-1]) + str(seq[0])
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

	if over_type == '3':	
		termpairA = seq[length-1] + compseq[0] + overA
		termpairB =  compseq[length-1] + seq[0] + overB 
	else:
		termpairA = compseq[length-1] +seq[0]+ overA
		termpairB =seq[length-1] +  compseq[0] + overB 
			
	if over_type == '3': 
		#print 'Adding 3',termpairA, ' and ', termpairB, DH_over_vals_3[termpairA], DH_over_vals_3[termpairB] 
		overDHA = DH_over_vals_3[termpairA]   
		overDSA = DS_over_vals_3[termpairA]
		overDHB = DH_over_vals_3[termpairB]
		overDSB = DS_over_vals_3[termpairB]
	elif over_type == '5':
		#print 'Adding 5',termpairA, ' and ', termpairB, DH_over_vals_5[termpairA], DH_over_vals_5[termpairB] 
		overDHA = DH_over_vals_5[termpairA]   
		overDSA = DS_over_vals_5[termpairA]
		overDHB = DH_over_vals_5[termpairB]
		overDSB = DS_over_vals_5[termpairB]
  	else:	
		print 'Error, unknown ending',over_type
		return 0		
	DH += overDHA + overDHB
	DS += overDSA + overDSB
	molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)

	
	RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )

	#print DH,DS,molcon/4	
	return RealTemp	

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
		

if len(sys.argv) < 4:
	print "Usage: ./%s length base_sequence overtype [box=20]" % (sys.argv[0])
	sys.exit(1)
 
length = int(sys.argv[1])
base_sequence = sys.argv[2]
length = len(base_sequence)

overtype = sys.argv[3]
if(overtype != '3' and overtype != '5'):
	print 'Error, overtype is 3 or 5'
	sys.exit(1)
	 
box = 20.

if len (sys.argv) >= 5:
	box = float(sys.argv[4])



newds3, newdh3, newds5, newdh5 = ReadDSandDH35(hstackDH,hstackDG)

newds_stack, newdh_stack = ReadDSandDH_stack(stackDH,stackDG)


base_compsek = ''
for j in range(length):
	base_compsek += '*'
for i in range(length):
	base = base_sequence[i] 
	idx = length - 1  -i
	base_compsek = base_compsek[0:idx] + inverted[base] + base_compsek[idx+1:]




for base_a in bases:
	for base_b in bases:
		if are_complementary(base_a, base_b):
			for overT in bases:
				
				sek, compsequence = base_sequence,base_compsek
				sek = base_a + sek[1:-1] + base_b
				compsequence =  base_a + compsequence[1:-1] + base_b
				
				sek = sek[::-1]
				compsequence = compsequence[::-1]
				overA = overT
				overB = overT
				RealTemp = GetTM(sek,compsequence,newdh_stack,newds_stack,newdh3,newds3,newdh5,newds5,overtype,overA,overB)


				#now we inverse the sequences, since they will be printed in 3' 5' direction
				sek = sek[::-1]
				compsequence = compsequence[::-1]

				if overtype == '3':
					sek = overA + sek
					compsequence = overB + compsequence
				if overtype == '5':
					sek = sek + overA
					compsequence = compsequence + overB

				print sek, compsequence , RealTemp 
