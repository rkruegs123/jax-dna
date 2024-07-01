#!/usr/bin/env python


import sys
import math
import random

GAS_CONSTANT = 1.9858775 * 10**(-3)


DATAFILEPATH = '/home/petr/studium/oxford/rna_prediction_tools/RNAstructure/data_tables/' 
#hstackDG = DATAFILEPATH + 'tstacki.dat'
#hstackDH = DATAFILEPATH + 'tstacki.dh'



stackDG = DATAFILEPATH + 'stack.dat'
stackDH = DATAFILEPATH + 'stack.dh'



loopDG = {1 : 3.8, 2 : 2.8, 3 : 3.2, 4 : 3.6, 5 : 4.0, 6 : 4.4, 7 : 4.6, 8 : 4.7, 9 : 4.8, 10 : 4.9, 11 : 5.0, 12 : 5.1, 13 : 5.2, 14 : 5.3, 15 : 5.4, 16 : 5.4, 17 : 5.5, 18 : 5.5, 19 : 5.6, 20 : 5.7, 21 : 5.7, 22 : 5.8, 23 : 5.8, 24 : 5.8, 25 : 5.9, 26 : 5.9, 27 : 6.0, 28 : 6.0, 29 : 6.0, 30 : 6.1}

loopDH = {1 : 10.6, 2 : 7.1, 3 : 7.1, 4 : 7.1, 5 : 7.1, 6 : 7.1, 7 : 7.1, 8 : 7.1, 9 : 7.1, 10 : 7.1, 11 : 7.1, 12 : 7.1, 13 : 7.1, 14 : 7.1, 15 : 7.1, 16 : 7.1, 17 : 7.1, 18 : 7.1, 19 : 7.1, 20 : 7.1, 21 : 7.1, 22 : 7.1, 23 : 7.1, 24 : 7.1, 25 : 7.1, 26 : 7.1, 27 : 7.1, 28 : 7.1, 29 : 7.1, 30 : 7.1}

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


def get_avg_mis_HS():
	DH_vals = []
	DS_vals = []
	newds, newdh = ReadDSandDH(hstackDH,hstackDG)
	print len(newds.keys())	
	for key in newds.keys():
		DH_vals.append(newdh[key])
		DS_vals.append(newds[key])
	return np.mean(DH_vals), np.mean(DS_vals)

def GetTM(sequence,compseq,loop_start,looplength,stack_dh,stack_ds,box=20):
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

	if looplength > 1:
		initpair = sequence[loop_start-1] + compseq[-loop_start]
		print 'Init Pair is', initpair
		if initpair in  ['GU','UG','AU','UA']:
			DH += 3.72
			DS += 10.5
		initpair = sequence[loop_start+looplength] + compseq[-1-loop_start]
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

	
	#length = stemlength
	for i in range(0,loop_start-1):
		pair = sequence[i] + compseq[-1-i] + sequence[i+1] + compseq[-i-1-1]
		print 'Pair is ',pair, ' and dh is ', stack_dh[pair]
		DH += stack_dh[pair]
		DS += stack_ds[pair]

	if looplength == 1:
		#efn2 also applies 0.4 dg penaty for each au base pair next to the bulge....
		print 'Middle pair'
		pair = sequence[loop_start-1] + compseq[-loop_start] + sequence[loop_start+1] + compseq[-loop_start-1]
		print 'Pair is ',pair, ' with dh', stack_dh[pair]
		DH += stack_dh[pair]
		DS += stack_ds[pair]
		
	for i in range(loop_start+looplength,len(sequence)-1):
		pair = sequence[i] + compseq[-1-i+looplength] + sequence[i+1] + compseq[-i-1-1+looplength]
		print 'Pair is ',pair, ' with dh ', stack_dh[pair]
		DH += stack_dh[pair]
		DS += stack_ds[pair]

	#mismatches 


	
	molcon = (2/(8.518*10.0**(-9)*box)**3)/(6.02214129*10**23)
	print 'Final DH, DS,DG37', DH, DS,(DH*1000 - (273.15+37)*DS)/1000.
	#RealTemp = (DH )*1000 / (DS)
	RealTemp = (DH )*1000 / (DS + 1.9859* math.log(molcon/4) )
	return RealTemp


if len(sys.argv) < 6:
	print 'Usage: ./program sequence compSequence(or W/O) bulge_start no_of_nucs_in_the_bulge box'
	sys.exit(1)




#sys.exit(1)
sequence = sys.argv[1]

sequence = sys.argv[1]
compsequence = sys.argv[2]

bulge_start = int(sys.argv[3])
bulge  = int(sys.argv[4])
box = float(sys.argv[5])

looplength = bulge

newds_stack, newdh_stack = ReadDSandDH_stack(stackDH,stackDG)


length = len(sequence) - looplength

if compsequence == 'W' or compsequence == 'O': #WC autocomplete
        compsek = ''
        for j in range(length):
                compsek += '*'
        for i in range(bulge_start):
                base = sequence[i]  
                idx = length - 1  -i
                if compsequence == 'W':
                        inversion = inverted[base]
                else:
                        inversion = inverted_wobble[base]

                compsek = compsek[0:idx] + inversion + compsek[idx+1:]
        for i in range(bulge_start+looplength,len(sequence)):
                base = sequence[i]  
                idx = length - 1  -(i -looplength)
                if compsequence == 'W':
                        inversion = inverted[base]
                else:
                        inversion = inverted_wobble[base]

                compsek = compsek[0:idx] + inversion + compsek[idx+1:]


        compsequence = compsek
print sequence, compsequence
#GetTM takes sequnce in 5-3 order, so they need to be reversed"
sequence = sequence[::-1]
compsequence = compsequence[::-1]

'''
for j in range(len(sequence)):
        if not are_complementary(sequence[j],compsequence[len(sequence)-1-j]):
                print 'Error, bases are not complementary'
                sys.exit(1)
stemlength = int(sys.argv[1]) - 2 
box = 20.
'''
#bulge_start = len(sequence)-bulge_start-1-looplength
RealTemp = GetTM(sequence[::-1],compsequence[::-1],bulge_start,looplength, newdh_stack, newds_stack,box)

print sequence ,compsequence, RealTemp 


#print RealTemp - 273.15, RealTemp
