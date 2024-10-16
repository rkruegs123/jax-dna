#!/usr/bin/env python

#This script calls mutiple times the INT_11 seqdep script to obtain multiple examples of Tm
#exec_command = './Rmelt_SEQDEP_INT11mismatch_from_file.py'

import sys
import random
import os

num_to_base = ['A','C','G','U']
base_to_num = {'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3}

bases = ['A','C','G','U']
inverted = {'A' : 'U', 'C' : 'G', 'G' : 'C', 'U' : 'A' }
 
def are_complementary(baseA, baseB):
	numA = base_to_num[baseA]
	numB = base_to_num[baseB]
	if(numA  + numB  == 3 or numA + numB == 5):
		return True
	else:
		return False

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

def GenMiddleBulSeq(total_length,mis_start, mis_length):
	seq, compseq = GenRanSeq(total_length)

	is_correct_middle_mis = 0
	misA, compMis = GenRanSeq(total_length)
	
	altBul, nic = GenRanSeq(mis_length)
		
	
	finalseq =  misA[0:mis_start] + altBul + misA[mis_start:]
	finalcompseq =  compMis #compMis[0:(total_length-mis_start-mis_length)] + altMis + compMis[total_length - mis_start:]
	return finalseq, finalcompseq 



if len(sys.argv) != 6:
	print 'Usage: %s sequence_length(counting middle mismatch) bulge_start bulge_length howmany box' % (sys.argv[0])
	sys.exit(1)
		
	
seq_length = int(sys.argv[1])
mis_start = int(sys.argv[2])
mis_length = int(sys.argv[3])
seq_count = int(sys.argv[4])
box = float(sys.argv[5])



exec_command = './Rmelt_SEQDEP_BULGEX.from_file.EXACT.py'



	
for i in xrange(seq_count):
	seq,compseq = GenMiddleBulSeq(seq_length,mis_start,mis_length)
	#print seq, compseq
	launchcommand = ' %s W %d %d %f ' % (seq, mis_start, mis_length, box )
	exe_string = exec_command + launchcommand + ' | tail -n 1 '
	#print exe_string
	os.system(exe_string) 
	 
	
