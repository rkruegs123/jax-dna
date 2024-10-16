#!/usr/bin/env python

#This script calls mutiple times the INT_11 seqdep script to obtain multiple examples of Tm
exec_command = './Rmelt_SEQDEP_INT11mismatch_from_file.py'

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

def GenMiddleMisSeq(total_length,mis_start, mis_length):
	seq, compseq = GenRanSeq(total_length)

	is_correct_middle_mis = 0
	misA, compMis = GenRanSeq(total_length)
	
	while not is_correct_middle_mis:
		altMis, nic = GenRanSeq(mis_length)
		is_correct_middle_mis = 1
		for id in range(mis_length):
			if are_complementary(misA[mis_start+id],altMis[mis_length - 1 - id]):
				is_correct_middle_mis = 0
	
	
	finalseq =  misA 
	finalcompseq = compMis[0:(total_length-mis_start-mis_length)] + altMis + compMis[total_length - mis_start:]
	return finalseq, finalcompseq 



if len(sys.argv) != 6:
	print 'Usage: %s sequence_length(counting middle mismatch) mis_start mis_length howmany box' % (sys.argv[0])
	sys.exit(1)
		
	
seq_length = int(sys.argv[1])
mis_start = int(sys.argv[2])
mis_length = int(sys.argv[3])
seq_count = int(sys.argv[4])
box = float(sys.argv[5])


if mis_length == 1:
	exec_command = './Rmelt_SEQDEP_INT11mismatch_from_file.py'
elif mis_length == 2:
	exec_command = './Rmelt_SEQDEP_INT22mismatch_from_file.py'
elif mis_length == 3:
	exec_command = './Rmelt_SEQDEP_INTXXmismatch_from_file.py'
else:
	print 'Wrong usage of mismatch segment length. It should be between 1 to 3'
	sys.exit(1)
	
for i in xrange(seq_count):
	seq,compseq = GenMiddleMisSeq(seq_length,mis_start,mis_length)
	#print seq, compseq
	launchcommand = ' %s %s %d %d %f ' % (seq, compseq, mis_start, mis_length, box )
	exe_string = exec_command + launchcommand + ' | tail -n 1 '
	#print exe_string
	os.system(exe_string) 
	 
	
