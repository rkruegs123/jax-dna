#!/usr/bin/env python

#This script calls mutiple times the INT_11 seqdep script to obtain multiple examples of Tm
#remove the 2 for lengths 5
exec_command = './Rmelttemp_with_coax_stabilization.WC.SEQDEP.twittled.2.py' 

import sys
import random
import os

random.seed(9)

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


def GenRanCXpair():
	base = random.choice(bases)
	ibase = inverted[base]
	return base, ibase	

if len(sys.argv) != 4:
	print 'Usage: %s sequence_length howmany box' % (sys.argv[0])
	sys.exit(1)
		
	
seq_length = int(sys.argv[1])
seq_count = int(sys.argv[2])
box = float(sys.argv[3])

cxA, cxB = GenRanCXpair()
	
for i in xrange(seq_count):
	seq,compseq = GenRanSeq(seq_length)
	#print seq, compseq
	launchcommand = ' %s %s %s %s %f ' % (seq, 'W', cxA, cxB, box )
	exe_string = exec_command + launchcommand + ' | tail -n 1 '
	#print exe_string
	os.system(exe_string) 
	 
	
