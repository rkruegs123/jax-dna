#!/usr/bin/env python

import sys


def HasGUGU(sek,compsek,gutypes):
	double_GU = 0
	correct_gutypes = 0
	for i in xrange(len(sek)-1):
		pairA = sek[i] + compsek[len(sek)-1-i]
		pairB = sek[i+1] + compsek[len(sek) - 1 - i-1]
		if pairA in ['GU','UG'] and pairB in ['GU','UG']:
			double_GU  += 1
			if (pairA + pairB in gutypes):
				correct_gutypes += 1
		
	return double_GU, correct_gutypes


def has_gu(sekA,sekB):
        counter = 0
        lastgu = -2
        neighbor = 0
        for i in range(len(sekA)):
                if (sekA[i] == 'G' and sekB[len(sekB) - i - 1] == 'U' ) or (sekA[i] == 'U' and sekB[len(sekB) - i - 1] == 'G') :
                        counter += 1
                        if(lastgu == i - 1):
                                neighbor += 1
                        lastgu = i

        if neighbor > 0:
                print 'Warning, neighboring GUGU', sekA, sekB
        return counter

myfile = sys.argv[1]
gutype = sys.argv[2]

inp = open(myfile,'r')

nogu = 0
gu_counter = 0
count = 0

for line in inp.readlines():
        valsA = line.split()[0].strip()
        valsB = line.split()[1].strip()
        if( has_gu(valsA,valsB) ):
                gu_counter += has_gu(valsA,valsB)
                count += 1
        else:
                #print line
                nogu += 1

	if HasGUGU(valsA,valsB,gutype):
		print line,

print >> sys.stder, 'Gu: ', gu_counter, 'nogu', nogu, 'with at least 1 gu: ', count
