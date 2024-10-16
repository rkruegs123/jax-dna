#!/usr/bin/env python27
import math
import sys,os

def is_symmetric(seq):
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'T') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'T' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
  
  return symmetric 

  
def get_TM(seq,boxsize=20,salt_conc=0.5):
  length = len(seq);
  deltaH = 0;
  deltaS = 0;
  molarconcentration = 2.6868994 / (boxsize*boxsize*boxsize) ;
 
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'T') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'T' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
  
  for  i in range (length-1) :
        
     pair = seq[i] + seq[i+1];

     if(pair ==  "AA" ):  
	 deltaH += -7.6 
	 deltaS += -21.3; 
     if (pair == "TT"):
	 deltaH += -7.6
	 deltaS += -21.3; 

     if (pair == "TA"):
	 deltaH += -7.2
	 deltaS += -20.4
      
     if (pair == "AT"):
	 deltaH += -7.2
	 deltaS += -21.3
      
     if (pair == "AC"):
	 deltaH += -8.5
	 deltaS += -22.7

     if (pair == "GT"):
	 deltaH += -8.5
	 deltaS += -22.7
      
     if (pair == "TG"):
	deltaH += -8.4
	deltaS += -22.4; 

     if (pair == "CA"):
	 deltaH += -8.4
	 deltaS += -22.4; 
        

     if (pair == "TC"):
	 deltaH += -7.8
	 deltaS += -21.0; 

     if (pair == "GA"):
	 deltaH += -7.8
	 deltaS += -21.0; 
      
     if (pair == "AG"):
 	 deltaH += -8.2;
 	 deltaS += -22.2; 

     if (pair == "CT"):
	 deltaH += -8.2
	 deltaS += -22.2; 
      
 
     if (pair == "GC"):
	 deltaH += -10.6
	 deltaS +=  -27.2; 
      
     if (pair == "CG"):
	 deltaH += -9.8
	 deltaS += -24.4;

     if (pair == "GG"):
	 deltaH += -8.0;
	 deltaS += -19.9; 

     if (pair == "CC"):
	 deltaH += -8.0;
	 deltaS += -19.9; 
       
      
  
    
    
  if(seq[0] == 'A' or seq[0] == 'T'):
   	deltaH += 2.2;
	deltaS += 6.9;
   



  if(seq[length-1] == 'A' or seq[length-1] == 'T'):
	deltaH += 2.2;
	deltaS += 6.9;
   

  deltaH += 0.2;
  deltaS += -5.7;

  deltaH *= 1000.0;

  divisor = 2.0;
 
  if(symmetric):
     deltaS += -1.4;
     divisor = 0.5;

  salt_correction = 0.368 * (len(seq) - 1.0)  * math.log(salt_conc)
  deltaS += salt_correction

 
  GAS_CONSTANT =  1.9858775 


  Tm = deltaH / (deltaS + GAS_CONSTANT * math.log(molarconcentration/divisor) );
  return Tm - 273.15 

import itertools
import numpy as np

def get_diffr_st(sek):
	temp = []
	
	mysek = list(sek)
	seks = []

	for g in itertools.permutations(mysek):
		jsek = ''
		for m in g:
			jsek = jsek + m
		if not is_symmetric(jsek):
			temp.append(get_TM(jsek))
			seks.append(jsek)
	
	
	return max(temp) - min(temp),seks[np.argmin(temp)],seks[np.argmax(temp)]
	

import random 

def get_ran_diff_st(sek,iters=5000):
	temp = []
    	mytm = get_TM(sek)	
	mysek = list(sek[1:-1])
	seks = []
	avgdiff = 0
	for i in range(iters):
		if random.random() < 0.5:
			random.shuffle(mysek)
		else :
			r =random.randint(0,len(mysek)-1)
			if mysek[r] == 'A':
				mysek[r] = 'T'
			elif mysek[r] == 'T':
				mysek[r] = 'A'
			elif mysek[r] == 'C':
				mysek[r] = 'G'
			elif mysek[r] == 'G':
				mysek[r] = 'C'
		jsek = ''
			
		for m in mysek:
			jsek = jsek + m
		realsek = sek[0] + jsek + sek[-1]
		#print 'Temperature of ',realsek,
		if not is_symmetric(realsek):
			tm = get_TM(realsek)
			#print tm
			temp.append(tm)
			seks.append(realsek)
			avgdiff += abs(tm - mytm)
			
	
	
	return max(temp) - min(temp),seks[np.argmin(temp)],seks[np.argmax(temp)],avgdiff 

if len(sys.argv) < 2:
	print "Usage: ./sek [length] cases"
	sys.exit(1)
 
#sek = sys.argv[1]
cases = 50000

if len(sys.argv) >= 3:
	cases = int(sys.argv[2])

length = int(sys.argv[1])

import random
import time

seed = int(time.time())


bases = ['A','C','G','T']

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

for it in range(cases):
	sek  = ''
	for i in range(length):
		sek = sek + random.choice(bases)

	temp = get_TM(sek)
	if not is_symmetric(sek):
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
	canmax,canA,canB,mavg = get_ran_diff_st(j)
	myavgs += (mavg)
	if(canmax > mymax):
		mymax = canmax
		canalA = canA
		canalB = canB 
		myavgs += mavg

print length,mymax, myavgs / float(cases * 5000), canalA, canalB, maxT, minT
#print get_TM('CAGGTCG',20,1.0)
#print get_TM('TAGAAATGCAAG',20,0.5)

#sys.exit(1)

