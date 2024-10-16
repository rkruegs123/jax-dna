#!/usr/bin/env python
import math
import sys,os

inverted = {'A' : 'U', 'C' : 'G', 'G' : 'C', 'U' : 'A' }

def get_overhang_3(last_pair,last_nuc):
	#taken from Turner 95;
	# overhangs of type 5-lastpair_53-3 lastnuc
	# 5AX3 + U  will be denoted as AUX in the array:
	deltaH =  { 'AUA' : -4.9, 'AUC' : -0.9, 'AUG' : -5.5, 'AUU' : -2.3   ,
		   'CGA' : -9.0, 'CGC' : -4.1, 'CGG' : -8.6,  'CGU' : -7.5, 
		   'GCA' : -7.4, 'GCC': -2.8, 'GCG' : -6.4, 'GCU' : -3.6,
		   'UAA' : -5.7, 'UAC' : -0.7, 'UAG' : -5.8, 'UAU' : -2.2  }

 	deltaS =  { 'AUA' : -13.2,  'AUC' : -1.2, 'AUG' : -15.0, 'AUU' : -5.4  , 
		   'CGA' : -23.4, 'CGC' : -10.7, 'CGG' : -22.2,  'CGU' : -20.4, 
		   'GCA' : -20.0, 'GCC': -7.9, 'GCG' : -16.6, 'GCU' : -9.7,
		   'UAA' : -16.4, 'UAC' : -1.8, 'UAG' : -16.4, 'UAU' : -6.8  }
	triplet_id = last_pair[0] + last_nuc + last_pair[1]
	
	return deltaH[triplet_id],deltaS[triplet_id]
	
def get_overhang_5(last_pair,last_nuc):
	#taken from Turner 95;
	# overhangs of type 5-lastpair_53-3 lastnuc
	# 5AX3 + U  will be denoted as AUX in the array:
	deltaH =  { 'AUA' : 1.6, 'AUC' : 2.2, 'AUG' : 0.7, 'AUU' : 3.1   ,
		   'CGA' : -2.4, 'CGC' : 3.3, 'CGG' : 0.8,  'CGU' : -1.4, 
		   'GCA' : -1.6, 'GCC':  0.7, 'GCG' : -4.6, 'GCU' : -0.4,
		   'UAA' : -0.5, 'UAC' : 6.9, 'UAG' : 0.6, 'UAU' : 0.6  }

 	deltaS =  { 'AUA' : 6.1,  'AUC' : 7.9, 'AUG' : 3.4, 'AUU' : 10.6 , 
		   'CGA' : -6.0, 'CGC' : 11.8, 'CGG' : 3.4,  'CGU' : -4.3, 
		   'GCA' : -4.5, 'GCC': 3.1, 'GCG' : -14.8, 'GCU' : -1.2,
		   'UAA' : -0.7, 'UAC' : 22.8, 'UAG' : 2.7, 'UAU' : 2.7  }


	triplet_id = last_pair[1] + last_nuc + last_pair[0]
	
	return deltaH[triplet_id],deltaS[triplet_id]





def is_symmetric(seq):
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'U') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'U' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
  
  return symmetric 

def get_double_over_TM(seq,overhang_type,overA,overB,boxsize=20):
  sek = seq
  length = len(sek);
  deltaH = 0;
  deltaS = 0;
  molarconcentration = 2.686899657 / (boxsize*boxsize*boxsize) ;
 
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'U') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'U' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
  
  for  i in range (length-1) :
        
     pair = seq[i] + seq[i+1];

     if(pair ==  "AA" ):  
	 deltaH += -6.82; 
	 deltaS += -19.0; 
     if (pair == "UU"):
	 deltaH += -6.82;
	 deltaS += -19.0; 

     if (pair == "UA"):
	 deltaH += -9.38;
	 deltaS += -26.7;
      
     if (pair == "AU"):
	 deltaH += -7.69;
	 deltaS += -20.5;
      
     if (pair == "AC"):
	 deltaH += -10.44;
	 deltaS += -26.9;

     if (pair == "GU"):
	 deltaH += -10.44;
	 deltaS += -26.9;
      
     if (pair == "UG"):
	deltaH += -11.4;
	deltaS += -29.5; 

     if (pair == "CA"):
	 deltaH += -11.4;
	 deltaS += -29.5; 
        

     if (pair == "UC"):
	 deltaH += -10.48;
	 deltaS += -27.1; 

     if (pair == "GA"):
	 deltaH += -10.48;
	 deltaS += -27.1; 
      
     if (pair == "AG"):
 	 deltaH += -12.44;
 	 deltaS += -32.5; 

     if (pair == "CU"):
	 deltaH += -12.44;
	 deltaS += -32.5; 
      
 
     if (pair == "GC"):
	 deltaH += -10.64;
	 deltaS +=  -26.7; 
      
     if (pair == "CG"):
	 deltaH += -14.88;
	 deltaS += -36.9;

     if (pair == "GG"):
	 deltaH += -13.39;
	 deltaS += -32.7; 

     if (pair == "CC"):
	 deltaH += -13.39;
	 deltaS += -32.7; 
       
      
  
    
    
  if(seq[0] == 'A' or seq[0] == 'U'):
   	deltaH += 3.72;
	deltaS += 10.5;
   



  if(seq[length-1] == 'A' or seq[length-1] == 'U'):
	deltaH += 3.72;
	deltaS += 10.5;
   

  deltaH += 3.61;
  deltaS += -1.5;

  compsek = ' ' * len(sek)
  for i in range(length):
	base = sek[i] 
	idx = length - 1  -i
	compsek = compsek[0:idx] + inverted[base] + compsek[idx+1:]

  #now deal with the overhangs:
  if overhang_type == '3':
	sek = overA + sek
	compsek = overB + compsek 
	last_pair_53 = sek[1] + sek[0]
	last_nuc = compsek[-1]
	extraH1, extraS1 = get_overhang_3(last_pair_53,last_nuc)

	last_pair_53 = compsek[1] + compsek[0]
	last_nuc = sek[-1] 
	extraH2, extraS2 = get_overhang_3(last_pair_53,last_nuc)  

	deltaH += extraH1 + extraH2
	deltaS += extraS1 + extraS2
 
  elif overhang_type == '5':
	sek = sek + overA 
	compsek = compsek + overB  
	last_pair_53 = sek[-1] + sek[-2]
	last_nuc = compsek[0]
	extraH1, extraS1 = get_overhang_5(last_pair_53,last_nuc)

	last_pair_53 = compsek[-1] + compsek[-2]
	last_nuc = sek[0] 
	extraH2, extraS2 = get_overhang_5(last_pair_53,last_nuc)  

	deltaH += extraH1 + extraH2
	deltaS += extraS1 + extraS2
 
	

  else:
       print 'Wrong overhang type!'

  deltaH *= 1000.0;

  divisor = 2.0;
 

  #salt_correction = 0.368 * (len(seq) - 1.0)  * math.log(salt_conc)
  #deltaS += salt_correction

 
  GAS_CONSTANT =  1.9858775 
  print 'Molar concentration is ',molarconcentration
  print 'DH=',deltaH, 'DS=',deltaS
  #deltaH = -109.57 * 1000
  #deltaS = -274.3
  Tm = deltaH / (deltaS + GAS_CONSTANT * math.log(molarconcentration/divisor) );
  return Tm - 273.15 


  
def get_TM(seq,boxsize=20):
  length = len(seq);
  deltaH = 0;
  deltaS = 0;
  molarconcentration = 2.686899657 / (boxsize*boxsize*boxsize) ;
 
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'U') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'U' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
  
  for  i in range (length-1) :
        
     pair = seq[i] + seq[i+1];

     if(pair ==  "AA" ):  
	 deltaH += -6.82; 
	 deltaS += -19.0; 
     if (pair == "UU"):
	 deltaH += -6.82;
	 deltaS += -19.0; 

     if (pair == "UA"):
	 deltaH += -9.38;
	 deltaS += -26.7;
      
     if (pair == "AU"):
	 deltaH += -7.69;
	 deltaS += -20.5;
      
     if (pair == "AC"):
	 deltaH += -10.44;
	 deltaS += -26.9;

     if (pair == "GU"):
	 deltaH += -10.44;
	 deltaS += -26.9;
      
     if (pair == "UG"):
	deltaH += -11.4;
	deltaS += -29.5; 

     if (pair == "CA"):
	 deltaH += -11.4;
	 deltaS += -29.5; 
        

     if (pair == "UC"):
	 deltaH += -10.48;
	 deltaS += -27.1; 

     if (pair == "GA"):
	 deltaH += -10.48;
	 deltaS += -27.1; 
      
     if (pair == "AG"):
 	 deltaH += -12.44;
 	 deltaS += -32.5; 

     if (pair == "CU"):
	 deltaH += -12.44;
	 deltaS += -32.5; 
      
 
     if (pair == "GC"):
	 deltaH += -10.64;
	 deltaS +=  -26.7; 
      
     if (pair == "CG"):
	 deltaH += -14.88;
	 deltaS += -36.9;

     if (pair == "GG"):
	 deltaH += -13.39;
	 deltaS += -32.7; 

     if (pair == "CC"):
	 deltaH += -13.39;
	 deltaS += -32.7; 
       
      
  
    
    
  if(seq[0] == 'A' or seq[0] == 'U'):
   	deltaH += 3.72;
	deltaS += 10.5;
   



  if(seq[length-1] == 'A' or seq[length-1] == 'U'):
	deltaH += 3.72;
	deltaS += 10.5;
   

  deltaH += 3.61;
  deltaS += -1.5;

  deltaH *= 1000.0;

  divisor = 2.0;
 
  if(symmetric):
     print 'Symmetrix!'
     deltaS += -1.4;
     divisor = 0.5;

  #salt_correction = 0.368 * (len(seq) - 1.0)  * math.log(salt_conc)
  #deltaS += salt_correction

 
  GAS_CONSTANT =  1.9858775 
  print 'Molar concentration is ',molarconcentration
  print deltaH, deltaS
  #deltaH = -109.57 * 1000
  #deltaS = -274.3
  Tm = deltaH / (deltaS + GAS_CONSTANT * math.log(molarconcentration/divisor) );
  return Tm - 273.15 



if len(sys.argv) != 5:
	print "Usage: ./sek sek double_overhang_type over-baseA over-baseB"
	sys.exit(1)
 
sek = sys.argv[1]
overhang_type = sys.argv[2]
overA = sys.argv[3]
overB = sys.argv[4]


Tm =  get_double_over_TM(sek,overhang_type,overA,overB)

print sek, Tm+273.15
#print get_TM(sek)

#print get_TM('CAGGTCG',20,1.0)
#print get_TM('TAGAAATGCAAG',20,0.5)

#sys.exit(1)

