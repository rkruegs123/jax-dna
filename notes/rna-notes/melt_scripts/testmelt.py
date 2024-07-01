#!/usr/bin/env python
import math
import sys,os

def is_symmetric(seq):
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'U') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'U' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
  
  return symmetric 

  
def get_TM(seq,boxsize=20):
  length = len(seq);
  deltaH = 0;
  deltaS = 0;
  molarconcentration = 2.686899657 / (boxsize*boxsize*boxsize) ;
#  molarconcentration =  0.1 * 10.**(-3.) 
  symmetric_points = 0
  for i in range(length):
   	if (seq[i] == 'A' and seq[length - 1 -i] == 'U') or  (seq[i] == 'C' and seq[length - 1 -i] == 'G') or  (seq[i] == 'G' and seq[length - 1 -i] == 'C')  or (seq[i] == 'U' and seq[length - 1 -i] == 'A'):
            symmetric_points += 1

  symmetric = 0

  if(symmetric_points == len(seq)):
        symmetric = 1
 
   
  for  i in range (length-1) :
     oldDH = deltaH
     oldDS = deltaS
            
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
       
#     print pair, 'dG = ', (deltaH - oldDH)  - (273. + 37)*(deltaS - oldDS)/1000. , 'dH = ',deltaH - oldDH, 'dS = ',deltaS - oldDS    
  
    
    
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
  print deltaH - (273.15+37)*deltaS
  Tm = deltaH / (deltaS + GAS_CONSTANT * math.log(molarconcentration/divisor) );
  
  return Tm - 273.15 



if len(sys.argv) != 2:
	print "Usage: ./sek sek"
	sys.exit(1)
 
sek = sys.argv[1]





print get_TM(sek)

#print get_TM('CAGGTCG',20,1.0)
#print get_TM('TAGAAATGCAAG',20,0.5)

#sys.exit(1)

