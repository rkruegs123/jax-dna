#!/usr/bin/env python

import base
import readers
try:
    import numpy as np
except:
    import mynumpy as np
import os.path
import sys
import math

prune = 1


def get_lj(s,nucid):
    #this function returns vector pointing from base midpoint of nucid-th base pair to (nudic+1)th midpointt of bp
    box = s._box
    i1A = nucid
    i1B = len(s._strands[1]._nucleotides)-1-nucid
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    secondA = s._strands[0]._nucleotides[i1A+1]
    secondB = s._strands[1]._nucleotides[i1B-1]
    first_midpos = (firstA.get_pos_base() + firstB.get_pos_base()) / 2.0  
    second_midpos = (secondA.get_pos_base() + secondB.get_pos_base()) / 2.0  
    lj = second_midpos - first_midpos 		
    lj -= box * np.rint(lj / box)
    return lj


def get_end_j(s,nid1,nid2):
    i1A = nid1
    i2A = nid2
    i1B = len(s._strands[1]._nucleotides) - nid1 - 1
    i2B = len(s._strands[1]._nucleotides) - nid2 - 1 
    
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    lastA = s._strands[0]._nucleotides[i2A]
    lastB = s._strands[1]._nucleotides[i2B]
   
    first_midpos = (firstA.get_pos_base() + firstB.get_pos_base()) / 2.0  
    last_midpos = (lastA.get_pos_base() + lastB.get_pos_base()) / 2.0  
  
    box = s._box  
    r0N = last_midpos - first_midpos 
    r0N -= box * np.rint (r0N / box)
    return math.sqrt(np.dot(r0N,r0N)) / float((nid2 - nid1))


def get_inclination(s,nid1,hel_vector):
    i1A = nid1
    i1B = len(s._strands[1]._nucleotides) - nid1 - 1
    
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
   
    firstAv = (firstA.get_pos_base() -  firstA.get_pos_back())
    firstBv = (firstB.get_pos_base() -  firstB.get_pos_back())
  
    firstAv = firstAv / math.sqrt(np.dot(firstAv,firstAv))
    firstBv = firstBv / math.sqrt(np.dot(firstBv,firstBv))
    
    return math.acos(np.dot(hel_vector,firstAv)),math.acos(np.dot(hel_vector,firstBv))

def get_ss_inclination(s,strand_id,nid1):
    i1A = nid1
    i1B = nid1 + 1
    
    firstA = s._strands[strand_id]._nucleotides[i1A].get_pos_stack()
    firstB = s._strands[strand_id]._nucleotides[i1B].get_pos_stack() 	
    
    A3 = s._strands[strand_id]._nucleotides[i1A]._a3
    B3 = s._strands[strand_id]._nucleotides[i1B]._a3	
   
    ss_vector = firstB - firstA
    ss_vector = ss_vector / math.sqrt(np.dot(ss_vector,ss_vector))
    
    
    return math.acos(np.dot(ss_vector,A3)),math.acos(np.dot(ss_vector,B3))

def get_ss_inclination_to_axis(s,nid1,hel_axis):
    i1A = nid1
    i1B = len(s._strands[1]._nucleotides) - nid1 - 1
    firstA = s._strands[0]._nucleotides[i1A].get_pos_stack()
    firstB = s._strands[1]._nucleotides[i1B].get_pos_stack() 	
    
    A3 = s._strands[0]._nucleotides[i1A]._a3
    B3 = s._strands[1]._nucleotides[i1B]._a3	
   
    ss_vector = firstB - firstA
    ss_vector = ss_vector / math.sqrt(np.dot(ss_vector,ss_vector))
    
    return math.acos(np.dot(hel_axis,A3)),math.acos(np.dot(hel_axis,B3))
	
def BADget_rise_per_bp(s,nucid): #DO NOT USE!!!
    #this function returns the distance between midpoints of hb sites
    box = s._box
    i1A = nucid
    i1B = len(s._strands[1]._nucleotides)-1-nucid
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    secondA = s._strands[0]._nucleotides[i1A+1]
    secondB = s._strands[1]._nucleotides[i1B-1]
    first_midpos = (firstA.get_pos_base() + firstB.get_pos_base()) / 2.0  
    second_midpos = (secondA.get_pos_base() + secondB.get_pos_base()) / 2.0  
    lj = second_midpos - first_midpos 		
    lj -= box * np.rint(lj / box)
    return lj, math.sqrt(np.dot(lj,lj))

def get_rise_per_bp(s,nucid,hel_axis):
    #this function returns the distance between midpoints of hb sites
    box = s._box
    i1A = nucid
    i1B = len(s._strands[1]._nucleotides)-1-nucid
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    secondA = s._strands[0]._nucleotides[i1A+1]
    secondB = s._strands[1]._nucleotides[i1B-1]
    first_midpos = (firstA.get_pos_base() + firstB.get_pos_base()) / 2.0  
    second_midpos = (secondA.get_pos_base() + secondB.get_pos_base()) / 2.0  
    lj = second_midpos - first_midpos 	
    lj -= box * np.rint(lj / box)
    riseperbp = np.dot(lj, hel_axis)
   
    return riseperbp

def get_local_helical_axis(s,nid1):
    i1A = nid1
    i2A = nid1+1
    i1B = len(s._strands[1]._nucleotides) - nid1 - 1
    i2B = len(s._strands[1]._nucleotides) - nid1 - 1  - 1 
    
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    lastA = s._strands[0]._nucleotides[i2A]
    lastB = s._strands[1]._nucleotides[i2B]
   
    first_midpos = (firstA.get_pos_base() + firstB.get_pos_base()) / 2.0  
    last_midpos = (lastA.get_pos_base() + lastB.get_pos_base()) / 2.0  
  
    box = s._box  
    r0N = last_midpos - first_midpos 
    r0N -= box * np.rint (r0N / box)
    return r0N	/ math.sqrt(np.dot(r0N,r0N))


def get_helical_axis(s,nid1,nid2):
    i1A = nid1
    i2A = nid2
    i1B = len(s._strands[1]._nucleotides) - nid1 - 1
    i2B = len(s._strands[1]._nucleotides) - nid2 - 1 
    
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    lastA = s._strands[0]._nucleotides[i2A]
    lastB = s._strands[1]._nucleotides[i2B]
   
    first_midpos = (firstA.get_pos_base() + firstB.get_pos_base()) / 2.0  
    last_midpos = (lastA.get_pos_base() + lastB.get_pos_base()) / 2.0  
  
    box = s._box  
    r0N = last_midpos - first_midpos 
    r0N -= box * np.rint (r0N / box)
    return r0N	/ math.sqrt(np.dot(r0N,r0N))

def get_bb_dist(s,nucid): 
    #this function measures the distance between backbone sites of the opposite nucleotides
    box = s._box
    i1A = nucid
    i1B = len(s._strands[1]._nucleotides)-1-nucid
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    first_midpos = (firstA.get_pos_back() - firstB.get_pos_back())   
    return math.sqrt(np.dot(first_midpos,first_midpos)) 

def get_back_back_distance(s,nucid):
    box = s._box
    i1A = nucid
    i1B = len(s._strands[1]._nucleotides)-1-nucid
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    secondA = s._strands[0]._nucleotides[i1A+1]
    secondB = s._strands[1]._nucleotides[i1B-1]
    first_midpos = (firstA.get_pos_back() - secondA.get_pos_back())   
    second_midpos = (firstB.get_pos_back() - secondB.get_pos_back())
    return math.sqrt( np.dot(first_midpos,first_midpos)), math.sqrt(np.dot(second_midpos,second_midpos))


def get_turn_per_bp(s,nucid,hel_vector):
    #this function returns the distance between midpoints of hb sites
    box = s._box
    i1A = nucid
    i1B = len(s._strands[1]._nucleotides)-1-nucid
    firstA = s._strands[0]._nucleotides[i1A]
    firstB = s._strands[1]._nucleotides[i1B] 	
    secondA = s._strands[0]._nucleotides[i1A+1]
    secondB = s._strands[1]._nucleotides[i1B-1]
    first_midpos = (firstA.get_pos_back() - firstB.get_pos_back())   
    second_midpos = (secondA.get_pos_back() - secondB.get_pos_back())   
    first_midpos /= math.sqrt(np.dot(first_midpos,first_midpos))
    second_midpos /= math.sqrt(np.dot(second_midpos,second_midpos))
    first_midpos = first_midpos - np.dot(hel_vector,first_midpos) * hel_vector
    second_midpos = second_midpos - np.dot(hel_vector,second_midpos) * hel_vector
    first_midpos /= math.sqrt(np.dot(first_midpos,first_midpos))
    second_midpos /= math.sqrt(np.dot(second_midpos,second_midpos))
 
    #angle = math.acos(np.dot(first_midpos,hel_vector)) - math.acos(np.dot(second_midpos,hel_vector))
    angle = math.acos(np.dot(first_midpos,second_midpos))
    return angle 




if len(sys.argv) < 3:
    base.Logger.log("Usage is %s configuration topology [offset=4] [prune=1]" % sys.argv[0], base.Logger.CRITICAL)
    sys.exit()

l = readers.LorenzoReader(sys.argv[1], sys.argv[2])
s = l.get_system()

offset = 4
prune = 1
if len(sys.argv) >= 4:
    offset = int(sys.argv[3])

if len(sys.argv) >= 5:
    prune = int(sys.argv[4])
    


try:
    i1A = offset
    i1B = len(s._strands[1]._nucleotides) - offset - 1

    print  "#Nucleotides", i1A, i1B

except:
    print >> sys.stderr, "Supply nucleotides... Aborting"
    sys.exit (-1)

L2 = 0.
l0 = 0.
Ll0 = 0.
niter = 0
rises = []
read_confs = 1
end_rises = []
angles = []
helix_widths = []
incsA = []
incsB = []

bbAs = []
bbBs = []

while s:
    if(read_confs % prune != 0):
	read_confs += 1
    	s = l.get_system()
	continue      
    box = s._box  
    hel_axis = get_helical_axis(s,i1A,i1B)
    end_rises.append(get_end_j(s,i1A,i1B))

    for j in range (offset,len(s._strands[0]._nucleotides) - offset - 2):
	jA = j
        #rise = get_rise_per_bp(s,jA,hel_axis)
        rise = get_rise_per_bp(s,jA,get_local_helical_axis(s,jA))
        rises.append(rise)
	#angle = get_turn_per_bp(s,jA,hel_axis)
	angle = get_turn_per_bp(s,jA,get_local_helical_axis(s,jA))
	angles.append(angle)     
        width = get_bb_dist(s,jA)
        helix_widths.append(width)
        #inclinationA, inclinationB = get_inclination(s,jA,hel_axis)
        #inclinationA, inclinationB = get_inclination(s,jA,get_local_helical_axis(s,jA))
        inclinationA, inclinationB = get_ss_inclination(s,0,jA)
        inclinationA, inclinationB = get_ss_inclination_to_axis(s,jA,hel_axis)
	incsA.append(inclinationA)
	incsB.append(inclinationB)
	bbA,bbB = get_back_back_distance(s,jA)
	bbAs.append(bbA)
	bbBs.append(bbB)

    s = l.get_system()
    niter += 1
    read_confs += 1


print '# ',sys.argv[1]
print '# read configurations: ', niter
mangle = np.mean(angles)

print 'Rise (computed per bp): %f (+/- %f), Rise (from end-end-dist): %f, Angle (per bp): %f (ie %f)  (+/- %f), which is pitch %f  (+/- %f), width %f (+/- %f), inclination A: %f (+- %f), inclination B: %f (+- %f), back-back distances:A %f (+- %f), B: %f (+- %f), \n' % ( np.mean(rises) * 8.518 ,np.std(rises) * 8.518,np.mean(end_rises)*8.518,mangle,180. * mangle /  math.pi,np.std(angles),2.*math.pi / mangle , np.std(2*math.pi / np.array(angles)), np.mean(helix_widths)*8.518, np.std(helix_widths)*8.518 ,np.rad2deg(np.mean(incsA)),np.rad2deg(np.std(incsA)), np.rad2deg(np.mean(incsB)), np.rad2deg(np.std(incsB)), np.mean(bbAs),np.std(bbAs),np.mean(bbBs),np.std(bbBs))
