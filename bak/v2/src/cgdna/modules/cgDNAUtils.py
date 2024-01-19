import sys, random
import numpy as np
import pandas as pd
import scipy, scipy.io
from scipy.linalg import sqrtm
from scipy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import gridspec
import os, sys, time, copy
import scipy.io as sio
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import matplotlib.colors as mcolors

from cgdna.modules.E_transform import Etrans
import cgdna.modules.RotationUtils as Rot


plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 0.5
plt.rcParams['axes.facecolor'] = 'white'

# FIXME: don't hardcode
params_basedir = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/v2/src/cgdna/params/"

def constructSeqParms(sequence,ps_name):

    ps = scipy.io.loadmat(params_basedir + ps_name)

	#### Following loop take every input sequence and construct shape and stiff matrix ###
    s_seq = seq_edit(sequence)
    nbp = len(s_seq.strip())
    N = 24*nbp-18

	#### Initialise the sigma vector ###
    s = np.zeros((N,1))

    #### Error report if sequence provided is less than 2 bp ####

    if nbp <= 3:
        print("sequence length must be greater than or equal to 2")
        sys.exit()

    elif nbp > 3 :

        data,row,col = {},{},{}

        ### 5' end ####
        tmp_ind = np.nonzero(ps['stiff_end5'][s_seq[0:2]][0][0][0:36,0:36])
        row[0],col[0] = tmp_ind[0][:],tmp_ind[1][:]
        data[0] = ps['stiff_end5'][s_seq[0:2]][0][0][row[0],col[0]]

        s[0:36] = ps['sigma_end5'][s_seq[0:2]][0][0][0:36]
        #### interior blocks  ###
        for i in range(2,nbp-1):
            tmp_ind = np.nonzero(ps['stiff_int'][s_seq[i-1:i+1]][0][0][0:42, 0:42])
            data[i-1] = ps['stiff_int'][s_seq[i-1:i+1]][0][0][tmp_ind[0][:], tmp_ind[1][:]]

            di = 24*(i-2)+18
            row[i-1] = tmp_ind[0][:]+np.ones((1,np.size(tmp_ind[0][:])))*di
            col[i-1] = tmp_ind[1][:]+np.ones((1,np.size(tmp_ind[1][:])))*di

            s[di:di+42] = np.add(s[di:di+42],ps['sigma_int'][s_seq[i-1:i+1]][0][0][0:42])

		#### 3' end ####
        tmp_ind = np.nonzero(ps['stiff_end3'][s_seq[nbp-2:nbp]][0][0][0:36, 0:36])
        data[nbp-1] = ps['stiff_end3'][s_seq[nbp-2:nbp]][0][0][tmp_ind[0][:], tmp_ind[1][:]]

        di = 24*(nbp-3)+18
        row[nbp-1] = tmp_ind[0][:]+np.ones((1,np.size(tmp_ind[0][:])))*di
        col[nbp-1] = tmp_ind[1][:]+np.ones((1,np.size(tmp_ind[1][:])))*di
        s[N-36:N] = s[N-36:N] + ps['sigma_end3'][s_seq[nbp-2:nbp]][0][0][0:36]

        tmp = list(row.values())
        row = np.concatenate(tmp,axis=None)

        tmp = list(col.values())
        col = np.concatenate(tmp,axis=None)

        tmp = list(data.values())
        data = np.concatenate(tmp,axis=None)


    #### Create the sparse Stiffness matrix from data,row_ind,col_ind  ###
        stiff =  csc_matrix((data, (row,col)), shape =(N,N))

	#### Groudstate calculation ####
        ground_state = spsolve(stiff, s)

    return ground_state,stiff


def BasepairFrames(s):
    (intra_r,intra_t,pho_W_r,pho_W_t,inter_r,inter_t,pho_C_r,pho_C_t) = DecomposeCoord(s)
    del intra_r,intra_t,pho_W_r,pho_W_t,pho_C_r,pho_C_t
    R,r = {},{}
    R[0] = np.identity(3)   # absolute coordinates for the first matrix
    r[0] =np.zeros(1,3) # absolute coordinates of the first basepair
    nbp = (len(s)+18)/24
    for i in range(1,nbp-1):
        ru = Rot.Cay(inter_r[:,i],5)
        R[i] = np.matmul(R[i-1],ru)
        H = Rot.midFrame(R[i-1],ru)
        r[i] = np.add(r[i-1],np.matmul(H,np.transpose(inter_t[i])))

    return R,r


def frames(s):
    (intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t) = DecomposeCoord(s) #relative coordinates of the oligomer
    G = np.identity(3)   # absolute coordinates for the first matrix
    q =np.transpose(np.zeros(3)) # absolute coordinates of the first basepair
    R,r,Rc,rc,Rw,rw,Rpw,rpw,Rpc,rpc={},{},{},{},{},{},{},{},{},{}

    nbp = int((len(s)+18)/24)
    Rpw[0],rpw[0],Rpc[nbp-1],rpc[nbp-1] = [],[],[],[]
    for i in range(nbp):
        R[i]=G #merging everything at origin
        r[i]=q #merging everything at origin
        L = Rot.Cay(intra_r[:,i],5) #basepair
        Gw = np.matmul(G,(np.transpose(intra_t[:,i]))) #basepair
        Rc[i] = np.matmul(G,np.transpose(np.real(sqrtm(L))))   #complementray strand
        rc[i] = np.subtract(q,np.multiply(0.5,Gw))   #complementray strand
        Rw[i] = np.matmul(Rc[i],L) 	#original strand
        rw[i] = np.add(rc[i],Gw)        #original strand
        if i < nbp-1:
            ru= Rot.Cay(inter_r[:,i],5)
            H = Rot.midFrame(G,ru)
 ################## Next base pair #################
            G = np.matmul(G,ru)
            q = np.add(q,np.matmul(H,np.transpose(inter_t[:,i])))

################### Compute the phosphate groups' frames ##################
    for i in range(nbp-1):
        Rpw[i+1]=np.matmul(Rw[i+1],Rot.Cay(pho_W_r[:,i],5))
        rpw[i+1]=np.add(rw[i+1],np.matmul(Rw[i+1],pho_W_t[:,i]))

        Pmat = np.identity(3)
        Pmat[1,1]=-1
        Pmat[2,2]=-1
        RcP=np.matmul(Rc[i],Pmat)
        Rpc[i]=np.matmul(RcP,Rot.Cay(pho_C_r[:,i],5))
        rpc[i]=np.add(rc[i],np.matmul(RcP,pho_C_t[:,i]))

    return R, r, Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc


def DecomposeCoord(s):

    nbp= int((len(s)+18)/24)
    intra_r=np.zeros((nbp,3))
    intra_t=np.zeros((nbp,3))
    inter_r=np.zeros((nbp-1,3))
    inter_t=np.zeros((nbp-1,3))
    pho_W_r=np.zeros((nbp-1,3))
    pho_W_t=np.zeros((nbp-1,3))
    pho_C_r=np.zeros((nbp-1,3))
    pho_C_t=np.zeros((nbp-1,3))

    l = 4*nbp-3;
    s = np.transpose(np.reshape(s, (l,6)))

    intra = s[:,::4]
    pho_C = s[:,1::4]
    inter = s[:,2::4]
    pho_W = s[:,3::4]


    intra_r,intra_t = intra[0:3,:],intra[3:6,:]
    pho_C_r,pho_C_t = pho_C[0:3,:],pho_C[3:6,:]
    inter_r,inter_t = inter[0:3,:],inter[3:6,:]
    pho_W_r,pho_W_t = pho_W[0:3,:],pho_W[3:6,:]


    return intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t

def TanTan(s,R0):
    nbp = int((len(s)+18)/24)
    l = 4*nbp-3;
    s = np.transpose(np.reshape(s, (l,6)))
    u = s[0:3,2::4]

    TanTan=np.zeros(nbp-1)

    q=Rot.Rot2Quat(R0)
    qinv = Rot.QuatInv(q)

    for i in range(nbp-1):
        q = Rot.QuatMult(q,Rot.Cay2Quat(u[:,i],5))
        qtantan = Rot.QuatMult(qinv,q)
        TanTan[i] = Rot.Quat2Rot_33(qtantan)

    return TanTan

def finder(seq):
	istart = []
	end = {}
	start = []
	for i, c in enumerate(seq):
		if c == '[':
			istart.append(i)
			start.append(i)
		if c == ']':
			try:
				end[istart.pop()] = i
			except IndexError:
				print('Too many closing parentheses')
	if istart:  # check if stack is empty afterwards
		print('Too many opening parentheses')
	return end, start

def mult(seq):
	i =seq.rfind('_')
	if seq[i+1].isdigit():
		a = seq[i+1]
		if seq[i+2].isdigit():
			a = a + seq[i+2]
			if seq[i+3].isdigit():
				a = a + seq[i+3]
				if seq[i+4].isdigit():
					a = a + seq[i+4]
					if seq[i+5].isdigit():
						a = a + seq[i+5]
	return a

def seq_edit(seq):
	s = seq.upper()
	while s.rfind('_')>0:
		if s[s.rfind('_')-1].isdigit():
			print("Please write the input sequence correctly. Two or more _ can't be put consequently. You can use the brackets. i.e. A_2_2 can be written as [A_2]_2")
			exit()
		if s[s.rfind('_')-1] != ']':
			a = int(mult(s))
			s = s[:s.rfind('_')-1]+ s[s.rfind('_')-1]*a +  s[s.rfind('_')+1+len(str((a))):]
		if s[s.rfind('_')-1] == ']':
			end,start = finder(s)
			ka=(2,len(start))
			h=np.zeros(ka)
			for i in range(len(start)):
				h[0][i] = start[i]
				h[1][i] = end[start[i]]
			ss=  int(max(h[1]))
			ee=  int(h[0][np.argmax(h[1])])
			a = int(mult(s))
			s =  s[0:ee] + s[ee+1:ss]*a + s[ss+2+len(str((a))):]
	return s

def comp(base):
    complement = ''
    for i in base:
        if i=="C":
            com = "G"
        elif i=="G":
            com = "C"
        elif i=="A":
            com = "T"
        elif i=="T":
            com = "A"
        elif i=="M":
            com = "N"
        elif i=="N":
            com = "M"
        complement = com + complement
    return complement

def swap(x, y):
  return (copy.copy(y), copy.copy(x))


#####################################
########### Groove width codes
#####################################


def GrooveWidths(w):     ####without cubic spline fitting, an approximation
    _,_,_,_,_,_,_,rpw,_,rpc = frames(w)
    nbp = int((len(w)+18)/24)
    pos = int(nbp/2)
    dist= pos
    rpw = rpw[pos]
    res_up = np.zeros(dist-1)
    res_down = np.zeros(dist-1)
    for i in range(1,dist):   ## in this loop last up pos dist is not computed while the opposite pos dist
        rpc_up = rpc[pos+i-1]
        rpc_down = rpc[pos-i-1]
        res_up[i-1] = norm(np.subtract(rpc_up,rpw))
        res_down[i-1]= norm(np.subtract(rpc_down,rpw))
    maxG = res_up.min()
    minG = res_down.min()
    return minG, maxG

def find_tangent(rw,nbp):
    Tp={}
    for i in range(nbp):
        try:
            tmp = rw[i+1]-rw[i-1]
            Tp[i] = tmp/norm(tmp)
        except:
            Tp[i] = []
    return Tp

def cubic_spline(p1,d,d_norm,T1,T2,r):
    # See lavery grooves article for details
    f = d_norm*(r - 2*r**2 + r**3)*T1 + (3*r**2-2*r**3)*d + d_norm*(-r**2+r**3)*T2 + p1
    return f


def GrooveWidths_CS(w):
    _,_,_,_,_,_,_,rpw,_,rpc = frames(w)
    nbp = (len(w)+18)//24
    pos = nbp//2
    rpw_mid,rpc_mid = rpw[pos],rpc[pos]

    rpc_fine,rpw_fine = [],[]
    res_c, res_w = [],[]

    Tpc = find_tangent(rpc,nbp)
    for i in range(1,nbp-3):
        rpc_fine.append(rpc[i])
        if i ==pos:
            mid_posc = len(rpc_fine)-1
        d = rpc[i+1] - rpc[i]
        d_norm = norm(d)
        for r in np.linspace(0.1,0.9,9):
            rpc_fine.append(cubic_spline(rpc[i],d,d_norm,Tpc[i],Tpc[i+1],r))
    for res in rpc_fine:
        res_c.append(norm(np.subtract(res,rpw_mid)))

    Watson=False
    if Watson == True:
        Tpw = find_tangent(rpw,nbp)
        for i in range(2,nbp-2):
            rpw_fine.append(rpw[i])
            if i == pos:
                mid_posw = len(rpw_fine)-1
            d = rpw[i+1] - rpw[i]
            d_norm = norm(d)
            for r in np.linspace(0.1,0.9,9):
                rpw_fine.append(cubic_spline(rpw[i],d,d_norm,Tpw[i],Tpw[i+1],r))
        for res in rpw_fine:
            res_w.append(norm(np.subtract(res,rpc_mid)))

#    plt.plot(np.arange(len(res_c)),res_c)
    # plt.plot(np.arange(len(res_w)),res_w)
    minG = min(res_c[:mid_posc])
    maxG = min(res_c[mid_posc:])
    return minG, maxG
