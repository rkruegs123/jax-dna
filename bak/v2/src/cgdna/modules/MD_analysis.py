import sys, random, re
import itertools, string
from cgDNAclass import cgDNA
from Init_MD import init_MD_data
import numpy as np
import pandas as pd
import RotationUtils as Rot
import scipy, scipy.io
from scipy.linalg import sqrtm
from scipy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv as sinv
from numpy.linalg import inv as ninv
from matplotlib import gridspec
import os, sys, time, copy
import scipy.io as sio
path = os.getcwd()
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import matplotlib.colors as mcolors
import seaborn as sns#; sns.set()
import tqdm
import time
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D


from cgdna.modules.E_transform import Etrans
from cgdna.modules.cgDNAUtils import *



c1 = ['blue','red','k']
c3 = ['royalblue','indianred','dimgray']
plt.rcParams['font.family'] = ['sans-serif']
ps_list = {'DNA':'ps2_cgf', 'MDNA':'ps_mdna', 'HDNA':'ps_hdna'}
ang = "Å"
ang = "\u212B"

c1 = ['red','blue','green']
c2 = ['darkred','navy','olivedrab']
c3 = ['indianred','royalblue','limegreen']
c4 = ['maroon','dodgerblue','limegreen']

#############################################################################
#############################################################################
DNA_sym = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,False,False,True,True,True,False]
PDNA_sym = DNA_sym[0:16]
RNA_sym = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,False,False,True,True,True]
HYB_sym = [False]*24
MDNA_sym=[True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,False,False,False]
MDNA_map = [0,1,2,3,4,5,8,9,12,13,14,16,6,7,10,11,15,17,18,19,20,21]
cgDNA_name = ["Buckle","Propeller","Opening","Shear","Stretch","Stagger",
              "WRot1","WRot2","WRot3","WTra1","WTra2","WTra3",
              "Tilt","Roll","Twist","Shift","Slide","Rise",
              "CRot1","CRot2","CRot3","CTra1","CTra2","CTra3",
              "Buckle","Propeller","Opening","Shear","Stretch","Stagger"]
cgDNA_name_web = ["Buckle","Propeller","Opening","Shear","Stretch","Stagger",
              "\eta_1","\eta_2","\eta_3","w_1","w_2","w_3",
              "Tilt","Roll","Twist","Shift","Slide","Rise",
              "\eta_1","\eta_2","\eta_3","w_1","w_2","w_3",
              "Buckle","Propeller","Opening","Shear","Stretch","Stagger"]
cgDNA_units = ([' (rad/5)']*3+[' ('+ang+')']*3)*5

dimer_16_RY = ['YR','RR','YY','RY']
dimer_16 = ['TA','CG','CA','TG','AA','AG','GA','GG', 'TT','CT','TC','CC', 'GC','AT','AC','GT']

dimer_17 = ['TA','CG','CA','TG','AA','AG','GA','GG', 'TT','CT','TC','CC', 'GC','AT','AC','GT','Avg']
dimer_10 = ['TA','CG','CA','AA','AG','GA','GG','GC','AT','AC']
mon = ['A','T','G','C']
color_16 = ['darkgreen', 'orange', 'slateblue', 'sienna', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','darkgoldenrod','skyblue','navy','lime','tan','gold']
color_16_RY = ['darkgreen', 'darkgreen', 'darkgreen', 'darkgreen', 'magenta', 'magenta', 'magenta', 'magenta', 'gray', 'gray','gray','gray','brown','brown','brown','brown','k']
color_4_RY = ['darkgreen','darkgreen','m','m','k']
def ttl_256():
    ttl = []
    for d in dimer_16:
        for i in mon:
            for j in mon:
                ttl.append(i+d+j)
    return ttl

def trimer_64():
    ttl = []
    for d in mon:
        for i in mon:
            for j in mon:
                ttl.append(i+d+j)
    return ttl

def epi_combination(what):
    mon_front = mon+['N']
    mon_back = mon+['M']
    mono6 = mon + ['M','N']
    if what =='monomer':
        comb = ['M','N']
    elif what == 'dimer':
        comb = ['MN','MG','CN','NM','AM','TM','CM','GM','NA','NT','NG','NC']
    elif what == 'trimer':
        comb = [ j + i for i in ['MN','MG'] for j in mon_front] + [ i + j for i in ['MN','CN'] for j in mon_back]
    elif what == 'tetramer': ## tetramer and tetramer2 gives the same result ..just different method
        comb =        [i + k for i in ['MN','CN'] for k in dimer_16 + ['MN','MG','CN','AM','TM','CM','GM'] ]
        comb = comb + [i + j + k for i in mon_front for j in ['MN','CN','MG'] for k in mon_back]
        comb = comb + [i + k for i in dimer_16 + ['MG','NA','NT','NG','NC'] for k in ['MN','MG'] ]
    elif what == 'tetramer2':
        d = epi_combination('dimer')
        comb =        [i + j + k for i in mon_front for j in d[0:3]   for k in mon_back]     #75 + 4 + 30 + 12 + 42
        comb = comb + [i + j + k for i in ['C','M'] for j in d[3:4]   for k in ['G','N']]
        comb = comb + [i + j + k for i in mon_front for j in d[4:7]   for k in ['G','N']]
        comb = comb + [i + j + k for i in mono6     for j in d[7:8]   for k in ['G','N']]
        comb = comb + [i + j + k for i in ['C','M'] for j in d[8:11]  for k in mon_back]
        comb = comb + [i + j + k for i in ['C','M'] for j in d[11:12] for k in mono6 ]
    elif what == 'tetramer_present':  ## list of tetramers present in the training library
        tmp_comb = epi_combination('tetramer2')  ## 82 are present
        present = [ 1,3,4,6,11,15,18,19,20,23,25,28,29,30,31,32,34,35,39,40,42,47,48,49,52,53,57,59,60,61,62,63,64,65,67,73,75,76,77,
                         78,  80,  81,  82,  84,  85,  86,  89,  90,  92,  93,  94,  95,
                         96,  98, 100, 102, 109, 110, 112, 116, 117, 122, 124, 126, 131,
                         134, 137, 138, 139, 141, 142, 143, 144, 146, 147, 148, 149, 152,156, 157, 158, 159]
        comb = [tmp_comb[i] for i in present]
    return comb


#############################################################################
#############################################################################
# Utilitiy codes
#############################################################################
#############################################################################
# this code set ideal limits for x and y
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 0.5
plt.rcParams['axes.facecolor'] = 'white'

def all_Nmers(N):
    return ["".join(item) for item in itertools.product("ATCG", repeat=N)]

def all_YR(N):
    return ["".join(item) for item in itertools.product("RY", repeat=N)]

def Met_all_Nmers(N):
    list_all = ["".join(item) for item in itertools.product(['A','T','C','G','MN','MG','CN'], repeat=N)]
    final_list = []
    for seq in list_all:
        if 'M' in seq:
            final_list.append(seq)
    return final_list

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def to_YR(seq):
    seq_YR = ''
    for s in seq:
        if s in "AG":
            seq_YR += "R"
        elif s in "CT":
            seq_YR += "Y"
        else:
            print("Unrecongnized base type --->", s)
            sys.exit()
    return seq_YR

def set_lim(x,y,ax,frac=0.09):
    xlim_min = min(x)
    xlim_max = max(x)
    ylim_min = min(y)
    ylim_max = max(y)
    xr = xlim_max - xlim_min
    yr = ylim_max - ylim_min
    xf = (xlim_min - 0.05*xr, xlim_max + 0.05*xr)
    yf = (ylim_min - 0.05*yr, ylim_max + 0.05*yr)
    ax.set_xlim(xf)
    ax.set_ylim(yf)
    lim_r = (xlim_min - (frac+0.01)*xr,ylim_max + frac*yr)
    return lim_r

# return randon sequence of length N
def random_seq(length):
    Base = ['A','T','C','G']
    seq  = ''
    for i in range(length):
        seq = seq + random.choice(Base)
    return seq

def stencil_42(nbp,typ='cg+'):
    x1 = np.zeros(nbp-1,dtype=int)
    x2 = np.zeros(nbp-1,dtype=int)
    if typ=='cg+':
        for i in range(nbp-2):
            x1[i+1] = 18 + 24*i
            x2[i] = 36 + 24*(i)
        x2[nbp-2] = 24*nbp -18
    ############################
    if typ=='cg':
        for i in range(nbp-2):
            x1[i+1] = 12 + 12*i
            x2[i]   = 18 + 12*(i)
        x2[nbp-2] = 12*nbp -6
    ############################
    if typ=='inter':
        for i in range(nbp-2):
            x1[i+1] = 6 + 6*i
            x2[i]   = 6 + 6*(i)
        x2[nbp-2] = 6*nbp -6


    return x1,x2


def replacenth(string, sub, wanted, where):
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return newString

def dimer_percent(seq,dim):
    loc = [m.start() for m in re.finditer(dim, seq)]
    return len(loc)

def mdna_random_seq(how_many):
    for i in range(how_many):
        seq = random_seq(216)
        loc = [m.start() for m in re.finditer('CG', seq)]
        print(seq,dimer_percent(seq,'CG'),dimer_percent(seq,'MN'),dimer_percent(seq,'MG') )
        for u in range(1,len(loc),1):
            for q in range(4):
                tmp_seq = seq
                for l in random.choices(loc,k=u):
                    hell = random.choice(['MN','MG'])
                    tmp_seq = replacenth(tmp_seq, 'CG', hell, l)
                print(tmp_seq,dimer_percent(tmp_seq,'CG'),dimer_percent(tmp_seq,'MN'),dimer_percent(tmp_seq,'MG') )
        for u in range(1,len(loc)):
            for s in ['MN','MG']:
                tmp_seq = seq
                for l in random.choices(loc,k=u):
                    tmp_seq = replacenth(tmp_seq, 'CG', s, l)
                print(tmp_seq,dimer_percent(tmp_seq,'CG'),dimer_percent(tmp_seq,'MN'),dimer_percent(tmp_seq,'MG'), "---")


###----------------------------------------------------------------------------------------
## Mahalanobis distance --------------------------------------
###----------------------------------------------------------------------------------------
def Mahal(mu1,mu2,A):
    try:
        A = A.to_numpy(dtype='float')
        mu1, mu2 = mu1.to_numpy(dtype='float')[:,np.newaxis],mu2.to_numpy(dtype='float')[:,np.newaxis]
    except:
        None
    ### Note A must be stiffness matrix
    dis = scipy.spatial.distance.mahalanobis(mu1,mu2,A)/len(mu1)
    return dis
###----------------------------------------------------------------------------------------
## Symmertic Mahalanobis distance --------------------------------------
###----------------------------------------------------------------------------------------
def Mahal_sym(mu1,mu2,A1,A2):
    try:
        A1 = A1.to_numpy(dtype='float')
        A2 = A2.to_numpy(dtype='float')
        mu1, mu2 = mu1.to_numpy(dtype='float')[:,np.newaxis],mu2.to_numpy(dtype='float')[:,np.newaxis]
    except:
        None
    # Note A1,A2 are stiffness matrix
    dis1 = Mahal(mu1,mu2,A1)
    dis2 = Mahal(mu1,mu2,A2)
    dis = (dis1+dis2)/2
    return dis

def numerical_KLD_sym(p,q,x):
    f1 = scipy.special.rel_entr(p, q)
    f2 = scipy.special.rel_entr(q, p)
    d = 0.5*(np.sum(f1) + np.sum(f2))
    d = d*(x[1]-x[0])
    return d

def kl_mvn(m0, K0, m1, K1):
    N = m0.shape[0]
    S0 = np.linalg.inv(K0)
    diff = m1 - m0
    # kl is made of three terms
    tr_term   = np.trace(np.matmul(S0, K1))
    det_term  = np.linalg.slogdet(K0)[1] - np.linalg.slogdet(K1)[1]
    quad_term = np.matmul(np.matmul(diff.T, K1), diff)
    # per dof
    kl = .5 * (tr_term + det_term + quad_term - N)
    return kl/N

def kl_mvn_sym(m0, K0, m1, K1):
    k1 = kl_mvn(m0, K0, m1, K1)
    k2 = kl_mvn(m1, K1, m0, K0)
    kl = 0.5*(k1+k2)
    return kl

def palin_err_mu_norm(data):
    wc = copy.deepcopy(data.shape[0])
    l = np.size(wc)
    nbp = copy.deepcopy(data.nbp[0])
    err = wc - np.matmul(Etrans(nbp),wc)
    err = np.linalg.norm(err)/l
    return err

def palin_err_K_norm(data):

    if hasattr(data, 's1b'):
        wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp

    wc_inside = np.zeros((24*nbp-18,24*nbp-18))
    x1,x2 = stencil_42(nbp)
    for i,j in zip(x1,x2):
        wc_inside[i:j,i:j] = wc[i:j,i:j]

    err = wc_inside - np.matmul(np.matmul(Etrans(nbp),wc_inside),Etrans(nbp))
    l = 36*36*2 + 42*42*(nbp-3) - 18*18*(nbp-2)
    err = np.linalg.norm(err)/l
    return err

def palin_err_Mahal_sym(data):
    wc = copy.deepcopy(data.shape[0])
    Kc = copy.deepcopy(data.s1b[0])
    nbp = copy.deepcopy(data.nbp[0])
    E = Etrans(nbp)
    wk = np.matmul(E,wc)
    Kk = np.matmul(np.matmul(E,Kc),E)
    err = Mahal_sym(wk,wc,Kk,Kc)
    return err

def palin_err_KL_sym(data):
    wc = copy.deepcopy(data.shape[0])
    Kc = copy.deepcopy(data.s1b[0])
    nbp = copy.deepcopy(data.nbp[0])
    E = Etrans(nbp)
    wk = np.matmul(E,wc)
    Kk = np.matmul(np.matmul(E,Kc),E)
    err = kl_mvn_sym(wc, Kc, wk, Kk)
    return err

def recons_err_mu_norm(data,ps,sym):
    seq = copy.deepcopy(data.seq[0])
    seq = seq.replace('U','T')
    if sym==False:
        wm = copy.deepcopy(data.shape[0])
    elif seq==comp(seq):
        wm = copy.deepcopy(data.shape_sym[0])
    else:
        wm = copy.deepcopy(data.shape[0])

    res = cgDNA(seq,ps)
    wr = res.ground_state
    l = np.size(wm)
    err = wm-wr
    err = np.linalg.norm(err)/l
    return err

def recons_err_K_norm(data,ps,sym):

    if hasattr(data, 's1b'):
        seq = copy.deepcopy(data.seq[0])
        nbp = data.nbp[0]
        if sym==False:
            wc = copy.deepcopy(data.s1b[0])
        elif seq==comp(seq):
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        seq = seq.replace('U','T')

    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
        seq = copy.deepcopy(data.seq)
        seq = seq.replace('U','T')

    wc_inside = np.zeros((24*nbp-18,24*nbp-18))
    x1,x2 = stencil_42(nbp)
    for i,j in zip(x1,x2):
        wc_inside[i:j,i:j] = wc[i:j,i:j]

    res = cgDNA(seq,ps)
    Kr = res.stiff.todense()

    err = wc_inside - Kr
    l = 36*36*2 + 42*42*(nbp-3) - 18*18*(nbp-2)
    err = np.linalg.norm(err)/l
    return err

def recons_err_Mahal_sym(data,ps,sym):
    seq = copy.deepcopy(data.seq[0])
    seq = seq.replace('U','T')
    if sym==False:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])
    elif seq==comp(seq):
        wm = copy.deepcopy(data.shape_sym[0])
        Km = copy.deepcopy(data.s1b_sym[0])
    else:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])

    res = cgDNA(seq,ps)
    wr, Kr = res.ground_state,res.stiff.todense()
    err = Mahal_sym(wm,wr,Km,Kr)
    return err


def recons_err_KL_sym(data,ps,sym):
    seq = copy.deepcopy(data.seq[0])
    seq = seq.replace('U','T')
    if sym==False:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])
    elif seq==comp(seq):
        wm = copy.deepcopy(data.shape_sym[0])
        Km = copy.deepcopy(data.s1b_sym[0])
    else:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])

    res = cgDNA(seq,ps)
    wr, Kr = res.ground_state,res.stiff.todense()
    err = kl_mvn_sym(wm, Km, wr, Kr)
    return err

def locality_err_KL_sym(data,ps,sym):
    seq = copy.deepcopy(data.seq[0])
    seq = seq.replace('U','T')
    if sym==False:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.stiff_me[0])
    else:
        wm = copy.deepcopy(data.shape_sym[0])
        Km = copy.deepcopy(data.stiff_me_sym[0])

    res = cgDNA(seq,ps)
    wr, Kr = res.ground_state,res.stiff.todense()
    err_KL = kl_mvn_sym(wm, Km, wr, Kr)
    err_M = Mahal_sym(wm,wr,Km,Kr)
    return err_M,err_KL



def difference_mu(data1,data2,sym):

    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
    elif hasattr(data1, 'stiff'):
        wm = copy.deepcopy(data1.ground_state)

    if hasattr(data2, 's1b'):
        if sym==True:
            wr = copy.deepcopy(data2.shape_sym[0])
        else:
            wr = copy.deepcopy(data2.shape[0])
    elif hasattr(data2, 'stiff'):
        wr = copy.deepcopy(data2.ground_state)

    l = np.size(wm)
    err = wm-wr
    err = np.linalg.norm(err)/l
    return err

def difference_K(data1,data2,sym):

    if hasattr(data1, 's1b'):
        if sym==True:
            Km = copy.deepcopy(data1.s1b_sym[0])
        else:
            Km = copy.deepcopy(data1.s1b[0])
        nbp = data1.nbp[0]
    elif hasattr(data1, 'stiff'):
        Km = copy.deepcopy(data1.stiff.todense())
        nbp = data1.nbp

    if hasattr(data2, 's1b'):
        if sym==True:
            Kr = copy.deepcopy(data2.s1b_sym[0])
        else:
            Kr = copy.deepcopy(data2.s1b[0])
    elif hasattr(data2, 'stiff'):
        Kr = copy.deepcopy(data2.stiff.todense())

    Km_inside = np.zeros((24*nbp-18,24*nbp-18))
    Kr_inside = np.zeros((24*nbp-18,24*nbp-18))
    x1,x2 = stencil_42(nbp)
    for i,j in zip(x1,x2):
        Km_inside[i:j,i:j] = Km[i:j,i:j]
        Kr_inside[i:j,i:j] = Kr[i:j,i:j]

    err = Km_inside - Kr_inside
    l = 36*36*2 + 42*42*(nbp-3) - 18*18*(nbp-2)
    err = np.linalg.norm(err)/l
    return err


def difference_Mahal_sym(data1,data2,sym):

    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
            Km = copy.deepcopy(data1.s1b_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
            Km = copy.deepcopy(data1.s1b[0])
    elif hasattr(data1, 'stiff'):
        wm = copy.deepcopy(data1.ground_state)
        Km = copy.deepcopy(data1.stiff.todense())

    if hasattr(data2, 's1b'):
        if sym==True:
            wr = copy.deepcopy(data2.shape_sym[0])
            Kr = copy.deepcopy(data2.s1b_sym[0])
        else:
            wr = copy.deepcopy(data2.shape[0])
            Kr = copy.deepcopy(data2.s1b[0])
    elif hasattr(data2, 'stiff'):
        wr = copy.deepcopy(data2.ground_state)
        Kr = copy.deepcopy(data2.stiff.todense())

    err = Mahal_sym(wm,wr,Km,Kr)
    return err


def Truncation_KL_sym(data1,sym):
    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
            Km = copy.deepcopy(data1.s1b_sym[0])
            Kr = copy.deepcopy(data1.stiff_me_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
            Km = copy.deepcopy(data1.s1b[0])
            Kr = copy.deepcopy(data1.stiff_me[0])
    err = kl_mvn_sym(wm, Km, wm, Kr)
    return err


def Truncation_KL_sym_cg(data1,sym):
    if hasattr(data1, 's1b_cg'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym_cg[0])
            Km = copy.deepcopy(data1.s1b_sym_cg[0])
            Kr = copy.deepcopy(data1.stiff_me_sym_cg[0])
        else:
            wm = copy.deepcopy(data1.shape_cg[0])
            Km = copy.deepcopy(data1.s1b_cg[0])
            Kr = copy.deepcopy(data1.stiff_me_cg[0])
    err = kl_mvn_sym(wm, Km, wm, Kr)
    return err

def Truncation_KL_sym_inter(data1,sym):
    if hasattr(data1, 's1b_inter'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym_inter[0])
            Km = copy.deepcopy(data1.s1b_sym_inter[0])
            Kr = copy.deepcopy(data1.stiff_me_sym_inter[0])
        else:
            wm = copy.deepcopy(data1.shape_inter[0])
            Km = copy.deepcopy(data1.s1b_inter[0])
            Kr = copy.deepcopy(data1.stiff_me_inter[0])
    err = kl_mvn_sym(wm, Km, wm, Kr)
    return err



def difference_KL_sym(data1,data2,sym):
    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
            Km = copy.deepcopy(data1.s1b_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
            Km = copy.deepcopy(data1.s1b[0])
    elif hasattr(data1, 'stiff'):
        wm = copy.deepcopy(data1.ground_state)
        Km = copy.deepcopy(data1.stiff.todense())

    if hasattr(data2, 's1b'):
        if sym==True:
            wr = copy.deepcopy(data2.shape_sym[0])
            Kr = copy.deepcopy(data2.s1b_sym[0])
        else:
            wr = copy.deepcopy(data2.shape[0])
            Kr = copy.deepcopy(data2.s1b[0])
    elif hasattr(data2, 'stiff'):
        wr = copy.deepcopy(data2.ground_state)
        Kr = copy.deepcopy(data2.stiff.todense())

    err = kl_mvn_sym(wm, Km, wr, Kr)
    return err





#############################################################################
#############################################################################
# CODE BELOW Are main code
#############################################################################
#############################################################################
def compare_shape(w_list, seq_list, save_name,lss,color,type_of_var='cg+',multi_xlabel=False,bottom=0.06):
    plt.close()
    RotDeg=0
    if RotDeg == 1:
        y2=' ($^\circ$)'
    elif RotDeg==0:
        y2=' rad/5'
    else:
        print('Wrong value for the Keyword RotDeg.')
    lww = 0.5
    legend_size = 4
    fs=6
    fs2 = 6
    if type_of_var=='cg+':
        print("plotting only cgDNA variables -  ---- ")
        fig = plt.figure(constrained_layout=False)
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.09, right=0.98,top=0.98,bottom=bottom)
        hs,he = 46,54
        vdiff,vshift =15,4

        ax00 = plt.subplot(gs1[0:50,       0:hs])
        ax01 = plt.subplot(gs1[0:50,       he:100])

        ax10 = plt.subplot(gs1[50 + vdiff - vshift:140,    0:hs])
        ax11 = plt.subplot(gs1[50 + vdiff - vshift:140,    he:100])

        ax20 = plt.subplot(gs1[140 + vdiff:260,    0:hs])
        ax21 = plt.subplot(gs1[140 + vdiff:260,    he:100])

        ax30 = plt.subplot(gs1[260 + vdiff:380,    0:hs])
        ax31 = plt.subplot(gs1[260 + vdiff:380,    he:100])

        ax40 = plt.subplot(gs1[380 + vdiff:500,    0:hs])
        ax41 = plt.subplot(gs1[380 + vdiff:500,    he:100])

        ax_list = [ax00,ax10, ax01,ax11, ax20,ax21, ax30,ax31, ax40,ax41]
        ax_list_str = ['ax00','ax10','ax01','ax11','ax20','ax21','ax30','ax31','ax40','ax41']
        ax_sub_list = [ax00,ax10,ax01,ax11,ax20,ax21,ax30,ax31]
        ax_sub_list2 = [ax00,ax10,ax01,ax11]
        ax_min = dict.fromkeys(ax_list_str, 0)
        ax_max = dict.fromkeys(ax_list_str, 0)
        ax_range = dict.fromkeys(ax_list_str, 0)

        count = 0
        for w,seq in zip(w_list,seq_list):
            c=color[count]
            if count > 0:
                coord = ['_no_legend']*12
                coordp = ['_no_legend']*12
            else:
                coord = ['Buckle','Propeller','Opening','Shear','Stretch','Stagger','Tilt','Roll','Twist','Shift','Slide','Rise']
                coordp = ['WRot1','WRot2','WRot3','WTra1','WTra2','WTra3','CRot1','CRot2','CRot3','CTra1','CTra2','CTra3']
            nbp = len(seq)
            ind = np.arange(nbp)
            inds = np.arange(0.5,nbp-1,1)
            intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t = DecomposeCoord(w)
            ## think about broken axis

            for i in range(3):

                ax20.plot(inds,pho_C_r[i::3].T,color=c[i],lw=lww,label=coordp[0+i],ls = lss[count])
                ax21.plot(inds,pho_W_r[i::3].T,color=c[i],lw=lww,label=coordp[6+i],ls = lss[count])

                if i == 1:
                    ax00.plot(inds,pho_C_t[i::3].T,color=c[i],lw=lww,label=coordp[3+i],ls = lss[count])
                    ax01.plot(inds,pho_W_t[i::3].T,color=c[i],lw=lww,label=coordp[9+i],ls = lss[count])
                else:
                    ax10.plot(inds,pho_C_t[i::3].T,color=c[i],lw=lww,label=coordp[3+i],ls = lss[count])
                    ax11.plot(inds,pho_W_t[i::3].T,color=c[i],lw=lww,label=coordp[9+i],ls = lss[count])

                ax40.plot(ind,intra_r[i::3].T,color=c[i],lw=lww,label=coord[0+i],ls = lss[count])
                ax30.plot(ind,intra_t[i::3].T,color=c[i],lw=lww,label=coord[3+i],ls = lss[count])

                ax41.plot(inds,inter_r[i::3].T,color=c[i],lw=lww,label=coord[6+i],ls = lss[count])
                ax31.plot(inds,inter_t[i::3].T,color=c[i],lw=lww,label=coord[9+i],ls = lss[count])

                for axl in ax_list:
                    axl.tick_params(axis='both', which='major', labelsize=fs2)
                    axl.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.1))
                for axl in ax_sub_list2:
                    axl.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.25))

            ax41.set_xticks(ind)
            ax41.set_xticklabels(list(seq),fontsize=fs2)
            ax40.set_xticks(ind)
            ax40.set_xticklabels(list(seq),fontsize=fs2)

            if multi_xlabel == True:
                secax1 = ax40.secondary_xaxis('bottom')
                secax1.set_xticks(ind)
                secax1.set_xticklabels(list(seq_list[0]),fontsize=fs2)
                secax1.tick_params(axis='both', pad=10,length=3,width=0.5,direction= 'inout',rotation=0)
                secax2 = ax41.secondary_xaxis('bottom')
                secax2.set_xticks(ind)
                secax2.set_xticklabels(list(seq_list[0]),fontsize=fs2)
                secax2.tick_params(axis='both', pad=10,length=3,width=0.5,direction= 'inout',rotation=0)
                multi_xlabel = False

            ax10.set_ylabel('Å',fontsize=fs2)
            ax20.set_ylabel(y2,fontsize=fs2)
            ax30.set_ylabel('Å',fontsize=fs2)
            ax40.set_ylabel(y2,fontsize=fs2)

            d = .025  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them

            kwargs = dict(transform=ax00.transAxes, color='k', clip_on=False,lw=0.5)
            ax00.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax00.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax10.transAxes)  # switch to the bottom axes
            ax10.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax10.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            kwargs.update(transform=ax01.transAxes)  # switch to the bottom axes
            ax01.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax01.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax11.transAxes)  # switch to the bottom axes
            ax11.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax11.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            ax00.spines['bottom'].set_visible(False)
            ax10.spines['top'].set_visible(False)
            ax01.spines['bottom'].set_visible(False)
            ax11.spines['top'].set_visible(False)
            for axl in ax_sub_list:
                axl.set_xticks([])
            for axl in ax_list:
                axl.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

################function below set the and ylim
            variable = [pho_C_t[1],pho_C_t[[0,2]],pho_W_t[1],pho_W_t[[0,2]],pho_C_r,pho_W_r,intra_t,inter_t,intra_r,inter_r]
            for axl,var_tmp in zip(ax_list_str,variable):
                tmp1,tmp2 = np.amin(var_tmp),np.amax(var_tmp)
                if count == 0:
                    ax_min[axl] = tmp1
                    ax_max[axl] = tmp2
                else:
                    if ax_min[axl] > tmp1:
                        ax_min[axl] = tmp1
                    if ax_max[axl] < tmp2:
                        ax_max[axl] = tmp2
################function above set the and ylim------------------
            count=count+1

        for axl,axl_str in zip(ax_list,ax_list_str):
            ax_range[axl_str] = 0.15*(ax_max[axl_str] - ax_min[axl_str])
            axl.set_ylim(ax_min[axl_str]-ax_range[axl_str],ax_max[axl_str]+ax_range[axl_str])

######################------------type_of_var---------------###################
    if type_of_var=='cg':
        print("plotting only cgDNA variables -  ---- ")
        fig = plt.figure(constrained_layout=False)
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.06)
        hs,he = 47,53
        vdiff,vshift =15,4


        ax30 = plt.subplot(gs1[0 :250,    0:hs])
        ax31 = plt.subplot(gs1[0 :250,    he:100])

        ax40 = plt.subplot(gs1[250 + vdiff:500,    0:hs])
        ax41 = plt.subplot(gs1[250 + vdiff:500,    he:100])

        ax_list = [ax30,ax31, ax40,ax41]
        ax_list_str = ['ax30','ax31','ax40','ax41']
        ax_min = dict.fromkeys(ax_list_str, 0)
        ax_max = dict.fromkeys(ax_list_str, 0)
        ax_range = dict.fromkeys(ax_list_str, 0)

        count = 0
        for w,seq in zip(w_list,seq_list):
            if count > 0:
                coord = ['_no_legend']*12
                coordp = ['_no_legend']*12
            else:
                coord = ['buckle','propeller','opening','shear','stretch','stagger','tilt','roll','twist','shift','slide','rise']
                coordp = ['WRot1','WRot2','WRot3','WTra1','Wtra2','WTra3','CRot1','CRot2','CRot3','CTra1','CTra2','CTra3']
            nbp = len(seq)
            ind = np.arange(nbp)
            inds = np.arange(0.5,nbp-1,1)
            intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t = DecomposeCoord(w)
            ## think about broken axis

            for i in range(3):

                ax40.plot(ind,intra_r[i::3].T,color=c[i],lw=lww,label=coord[0+i],ls = lss[count])
                ax30.plot(ind,intra_t[i::3].T,color=c[i],lw=lww,label=coord[3+i],ls = lss[count])

                ax41.plot(inds,inter_r[i::3].T,color=c[i],lw=lww,label=coord[6+i],ls = lss[count])
                ax31.plot(inds,inter_t[i::3].T,color=c[i],lw=lww,label=coord[9+i],ls = lss[count])

                for axl in ax_list:
                    axl.tick_params(axis='both', which='major', labelsize=fs2)
                    axl.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.05))

            ax41.set_xticks(ind)
            ax41.set_xticklabels(list(seq),fontsize=fs2)
            ax40.set_xticks(ind)
            ax40.set_xticklabels(list(seq),fontsize=fs2)

            ax30.set_ylabel('Å',fontsize=fs2)
            ax40.set_ylabel(y2,fontsize=fs2)

            for axl in [ax30,ax31]:
                axl.set_xticks([])
            for axl in ax_list:
                axl.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

################function below set the and ylim
            variable = [intra_t,inter_t,intra_r,inter_r]
            for axl,var_tmp in zip(ax_list_str,variable):
                tmp1,tmp2 = np.amin(var_tmp),np.amax(var_tmp)
                if count == 0:
                    ax_min[axl] = tmp1
                    ax_max[axl] = tmp2
                else:
                    if ax_min[axl] > tmp1:
                        ax_min[axl] = tmp1
                    if ax_max[axl] < tmp2:
                        ax_max[axl] = tmp2
################function above set the and ylim------------------
            count=count+1

        for axl,axl_str in zip(ax_list,ax_list_str):
            ax_range[axl_str] = 0.15*(ax_max[axl_str] - ax_min[axl_str])
            axl.set_ylim(ax_min[axl_str]-ax_range[axl_str],ax_max[axl_str]+ax_range[axl_str])

    for ax in [ax00,ax10, ax01,ax11, ax20,ax21, ax30,ax31, ax40,ax41]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    plt.show()
    fig.savefig("./Plots/" + save_name +".pdf",dpi=600)
    return fig,ax_list

##################################---------------------------------------------
# persistence length
##################################---------------------------------------------
colo = [c1[0] , c3[0] , c1[1] , c3[1] , c1[2] , c3[2]]
colo = c1 + c3
def read_persist_data(path):
    all_files = np.concatenate((np.arange(1,101),np.arange(501,554)))
    all_files = np.arange(1,101)
    print("only reading random files")
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()

def read_persist_data_rand(path):
    all_files = np.arange(1,101)
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()

def read_persist_data_sub(path):
    all_files = np.arange(1,101)
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()

def read_persist_data_tandem(path):
    all_files = np.arange(501,554)
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()


def read_persist_data_NA(path):
    all_files = np.concatenate((np.arange(1,101),np.arange(501,554)))
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def identify_file_index(ind):
    ind = ind+1
    file = ind//20000 + 1
    line = ind%20000
    if file > 100:
        file = file + 400
    return file, line

def read_persis_file_index(path,ind):
    print(path)
    file, line = identify_file_index(ind)
    print(file,line)
    df = pd.read_csv(path+'/results_'+str(file)+'.txt', engine='python',sep=" ",header=None)
    print(df)
    print(df.loc[line-1])

def update_prop(handle, orig):
    handle.update_from(orig)
    handle.set_marker("")

def plot_persistence_length(names):
    fig = plt.subplots(figsize=(5,5.7))   ###uuuuuuuuuuu
    gs1 = gridspec.GridSpec(600, 600)
    gs1.update(left=0.14, right=0.97,top=0.98,bottom=0.06)
    d = 15
    ax1 = plt.subplot(gs1[000+0:300-d, 000:600])
    ax2 = plt.subplot(gs1[300+d:600-d, 000:600])
#    ax3 = plt.subplot(gs1[600+d:900-0, 000:600])
    count=0
    data = {}
    fs = 10
    label1 =  ["$\ell_{p}^{DNA}$", "$\ell_{p}^{RNA}$", "$\ell_{p}^{DRH}$"]
    label2 =  ["$\ell_{d}^{DNA}$", "$\ell_{d}^{RNA}$", "$\ell_{d}^{DRH}$"]
    label11 =  ["$\ell_{p}^{RNA} - \ell_{p}^{DNA}$", "$\ell_{p}^{DRH} - \ell_{p}^{DNA}$"]
    label22 =  ["$\ell_{d}^{RNA} - \ell_{d}^{DNA}$", "$\ell_{d}^{DRH} - \ell_{d}^{DNA}$"]
    label3 =  ["$\ell_{d}^{DNA}-\ell_{p}^{DNA}$", "$\ell_{d}^{RNA}-\ell_{p}^{RNA}$", "$\ell_{d}^{DRH}-\ell_{p}^{DRH}$"]
    for enum,n in enumerate(names):
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data[n] = read_persist_data_rand(path)
        print("Note which data is Reading the ----------")
        print(names[enum],"-lp-","max, min --",max(data[n][0]),min(data[n][0]),"range --",max(data[n][0]) - min(data[n][0]),"mean --", np.mean(data[n][0]))
        print(names[enum],"-ld-","max, min --",max(data[n][1]),min(data[n][1]),"range --",max(data[n][1]) - min(data[n][1]),"mean --", np.mean(data[n][1]))
        ax1.hist(data[n][0] ,histtype = 'step',bins = 1000,color=colo[enum] ,lw=1, label = label1[enum], density=True )
        ax1.hist(data[n][1] ,histtype = 'step',bins = 500,color=colo[enum+3],lw=1, label = label2[enum], density=True )
        u0 = data[n][0]- data['DNA_BSTJ_CGF'][0]
        u1 = data[n][1]- data['DNA_BSTJ_CGF'][1]
        v0 =-data[n][0]+data[n][1]
        if count>0:
            ax2.hist(u0.dropna() ,histtype = 'step',bins = 1000,color=colo[enum] ,lw=1, label = label11[enum-1], density=True)
            ax2.hist(u1.dropna() ,histtype = 'step',bins = 500,color=colo[enum+3],lw=1, label = label22[enum-1], density=True)
            None
#        ax3.hist(v0,histtype = 'step',bins = 500,color=colo[enum],lw=1, label = label3[enum] , density=True)
        count = count+2
    loc = [2,2,1]
    for enum,ax in enumerate([ax1,ax2]):
        ax.legend(fontsize=fs,ncol=2,loc=loc[enum])
        ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
        ax.set_xlabel("length in base-pairs",fontsize=fs)
    ax2.set_ylabel("Normalized histogram",fontsize=fs)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.savefig('./Plots/persistence_length'+".pdf",dpi=600)
    plt.show()
    return data


def plot_persistence_length_tandem_vs_random(names):
    fig,ax = plt.subplots(figsize=(6,3))
    count=0
    data1, data2 = {},{}
    fs = 8
    label1 =  ["$\ell_{p}^{DNA}$", "$\ell_{p}^{RNA}$", "$\ell_{p}^{DRH}$"]
    label2 =  ["$\ell_{d}^{DNA}$", "$\ell_{d}^{RNA}$", "$\ell_{d}^{DRH}$"]
    for n in names:
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data1[n] = read_persist_data_rand(path)
        data2[n] = read_persist_data_tandem(path)
        plt.hist(data1[n][0] ,histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = n[:3] +'_' + 'app_rand', ls=(0, (5, 10))  )
        plt.hist(data1[n][1] ,histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = n[:3] +'_' + 'dyn_rand', ls=(0, (5, 10))  )
        plt.hist(data2[n][0] ,histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = n[:3] +'_' + 'app_tand'  )
        plt.hist(data2[n][1] ,histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = n[:3] +'_' + 'dyn_tand'  )
        print(min(data1[n][0]), min(data1[n][1]), min(data2[n][0]), min(data2[n][1]))
        print(max(data1[n][0]), max(data1[n][1]), max(data2[n][0]), max(data2[n][1]))
        count = count+2
    ax.legend(fontsize=fs,ncol=2)
    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_xlabel("Persistence length in number of bp",fontsize=fs)
    plt.savefig('./Plots/'+'_persistence_length_rand_vs_tand'+".pdf",dpi=600)
    plt.show()
    return None


def read_special_persist_data(path):

    dimer_files= ['AA','TT','GC','CG', 'TC','CT','TG','GT', 'CC','GG', 'AC','CA','AG','GA', 'AT','TA' ]
    all_files = [501,518,549,539, 521,532,525,546, 535,553, 507,528, 511, 542, 514, 504]
    lines_index = [1,9526,18671,8956, 19431,9146,9336,8766,19051,8576,19811, 19241, 9716,18861, 19621,  9906 ]
    li  = np.zeros((len(all_files),2))
    for enum,i in enumerate(all_files):
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li[enum]=df.loc[lines_index[enum]-1]
    return li, dimer_files

def plot_special_persistence_length(names):
    fig = plt.subplots(figsize=(5,2.6))   ###uuuuuuuuuuu
    gs1 = gridspec.GridSpec(300, 300)
    gs1.update(left=0.1, right=0.91,top=0.91,bottom=0.105)
    d = 15
    ax = plt.subplot(gs1[000+0:300, 000:300])
    count=0
    data = {}
    fs = 10
    label1 =  ["$\ell_{p}^{DNA}$", "$\ell_{p}^{RNA}$", "$\ell_{p}^{DRH}$"]
    label2 =  ["$\ell_{d}^{DNA}$", "$\ell_{d}^{RNA}$", "$\ell_{d}^{DRH}$"]
    subset = [0,1,2,4, 6,8,9,10, 12,14]
    for enum,n in enumerate(names):
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data[n], dimer_files = read_special_persist_data(path)
        print(n,data[n][:,0],"lp")
        print(n,data[n][:,1],"ld")
        ax.scatter(np.arange(10),data[n][:,0][subset] ,color=colo[count],s=10, label = label1[enum])
        ax.scatter(np.arange(10),data[n][:,1][subset] ,color=colo[count],marker='_',label = label2[enum])
        count = count+1

    ax.legend(fontsize=fs,ncol=2,loc=1, bbox_to_anchor=(1.125, 1.125))
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([dimer_files[k] for k in subset], minor=False,rotation=90)

    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_ylabel("Persistence length in bp",fontsize=fs)
    plt.savefig('./Plots/persistence_length_sp'+".pdf",dpi=600)
    plt.show()
    return data


##################################---------------------------------------------
# plot stencil in stiffness matrix
##################################---------------------------------------------

def fit_stencil_in_matrix(data,sym,save_name):
    if hasattr(data, 's1b'):
        if sym==True:
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
    wc = np.nan_to_num(wc)

    x1,x2 = stencil_42(nbp)

    x6 = np.arange(0,24*nbp-12,6)
    fig,axr = plt.subplots(1)
    sns.heatmap(wc,ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.5,0.05
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='limegreen', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='limegreen', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='limegreen', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='limegreen', lw=lwl)
    # for j in x6:
    #     axr.axvline(x=j,color='black', lw=lws)
    #     axr.axhline(y=j,color='black', lw=lws)

    ind = np.arange(0,24*nbp-18,12)
    axr.set_xticks(ind)
    axr.set_yticks(ind)
    axr.set_xticklabels(ind)
    axr.set_yticklabels(ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=6, pad=3,length=3,width=0.5,direction= 'inout')
    axr.set_xlim(0,279)
    axr.set_ylim(0,279)
    axr.grid(False)
    for uu in [0,278]:
        axr.axhline(y=uu, color='k',linewidth=0.5)
        axr.axvline(x=uu, color='k',linewidth=0.5)
    axr.invert_yaxis()
    plt.savefig('./Plots/'+save_name+'.pdf',dpi=600)

def fit_stencil_in_submatrix(data,sym,save_name):
    if hasattr(data, 's1b'):
        if sym==True:
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
    wc = np.nan_to_num(wc)

    x1,x2 = stencil_42(nbp)

    x6 = np.arange(0,24*nbp-12,6)
    fig,axr = plt.subplots(1)
    sns.heatmap(wc,ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 2,0.1
    for j in x6:
        axr.axvline(x=j,color='black', lw=lws)
        axr.axhline(y=j,color='black', lw=lws)
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='limegreen', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='limegreen', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='limegreen', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='limegreen', lw=lwl)

    ind = np.arange(0,24*nbp-18,12)
    axr.set_xticks(ind)
    axr.set_yticks(ind)
    axr.set_xticklabels(ind)
    axr.set_yticklabels(ind)
    xx1,xx2 = 240-12,318+12
    axr.set_xlim(xx1,xx2)
    axr.set_ylim(xx1,xx2)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=6, pad=3,length=3,width=0.5,direction= 'inout')
    axr.invert_yaxis()
    for uu in [xx1,xx2]:
        axr.axhline(y=uu, color='k',linewidth=0.8)
        axr.axvline(x=uu, color='k',linewidth=0.8)
#    axr.invert_yaxis()
    axr.grid(False)
    plt.savefig('./Plots/'+save_name+'.pdf',dpi=600)



##################################---------------------------------------------
# plot palindromic error
##################################---------------------------------------------
def check_palin(seq):
    sym = False
    if comp(seq) == seq:
        sym = True
#    print(sym)
    return sym

def plot_below1(mat,fs,center,clabel,save_name):
    fig,ax = plt.subplots()
    sns.heatmap(mat,ax=ax,cmap="RdBu_r",cbar=1,annot=True,fmt='.4f',center=center,annot_kws={"size":fs-2},cbar_kws={"shrink": .7,"pad":0.018}).invert_yaxis()
    s = np.shape(mat)
    ax.tick_params(axis='both', length = 3, width=0.5, direction='inout',pad=3)
    ax.figure.axes[-1].set_ylabel(clabel,size=fs)
    ax.figure.axes[-1].tick_params(labelsize=fs-2)
    ax.set_xticks(np.arange(s[1])+.5)
    ax.set_yticks(np.arange(s[0])+.5)
    ax.set_xticklabels(np.arange(s[1])+1,fontsize=fs)
    ax.set_yticklabels(np.arange(s[0])+1,fontsize=fs)
    ax.set_ylabel('Index of sequences in the training library',size=fs)
    ax.set_xlabel('Simulation length in $\mu$s',size=fs)
    plt.tight_layout()
    fig.savefig('./Plots/'+save_name+'.pdf',dpi=600)
    plt.show()
    plt.close()


def plot_heatmap_palin_err(data_name,name):
    sims = 10
    mat1 = np.zeros((16,sims))
    mat2 = np.zeros((16,sims))
    mat3 = np.zeros((16,sims))
    mat4 = np.zeros((16,sims))
    for file_index in np.arange(1,sims+1,1):
        data_path_tmp = data_name.replace('XX',str(file_index))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for seq in np.arange(0,16,1):
            mat1[seq,file_index-1] = palin_err_mu_norm(data_tmp.choose_seq([seq]))
            mat2[seq,file_index-1] = palin_err_K_norm(data_tmp.choose_seq([seq]))
            mat3[seq,file_index-1] = palin_err_Mahal_sym(data_tmp.choose_seq([seq]))
            mat4[seq,file_index-1] = palin_err_KL_sym(data_tmp.choose_seq([seq]))
    plot_below1(mat1,fs=8,center=0.0035,clabel="Palindromic error in groundstate, |$\mu$-E$\mu$|", save_name=name+'_palin_err_mu_norm')
    plot_below1(mat2,fs=8,center=0.005, clabel="Palindromic error in stiffness, |K-EKE|",          save_name=name+'_palin_err_K_norm')
    plot_below1(mat3,fs=8,center=0.0023,clabel="Palindromic error, Symmetric Mahalanobis distance",save_name=name+'_palin_err_Mahal_sym')
    plot_below1(mat4,fs=8,center=0.04,  clabel="Palindromic error, Symmetric KL divergence",       save_name=name+'_palin_err_KL_sym')
    print(np.mean(mat3,axis=0))
    print(np.mean(mat4,axis=0))
    plt.show()
    plt.close()
    fs2 = 14
    fig,ax = plt.subplots(2,sharex=True)
    ind = np.arange(10)
    ax[0].plot(ind,np.mean(mat3,axis=0),color='red')
    ax[1].plot(ind,np.mean(mat4,axis=0))
    ax[0].scatter(ind,np.mean(mat3,axis=0),color='red')
    ax[1].scatter(ind,np.mean(mat4,axis=0))
    ax[1].set_xticks(ind)
    ax[1].set_xticklabels(ind)
    ax[1].set_yticks([0.01,0.02,0.03,0.04,0.05])
    ax[1].set_yticklabels([0.01,0.02,0.03,0.04,0.05])
    ax[1].set_xlabel("Simulation length ($\mu$s)",fontsize=10)
    ax[0].set_ylabel('$\epsilon_{M}^{palin}$',fontsize=fs2,rotation=90)
    ax[1].set_ylabel('$\epsilon_{KL}^{palin}$',fontsize=fs2,rotation=90)
    plt.tight_layout()
    fig.savefig('./Plots/palin_err_plot_'+name+'.pdf',dpi=600)
    plt.show()
    plt.close()
    return None

def plot_heatmap_palin_err_epi(data_name,name):
    mat1 = np.zeros((12,10))
    mat2 = np.zeros((12,10))
    mat3 = np.zeros((12,10))
    mat4 = np.zeros((12,10))
    for file_index in np.arange(1,11,1):
        data_path_tmp = data_name.replace('XX',str(file_index))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for enum,seq in enumerate(MDNA_map[0:12]):
            mat1[enum,file_index-1] = palin_err_mu_norm(data_tmp.choose_seq([seq]))
            mat2[enum,file_index-1] = palin_err_K_norm(data_tmp.choose_seq([seq]))
            mat3[enum,file_index-1] = palin_err_Mahal_sym(data_tmp.choose_seq([seq]))
            mat4[enum,file_index-1] = palin_err_KL_sym(data_tmp.choose_seq([seq]))
    plot_below1(mat1,fs=8,center=0.0035,clabel="Palindromic error in groundstate, |$\mu$-E$\mu$|", save_name=name+'_palin_err_mu_norm')
    plot_below1(mat2,fs=8,center=0.005, clabel="Palindromic error in stiffness, |K-EKE|",          save_name=name+'_palin_err_K_norm')
    plot_below1(mat3,fs=8,center=0.0023,clabel="Palindromic error, Symmetric Mahalanobis distance",save_name=name+'_palin_err_Mahal_sym')
    plot_below1(mat4,fs=8,center=0.04,  clabel="Palindromic error, Symmetric KL divergence",       save_name=name+'_palin_err_KL_sym')
    print(np.mean(mat3,axis=0))
    print(np.mean(mat4,axis=0))
    return None

##################################---------------------------------------------
# Set scale for training/test error, and palindromic error
##################################---------------------------------------------
def set_scale_for_error(path1,sym,label='DNA'):
    print('set_scale_for_error')
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 95)
    gs1.update(left=0.03, right=0.95,top=0.98,bottom=0.08)
    ax1 = plt.subplot(gs1[0:280,         0:95])
    ax2 = plt.subplot(gs1[320:600,         0:95])

    DNA = init_MD_data().load_data(path1)
    nseq = 16
    nent =  nseq*(nseq-1)/2
    mat1 = np.zeros((nseq,nseq))
    mat2 = np.zeros((nseq,nseq))
    mat3 = np.zeros((nseq,nseq))
    mat4 = np.zeros((nseq,nseq))
    for i in range(nseq):
        for j in range(nseq):
            if i >j:
                mat1[i,j]=difference_mu(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat2[i,j]=difference_K(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat3[i,j]=difference_Mahal_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat4[i,j]=difference_KL_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                None
#    mat3 = np.eye(16)
#    mat4 = np.eye(16)
    clabel = ["SM","SKL"]
    fs=6
    sns.heatmap(mat3+mat3.T,ax=ax1,cmap="RdBu_r",cbar=1,square=1,cbar_kws={"shrink": .7,"pad":0.018},linewidths=0.1, linecolor='k').invert_yaxis()
    ax1.figure.axes[-1].set_ylabel("SM",size=fs)
    ax1.figure.axes[-1].tick_params(labelsize=fs,width=0.5, direction='inout',pad=3)

    sns.heatmap(mat4+mat4.T,ax=ax2,cmap="RdBu_r",cbar=1,square=1,cbar_kws={"shrink": .7,"pad":0.018},linewidths=0.1, linecolor='k').invert_yaxis()
    ax2.figure.axes[-1].set_ylabel("SKL",size=fs)
    ax2.figure.axes[-1].tick_params(labelsize=fs,width=0.5, direction='inout',pad=3)
    for enum,ax in enumerate([ax1,ax2]):
        ax.tick_params(axis='both', length = 3, width=0.5, direction='inout',pad=3)
        ax.set_xticks(np.arange(16)+.5)
        ax.set_yticks(np.arange(16)+.5)
        ax.set_xticklabels(np.arange(16)+1,fontsize=fs)
        ax.set_yticklabels(np.arange(16)+1,fontsize=fs,rotation=0)
        ax.set_ylabel('Index of sequences in the training library',size=fs)
    ax2.set_xlabel('Index of sequences in the training library',size=fs)

    plt.savefig('./Plots/scale_fig_'+label+'.pdf',dpi=600)

#    sys.exit()
    scale1, min1 = np.sum(mat1)/nent, np.min(mat1[np.nonzero(mat1)])
    scale2, min2 = np.sum(mat2)/nent, np.min(mat2[np.nonzero(mat2)])
    scale3, min3 = np.sum(mat3)/nent, np.min(mat3[np.nonzero(mat3)])
    scale4, min4 = np.sum(mat4)/nent, np.min(mat4[np.nonzero(mat4)])
    print("\\textbf{Scale1}"," & ",str(scale1)[0:6]," & ",str(scale2)[0:6]," & ",str(scale3)[0:6]," & ",str(scale4)[0:6]," \\\\ ")
    print("\\textbf{Scale2}"," & ",str(min1)[0:6]," & ",str(min2)[0:6]," & ",str(min3)[0:6]," & ",str(min4)[0:6]," \\\\ ")

def set_scale_for_error_epi(DNA,sym):
    nseq = 12
    nent =  nseq*(nseq-1)/2
    mat1 = np.zeros((nseq,nseq))
    mat2 = np.zeros((nseq,nseq))
    mat3 = np.zeros((nseq,nseq))
    mat4 = np.zeros((nseq,nseq))
    which = [0,1,2,3,4,5,8,9,12,13,14,16]
    for enum1,i in enumerate(which):
        for enum2,j in enumerate(which):
            if enum1 > enum2:
                mat1[enum1,enum2]=difference_mu(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat2[enum1,enum2]=difference_K(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat3[enum1,enum2]=difference_Mahal_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat4[enum1,enum2]=difference_KL_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
    scale1, min1 = np.sum(mat1)/nent, np.min(mat1[np.nonzero(mat1)])
    scale2, min2 = np.sum(mat2)/nent, np.min(mat2[np.nonzero(mat2)])
    scale3, min3 = np.sum(mat3)/nent, np.min(mat3[np.nonzero(mat3)])
    scale4, min4 = np.sum(mat4)/nent, np.min(mat4[np.nonzero(mat4)])
    print("\\textbf{Scale1}"," & ",str(scale1)[0:6]," & ",str(scale2)[0:6]," & ",str(scale3)[0:6]," & ",str(scale4)[0:6]," \\\\ ")
    print("\\textbf{Scale2}"," & ",str(min1)[0:6]," & ",str(min2)[0:6]," & ",str(min3)[0:6]," & ",str(min4)[0:6]," \\\\ ")

##################################---------------------------------------------
# plot training/test error, plot_heatmap_training_err
##################################---------------------------------------------
def plot_below2(mat,fs,center,clabel,save_name):
    fig,ax = plt.subplots(edgecolor='k',facecolor='w')
    sns.heatmap(mat,ax=ax,cmap="RdBu_r",cbar=1,annot=True,fmt='.4f',center=center,annot_kws={"size":fs-2},cbar_kws={"shrink": .7,"pad":0.018},linewidths=0.1, linecolor='k').invert_yaxis()
    s = np.shape(mat)
    ax.tick_params(axis='both', length = 3, width=0.5, direction='inout',pad=3)
    ax.figure.axes[-1].set_ylabel(clabel,size=fs)
    ax.figure.axes[-1].tick_params(labelsize=fs-2)
    ax.set_xticks(np.arange(s[1])+.5)
    ax.set_yticks(np.arange(s[0])+.5)
    ax.set_xticklabels(np.arange(s[1])+1,fontsize=fs)
    ax.set_yticklabels(np.arange(s[0])+1,fontsize=fs)
    ax.set_ylabel('Index of sequences in the training library',size=fs)
    ax.set_xlabel('Error definition',size=fs)
    plt.tight_layout()
    fig.savefig('./Plots/'+save_name+'.png',dpi=600)
    plt.show()
    plt.close()

def plot_heatmap_training_err(data_tmp,ps,sym):
    nseq=len(data_tmp.nbp)
    for i in range(nseq):
        print(i+1,"& \\ttfamily",data_tmp.seq[i],"\\\\")
    mat1 = np.zeros((nseq,4))
    for seq in np.arange(0,16,1):
        mat1[seq,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(seq+1, "&",str(mat1[seq,0])[0:6], "&",str(mat1[seq,1])[0:6], "&",str(mat1[seq,2])[0:6], "&",str(mat1[seq,3])[0:6], "\\\\")
    print("\hline")
    print("\\textbf{Average training error}", "&",str(np.mean(mat1[0:16,0]))[0:6], "&",str(np.mean(mat1[0:16,1]))[0:6], "&",str(np.mean(mat1[0:16,2]))[0:6], "&",str(np.mean(mat1[0:16,3]))[0:6], "\\\\")
    print("\hline")
    for seq in np.arange(16,nseq,1):
        mat1[seq,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(seq+1, "&",str(mat1[seq,0])[0:6], "&",str(mat1[seq,1])[0:6], "&",str(mat1[seq,2])[0:6], "&",str(mat1[seq,3])[0:6], "\\\\")
    print("\hline")
    print("\\textbf{Average test error}", "&",str(np.mean(mat1[16:nseq,0]))[0:6], "&",str(np.mean(mat1[16:nseq,1]))[0:6], "&",str(np.mean(mat1[16:nseq,2]))[0:6], "&",str(np.mean(mat1[16:nseq,3]))[0:6])


    return mat1[:,2],mat1[:,3]

def unmodify(seq):
    seq = seq.replace('M','C')
    seq = seq.replace('N','G')
    seq = seq.replace('H','C')
    seq = seq.replace('K','G')
    seq = seq.replace('U','T')
    return seq

def plot_heatmap_training_err_epi(data_tmp,data3,ps):
    nseq=len(data_tmp.nbp)
    for enum,seq in enumerate([0,1,2,3,4,5,8,9,12,13,14,16]):
        print(enum+1,"& \\ttfamily",data_tmp.seq[seq],"\\\\")
    for enum,seq in enumerate([6,7,10,11,15,17]):
        print(enum+13,"& \\ttfamily",data_tmp.seq[seq],"\\\\")
    for enum,seq in enumerate([19,20,21]):
        print(seq,"& \\ttfamily",data3.seq[enum],"\\\\")
    mat1 = np.zeros((nseq,4))
    count = 0
    for enum,seq in enumerate([0,1,2,3,4,5,8,9,12,13,14,16]):
        sym = check_palin(data_tmp.seq[seq])
        mat1[count,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(count+1, "&",str(mat1[count,2])[0:6], "&",str(mat1[count,3])[0:6], "\\\\")
        count = count+1
    print("\hline")
    print("\\textbf{Average training error}", "&",str(np.mean(mat1[0:12,2]))[0:6], "&",str(np.mean(mat1[0:12,3]))[0:6], "\\\\")
    print("\hline")
    for enum,seq in enumerate([6,7,10,11,15,17]):
        sym = check_palin(data_tmp.seq[seq])
        mat1[count,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(count+1,  "&",str(mat1[count,2])[0:6], "&",str(mat1[count,3])[0:6], "\\\\")
        count = count+1
    for enum,seq in enumerate([0,1,2]):
        sym = check_palin(data3.seq[seq])
        mat1[count,0] = recons_err_mu_norm(data3.choose_seq([seq]),ps,sym)
        mat1[count,1] = recons_err_K_norm(data3.choose_seq([seq]),ps,sym)
        mat1[count,2] = recons_err_Mahal_sym(data3.choose_seq([seq]),ps,sym)
        mat1[count,3] = recons_err_KL_sym(data3.choose_seq([seq]),ps,sym)
        print(count+1,  "&",str(mat1[count,2])[0:6], "&",str(mat1[count,3])[0:6], "\\\\")
        count = count+1

    print("\hline")
    print("\\textbf{Average test error}", "&",str(np.mean(mat1[12:nseq-1,2]))[0:6], "&",str(np.mean(mat1[12:nseq-1,3]))[0:6])

    return None


##################################---------------------------------------------
# plot_gs_vs_MD_shape
##################################---------------------------------------------
def Truncation_error(data,NA_type_list):
    mat1= {}
    count = 0
    for data_tmp, NA_type in zip(data,NA_type_list):
        if NA_type == 'DNA':
            NA_sym = DNA_sym
            pam = np.arange(16)
            print("Note only computing for 16 seq")
        elif NA_type == 'PDNA':
            NA_sym = PDNA_sym
            pam = np.arange(16)
        elif NA_type == 'RNA':
            NA_sym = RNA_sym
            pam = np.arange(24)
            pam = np.arange(16)
        elif NA_type == 'HYB':
            NA_sym = HYB_sym
            pam = np.arange(24)
            pam = np.arange(16)
        elif NA_type == 'MDNA':
            NA_sym = MDNA_sym
            pam = MDNA_map[0:12]
        elif NA_type == 'HDNA':
            NA_sym = MDNA_sym
            pam = MDNA_map[0:12]
        else:
            print("------------Error provide argiment for sym------------------")
        mat1[count] = []
        for enum,seq in enumerate(pam):
            mat1[count].append(Truncation_KL_sym(data_tmp.choose_seq([seq]), NA_sym[seq]))
        count = count+1

    print("printing table for M/H DNA thesis ---")
    for i in range(12):
        print(i+1," & ", np.around(mat1[1][i],4)," & ", np.around(mat1[2][i],4), "\\\\")
    print("\\textbf{Average}"," & ", np.around(np.mean(mat1[1]),4)," & ", np.around(np.mean(mat1[2]),4), "\\\\")
    return mat1


def locality_error(data,NA_type,ps):
    if NA_type in ['DNA','RNA','MDNA','HDNA']:
        sym=True
    elif NA_type in ['HYB']:
        sym=False
    else:
        print("------------Error provide argiment for sym------------------")
    mat1,mat2= [],[]
    pam = np.arange(16)
    for enum,seq in enumerate(pam):
        u1,u2 = locality_err_KL_sym(data.choose_seq([seq]),ps, sym)
        mat1.append(u1)
        mat2.append(u2)
    return mat1,mat2

def locality_error_epi(data,NA_type_list):
    mat1 = {}
    count = 0
    for data_tmp, NA_type in zip(data,NA_type_list):
        mat1[count] = np.zeros((12,2))
        for enum,i in enumerate(MDNA_map[0:12]):
            mat1[count][enum,0],mat1[count][enum,1] = locality_err_KL_sym(data_tmp.choose_seq([i]),ps_list[NA_type], check_palin(data_tmp.seq[i]))
        count = count + 1
    print("printing table for M/H DNA thesis ---")
    for i in range(12):
        print(i+1," & ", np.around(mat1[1][i,0],4)," & ", np.around(mat1[1][i,1],4)," & ", np.around(mat1[2][i,0],4)," & ", np.around(mat1[2][i,1],4), "\\\\")
    print("\\textbf{Average}"," & ", np.around(np.mean(mat1[1][:,0]),4)," & ", np.around(np.mean(mat1[1][:,1]),4)," & ", np.around(np.mean(mat1[2][:,0]),4)," & ", np.around(np.mean(mat1[2][:,1]),4), "\\\\")
    return mat1

def Truncation_error_marg(data,NA_type_list):
    for data_tmp, NA_type in zip(data,NA_type_list):
        if NA_type == 'DNA':
            NA_sym = DNA_sym
            pam = np.arange(16)
            print("Note only computing for 16 seq")
        mat1,mat2,mat3 = [],[],[]
        for enum,seq in enumerate(pam):
            mat1.append(Truncation_KL_sym_inter(data_tmp.choose_seq([seq]), NA_sym[seq]))
            mat2.append(Truncation_KL_sym_cg(data_tmp.choose_seq([seq]), NA_sym[seq]))
            mat3.append(Truncation_KL_sym(data_tmp.choose_seq([seq]), NA_sym[seq]))

    return mat1,mat2,mat3

##################################---------------------------------------------
# plot_gs_vs_MD_shape
##################################---------------------------------------------
def plot_gs_vs_MD_shape(path1,path2,path3):
#########-------------------------DNA------------------
    c1 = ['red','blue','green']
    c2 = ['maroon','dodgerblue','limegreen']
    linestyles= ['-', '--', '-.', ':']
    DNA = init_MD_data().load_data(path1)
    nseq = len(DNA.nbp)
    for i in range(nseq):
        res = cgDNA(DNA.seq[i],'ps2_cgf')
        if comp(DNA.seq[i]) ==DNA.seq[i]:
            compare_shape([DNA.shape_sym[i],res.ground_state],[DNA.seq[i],res.seq],'DNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
        else:
            compare_shape([DNA.shape[i],res.ground_state],[DNA.seq[i],res.seq],'DNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')

#########-------------------------RNA------------------
    RNA = init_MD_data().load_data(path2)
    nseq = len(RNA.nbp)
    for i in range(nseq):
        seqD = (RNA.seq[i]).replace('U','T')
        res = cgDNA(seqD,'ps_rna')
        if comp(seqD) == seqD:
            compare_shape([RNA.shape_sym[i],res.ground_state],[RNA.seq[i],RNA.seq[i]],'RNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
        else:
            compare_shape([RNA.shape[i],res.ground_state],[RNA.seq[i],RNA.seq[i]],'RNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')

#########-------------------------HYB------------------
    HYB = init_MD_data().load_data(path3)
    nseq = len(HYB.nbp)
    for i in range(nseq):
        seqD = (HYB.seq[i]).replace('U','T')
        res = cgDNA(seqD,'ps_hyb')
        compare_shape([HYB.shape[i],res.ground_state],[HYB.seq[i],HYB.seq[i]],'HYB_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')





def plot_gs_vs_MD_shape_epi(MDNA,HDNA):
#########-------------------------MDNA------------------
    c1 = ['red','blue','green']
    c2 = ['maroon','dodgerblue','limegreen']
    linestyles= ['-', '--', '-.', ':']
    for DNA, dna in zip([MDNA,HDNA],['mdna','hdna']):
        nseq = len(DNA.nbp)
        for enum, i in enumerate(MDNA_map):
            res = cgDNA(DNA.seq[i],'ps_'+dna)
            if comp(DNA.seq[i]) ==DNA.seq[i]:
                compare_shape([DNA.shape_sym[i],res.ground_state],[DNA.seq[i],res.seq],dna.upper() +'_gs_res_compare_'+str(enum+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
            else:
                compare_shape([DNA.shape[i],res.ground_state],[DNA.seq[i],res.seq],dna.upper() +'_gs_res_compare_'+str(enum+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')

#########-------------------------HDNA------------------
    DNA = HDNA
    nseq = len(DNA.nbp)
    for i in range(nseq):
        res = cgDNA(DNA.seq[i],'ps_mdna')
        if comp(DNA.seq[i]) ==DNA.seq[i]:
            compare_shape([DNA.shape_sym[i],res.ground_state],[DNA.seq[i],res.seq],'MDNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
        else:
            compare_shape([DNA.shape[i],res.ground_state],[DNA.seq[i],res.seq],'MDNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')










##################################---------------------------------------------
# plot and compare same sequences across data
##################################---------------------------------------------
def compare_same_seq_across_data(d1,d2,d3,color,seq_id,save_name):
    lss=['-', '-','-','--','--','--']
    lss=['-', ':','--','--','--','--']
    a = d1.shape_sym[seq_id]
    b = d2.shape_sym[seq_id]
    c = d3.shape[seq_id]
    asq = d1.seq[seq_id]
#    ar = cgDNA(asq,'ps2_cgf').ground_state
#    br = cgDNA(asq,'ps_rna').ground_state
#    cr = cgDNA(asq,'ps_hyb').ground_state
#    shp = [a,b,c,ar,br,cr]
    shp = [a,b,c]
#    seq = [asq]*6
    seq = [asq]*3
    compare_shape(shp,seq,save_name+'_'+str(seq_id+1),lss,color=color*2,type_of_var='cg+')


def compare_same_seq_across_data_epi(d1,d2,d3_DNA,color,seq_id,save_name):
#    lss=['-', '-','-','--','--','--']
    asq = d1.seq[seq_id]
    if asq == comp(asq):
        a = d1.shape_sym[seq_id]
        b = d2.shape_sym[seq_id]
    else:
        a = d1.shape[seq_id]
        b = d2.shape[seq_id]
        print("This sequence is not palindrome", asq)

    dna_19 = d3_DNA.shape[19]
    a_res = cgDNA(asq,'ps_mdna').ground_state
    b_res = cgDNA(asq,'ps_hdna').ground_state
    tmp = min(np.linalg.eigvals(cgDNA(asq,'ps_mdna').stiff.todense() ))

    seq = asq.replace('M','C')
    seq = seq.replace('N','G')
    dna_res = cgDNA(seq,'ps_mdna').ground_state
    print(seq_id,seq,"--min eigvals-->    ",tmp)
    if tmp < 0 :
        print("Non-positive definite reconstruction ------- XXXXXX")
        sys.exit()

    shp = [dna_19,dna_res,a,a_res]
    seq = [asq]*4
    lss=['-', '--','-','--']
    color = [c1,c1,c4,c4]
    fig,ax_list = compare_shape(shp,seq,save_name+'_'+str(seq_id+1),lss,color=color,type_of_var='cg+')
    sf = list(asq)
    sf = list(map(lambda x: x.replace('M', 'C/M'), sf))
    sf = list(map(lambda x: x.replace('N', 'G/N'), sf))
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/compare_seq_MDNA_DNA_'+str(seq_id+1)+'.pdf',dpi=600)
    plt.close()

    lss=['-', '--','-','--']
    color = [c1,c1,c4,c4]
    shp = [dna_19,dna_res,b,b_res]
    fig,ax_list = compare_shape(shp,seq,save_name+'_'+str(seq_id+1),lss,color=color,type_of_var='cg+')
    sf = list(asq)
    sf = list(map(lambda x: x.replace('M', 'C/H'), sf))
    sf = list(map(lambda x: x.replace('N', 'G/K'), sf))
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/compare_seq_HDNA_DNA_'+str(seq_id+1)+'.pdf',dpi=600)

    lss=['-', '-','--','--','--']
    color = [c1,c4,c4]
    shp = [dna_res,a_res,b_res]
    fig,ax_list = compare_shape(shp,seq,save_name+'_'+str(seq_id+1),lss,color=color*2,type_of_var='cg+',bottom=0.075)
    sf = list(asq)
    sf = list(map(lambda x: x.replace('M', 'C/M/H'), sf))
    sf = list(map(lambda x: x.replace('N', 'G/N/K'), sf))
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/compare_seq_MDNA_HDNA_'+str(seq_id+1)+'.pdf',dpi=600)




###################
# compare_cpg_island_seq_across_data_epi
####################

def compare_cpg_island_seq_across_data_epi(d1,d2,d3,color,save_name):
#    lss=['-', '-','-','--','--','--']
    lss=['-', '-','--','--','--']

    ### seq 1 is palindrome ..rest are not
    dna_seq  = d1.seq[0]
    mdna_seq = {}
    mdna_shape = {}
    hdna_shape = {}
    mdna_shape_res = {}
    hdna_shape_res = {}
    dna_shape = d1.shape_sym[0]
    dna_shape_res = cgDNA(dna_seq,'ps_mdna').ground_state

    for i in range(3):
        if i == 0 :
            mdna_shape[i] = d2.shape_sym[i]
            hdna_shape[i] = d3.shape_sym[i]
        else:
            mdna_shape[i] = d2.shape[i]
            hdna_shape[i] = d3.shape[i]
        mdna_seq[i] = d2.seq[i]
        mdna_shape_res[i] = cgDNA(mdna_seq[i],'ps_mdna').ground_state
        hdna_shape_res[i] = cgDNA(mdna_seq[i],'ps_hdna').ground_state

    shp = [dna_shape,mdna_shape[2],hdna_shape[2]]
    seq = [dna_seq]*3
    fig,ax_list = compare_shape(shp,seq,save_name,lss,color=color*2,type_of_var='cg+',bottom=0.075)
    sf = list(mdna_seq[2])
    sf = list(map(lambda x: x.replace('M', 'C/M/H'), sf))
    sf = list(map(lambda x: x.replace('N', 'G/N/K'), sf))
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/'+save_name+'1.pdf',dpi=600)
    plt.close()

    color= [c1,c4]
    shp = [mdna_shape[0],mdna_shape[1],mdna_shape_res[0],mdna_shape_res[1]]
    seq = [dna_seq]*4
    fig,ax_list = compare_shape(shp,seq,save_name,lss,color=color*2,type_of_var='cg+',bottom=0.075)
    sf = list(mdna_seq[0])
    sf = list(map(lambda x: x.replace('M', 'M'), sf))
    sf = list(map(lambda x: x.replace('N', 'N/G'), sf))
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/'+save_name+'2.pdf',dpi=600)
    plt.close()

##################################---------------------------------------------
# plot and compare different sequences within same data
##################################---------------------------------------------
def compare_diff_seq_within_data(d,seq_id,ps,color,sym,save_name):
    lss=['-', '--','-', '--','-', '--','-', '--']
    shp=[]
    seq=[]
    ids = ''
    for enum,i in enumerate(seq_id):
        s = d.seq[i]
        s = s.replace('U','T')
        if sym[enum] == True:
            shp.append(d.shape_sym[i])
        else:
            shp.append(d.shape[i])
        shp.append(cgDNA(s,ps).ground_state)
        seq.append(s)
        seq.append(s)
        ids = ids + '_'+ str(i+1)

    if ps == 'ps_rna':
        for enum,s in enumerate(seq):
            seq[enum] = s.replace('T','U')
    compare_shape(shp,seq,save_name+ids,lss,color=color,type_of_var='cg+',multi_xlabel=True)


##################################---------------------------------------------
# plot oligomer level eigenvalues for DNA,RNA, Hybrid
##################################---------------------------------------------
def plot_eig_olig_first(d1,d2,d3,sym,color,save_name):
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 100)
    gs1.update(left=0.09, right=0.98,top=0.94,bottom=0.06)

    ax0 = plt.subplot(gs1[0:247,       0:100])
    ax1 = plt.subplot(gs1[253:500,       0:100])
    ax2 = plt.subplot(gs1[505:600,       0:100])
    fs=10
    if sym[0] == True:
        s1 = d1.s1b_sym[0]
        e1 = np.linalg.eigvals(s1)
    else:
        s1 = d1.s1b[0]
        e1 = np.linalg.eigvals(s1)

    if sym[1] == True:
        s2 = d2.s1b_sym[0]
        e2 = np.linalg.eigvals(s2)
    else:
        s2 = d2.s1b[0]
        e2 = np.linalg.eigvals(s2)
    sq = d1.seq[0]
    s3 = d3.s1b[0]
    e3 = np.linalg.eigvals(s3)
    s4 = cgDNA(sq,'ps2_cgf').stiff.todense()
    s5 = cgDNA(sq,'ps_rna' ).stiff.todense()
    s6 = cgDNA(sq,'ps_hyb' ).stiff.todense()
    e4 = np.linalg.eigvals(s4)
    e5 = np.linalg.eigvals(s5)
    e6 = np.linalg.eigvals(s6)


    g1 = scipy.linalg.eigh(s1, s4, eigvals_only=True)
    g2 = scipy.linalg.eigh(s2, s5, eigvals_only=True)
    g3 = scipy.linalg.eigh(s3, s6, eigvals_only=True)
#    print(g1,g2)

    s = np.shape(e1)
    ax0.scatter(np.arange(s[0])+1,np.sort(e2)[::-1],s=4,color=color[1],label='RNA_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e3)[::-1],s=4,color=color[2],label='HYB_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e1)[::-1],s=4,color=color[0],label='DNA_MD')
    ax1.scatter(np.arange(s[0])+1,np.sort(e5)[::-1],s=4,color=color[1],label='RNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e6)[::-1],s=4,color=color[2],label='HYB_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e4)[::-1],s=4,color=color[0],label='DNA_cg')

    ax2.scatter(np.arange(s[0])+1,np.sort(g2)[::-1],s=3,color=color[1],label='RNA_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g3)[::-1],s=3,color=color[2],label='HYB_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g1)[::-1],s=3,color=color[0],label='DNA_gen')

    for ax in [ax0,ax1,ax2]:
        ax.legend(fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

    ax1.set_ylabel("Eigenvalues for stiffness matrix",fontsize=fs)
    ax2.set_xlabel("Eigenvalue Index",fontsize=fs)
    ax0.set_title(sq+", Seq-length = " + str(len(sq)), fontsize=fs)
    plt.savefig('./Plots/'+save_name+".pdf",dpi=600)
    plt.show()
    plt.close()



##################################---------------------------------------------
# plot oligomer level eigenvalues for DNA,MDNA, HDNA
##################################---------------------------------------------
def plot_eig_olig_first_epi(d1,d2,d3,sym,color,save_name):
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 100)
    gs1.update(left=0.09, right=0.98,top=0.94,bottom=0.06)

    ax0 = plt.subplot(gs1[0:247,       0:100])
    ax1 = plt.subplot(gs1[253:500,       0:100])
    ax2 = plt.subplot(gs1[505:600,       0:100])
    fs=10
    if sym[0] == True:
        s1 = d1.stiff.todense()
        e1 = np.linalg.eigvals(s1)
    else:
        s1 = d1.stiff.todense()
        e1 = np.linalg.eigvals(s1)

    if sym[1] == True:
        s2 = d2.s1b_sym[0]
        e2 = np.linalg.eigvals(s2)
    else:
        s2 = d2.s1b[0]
        e2 = np.linalg.eigvals(s2)

    if sym[2] == True:
        s3 = d3.s1b_sym[0]
        e3 = np.linalg.eigvals(s3)
    else:
        s3 = d3.s1b[0]
        e3 = np.linalg.eigvals(s3)

    sq = d2.seq[0]
    e3 = np.linalg.eigvals(s3)
    s4 = s1
    s5 = cgDNA(sq,'ps_mdna' ).stiff.todense()
    s6 = cgDNA(sq,'ps_hdna' ).stiff.todense()
    e4 = np.linalg.eigvals(s4)
    e5 = np.linalg.eigvals(s5)
    e6 = np.linalg.eigvals(s6)


    g1 = scipy.linalg.eigh(s1, s4, eigvals_only=True)
    g2 = scipy.linalg.eigh(s2, s5, eigvals_only=True)
    g3 = scipy.linalg.eigh(s3, s6, eigvals_only=True)
#    print(g1,g2)

    s = np.shape(e1)
    ax0.scatter(np.arange(s[0])+1,np.sort(e2)[::-1],s=4,color=color[1],label='MDNA_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e3)[::-1],s=4,color=color[2],label='HDNA_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e1)[::-1],s=4,color=color[0],label='DNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e5)[::-1],s=4,color=color[1],label='MDNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e6)[::-1],s=4,color=color[2],label='HDNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e4)[::-1],s=4,color=color[0],label='DNA_cg')

    ax2.scatter(np.arange(s[0])+1,np.sort(g2)[::-1],s=3,color=color[1],label='MDNA_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g3)[::-1],s=3,color=color[2],label='HDNA_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g1)[::-1],s=3,color=color[0],label='DNA_gen')

    for ax in [ax0,ax1,ax2]:
        ax.legend(fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

    ax1.set_ylabel("Eigenvalues for stiffness matrix",fontsize=fs)
    ax2.set_xlabel("Eigenvalue Index",fontsize=fs)
    ax0.set_title(sq+", Seq-length = " + str(len(sq)), fontsize=fs)
    plt.savefig('./Plots/'+save_name+".pdf",dpi=600)
    plt.show()
    plt.close()

def sub_string_index(sbst, st):
    a = len(sbst)
    b = len(st)
    loc = []
    for i in range(b):
        if sbst == st[i:i+a]:
            loc.append(i)
    return loc
##################################---------------------------------------------
# Compare groundstate DNA,RNA,HDNA
##################################---------------------------------------------
def weighted_mean(mu1,w1):
    mu = copy.deepcopy(mu1)
    w = copy.deepcopy(w1)
    ttl = mu.index.to_list()
    for i,tet in enumerate(ttl):
        mu.loc[tet] = w[i]*mu.loc[tet]
    mu_avg = mu.sum()/sum(w)
    print("computing weighted mean ----------------------")
    return mu_avg

def weighted_shape_cov(mu1,w1):
    mu = copy.deepcopy(mu1)
    w = copy.deepcopy(w1)
    ttl = mu.index.to_list()
    mu_avg = weighted_mean(mu,w)
    s = np.zeros((18,18))
    for i,tet in enumerate(ttl):
        h = (mu.loc[tet]-mu_avg).to_numpy()[:,np.newaxis]
        s = s + w[i]* np.matmul(h,h.T)
    s = s/(sum(w)-1)
    return s

def confi_cov(mu1,cov1):
    mu_avg = np.mean(mu1,axis=0)
    H = np.zeros(cov1[0].shape)
    for enum,cov in enumerate(cov1):
        tmp = np.matmul(mu1[enum][:,np.newaxis],mu1[enum][:,np.newaxis].T)
        H = H + (cov + tmp)
    tut = np.matmul(mu_avg[:,np.newaxis],mu_avg[:,np.newaxis].T)
    cov_avg = H/(enum+1) -  tut #[:,np.newaxis]
    if H.shape == tut.shape == tmp.shape:
        None
    else:
        print("Dimensions incompatible while computing average covariance ----")
    return cov_avg

def extract_req_gs_data(MD,sym,epi,MD_or_model=None):
    if epi == False:
        mon_5 = mon
        tri_64 = trimer_64()
        tet_256 = ttl_256()
        dimer16 =  dimer_16
        nbr_seq = np.arange(16)
    else:
        mon_5   = epi_combination('monomer'  )
        tri_64  = epi_combination('trimer'    )
        tet_256 = epi_combination('tetramer_present'   )
        dimer16 = epi_combination('dimer')
        nbr_seq = MDNA_map[0:12]

    tmp_dict1 = dict([(key, []) for key in mon_5])
    tmp_dict2 = dict([(key, []) for key in tri_64])
    tmp_dict3 = dict([(key, []) for key in dimer16])
    tmp_dict4 = dict([(key, []) for key in tet_256])

    for seq_id in nbr_seq:
        tmp_seq = MD.choose_seq([seq_id]).seq[0].replace('U','T')
        if MD_or_model == 'DNA':
            tmp_shape = cgDNA(tmp_seq,'ps2_cgf').ground_state
        elif MD_or_model == 'RNA':
            tmp_shape = cgDNA(tmp_seq,'ps_rna').ground_state
        elif MD_or_model == 'HYB':
            tmp_shape = cgDNA(tmp_seq,'ps_hyb').ground_state
        elif MD_or_model == 'HDNA':
            tmp_shape = cgDNA(tmp_seq,'ps_hdna').ground_state
        elif MD_or_model == 'MDNA':
            tmp_shape = cgDNA(tmp_seq,'ps_mdna').ground_state
        elif MD_or_model == 'MD':
            if sym == True:
                tmp_shape = MD.choose_seq([seq_id]).shape_sym[0]
            elif sym == False:
                tmp_shape = MD.choose_seq([seq_id]).shape[0]
        else:
            print("in correct specification......")
            sys.exit()

        for tet in tmp_dict1.keys():
            for loc in sub_string_index(tet, tmp_seq[2:22]):
                tmp_dict1[tet].append(tmp_shape[24*(loc+2):24*(loc+2)+6])

        for tet in tmp_dict2.keys():
            for loc in sub_string_index(tet, tmp_seq[1:23]):
                tmp_dict2[tet].append(tmp_shape[24*(loc+2):24*(loc+2)+6])

        for dim in tmp_dict3.keys():
            for loc in sub_string_index(dim, tmp_seq[2:22]):
                tmp_dict3[dim].append(tmp_shape[24*(loc+2):24*(loc+2)+30])

        for tet in tmp_dict4.keys():
            for loc in sub_string_index(tet, tmp_seq[1:23]):
                tmp_dict4[tet].append(tmp_shape[24*(loc+2):24*(loc+2)+30])

    return tmp_dict1, tmp_dict2, tmp_dict3, tmp_dict4

def extract_mer_gs_data_from_MD(MD,sym,epi,MD_or_model=None):
    if epi==False:
        mat1 = np.zeros((6,5))
        mat2 = np.zeros((6,65))
        mat3 = np.zeros((30,17))
        mat4 = np.zeros((30,257))
    else:
        mat1 = np.zeros((6,3))
        mat2 = np.zeros((6,21))
        mat3 = np.zeros((30,13))
        mat4 = np.zeros((30,83))

    tmp_dict1, tmp_dict2, tmp_dict3, tmp_dict4 = extract_req_gs_data(MD,sym,epi,MD_or_model)

    for enum,tet in enumerate(tmp_dict1.keys()):
        mat1[:,enum] = np.mean(tmp_dict1[tet],axis=0)

    for enum,tet in enumerate(tmp_dict2.keys()):
        mat2[:,enum] = np.mean(tmp_dict2[tet],axis=0)

    for enum,dim in enumerate(tmp_dict3.keys()):
        mat3[:,enum] = np.mean(tmp_dict3[dim],axis=0)

    for enum,tet in enumerate(tmp_dict4.keys()):
        mat4[:,enum] = np.mean(tmp_dict4[tet],axis=0)
    if epi == False:
        mat1[:,4] = np.mean(mat1[:,0:4],axis=1)
        mat2[:,64] = np.mean(mat2[:,0:64],axis=1)
        mat3[:,16] = np.mean(mat3[:,0:16],axis=1)
        mat4[:,256] = np.mean(mat4[:,0:256],axis=1)
    else:
        mat1[:,2] = np.mean(mat1[:,0:2],axis=1)
        mat2[:,20] = np.mean(mat2[:,0:20],axis=1)
        mat3[:,12] = np.mean(mat3[:,0:12],axis=1)
        mat4[:,82] = np.mean(mat4[:,0:82],axis=1)

    return mat1, mat3, mat2, mat4

def cov_stiff(mat):
    ind = np.shape(mat)[0]
    for i in range(ind):
        mat[i] = ninv(mat[i])
    return mat

def stiff_diag(mat):
    ind = np.shape(mat)[0]
    vec = np.zeros(np.shape(mat)[0:2])
    for i in range(ind):
        vec[i] = 1/np.diag(mat[i])
    return vec.T

def extract_mer_stiff_data_from_MD(MD,sym,mat_form=False,MD_or_model=None,epi=False):
    gs_dict1, gs_dict2, gs_dict3, gs_dict4 = extract_req_gs_data(MD,sym,epi,MD_or_model)
    gs_mat1 , gs_mat3 , gs_mat2 , gs_mat4  = extract_mer_gs_data_from_MD(MD,sym,epi,MD_or_model)
    gs_mat1 , gs_mat3 , gs_mat2 , gs_mat4  = gs_mat1.T, gs_mat3.T, gs_mat2.T, gs_mat4.T

    if epi == False:
        mon_5 = mon
        tri_64 = trimer_64()
        tet_256 = ttl_256()
        dimer16 =  dimer_16
        nbr_seq = np.arange(16)
    else:
        mon_5   = epi_combination('monomer'  )
        tri_64  = epi_combination('trimer'    )
        tet_256 = epi_combination('tetramer_present')
        dimer16 = epi_combination('dimer')
        nbr_seq = MDNA_map[0:12]

    mat1 = np.zeros((len(mon_5)+1,6,6))
    tmp_dict1 = dict([(key, []) for key in mon_5])

    mat2 = np.zeros((len(tri_64)+1,6,6))
    tmp_dict2 = dict([(key, []) for key in tri_64])

    mat3 = np.zeros((len(dimer16)+1,30,30))
    tmp_dict3 = dict([(key, []) for key in dimer16])

    mat4 = np.zeros((len(tet_256)+1,30,30))
    tmp_dict4 = dict([(key, []) for key in tet_256])

    for seq_id in nbr_seq:
        tmp_seq = MD.choose_seq([seq_id]).seq[0].replace('U','T')
        if MD_or_model == 'DNA':
            a_tmp = cgDNA(tmp_seq,'ps2_cgf')
            tmp_cov = sinv(a_tmp.stiff).todense()
        elif MD_or_model == 'RNA':
            a_tmp = cgDNA(tmp_seq,'ps_rna')
            tmp_cov = sinv(a_tmp.stiff).todense()
        elif MD_or_model == 'HYB':
            a_tmp = cgDNA(tmp_seq,'ps_hyb')
            tmp_cov = sinv(a_tmp.stiff).todense()
        elif MD_or_model == 'HDNA':
            a_tmp = cgDNA(tmp_seq,'ps_hdna')
            tmp_cov = sinv(a_tmp.stiff).todense()
        elif MD_or_model == 'MDNA':
            a_tmp = cgDNA(tmp_seq,'ps_mdna')
            tmp_cov = sinv(a_tmp.stiff).todense()
        elif MD_or_model == 'MD':
            if sym == True:
                tmp_cov = ninv(MD.choose_seq([seq_id]).s1b_sym[0])
            elif sym == False:
                tmp_cov = ninv(MD.choose_seq([seq_id]).s1b[0])
        else:
            print("in correct specification......")
            sys.exit()

        for tet in mon_5:
            for loc in sub_string_index(tet, tmp_seq[2:22]):
                tmp = tmp_cov[24*(loc+2):24*(loc+2)+6,24*(loc+2):24*(loc+2)+6]
                tmp_dict1[tet].append(tmp)

        for tet in tri_64:
            for loc in sub_string_index(tet, tmp_seq[1:23]):
                tmp = tmp_cov[24*(loc+2):24*(loc+2)+6,24*(loc+2):24*(loc+2)+6]
                tmp_dict2[tet].append(tmp)

        for dim in dimer16:
            for loc in sub_string_index(dim, tmp_seq[2:22]):
                tmp = tmp_cov[24*(loc+2):24*(loc+2)+30,24*(loc+2):24*(loc+2)+30]
                tmp_dict3[dim].append(tmp)

        for tet in tet_256:
            for loc in sub_string_index(tet, tmp_seq[1:23]):
                tmp = tmp_cov[24*(loc+2):24*(loc+2)+30,24*(loc+2):24*(loc+2)+30]
                tmp_dict4[tet].append(tmp)

    ##########averaging covariance
    for enum,tet in enumerate(mon_5):
        mat1[enum,:,:] = confi_cov(gs_dict1[tet],tmp_dict1[tet])

    for enum,tet in enumerate(tri_64):
        mat2[enum,:,:] = confi_cov(gs_dict2[tet],tmp_dict2[tet])

    for enum,tet in enumerate(dimer16):
        mat3[enum,:,:] = confi_cov(gs_dict3[tet],tmp_dict3[tet])

    for enum,tet in enumerate(tet_256):
        try:
            mat4[enum,:,:] = confi_cov(gs_dict4[tet],tmp_dict4[tet])
        except:
            None   ### this is essential as some of the methylated tetramers are missing
    if epi == False:
        mat1[4,:,:] =  confi_cov(gs_mat1[0:4],mat1[0:4,:,:])
        mat2[64,:,:] = confi_cov(gs_mat2[0:64],mat2[0:64,:,:])
        mat3[16,:,:] = confi_cov(gs_mat3[0:16],mat3[0:16,:,:])
        mat4[256,:,:] = confi_cov(gs_mat4[0:256],mat4[0:256,:,:])
    else:
        mat1[2,:,:] =  confi_cov(gs_mat1[0:2],mat1[0:2,:,:])
        mat2[20,:,:] = confi_cov(gs_mat2[0:20],mat2[0:20,:,:])
        mat3[12,:,:] = confi_cov(gs_mat3[0:12],mat3[0:12,:,:])
        mat4[82,:,:] = confi_cov(gs_mat4[0:82],mat4[0:82,:,:])

    if mat_form==False:
        vec1, vec2, vec3, vec4 = stiff_diag(mat1), stiff_diag(mat2), stiff_diag(mat3), stiff_diag(mat4)
        return vec1, vec3, vec2, vec4
    elif mat_form == True:
        mat1, mat2, mat3, mat4 = cov_stiff(mat1), cov_stiff(mat2), cov_stiff(mat3), cov_stiff(mat4)
        return mat1, mat3, mat2, mat4


def decimal_printer(tmp2,how_many):
    pos = tmp2.find('.') + 1
    while len(tmp2) - pos < how_many:
        tmp2 = tmp2 + '0'
    return tmp2

def compare_gs_DNA_RNA(DNA, RNA, HYB,what,tetramer_points):
    if tetramer_points == True:
        x_shift_list = [-0.19,0,0.19]
    else:
        x_shift_list = [0,0,0]
    DNA_dimer_data, RNA_dimer_data, HYB_dimer_data  = [], [], []
    DNA_tri_data, RNA_tri_data, HYB_tri_data  = [], [], []
    DNA_tet_data, RNA_tet_data, HYB_tet_data  = [], [], []
    DNA_mon_data, RNA_mon_data, HYB_mon_data  = [], [], []

    if what == 'stiff':
        Dmat1, Dmat2, Dmat3, Dmat4 = extract_mer_stiff_data_from_MD(DNA,True ,False,'MD')
        Rmat1, Rmat2, Rmat3, Rmat4 = extract_mer_stiff_data_from_MD(RNA,True ,False,'MD')
        Hmat1, Hmat2, Hmat3, Hmat4 = extract_mer_stiff_data_from_MD(HYB,False,False,'MD')


        Dmat1cg, Dmat2cg, Dmat3cg, Dmat4cg = extract_mer_stiff_data_from_MD(DNA,True ,False,'DNA')
        Rmat1cg, Rmat2cg, Rmat3cg, Rmat4cg = extract_mer_stiff_data_from_MD(RNA,True ,False,'RNA')
        Hmat1cg, Hmat2cg, Hmat3cg, Hmat4cg = extract_mer_stiff_data_from_MD(HYB,False,False,'HYB')

    elif what == 'gs':
        Dmat1, Dmat2, Dmat3, Dmat4 = extract_mer_gs_data_from_MD(DNA,True ,epi,'MD')
        print(np.shape(Dmat1))
        print(np.shape(Dmat4))
        Rmat1, Rmat2, Rmat3, Rmat4 = extract_mer_gs_data_from_MD(RNA,True ,epi,'MD')
        Hmat1, Hmat2, Hmat3, Hmat4 = extract_mer_gs_data_from_MD(HYB,False,epi,'MD')

        Dmat1cg, Dmat2cg, Dmat3cg, Dmat4cg = extract_mer_gs_data_from_MD(DNA,True ,epi,'DNA')
        Rmat1cg, Rmat2cg, Rmat3cg, Rmat4cg = extract_mer_gs_data_from_MD(RNA,True ,epi,'RNA')
        Hmat1cg, Hmat2cg, Hmat3cg, Hmat4cg = extract_mer_gs_data_from_MD(HYB,False ,epi,'HYB')

    DNA_mon_data.append(Dmat1)
    RNA_mon_data.append(Rmat1)
    HYB_mon_data.append(Hmat1)
    DNA_mon_data.append(Dmat1cg)
    RNA_mon_data.append(Rmat1cg)
    HYB_mon_data.append(Hmat1cg)

    DNA_dimer_data.append(Dmat2)
    RNA_dimer_data.append(Rmat2)
    HYB_dimer_data.append(Hmat2)
    DNA_dimer_data.append(Dmat2cg)
    RNA_dimer_data.append(Rmat2cg)
    HYB_dimer_data.append(Hmat2cg)

    DNA_tri_data.append(Dmat3)
    RNA_tri_data.append(Rmat3)
    HYB_tri_data.append(Hmat3)
    DNA_tri_data.append(Dmat3cg)
    RNA_tri_data.append(Rmat3cg)
    HYB_tri_data.append(Hmat3cg)

    DNA_tet_data.append(Dmat4)
    RNA_tet_data.append(Rmat4)
    HYB_tet_data.append(Hmat4)
    DNA_tet_data.append(Dmat4cg)
    RNA_tet_data.append(Rmat4cg)
    HYB_tet_data.append(Hmat4cg)

    tet_data = [DNA_tet_data, RNA_tet_data,HYB_tet_data]
    tri_data = [DNA_tri_data, RNA_tri_data,HYB_tri_data]
    dim_data = [DNA_dimer_data, RNA_dimer_data,HYB_dimer_data]
    mon_data = [DNA_mon_data, RNA_mon_data,HYB_mon_data]

    ##### think about how to read HYB from RNA strand and E_trans method is wrong
    fig = plt.figure(constrained_layout=False,figsize=(7,8))
    d3 = 600
    gs1 = gridspec.GridSpec(2402-d3, 215)
    gs1.update(left=0.065, right=0.99,top=0.99,bottom=0.04)
    d = 15
    d2 = 1

    ax20 = plt.subplot(gs1[ 600-d3:800 -d2-d3,   0:100])
    ax21 = plt.subplot(gs1[ 800-d3:1000-d2-d3,   0:100])
    ax22 = plt.subplot(gs1[1000-d3:1200-d2-d3,   0:100])
    ax23 = plt.subplot(gs1[ 600-d3:800 -d2-d3,   100+d:200+d])
    ax24 = plt.subplot(gs1[ 800-d3:1000-d2-d3,   100+d:200+d])
    ax25 = plt.subplot(gs1[1000-d3:1200-d2-d3,   100+d:200+d])

    ax30 = plt.subplot(gs1[1200-d3:1400-d2-d3,   0:100])
    ax31 = plt.subplot(gs1[1400-d3:1600-d2-d3,   0:100])
    ax32 = plt.subplot(gs1[1600-d3:1800-d2-d3,   0:100])
    ax33 = plt.subplot(gs1[1200-d3:1400-d2-d3,   100+d:200+d])
    ax34 = plt.subplot(gs1[1400-d3:1600-d2-d3,   100+d:200+d])
    ax35 = plt.subplot(gs1[1600-d3:1800-d2-d3,   100+d:200+d])

    ax40 = plt.subplot(gs1[1800-d3:2000-d2-d3,   0:100])
    ax41 = plt.subplot(gs1[2000-d3:2200-d2-d3,   0:100])
    ax42 = plt.subplot(gs1[2200-d3:2400-d2-d3,   0:100])
    ax43 = plt.subplot(gs1[1800-d3:2000-d2-d3,   100+d:200+d])
    ax44 = plt.subplot(gs1[2000-d3:2200-d2-d3,   100+d:200+d])
    ax45 = plt.subplot(gs1[2200-d3:2400-d2-d3,   100+d:200+d])

    ax_list = [ax20,ax21,ax22,ax23,ax24,ax25, ax30,ax31,ax32,ax33,ax34,ax35, ax40,ax41,ax42,ax43,ax44,ax45]
    fs = 7
    color = ['blue','red','k','k']
    line_styles = ['-','-','-','--']
    for enum_MD, MD in enumerate(dim_data):
        x_shift = x_shift_list[enum_MD]
        tet_MD = tet_data[enum_MD]
        for enum, ax in enumerate(ax_list):
            enum = enum+6
            if enum_MD == 0:
                if what == 'gs':
                    leg = cgDNA_name[enum] + cgDNA_units[enum]
                elif what == 'stiff':
                    leg = cgDNA_name[enum] +'-'+cgDNA_name[enum]
                ax.set_ylabel(leg,fontsize=fs)
                ax.get_yaxis().set_label_coords(-0.09,0.5)
            else:
                leg = '_no_legend_'
            leg1 = '_no_legend_'
            ax.scatter(np.arange(17)+x_shift, MD[0][enum,:],color=color[enum_MD],s=1,label=leg1)
            ax.scatter(np.arange(17)+x_shift, MD[1][enum,:],color=color[enum_MD],s=7,marker="x",lw=0.6)

            ####### tetramer data
            if tetramer_points == True:
                for i1 in range(16):
                    small_x = np.zeros(16)+i1+x_shift
                    small_y = tet_MD[0][enum,16*i1:16*(i1+1)]
                    for i2 in range(16):
                        ax.scatter(small_x[i2],small_y[i2],label=leg1,marker='|',lw=0.4,s=5,color=color[enum_MD]) #color=color_16_RY[i2]
#                        ax.scatter(small_x[i2],small_y[i2],label=leg1,marker='|',lw=0.4,s=6,color=color_16_RY[i2])

            ax.plot(np.arange(17)+x_shift, MD[0][enum,:],color=color[enum_MD],lw=0.15,label=leg,ls=line_styles[enum_MD])
            ax.set_xlim(-0.35,16.3)
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            if what == 'gs':
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            elif what == 'stiff':
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #           ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    #            ax.legend(fontsize=fs)
    prin = False
    if prin == True:
        a1 = np.around(DNA_dimer_data[1],3)
        b1 = np.around(RNA_dimer_data[1],3)
        c1 = np.around(HYB_dimer_data[1],3)
        tmp = 'IC'
        for zi in range(17):
            tmp = tmp + ' & ' + str(dimer_17[zi])
        tmp = tmp + '\\\\'
        print(tmp)
        print('\\hline')
        print('\\hline')
        for yi in range(6,24):
            tmp = ''
            for zi in range(17):
                tmp = tmp + ' & ' + decimal_printer( str(a1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = '' + cgDNA_name[yi]
            for zi in range(17):
                tmp = tmp + ' & ' + decimal_printer( str(c1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = ''
            for zi in range(17):
                tmp = tmp + ' & ' + decimal_printer( str(b1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)
            print('\\hline')

    ind = np.arange(len(dimer_17))
    ax42.set_xticks(ind)
    ax42.set_xticklabels(dimer_17,fontsize=fs)
    ax45.set_xticks(ind)
    ax45.set_xticklabels(dimer_17,fontsize=fs)
    secax1 = ax42.secondary_xaxis('bottom')
    secax1.set_xticks(ind)
    secax1.set_xticklabels([diii.replace('T','U')for diii in dimer_17],fontsize=fs)
    secax1.tick_params(axis='both', pad=10,length=3,width=0.5,direction= 'inout',rotation=0)
    secax2 = ax45.secondary_xaxis('bottom')
    secax2.set_xticks(ind)
    secax2.set_xticklabels([diii.replace('T','U')for diii in dimer_17],fontsize=fs)
    secax2.tick_params(axis='both', pad=10,length=3,width=0.5,direction= 'inout',rotation=0)

    for ticklabel, tickcolor in zip(ax42.get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(ax45.get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(secax1.get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(secax2.get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)

#    plt.show()
    fig.savefig("./Plots/compare_dim_gs_DNA_RNA_HYB_"+what  +".pdf",dpi=600)
    plt.close()

    fig = plt.figure(constrained_layout=False,figsize=(6,3))
    gs1 = gridspec.GridSpec(602, 210)
    gs1.update(left=0.08, right=0.99,top=0.96,bottom=0.1)
    d = 20
    d2 = 1
    ax10 = plt.subplot(gs1[   0:200 -d2,   0:95])
    ax11 = plt.subplot(gs1[ 200:400 -d2,   0:95])
    ax12 = plt.subplot(gs1[ 400:600 -d2,   0:95])
    ax13 = plt.subplot(gs1[   0:200 -d2,   95+d:190+d])
    ax14 = plt.subplot(gs1[ 200:400 -d2,   95+d:190+d])
    ax15 = plt.subplot(gs1[ 400:600 -d2,   95+d:190+d])
    ax_list = [ax10,ax11,ax12,ax13,ax14,ax15]

    for enum_MD, mon_MD in enumerate(mon_data):
        x_shift = x_shift_list[enum_MD]
        tri_MD = tri_data[enum_MD]
        for enum, ax in enumerate(ax_list):
            if enum_MD == 0:
                leg = cgDNA_name[enum] + cgDNA_units[enum]
                ax.set_ylabel(leg,fontsize=fs)
                ax.get_yaxis().set_label_coords(-0.14,0.5)
            else:
                leg = '_no_legend_'
            leg1 = '_no_legend_'
            ax.scatter(np.arange(5)+x_shift, mon_MD[0][enum,:],color=color[enum_MD],s=1,label=leg1)
            ax.scatter(np.arange(5)+x_shift, mon_MD[1][enum,:],color=color[enum_MD],s=7,marker="x",lw=0.6)

            ####### tetramer data
            if tetramer_points == True:
                for i1 in range(4):
                    small_x = np.zeros(16)+i1+x_shift
                    small_y = tri_MD[0][enum,16*i1:16*(i1+1)]
                    for i2 in range(16):
                        ax.scatter(small_x[i2],small_y[i2],label=leg1,marker='|',lw=0.4,s=5,color=color[enum_MD]) #color=color_16_RY[i2]
#                        ax.scatter(small_x[i2],small_y[i2],label=leg1,marker='|',lw=0.4,s=6,color=color_16_RY[i2])

            ax.plot(np.arange(5)+x_shift, mon_MD[0][enum,:],color=color[enum_MD],lw=0.15,label=leg,ls=line_styles[enum_MD])
            ax.set_xlim(-0.35,4.3)
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    mon5 = ['A','T/U','G','C']+['Avg']
    ind = np.arange(len(mon5))
    ax12.set_xticks(ind)
    ax12.set_xticklabels(mon5,fontsize=fs)
    ax15.set_xticks(ind)
    ax15.set_xticklabels(mon5,fontsize=fs)
    for ticklabel, tickcolor in zip(ax12.get_xticklabels(), color_4_RY):ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(ax15.get_xticklabels(), color_4_RY):ticklabel.set_color(tickcolor)
    fig.savefig("./Plots/compare_mon_gs_DNA_RNA_HYB_"+what  +".pdf",dpi=600)
    plt.close()
    if prin == True:
        a1 = np.around(DNA_mon_data[1],3)
        b1 = np.around(RNA_mon_data[1],3)
        c1 = np.around(HYB_mon_data[1],3)
        for zi in range(5):
            tmp = tmp + ' & ' + str(mon5[zi])
        tmp = tmp + '\\\\'
        print(tmp)
        print('\\hline')
        print('\\hline')
        for yi in range(6):
            tmp = ''
            for zi in range(5):
                tmp = tmp + ' & ' + decimal_printer( str(a1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = '' + cgDNA_name[yi]
            for zi in range(5):
                tmp = tmp + ' & ' + decimal_printer( str(c1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = ''
            for zi in range(5):
                tmp = tmp + ' & ' + decimal_printer( str(b1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)
            print('\\hline')
    return None



##################################---------------------------------------------
##################################---------------------------------------------
##################################---------------------------------------------
def epi_dimer_in_tet_list(tet,dim):
    l = []
    ld = len(dim)
    for enum,t in enumerate(tet):
        if t[1:1+ld] == dim:
            l.append(enum)
    return l

def compare_gs_DNA_epi(DNA, MDNA, HDNA,what,tetramer_points):

    epi_mon_5   = epi_combination('monomer'  )
    epi_tri_64  = epi_combination('trimer'    )
    epi_tet_256 = epi_combination('tetramer_present')
    epi_dimer16 = epi_combination('dimer')

    un_mon_5   = [unmodify(s) for s  in epi_mon_5]
    un_tri_64  = [unmodify(s) for s  in epi_tri_64]
    un_tet_256 = [unmodify(s) for s  in epi_tet_256]
    un_dimer16 = [unmodify(s) for s  in epi_dimer16]

    mon_5 = mon
    tri_64 = trimer_64()
    tet_256 = ttl_256()
    dimer16 =  dimer_16

    epi_mon_index = [mon_5.index(i)   for i in un_mon_5]
    epi_dim_index = [dimer16.index(i) for i in un_dimer16]
    epi_tri_index = [tri_64.index(i)  for i in un_tri_64]
    epi_tet_index = [tet_256.index(i) for i in un_tet_256]

    if tetramer_points == True:
        x_shift_list = [-0.19,0,0.19]
    else:
        x_shift_list = [0,0,0]

    DNA_dimer_data, MDNA_dimer_data, HDNA_dimer_data  = [], [], []
    DNA_tri_data  , MDNA_tri_data  , HDNA_tri_data    = [], [], []
    DNA_tet_data  , MDNA_tet_data  , HDNA_tet_data    = [], [], []
    DNA_mon_data  , MDNA_mon_data  , HDNA_mon_data    = [], [], []

    if what == 'stiff':
        ###extract_mer_stiff_data_from_MD(MD,sym,mat_form=False,MD_or_model=None,epi=False):
        Dmat1, Dmat2, Dmat3, Dmat4 = extract_mer_stiff_data_from_MD(DNA,  True, False, 'MD', False)
        Mmat1, Mmat2, Mmat3, Mmat4 = extract_mer_stiff_data_from_MD(MDNA, True, False, 'MD', True)
        Hmat1, Hmat2, Hmat3, Hmat4 = extract_mer_stiff_data_from_MD(HDNA, True, False, 'MD', True)
        ## idea is to make the data comparable to epi
        Dmat1, Dmat2, Dmat3, Dmat4 = Dmat1[:,epi_mon_index],Dmat2[:,epi_dim_index],Dmat3[:,epi_tri_index],Dmat4[:,epi_tet_index]

        Dmat1cg, Dmat2cg, Dmat3cg, Dmat4cg = extract_mer_stiff_data_from_MD(DNA, True ,False,'DNA',False)
        Mmat1cg, Mmat2cg, Mmat3cg, Mmat4cg = extract_mer_stiff_data_from_MD(MDNA,True ,False,'MDNA',True)
        Hmat1cg, Hmat2cg, Hmat3cg, Hmat4cg = extract_mer_stiff_data_from_MD(HDNA,True ,False,'HDNA',True)
        Dmat1cg, Dmat2cg, Dmat3cg, Dmat4cg = Dmat1cg[:,epi_mon_index],Dmat2cg[:,epi_dim_index],Dmat3cg[:,epi_tri_index],Dmat4cg[:,epi_tet_index]

        Mmat2cg, Hmat2cg = Mmat2cg[:,0:12], Hmat2cg[:,0:12]
        Mmat2  , Hmat2   = Mmat2[:,0:12]  , Hmat2[:,0:12]

        Mmat1cg, Hmat1cg = Mmat1cg[:,0:2], Hmat1cg[:,0:2]
        Mmat1  , Hmat1   = Mmat1[:,0:2]  , Hmat1[:,0:2]

    elif what == 'gs':
        ## extract_mer_gs_data_from_MD(MD,sym,epi,MD_or_model=None):
        symm = True
        Dmat1, Dmat2, Dmat3, Dmat4 = extract_mer_gs_data_from_MD(DNA ,symm ,False,'MD')
        Dmat1, Dmat2, Dmat3, Dmat4 = Dmat1[:,epi_mon_index],Dmat2[:,epi_dim_index],Dmat3[:,epi_tri_index],Dmat4[:,epi_tet_index]
        Mmat1, Mmat2, Mmat3, Mmat4 = extract_mer_gs_data_from_MD(MDNA,symm ,True, 'MD')
        Hmat1, Hmat2, Hmat3, Hmat4 = extract_mer_gs_data_from_MD(HDNA,symm ,True, 'MD')

        Dmat1cg, Dmat2cg, Dmat3cg, Dmat4cg = extract_mer_gs_data_from_MD(DNA ,symm,False,'DNA')
        Dmat1cg, Dmat2cg, Dmat3cg, Dmat4cg = Dmat1cg[:,epi_mon_index],Dmat2cg[:,epi_dim_index],Dmat3cg[:,epi_tri_index],Dmat4cg[:,epi_tet_index]
        Mmat1cg, Mmat2cg, Mmat3cg, Mmat4cg = extract_mer_gs_data_from_MD(MDNA,symm,True ,'MDNA')
        Hmat1cg, Hmat2cg, Hmat3cg, Hmat4cg = extract_mer_gs_data_from_MD(HDNA,symm,True ,'HDNA')

        Mmat2cg, Hmat2cg = Mmat2cg[:,0:12], Hmat2cg[:,0:12]
        Mmat2  , Hmat2   = Mmat2[:,0:12]  , Hmat2[:,0:12]
        Mmat1cg, Hmat1cg = Mmat1cg[:,0:2], Hmat1cg[:,0:2]
        Mmat1  , Hmat1   = Mmat1[:,0:2]  , Hmat1[:,0:2]


    DNA_mon_data.append( Dmat1)
    MDNA_mon_data.append(Mmat1)
    HDNA_mon_data.append(Hmat1)
    DNA_mon_data.append( Dmat1cg)
    MDNA_mon_data.append(Mmat1cg)
    HDNA_mon_data.append(Hmat1cg)

    DNA_dimer_data.append( Dmat2)
    MDNA_dimer_data.append(Mmat2)
    HDNA_dimer_data.append(Hmat2)
    DNA_dimer_data.append( Dmat2cg)
    MDNA_dimer_data.append(Mmat2cg)
    HDNA_dimer_data.append(Hmat2cg)

    DNA_tri_data.append( Dmat3)
    MDNA_tri_data.append(Mmat3)
    HDNA_tri_data.append(Hmat3)
    DNA_tri_data.append( Dmat3cg)
    MDNA_tri_data.append(Mmat3cg)
    HDNA_tri_data.append(Hmat3cg)

    DNA_tet_data.append( Dmat4)
    MDNA_tet_data.append(Mmat4)
    HDNA_tet_data.append(Hmat4)
    DNA_tet_data.append( Dmat4cg)
    MDNA_tet_data.append(Mmat4cg)
    HDNA_tet_data.append(Hmat4cg)

    tet_data = [DNA_tet_data  , MDNA_tet_data   ,HDNA_tet_data]
    tri_data = [DNA_tri_data  , MDNA_tri_data   ,HDNA_tri_data]
    dim_data = [DNA_dimer_data, MDNA_dimer_data ,HDNA_dimer_data]
    mon_data = [DNA_mon_data  , MDNA_mon_data   ,HDNA_mon_data]

#################################################
#################### PLOTS ######################
#################################################
    fig = plt.figure(constrained_layout=False,figsize=(6.4,6))
    d3 = 1200
    gs1 = gridspec.GridSpec(2402-d3, 220)
    gs1.update(left=0.08, right=0.99,top=0.99,bottom=0.04)
    d = 20
    d2 = 1

    # ax20 = plt.subplot(gs1[ 600-d3:800 -d2-d3,   0:100])
    # ax21 = plt.subplot(gs1[ 800-d3:1000-d2-d3,   0:100])
    # ax22 = plt.subplot(gs1[1000-d3:1200-d2-d3,   0:100])
    # ax23 = plt.subplot(gs1[ 600-d3:800 -d2-d3,   100+d:200+d])
    # ax24 = plt.subplot(gs1[ 800-d3:1000-d2-d3,   100+d:200+d])
    # ax25 = plt.subplot(gs1[1000-d3:1200-d2-d3,   100+d:200+d])

    ax30 = plt.subplot(gs1[1200-d3:1400-d2-d3,   0:100])
    ax31 = plt.subplot(gs1[1400-d3:1600-d2-d3,   0:100])
    ax32 = plt.subplot(gs1[1600-d3:1800-d2-d3,   0:100])
    ax33 = plt.subplot(gs1[1200-d3:1400-d2-d3,   100+d:200+d])
    ax34 = plt.subplot(gs1[1400-d3:1600-d2-d3,   100+d:200+d])
    ax35 = plt.subplot(gs1[1600-d3:1800-d2-d3,   100+d:200+d])

    ax40 = plt.subplot(gs1[1800-d3:2000-d2-d3,   0:100])
    ax41 = plt.subplot(gs1[2000-d3:2200-d2-d3,   0:100])
    ax42 = plt.subplot(gs1[2200-d3:2400-d2-d3,   0:100])
    ax43 = plt.subplot(gs1[1800-d3:2000-d2-d3,   100+d:200+d])
    ax44 = plt.subplot(gs1[2000-d3:2200-d2-d3,   100+d:200+d])
    ax45 = plt.subplot(gs1[2200-d3:2400-d2-d3,   100+d:200+d])

    # ax_list = [ax20,ax21,ax22,ax23,ax24,ax25, ax30,ax31,ax32,ax33,ax34,ax35, ax40,ax41,ax42,ax43,ax44,ax45]
    ax_list = [ax30,ax31,ax32,ax33,ax34,ax35, ax40,ax41,ax42,ax43,ax44,ax45]
    fs = 8
    color = ['blue','red','k','k']
    line_styles = ['-','-','-','--']
    for enum_MD, MD in enumerate(dim_data):
        x_shift = x_shift_list[enum_MD]
        tet_MD = tet_data[enum_MD]
        for enum, ax in enumerate(ax_list):
            enum = enum+12
            if enum_MD == 0:
                if what == 'gs':
                    leg = cgDNA_name[enum] + cgDNA_units[enum]
                elif what == 'stiff':
                    leg = cgDNA_name[enum] +' - '+cgDNA_name[enum]
                ax.set_ylabel(leg,fontsize=fs,labelpad = 8)
                ax.get_yaxis().set_label_coords(-0.12,0.5)
            else:
                leg = '_no_legend_'
            leg1 = '_no_legend_'
            ax.scatter(np.arange(12)+x_shift, MD[0][enum,:],color=color[enum_MD],s=1,label=leg1)
            ax.scatter(np.arange(12)+x_shift, MD[1][enum,:],color=color[enum_MD],s=7,marker="x",lw=0.6)

            ####### tetramer data
            if tetramer_points == True:
                for i1 in range(12):
                    tet_sub_list_dim = epi_dimer_in_tet_list(epi_tet_256,epi_dimer16[i1])
                    small_x = np.zeros(len(tet_sub_list_dim))+i1+x_shift
                    small_y = tet_MD[0][enum,tet_sub_list_dim]
                    for i2 in range(len(tet_sub_list_dim)):
                        ax.scatter(small_x[i2],small_y[i2],label=leg1,marker='|',lw=0.4,s=5,color=color[enum_MD]) #color=color_16_RY[i2]

            ax.plot(np.arange(12)+x_shift, MD[0][enum,:],color=color[enum_MD],lw=0.15,label=leg,ls=line_styles[enum_MD])
            ax.set_xlim(-0.35,11.3)
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            if what == 'gs':
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            elif what == 'stiff':
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    prin = False
    if prin == True:
        a1 = np.around(DNA_dimer_data[1],3)
        b1 = np.around(MDNA_dimer_data[1],3)
        c1 = np.around(HDNA_dimer_data[1],3)
        tmp = 'IC'
        for zi in range(17):
            tmp = tmp + ' & ' + str(dimer_17[zi])
        tmp = tmp + '\\\\'
        print(tmp)
        print('\\hline')
        print('\\hline')
        for yi in range(6,24):
            tmp = ''
            for zi in range(17):
                tmp = tmp + ' & ' + decimal_printer( str(a1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = '' + cgDNA_name[yi]
            for zi in range(17):
                tmp = tmp + ' & ' + decimal_printer( str(c1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = ''
            for zi in range(17):
                tmp = tmp + ' & ' + decimal_printer( str(b1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)
            print('\\hline')

    ind = np.arange(len(epi_dimer16))

    indlab = [i.replace('M','X') for i in epi_dimer16]
    indlab = [i.replace('N','Z') for i in indlab]

    ax42.set_xticks(ind)
    ax42.set_xticklabels(indlab,fontsize=fs)
    ax45.set_xticks(ind)
    ax45.set_xticklabels(indlab,fontsize=fs)

    fig.savefig("./Plots/compare_dim_gs_DNA_MDNA_HDNA_"+what  +".pdf",dpi=600)
    plt.close()

    fig = plt.figure(constrained_layout=False,figsize=(6,3))
    gs1 = gridspec.GridSpec(602, 210)
    gs1.update(left=0.08, right=0.99,top=0.96,bottom=0.1)
    d = 20
    d2 = 1
    ax10 = plt.subplot(gs1[   0:200 -d2,   0:95])
    ax11 = plt.subplot(gs1[ 200:400 -d2,   0:95])
    ax12 = plt.subplot(gs1[ 400:600 -d2,   0:95])
    ax13 = plt.subplot(gs1[   0:200 -d2,   95+d:190+d])
    ax14 = plt.subplot(gs1[ 200:400 -d2,   95+d:190+d])
    ax15 = plt.subplot(gs1[ 400:600 -d2,   95+d:190+d])
    ax_list = [ax10,ax11,ax12,ax13,ax14,ax15]

    for enum_MD, mon_MD in enumerate(mon_data):
        x_shift = x_shift_list[enum_MD]
        tri_MD = tri_data[enum_MD]
        for enum, ax in enumerate(ax_list):
            if enum_MD == 0:
                if what == 'gs':
                    leg = cgDNA_name[enum] + cgDNA_units[enum]
                elif what == 'stiff':
                    leg = cgDNA_name[enum] +' - '+cgDNA_name[enum]
                ax.set_ylabel(leg,fontsize=fs, labelpad = 8)
                ax.get_yaxis().set_label_coords(-0.17,0.5)
            else:
                leg = '_no_legend_'
            leg1 = '_no_legend_'

            ax.scatter(np.arange(2)+x_shift, mon_MD[0][enum,:],color=color[enum_MD],s=1,label=leg1)
            ax.scatter(np.arange(2)+x_shift, mon_MD[1][enum,:],color=color[enum_MD],s=7,marker="x",lw=0.6)

            ####### tetramer data
            if tetramer_points == True:
                for i1 in range(2):
                    tri_sub_list_mono = epi_dimer_in_tet_list(epi_tri_64,epi_mon_5[i1])
                    small_x = np.zeros(len(tri_sub_list_mono))+i1+x_shift
                    small_y = tet_MD[0][enum,tri_sub_list_mono]
                    for i2 in range(len(tri_sub_list_mono)):
                        ax.scatter(small_x[i2],small_y[i2],label=leg1,marker='|',lw=0.4,s=5,color=color[enum_MD]) #color=color_16_RY[i2]
#                        ax.scatter(small_x[i2],small_y[i2],label=leg1,marker='|',lw=0.4,s=6,color=color_16_RY[i2])

            ax.plot(np.arange(2)+x_shift, mon_MD[0][enum,:],color=color[enum_MD],lw=0.15,label=leg,ls=line_styles[enum_MD])
            ax.set_xlim(-0.35,1.3)
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    mon5 = ['U','V']
    ind = np.arange(len(mon5))
    ax12.set_xticks(ind)
    ax12.set_xticklabels(mon5,fontsize=fs)
    ax15.set_xticks(ind)
    ax15.set_xticklabels(mon5,fontsize=fs)
    # for ticklabel, tickcolor in zip(ax12.get_xticklabels(), color_4_RY):ticklabel.set_color(tickcolor)
    # for ticklabel, tickcolor in zip(ax15.get_xticklabels(), color_4_RY):ticklabel.set_color(tickcolor)
    fig.savefig("./Plots/compare_mon_gs_DNA_MDNA_HDNA_"+what  +".pdf",dpi=600)
    plt.close()



    fig = plt.figure(constrained_layout=False,figsize=(6.4,6))
    d3 = 1200
    gs1 = gridspec.GridSpec(2402-d3, 220)
    gs1.update(left=0.08, right=0.99,top=0.99,bottom=0.07)
    d = 20
    d2 = 1

    # ax20 = plt.subplot(gs1[ 600-d3:800 -d2-d3,   0:100])
    # ax21 = plt.subplot(gs1[ 800-d3:1000-d2-d3,   0:100])
    # ax22 = plt.subplot(gs1[1000-d3:1200-d2-d3,   0:100])
    # ax23 = plt.subplot(gs1[ 600-d3:800 -d2-d3,   100+d:200+d])
    # ax24 = plt.subplot(gs1[ 800-d3:1000-d2-d3,   100+d:200+d])
    # ax25 = plt.subplot(gs1[1000-d3:1200-d2-d3,   100+d:200+d])

    ax30 = plt.subplot(gs1[1200-d3:1400-d2-d3,   0:100])
    ax31 = plt.subplot(gs1[1400-d3:1600-d2-d3,   0:100])
    ax32 = plt.subplot(gs1[1600-d3:1800-d2-d3,   0:100])
    ax33 = plt.subplot(gs1[1200-d3:1400-d2-d3,   100+d:200+d])
    ax34 = plt.subplot(gs1[1400-d3:1600-d2-d3,   100+d:200+d])
    ax35 = plt.subplot(gs1[1600-d3:1800-d2-d3,   100+d:200+d])

    ax40 = plt.subplot(gs1[1800-d3:2000-d2-d3,   0:100])
    ax41 = plt.subplot(gs1[2000-d3:2200-d2-d3,   0:100])
    ax42 = plt.subplot(gs1[2200-d3:2400-d2-d3,   0:100])
    ax43 = plt.subplot(gs1[1800-d3:2000-d2-d3,   100+d:200+d])
    ax44 = plt.subplot(gs1[2000-d3:2200-d2-d3,   100+d:200+d])
    ax45 = plt.subplot(gs1[2200-d3:2400-d2-d3,   100+d:200+d])

#    ax_list = [ax20,ax21,ax22,ax23,ax24,ax25, ax30,ax31,ax32,ax33,ax34,ax35, ax40,ax41,ax42,ax43,ax44,ax45]
    ax_list = [ax30,ax31,ax32,ax33,ax34,ax35, ax40,ax41,ax42,ax43,ax44,ax45]
    fs = 8
    color = ['blue','red','k','k']
    line_styles = ['-','-','-','--']
    for enum_MD, MD in enumerate(dim_data):
        x_shift = x_shift_list[enum_MD]
        tet_MD = tet_data[enum_MD]
        for enum, ax in enumerate(ax_list):
            enum = enum+ 6 + 6
            if enum_MD == 0:
                if what == 'gs':
                    leg = cgDNA_name[enum] + cgDNA_units[enum]
                elif what == 'stiff':
                    leg = cgDNA_name[enum] +' - '+cgDNA_name[enum]
                ax.set_ylabel(leg,fontsize=fs,labelpad = 8)
                ax.get_yaxis().set_label_coords(-0.12,0.5)
            else:
                leg = '_no_legend_'
            leg1 = '_no_legend_'
            tet_sub_list_dim = epi_dimer_in_tet_list(epi_tet_256,'MN')
            tet_sub_flank_label = [epi_tet_256[i][0] + '..' + epi_tet_256[i][3]  for i in tet_sub_list_dim]
            tet_sub_flank_label = [i.replace('M','X') for i in tet_sub_flank_label]
            tet_sub_flank_label = [i.replace('N','Z') for i in tet_sub_flank_label]
            small_x = np.arange(len(tet_sub_list_dim))+x_shift
            small_y = tet_MD[0][enum,tet_sub_list_dim]
            ax.scatter(small_x,small_y,label=leg1,s=1,color=color[enum_MD]) #color=color_16_RY[i2]
            ax.plot(   small_x,small_y,label=leg1,lw=0.15,color=color[enum_MD]) #color=color_16_RY[i2]

            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.set_xlim(-0.35,17.3)
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.set_xticklabels([])

            if what == 'gs':
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            elif what == 'stiff':
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax42.set_xticks(np.arange(len(tet_sub_flank_label)))
    ax42.set_xticklabels(tet_sub_flank_label,fontsize=fs,rotation=90,fontname='monospace')
    ax45.set_xticks(np.arange(len(tet_sub_flank_label)))
    ax45.set_xticklabels(tet_sub_flank_label,fontsize=fs,rotation=90,fontname='monospace')
    fig.savefig("./Plots/DNA_MDNA_HDNA_gs_MN_tet_context_"+what  +".pdf",dpi=600)
    plt.close()


    if prin == True:
        a1 = np.around( DNA_mon_data[1],3)
        b1 = np.around(MDNA_mon_data[1],3)
        c1 = np.around(HDNA_mon_data[1],3)
        for zi in range(5):
            tmp = tmp + ' & ' + str(mon5[zi])
        tmp = tmp + '\\\\'
        print(tmp)
        print('\\hline')
        print('\\hline')
        for yi in range(6):
            tmp = ''
            for zi in range(5):
                tmp = tmp + ' & ' + decimal_printer( str(a1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = '' + cgDNA_name[yi]
            for zi in range(5):
                tmp = tmp + ' & ' + decimal_printer( str(c1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)

            tmp = ''
            for zi in range(5):
                tmp = tmp + ' & ' + decimal_printer( str(b1[yi,zi]),3)
            tmp = tmp + '\\\\'
            print(tmp)
            print('\\hline')
    return None



#################################----------------------------------------------
###########---seq logo
#################################----------------------------------------------


def prob(arr):
    pro = np.zeros((4,2))

    for k1,k2 in zip([0,3],[0,1]):
        for tt in arr:
            if tt[k1] == 'A':
                pro[0,k2] = 1 + pro[0,k2]
            if tt[k1] == 'T':
                pro[1,k2] = 1 + pro[1,k2]
            if tt[k1] == 'C':
                pro[2,k2] = 1 + pro[2,k2]
            if tt[k1] == 'G':
                pro[3,k2] = 1 + pro[3,k2]
    pro = pro/len(arr)
    print("monomer logo for flank -----")
    return pro

def prob_16(arr):
    pro = np.zeros((16,1))
    for tt in arr:
        for i in range(16):
            if tt[0] + tt[3]  == dimer_16[i]:
                pro[i] = pro[i]+1
    pro = pro/len(arr)
    print("dimer logo for flanking -----")
    return pro


def prob_mid(arr):
    pro = np.zeros((16,1))
    for tt in arr:
        for i in range(16):
            if tt[1:3]  == dimer_16[i]:
                pro[i] = pro[i]+1
    pro = pro/len(arr)
    print("dimer logo for middle -----")
    return pro
import logomaker


def logo_plot(data,NA,ax,yaxis):

    if yaxis == "bits":
        H_i = -data*np.nan_to_num(np.log2(data))
        H_i = np.sum(H_i,axis=1)
        R_i = 2 - H_i
        for i in range(np.shape(R_i)[0]):
            data[i] = data[i]*R_i[i]
    else:
        None

    s = 6

    if NA == 'RNA':
        ppm = pd.DataFrame(data, columns=['A', 'U', 'C','G'])
        color_scheme_mon = {
            'A': 'green' ,
            'U': 'red' ,
            'C': 'blue' ,
            'G': 'orange'
        }
    else:
        ppm = pd.DataFrame(data, columns=['A', 'T', 'C','G'])
        color_scheme_mon = {
            'A': 'green' ,
            'T': 'red' ,
            'C': 'blue' ,
            'G': 'orange'
        }

    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False

    # create Logo object
    crp_logo = logomaker.Logo(ppm,
                              ax = ax,
                              color_scheme = color_scheme_mon,
                              shade_below=.5,
                              fade_below=.5,
                              font_name='Arial Rounded MT Bold')

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.ax.set_ylim(0,1.05)

    return ax


##################################---------------------------------------------
# plot Groovewidths
##################################---------------------------------------------
# In general, Major	Groove	wide	and	deep,	Minor	Groove	narrow	and	deep for BDNA
# In ADNA, Major Groove	narrow	and	deep,	Minor	Groove	wide	and	shallow
# in cgDNA+ model, interplay between A nd B DNA -- ?


def create_groovewidths_data(NA):
    if NA == 'DNA':
        ps = 'ps2_cgf'
    if NA == 'RNA':
        ps = 'ps_rna'
    if NA == 'HYB':
        ps = 'ps_hyb'
    seq_list = all_Nmers(10)
    orig_stdout = sys.stdout
    f = open('./Data/grooves_data/' + NA + 'grooves.txt','w')
    sys.stdout = f
    for seq in tqdm.tqdm(seq_list):
        seq = 'GC'+random_seq(4) + seq +random_seq(4)+ 'GC'
        Dmin, Dmax = GrooveWidths_CS(cgDNA(seq,ps).ground_state)
        print(seq,np.around(Dmin, 3),np.around(Dmax, 3))
    sys.stdout = orig_stdout
    f.close()
    return None

def initiate_groove_data():
    DNA_grooves = pd.read_csv('./Data/grooves_data/DNAgrooves.txt', engine='python',sep=" ",header=None)
    RNA_grooves = pd.read_csv('./Data/grooves_data/RNAgrooves.txt', engine='python',sep=" ",header=None)
    HYB_grooves = pd.read_csv('./Data/grooves_data/HYBgrooves.txt', engine='python',sep=" ",header=None)

    gDNA = {'name': 'DNA', 'seq':DNA_grooves[0], 'minor': DNA_grooves[1].to_numpy(), 'major': DNA_grooves[2].to_numpy(), 'diff': DNA_grooves[2].to_numpy() - DNA_grooves[1].to_numpy()}
    gRNA = {'name': 'RNA', 'seq':RNA_grooves[0], 'minor': RNA_grooves[1].to_numpy(), 'major': RNA_grooves[2].to_numpy(), 'diff': RNA_grooves[2].to_numpy() - RNA_grooves[1].to_numpy()}
    gHYB = {'name': 'HDR', 'seq':HYB_grooves[0], 'minor': HYB_grooves[1].to_numpy(), 'major': HYB_grooves[2].to_numpy(), 'diff': HYB_grooves[2].to_numpy() - HYB_grooves[1].to_numpy()}

    return gDNA, gRNA, gHYB

def plot_corr_major_minor():

    gDNA, gRNA, gHYB = initiate_groove_data()
    fig = plt.figure(figsize=(8,8))
    s = 10
    gs1 = gridspec.GridSpec(530, 100)
    gs1.update(left=0.09, right=0.98,top=0.95,bottom=0.1)
    ax0 = plt.subplot(gs1[  0:150,   0:100])
    ax1 = plt.subplot(gs1[190:340,   0:100])
    ax2 = plt.subplot(gs1[380:530,   0:100])
    ax_list = [ax0,ax1,ax2]
    NA = [gDNA, gRNA, gHYB]
    NA_name = ["DNA", "RNA", "HDR"]
    label_name = ["DNA", "RNA", "DRH"]
    for enum, ax in enumerate(ax_list):
        ax.scatter(NA[enum]['minor'],NA[enum]['major'],s=0.1)
        PC = scipy.stats.pearsonr(NA[enum]['minor'],NA[enum]['major'])
        PC = np.around(PC[0],2)
        ax.set_ylabel("Major Groove (in Å)",fontsize=s)
        ax.tick_params(axis='both', labelsize=s,   pad=3, length=3,width=0.5,direction= 'inout')
        ax.set_title(label_name[enum]+", PC = "+str(PC),fontsize=s)
    ax2.set_xlabel("Minor Groove (in Å)",fontsize=s)
    fig.savefig("./Plots/Grooves_DNA_RNA_HYB_P_correlation.png",dpi=600)

    plt.show()
    plt.close()
    return None


def plot_groovewidths_all():

    gDNA = pd.read_csv('./Data/grooves_data/DNAgrooves.txt', engine='python',sep=" ",header=None)
    gRNA = pd.read_csv('./Data/grooves_data/RNAgrooves.txt', engine='python',sep=" ",header=None)
    gHYB = pd.read_csv('./Data/grooves_data/HYBgrooves.txt', engine='python',sep=" ",header=None)
    gDNA.columns = ['seq','minor','major']
    gRNA.columns = ['seq','minor','major']
    gHYB.columns = ['seq','minor','major']

    for difference in [True,False]:
        fig = plt.figure(figsize=(4,6))
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.15, right=0.98,top=0.98,bottom=0.1)
        d = 3
        ax0 = plt.subplot(gs1[0:165,   0:100])
        ax1 = plt.subplot(gs1[165+d:335, 0:100],sharex=ax0)
        ax2 = plt.subplot(gs1[335+d:500, 0:100],sharex=ax0)

        ax_list = [ax0,ax1,ax2]
        Na = [gDNA, gRNA, gHYB]
        l,bins,fs = 1, 100,9
        count = 0
        label = ['DNA','RNA','DRH']
        for ax, NA in zip(ax_list,Na):
            ax.hist(NA['minor']-5.8,histtype = 'step', color='red' ,lw=l,bins=bins,density=True,label = label[count]+' '+'Minor')
            ax.hist(NA['major']-5.8,histtype = 'step', color='blue',lw=l,bins=bins,density=True,label = label[count]+' ''Major')
            ax.set_ylabel("Norm. hist.")
            ax.set_xlabel("groove widths in Å")
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.legend(fontsize=fs)
            count = count+1
        plt.show()
        fig.savefig("./Plots/Grooves_DNA_RNA_HYB.pdf",dpi=600)
    return None

def grooves_ind(NA,condition,groove_kind):
    cond_index = {}
    cond_len = len(condition[0])
    seq_len = len(NA['seq'][0])
    if groove_kind == 'major':
        half_seq_len = seq_len//2 - 1  ### includinde the middle base-pair
        for mid in tqdm.tqdm(condition):
            tmp = []
            for enum, seq in enumerate(NA['seq']):
                if mid == to_YR(seq[half_seq_len:half_seq_len+cond_len]):
                    tmp.append(enum)
            cond_index[mid] = tmp
    elif groove_kind == 'minor':
        half_seq_len = seq_len//2 + 1  ### includinde the middle base-pair
        for mid in tqdm.tqdm(condition):
            tmp = []
            for enum, seq in enumerate(NA['seq']):
                if mid == to_YR(seq[half_seq_len-cond_len:half_seq_len]):
                    tmp.append(enum)
            cond_index[mid] = tmp
    return cond_index

def plot_groovewidths_with_seq_condition(hist=False,box=False,violin=False):
    NA_name = ['DNA','RNA','HDR']
    gDNA, gRNA, gHYB = initiate_groove_data()
    tmp_cond = all_YR(5)
    condition = []
    # following loop is to sort R,Y steps in a desired manner
    for kk in ['RYY','RRY','RRR',  'RYR',  'YRY', 'YYY', 'YRR','YYR']:
        for tmp in tmp_cond:
            # if tmp[-2:] == kk:   ## last two elements
            if tmp[1:4] == kk:   ## middle 3 elements
                condition.append(tmp)

    cond_index_minor = grooves_ind(gDNA,condition,'minor')
    cond_index_major = grooves_ind(gDNA,condition,'major')
    box=True
    hist=True

    if hist==True:
        fig = plt.figure(figsize=(7,8))
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.1)
        ax0 = plt.subplot(gs1[0:160,   0:100])
        ax1 = plt.subplot(gs1[167:330, 0:100],sharex=ax0)
        ax2 = plt.subplot(gs1[337:500, 0:100],sharex=ax0)
        ax_list = [ax0,ax1,ax2]
        Na = [gDNA, gRNA, gHYB]
        l,ls, bins,fs = 1,0.3, 100,10
        for ax, NA in zip(ax_list,Na):
            sns.kdeplot(NA['minor'],color='red' ,linewidth=l,label = NA['name']+'_'+'Minor',ax=ax)
            sns.kdeplot(NA['major'],color='blue',linewidth=l,label = NA['name']+'_''Major' ,ax=ax)

            ax.set_ylabel("Norm. hist.")
            ax.set_xlabel("distance in Å")
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.legend(fontsize=fs)

            # for enum,mid in enumerate(dimer_16):
            #     sns.kdeplot(NA['minor'][cond_index_minor[mid]], color=color_16_RY[enum], label = '_no_legend_',linewidth=0.3,ax=ax)
            #     sns.kdeplot(NA['major'][cond_index_major[mid]], color=color_16_RY[enum], label = '_no_legend_',linewidth=0.3,ax=ax)

        plt.show()
        fig.savefig("./Plots/Grooves_DNA_RNA_HYB"  +".pdf",dpi=600)
        plt.close()

    if box==True:
        Na = [gDNA, gRNA, gHYB]
        for enum1,NA in enumerate(Na):
            fig = plt.figure(figsize=(7,8))
            gs1 = gridspec.GridSpec(500, 100)
            gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.1)
            ax0 = plt.subplot(gs1[0:220,    0:100])
            ax1 = plt.subplot(gs1[280:500,   0:100])
            ylabel = ['Minor groove width in Å', 'Major groove width in Å']
        #    ax_list = [ax0,ax1,ax2]
            data_min,data_max = [], []
            for enum,mid in enumerate(condition):
                data_min.append(NA['minor'][cond_index_minor[mid]])
                data_max.append(NA['major'][cond_index_major[mid]])
            ax0.boxplot(data_min)
            ax1.boxplot(data_max)

            for pnum,ax in enumerate([ax0,ax1]):
                ax.set_ylabel(ylabel[pnum])
                ax.set_xticks(1+np.arange(len(condition)))
                ax.set_xticklabels(condition,rotation=90)
            fig.savefig('./Plots/Grooves_box_plot_'+NA_name[enum1]+'.pdf',dpi=600)
            plt.show()
            plt.close()

    if violin==True:
        Na = [gDNA, gRNA, gHYB]
        for NA in Na:
            fig = plt.figure(figsize=(7,8))
            gs1 = gridspec.GridSpec(500, 100)
            gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.1)
            ax0 = plt.subplot(gs1[0:500,   0:100])
        #    ax1 = plt.subplot(gs1[167:330, 0:100],sharex=ax0)
        #    ax2 = plt.subplot(gs1[337:500, 0:100],sharex=ax0)

        #    ax_list = [ax0,ax1,ax2]
            data_min,data_max = {}, {}
            for enum,mid in enumerate(dimer_16):
                tmp1, tmp2 = np.empty(4**10), np.empty(4**10)
                tmp1[:], tmp2[:] = np.nan, np.nan
                tmp1[cond_index[mid]] =  NA['minor'][cond_index_minor[mid]]
                tmp2[cond_index[mid]] =  NA['major'][cond_index_major[mid]]
                data_min[mid] = tmp1
                data_max[mid] = tmp2
            data_min['Avg'] = NA['minor']
            data_max['Avg'] = NA['major']
            df_min = pd.DataFrame.from_dict(data_min)
            df_max = pd.DataFrame.from_dict(data_max)
            sns.violinplot(data=df_max,ax=ax0,palette="muted",linewidte=0.01)
            sns.violinplot(data=df_min,ax=ax0,palette="muted",linewidte=0.01)
#            ax0.set_xticks(np.arange(17))
#            ax0.set_xticklabels(dimer_17,fontsize=fs)
            for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)
            plt.show()
            fig.savefig("./Plots/Grooves_dimer_violin_"+ NA['name']  +".pdf",dpi=600)
            plt.close()

    return None

def compute_prob_list_of_seq(seqs,pos1,pos2):
    s = pos2-pos1
    prob_mat = np.zeros((s,4))  ### A,T,C,G
    for seq in seqs:
        for enum,base  in enumerate(seq[pos1:pos2]):
            if base == 'A':
                prob_mat[enum,0] = prob_mat[enum,0] + 1
            if base == 'T':
                prob_mat[enum,1] = prob_mat[enum,1] + 1
            if base == 'C':
                prob_mat[enum,2] = prob_mat[enum,2] + 1
            if base == 'G':
                prob_mat[enum,3] = prob_mat[enum,3] + 1
    prob_mat = prob_mat/np.sum(prob_mat[0,:])  ### division by numer of seq
    return prob_mat


def compute_prob_seq_logo(df,label):
    how_many = int(np.around(4**10*0.0015))  ## 0.15%
    minor_sort = np.argsort(df['minor'].to_numpy())
    major_sort = np.argsort(df['major'].to_numpy())

    minor_seq_list_low  = df['seq'].iloc[minor_sort[0:how_many]]
    minor_seq_list_high = df['seq'].iloc[minor_sort[-how_many:]]
    major_seq_list_low  = df['seq'].iloc[major_sort[0:how_many]]
    major_seq_list_high = df['seq'].iloc[major_sort[-how_many:]]
    count_TA = 0
    count_AA = 0
    count_AT = 0
    print(minor_seq_list_low,minor_seq_list_high,major_seq_list_low,major_seq_list_high)
    for seq in minor_seq_list_low:
        count_TA += count_appearances(seq[7:12],'TA')
        count_AA += count_appearances(seq[7:12],'AA') + count_appearances(seq[7:12],'TT')
        count_AT += count_appearances(seq[7:12],'AT')
    summ = count_TA+ count_AA+ count_AT
    print(count_TA/summ, count_AA/summ, count_AT/summ)
    prob_minor_low  = compute_prob_list_of_seq(minor_seq_list_low, 6,16) + 10**-6
    prob_minor_high = compute_prob_list_of_seq(minor_seq_list_high,6,16) + 10**-6
    prob_major_low  = compute_prob_list_of_seq(major_seq_list_low, 6,16) + 10**-6
    prob_major_high = compute_prob_list_of_seq(major_seq_list_high,6,16) + 10**-6

    fig,ax = plt.subplots(2,2,sharex = True,sharey = True,figsize=(4,2.9))
    logo_plot(prob_minor_low , label, ax[0,0],'bits')
    logo_plot(prob_minor_high, label, ax[1,0],'bits')
    logo_plot(prob_major_low , label, ax[0,1],'bits')
    logo_plot(prob_major_high, label, ax[1,1],'bits')
    fs =  8
    for i in range(2):
        for j in range(2):
            ax[i,j].tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax[i,j].set_ylim(0,2)
        ax[i,0].set_ylabel('bits',fontsize=fs)
        ax[1,i].set_xlabel('Index of base in decamer',fontsize=fs)
        ax[1,i].set_xticks(np.arange(10))
        ax[1,i].set_xticklabels(1+np.arange(10),fontsize=fs)
        ax[0,0].set_title("Narrowest minor grooves",fontsize=fs)
        ax[1,0].set_title("Widest minor grooves",fontsize=fs)
        ax[0,1].set_title("Narrowest major grooves",fontsize=fs)
        ax[1,1].set_title("Widest major grooves",fontsize=fs)
    plt.tight_layout()
    plt.show()
    fig.savefig("./Plots/"+label+"_groove_seq_logo.pdf",dpi=600)
    plt.close()
    return None

def plot_groovewidths_seq_logo():
    gDNA = pd.read_csv('./Data/grooves_data/DNAgrooves.txt', engine='python',sep=" ",header=None)
    gRNA = pd.read_csv('./Data/grooves_data/RNAgrooves.txt', engine='python',sep=" ",header=None)
    gHYB = pd.read_csv('./Data/grooves_data/HYBgrooves.txt', engine='python',sep=" ",header=None)
    gDNA.columns = ['seq','minor','major']
    gRNA.columns = ['seq','minor','major']
    gHYB.columns = ['seq','minor','major']
    compute_prob_seq_logo(gDNA,'DNA')
    compute_prob_seq_logo(gRNA,'RNA')
    compute_prob_seq_logo(gHYB,'HYB')
#    print(gDNA['seq'])
#    ['YR', 'RR', 'YY', 'RY']


#############################################################################
################################### KL Envelope ####################################
###################################################################################
def plot_envelope_KL(ax,mean=0,std=1,lw=0.5,color='red'):
    x_values = np.arange(-4, 4, 0.15)
    y_values = scipy.stats.norm(mean, std)

    ax.plot(x_values, y_values.pdf(x_values),lw,color=color)
    return ax

def KL_single(m1,s1,m2,s2):
    K = np.log(s2/s1) + -0.5 + (s1**2 + (m1-m2)**2)/(2*s2**2)
    return K

def envelope_KL():
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(1050, 900)
    gs1.update(left=0.09, right=0.98,top=0.92,bottom=0.1)

    ax00 = plt.subplot(gs1[1:250, 0:400])
    ax01 = plt.subplot(gs1[1:250, 500:900],sharex=ax00)

    ax10 = plt.subplot(gs1[400:650, 0:400])
    ax11 = plt.subplot(gs1[400:650, 500:900],sharex=ax00)

    ax20 = plt.subplot(gs1[800:1050, 0:400])
    ax21 = plt.subplot(gs1[800:1050, 500:900],sharex=ax00)

    what = [0.0001, 0.0005,0.001,0.005,0.01,0.05]
    ax_list = [ax00,ax01,ax10,ax11,ax20,ax21]
    around = [4,4,3,3,2,2]
    ss = [600000,600000,60000,60000,6000,6000]
    for ax,ar,KL,s in zip(ax_list,around,what,ss):
        count=0
        e = np.arange(-0.5,0.5,0.00001) + 0.000000001
        for i in range(s):
            e1 = random.choice(e)
            e2 = 1+ random.choice(e)
            u = 0.5*(KL_single(0,1,e1,e2) + KL_single(e1,e2,0,1))

            if KL == np.around(u,ar):
                count=count+1
                plot_envelope_KL(ax,e1,e2,0.02)
        print(count)

        plot_envelope_KL(ax,0,1,.025,color='k')
        ax.set_title("sym KL divergence = " + str(KL),fontsize=8)
    plt.show()
    fig.savefig("./Plots/KL_envelope.pdf",dpi=600)
    plt.close()
    return None

###################################################################################
#########  Difference between two MD protocols ####################################
###################################################################################

def compare_MD(DNA1,DNA2):
    A,B = [],[]
    for i in range(np.size(DNA2.seq)):
        A.append(difference_KL_sym(DNA1.choose_seq([i]),DNA2.choose_seq([i]),True))
        B.append(difference_Mahal_sym(DNA1.choose_seq([i]),DNA2.choose_seq([i]),True))
    return A,B

def compare_ps(DNA,ps1,ps2):
    A,B = [],[]
    for seq in DNA.seq:
        res1 = cgDNA(seq,ps1)
        res2 = cgDNA(seq,ps2)
        A.append(difference_KL_sym(res1,res2,True)[0,0])
        B.append(difference_Mahal_sym(res1,res2,True)[0,0])
    return A,B

def palin_training_err(data_tmp,ps):
    a,b= [],[]
    for seq in range(0,16):
        a.append(recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,DNA_sym[seq])[0,0])
        b.append(recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,DNA_sym[seq])[0,0])
    return a,b

def make_table(data,label):
    data = np.around(data,4)
    for i in range(len(data[0])):
        if i==0:
            print("Index" ,'&', label[0],'&', label[1], '&', label[2],'&',  label[3], '&',  label[4], '&',  label[5], '&', label[6],'&',  label[7], '&',  label[8], '&',  label[9], '\\\\')
            print("\\hline")
        print(i+1,'&', data[0][i],'&', data[1][i], '&', data[2][i],'&',  data[3][i], '&',  data[4][i], '&',  data[5][i], '&', data[6][i],'&',  data[7][i], '&',  data[8][i], '&',  data[9][i], '\\\\')
    print("\\hline")
    avg = np.average(data,axis=1)
    avg = np.around(avg,4)
    print("Average",'&', avg[0],'&', avg[1], '&', avg[2],'&',  avg[3], '&',  avg[4], '&',  avg[5], '&', avg[6],'&',  avg[7], '&',  avg[8], '&',  avg[9], '\\\\')



def make_table2(data,label):
    avg = np.average(data,axis=1)
    data = np.around(data,4)
    avg = np.around(avg,4)
    print("Index", '&', 1, '&', 2, '&', 3, '&', 4, '&', 5, '&', 6, '&', 7, '&', 8, '&', 9, '&', 10, '&', 11, '&', 12, '&', 13, '&', 14, '&', 15, '&', 16, '&', 'Avg' , '\\\\', "\\hline")
    for enum,i in enumerate(data):
        print(label[enum], '&', data[enum][0], '&', data[enum][1], '&', data[enum][2], '&', data[enum][3], '&', data[enum][4], '&', data[enum][5], '&', data[enum][6], '&', data[enum][7], '&', data[enum][8], '&', data[enum][9], '&', data[enum][10], '&', data[enum][11], '&', data[enum][12], '&', data[enum][13] , '&', data[enum][14] , '&', data[enum][15] , '&', avg[enum] , "\\\\" , "\\hline"             )


def print_palin_err(data_name,DNA_217):
    print("Index", '&', 1, '&', 2, '&', 3, '&', 4, '&', 5, '&', 6, '&', 7, '&', 8, '&', 9, '&', 10, '&', 11, '&', 12, '&', 13, '&', 14, '&', 15, '&', 16, '&', 17, '\\\\', "\\hline")
    print( '&',  '&',  '&',  '&', '&',  '&',  '&', 'SM', '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', '\\\\', "\\hline")

    for f in np.arange(1,11,1):
        mat3 = []
        data_path_tmp = data_name.replace('XX',str(f))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for seq in np.arange(0,17,1):
            mat3.append(palin_err_Mahal_sym(data_tmp.choose_seq([seq])))
        avg3 = np.around(np.mean(mat3),4)
        mat3 = np.around(mat3,4)
        print(f, '$\\mu$s', '&', mat3[0], '&', mat3[1], '&', mat3[2], '&', mat3[3], '&', mat3[4], '&', mat3[5], '&', mat3[6], '&', mat3[7], '&', mat3[8], '&', mat3[9], '&', mat3[10], '&', mat3[11], '&', mat3[12], '&', mat3[13], '&', mat3[14], '&', mat3[15], '&', mat3[16] , '\\\\', "\\hline")
    tmp = palin_err_Mahal_sym(DNA_217.choose_seq([0]))
    print(20, '$\\mu$s', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', np.around(tmp,4) , '\\\\', "\\hline")
    print( '&',  '&',  '&',  '&', '&',  '&',  '&', 'SKL', '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', '\\\\', "\\hline")

    for f in np.arange(1,11,1):
        mat4 = []
        data_path_tmp = data_name.replace('XX',str(f))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for seq in np.arange(0,17,1):
            mat4.append(palin_err_KL_sym(data_tmp.choose_seq([seq])))
        avg4 = np.around(np.mean(mat4),4)
        mat4 = np.around(mat4,4)

        print(f, '$\\mu$s', '&', mat4[0], '&', mat4[1], '&', mat4[2], '&', mat4[3], '&', mat4[4], '&', mat4[5], '&', mat4[6], '&', mat4[7], '&', mat4[8], '&', mat4[9], '&', mat4[10], '&', mat4[11], '&', mat4[12], '&', mat4[13], '&', mat4[14], '&', mat4[15], '&', mat4[16] , '\\\\', "\\hline")
    tmp = palin_err_KL_sym(DNA_217.choose_seq([0]))
    print(20, '$\\mu$s', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', np.around(tmp,4) , '\\\\', "\\hline")

    return None


def print_end_seq(DNA_ends):
    for i in range(15):
        print(4*i+1, '&', '${\Sfont', DNA_ends.seq[4*i], '}$', '&',  4*i+2, '&','${\Sfont', DNA_ends.seq[4*i+1], '}$', '&',  4*i+1+2, '&', '${\Sfont', DNA_ends.seq[4*i+2], '}$', '&',  4*i+1+3, '&', '${\Sfont', DNA_ends.seq[4*i+3], '}$',        '\\\\')


def hist_seq_training_lib(dna):
    seq_list = dna.choose_seq(np.arange(16)).seq
    mono = dict.fromkeys(['A','G'], 0)
    dimer_dict = dict.fromkeys(dimer_10, 0)
    trimer_list = ['AAA', 'AAT', 'AAC', 'AAG', 'TAA', 'TAT', 'TAC', 'TAG', 'CAA', 'CAT', 'CAC', 'CAG', 'GAA', 'GAT', 'GAC', 'GAG', 'AGA', 'AGT','AGC', 'AGG', 'TGA', 'TGT', 'TGC', 'TGG', 'CGA', 'CGT', 'CGC', 'CGG', 'GGA', 'GGT', 'GGC', 'GGG']
    trimer_dict = dict.fromkeys(trimer_list, 0)

    for seq in seq_list:
        for k in list(mono.keys()):
            mono[k] = mono[k] + seq[2:22].count(k)

        for u in list(dimer_dict.keys()):
            dimer_dict[u] = dimer_dict[u] + seq[2:22].count(u) + seq[2:22].count(comp(u))

        for u in list(trimer_dict.keys()):
            trimer_dict[u] = trimer_dict[u] + seq[1:23].count(u)  + seq[1:23].count(comp(u))

    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(500, 500)
    gs1.update(left=0.09, right=0.98,top=0.97,bottom=0.12)

    ax0 = plt.subplot(gs1[0 :215,    0:160])
    ax1 = plt.subplot(gs1[0 :215,    210:500])
    ax2 = plt.subplot(gs1[285 : 500, 0:500   ])

    ax0.bar(np.arange(2), list(mono.values()) , color = 'grey', width = 0.25)
    ax1.bar(np.arange(10), list(dimer_dict.values()) , color = 'grey', width = 0.25)
    ax2.bar(np.arange(32), list(trimer_dict.values())[0:32] , color = 'grey', width = 0.25)

    fs2 = 10
    ax0.set_xticks(np.arange(2))
    ax0.set_xticklabels(mono.keys(),fontsize=fs2)

    ax1.set_xticks(np.arange(10))
    ax1.set_xticklabels(dimer_dict.keys(),fontsize=fs2,rotation=90)


    ax2.set_xticks(np.arange(32))
    ax2.set_xticklabels(list(trimer_dict.keys())[0:32],fontsize=fs2,rotation=90)
    ax2.set_xlim(-0.5,31.5)

    ax_list = [ax0,ax1,ax2]
    for axl in ax_list:
        axl.tick_params(axis='both', labelsize=fs2, pad=3,length=3,width=0.5,direction= 'inout')

    ax0.text(-0.23, 1, '(a)', transform=ax0.transAxes, size=12)#, weight='bold')
    ax1.text(-0.12, 1, '(b)', transform=ax1.transAxes, size=12)#, weight='bold')
    ax2.text(-0.07, 1.02, '(c)', transform=ax2.transAxes, size=12)#, weight='bold')

    plt.show()
    fig.savefig("./Plots/hist_freq_DNA.pdf",dpi=600)
    plt.close()


    return None



##########---- plot stiffness matrix in 3 coordinates

##################################---------------------------------------------
# plot stencil in stiffness matrix
##################################---------------------------------------------

def fit_stencil_in_matrix_3types_combine(data,sym,save_name):
    if hasattr(data, 's1b'):
        if sym==True:
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
    seq = data.seq[0]
    wc = np.nan_to_num(wc)
    ind_inter = np.array([24*i+j+12 for i in range(23) for j in range(6)])
    ind_cg = np.array([12*i+j for i in range(24*2-1) for j in range(6)])
    ix_inter =  np.ix_(ind_inter,ind_inter)
    ix_cg    =  np.ix_(ind_cg,ind_cg)
    wc_cov = np.array(np.linalg.inv(wc))

    wc_inter = np.linalg.inv(wc_cov[ix_inter])
    wc_cg = np.linalg.inv(wc_cov[ix_cg])



##########----------------------------##################
    labelsize = 8
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 106)
    gs1.update(left=0.03, right=0.98,top=0.995,bottom=0.005)
    ax1 = plt.subplot(gs1[0:600,         0:30])
    ax2 = plt.subplot(gs1[0:600,        35:65])
    ax0 = plt.subplot(gs1[0:600,       70:106])

    x1,x2 = stencil_42(nbp)
    x6 = np.arange(0,24*nbp-12,6)
    shp = int(np.shape(wc)[0]/2)
    sns.heatmap(wc[0:shp,0:shp],ax=ax0,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.6,0.05
    for i in range(len(x1)):
        ax0.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax0.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax0.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        ax0.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        ax0.axhline(y=uu, color='k',linewidth=0.5)
        ax0.axvline(x=uu, color='k',linewidth=0.5)

#    for j in x6:
#        axr.axvline(x=j,color='black', lw=lws)
#        axr.axhline(y=j,color='black', lw=lws)
    ind = np.array([24*i+3 for i in range(int(nbp/2))])
#    ind[1::] = ind[1::]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    ax0.set_xticks(ind)
    ax0.set_yticks(ind)
    ax0.set_xticklabels(seq_ind)
    ax0.set_yticklabels(seq_ind)
    ax0.set_aspect('equal', adjustable='box')
    ax0.tick_params(axis='both', labelsize=labelsize, pad=3,length=3,width=0.5,direction= 'inout',rotation=0)
    ax0.grid(True)


##########----------------------------##################
    lwl,lws = 0.85,0.05
    x1,x2 = stencil_42(nbp,'inter')
    shp = int(np.shape(wc_inter)[0]/2)
    for i in range(len(x1)):
        ax1.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax1.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax1.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        ax1.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    sns.heatmap(wc_inter[0:shp,0:shp],ax=ax1,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})

    ind = [6*i for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    ax1.set_xticks(ind)
    ax1.set_yticks(ind)
    ax1.set_xticklabels(seq_ind)
    ax1.set_yticklabels(seq_ind)
    ax1.set_aspect('equal', adjustable='box')
    ax1.tick_params(axis='both', labelsize=labelsize, pad=3,length=3,width=0.5,direction= 'inout',rotation=0)
    for uu in [0,shp]:
        ax1.axhline(y=uu, color='k',linewidth=0.5)
        ax1.axvline(x=uu, color='k',linewidth=0.5)



##########----------------------------##################
    lwl,lws = 0.7,0.05
    shp = int(np.shape(wc_cg)[0]/2)
    sns.heatmap(wc_cg[0:shp,0:shp],ax=ax2,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    x1,x2 = stencil_42(nbp,'cg')
    for i in range(len(x1)):
        ax2.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax2.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax2.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        ax2.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        ax2.axhline(y=uu, color='k',linewidth=0.5)
        ax2.axvline(x=uu, color='k',linewidth=0.5)
    ind = [12*i+3 for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    ax2.set_xticks(ind)
    ax2.set_yticks(ind)
    ax2.set_xticklabels(seq_ind)
    ax2.set_yticklabels(seq_ind)
    ax2.set_aspect('equal', adjustable='box')
    ax2.tick_params(axis='both', labelsize=labelsize, pad=3,length=3,width=0.5,direction= 'inout',rotation=0)

    plt.savefig('./Plots/'+save_name+'_combine.pdf',dpi=600)
    plt.close()
    print('done')



def fit_stencil_in_matrix_3types(data,sym,save_name):
    if hasattr(data, 's1b'):
        if sym==True:
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
    seq = data.seq[0]
    wc = np.nan_to_num(wc)
    ind_inter = np.array([24*i+j+12 for i in range(23) for j in range(6)])
    ind_cg = np.array([12*i+j for i in range(24*2-1) for j in range(6)])
    ix_inter =  np.ix_(ind_inter,ind_inter)
    ix_cg    =  np.ix_(ind_cg,ind_cg)
    wc_cov = np.array(np.linalg.inv(wc))

    wc_inter = np.linalg.inv(wc_cov[ix_inter])
    wc_cg = np.linalg.inv(wc_cov[ix_cg])

##########----------------------------##################
    x1,x2 = stencil_42(nbp)
    x6 = np.arange(0,24*nbp-12,6)
    fig,axr = plt.subplots(1)
    shp = int(np.shape(wc)[0]/2)
    sns.heatmap(wc[0:shp,0:shp],ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.6,0.05
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        axr.axhline(y=uu, color='k',linewidth=0.5)
        axr.axvline(x=uu, color='k',linewidth=0.5)

#    for j in x6:
#        axr.axvline(x=j,color='black', lw=lws)
#        axr.axhline(y=j,color='black', lw=lws)
    ind = np.array([24*i+3 for i in range(int(nbp/2))])
#    ind[1::] = ind[1::]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    axr.set_xticks(ind)
    axr.set_yticks(ind)
    axr.set_xticklabels(seq_ind)
    axr.set_yticklabels(seq_ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')
    axr.grid(True)
    plt.savefig('./Plots/'+save_name+'_cg+.pdf',dpi=600)


##########----------------------------##################
    lwl,lws = 0.85,0.05
    fig,axr = plt.subplots(1)
    x1,x2 = stencil_42(nbp,'inter')
    shp = int(np.shape(wc_inter)[0]/2)
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    sns.heatmap(wc_inter[0:shp,0:shp],ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})

    ind = [6*i for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    axr.set_xticks(ind)
    axr.set_yticks(ind)
    axr.set_xticklabels(seq_ind)
    axr.set_yticklabels(seq_ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')
    for uu in [0,shp]:
        axr.axhline(y=uu, color='k',linewidth=0.5)
        axr.axvline(x=uu, color='k',linewidth=0.5)

    plt.savefig('./Plots/'+save_name+'_inter.pdf',dpi=600)
    plt.close()


##########----------------------------##################
    lwl,lws = 0.7,0.05
    fig,axr = plt.subplots(1)
    shp = int(np.shape(wc_cg)[0]/2)
    sns.heatmap(wc_cg[0:shp,0:shp],ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    x1,x2 = stencil_42(nbp,'cg')
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        axr.axhline(y=uu, color='k',linewidth=0.5)
        axr.axvline(x=uu, color='k',linewidth=0.5)
    ind = [12*i+3 for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    axr.set_xticks(ind)
    axr.set_yticks(ind)
    axr.set_xticklabels(seq_ind)
    axr.set_yticklabels(seq_ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')

    plt.savefig('./Plots/'+save_name+'_cg.pdf',dpi=600)
    plt.close()
    print('done')



def compare_seq_for_palindrome_article(d,save_name):
#    lss=['-', '-','-','--','--','--']
    lss=['-', '--','-','--',':',':']
    a = d.shape[19]
    b = d.shape[20]
    aseq = d.seq[19]
    bseq = d.seq[20]
    ar = cgDNA(aseq,'ps2_cgf').ground_state
    br = cgDNA(bseq,'ps2_cgf').ground_state

    ar2 = cgDNA(aseq,'dna_mle').ground_state
    br2 = cgDNA(bseq,'dna_mle').ground_state

    shp = [a,ar,b,br,ar2,br2]
    seq = [aseq]*6
    shp = [a,ar,b,br]
    seq = [aseq]*6
    color = ['red','blue','green','maroon','dodgerblue','limegreen']
    c1 = ['red','blue','green']
    c2 = ['maroon','dodgerblue','limegreen']

    fig, ax_list = compare_shape(shp,seq,save_name+'_19_20',lss,color=[c1,c1,c2,c2,c1,c2],type_of_var='cg+')
    s1 = 'GCGGATTACGCAGGC'
    s2 = 'GCGGATTCCGCAGGC'
    sf = list(s1)
    sf[7] = 'A/C'
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        print(seq_l[enum])
#        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=True)
#        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=8,minor=True)
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
#        ax_list[a].tick_params(axis='x', which='minor', pad=2)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
#        [t.set_color('maroon') for t in ax_list[a].xaxis.get_ticklabels(minor=True)]

    fig.savefig("./Plots/X" + save_name +".pdf",dpi=600)



def plot_persistence_length_palindrome_article(names):
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(100, 100)
    gs1.update(left=0.1, right=0.97,top=0.99,bottom=0.09)

    ax0 = plt.subplot(gs1[0:47,       0:100])
    ax1 = plt.subplot(gs1[53:100,      0:100])

    indices_dim = [0,3,6,9,12,14]
    color_dim = ['cyan','orange','red','blue','gray','magenta']
    count=0
    data,data_sp  = {},{}
    fs = 8
    for n in names:
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data[n] = read_persist_data_sub(path)
        data_sp[n], dimer_files = read_special_persist_data(path)
        print(names,max(data[n][0]),max(data[n][1]),min(data[n][0]),min(data[n][1]), )
    ax0.hist(data['DNA_BSTJ_MLE'][0],histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = "$\ell_{p}^{\mathcal{P}2}$"  ,density=1 )
    ax0.hist(data['DNA_BSTJ_MLE'][1],histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = "$\ell_{d}^{\mathcal{P}2}$" ,density=1  )
#            ['AA', 'TT', 'GC', 'CG', 'TC', 'CT', 'TG', 'GT', 'CC', 'GG', 'AC', 'CA', 'AG', 'GA', 'AT', 'TA']
    for j in range(6):
        ax0.scatter(data_sp['DNA_BSTJ_MLE'][indices_dim[j],0],[0.165],color=color_dim[j],marker='^', s=20 )
        ax0.scatter(data_sp['DNA_BSTJ_MLE'][indices_dim[j],1],[0.173],color=color_dim[j],marker='o',s=10 )
        print(dimer_files[indices_dim[j]], data_sp['DNA_BSTJ_MLE'][indices_dim[j],0],data_sp['DNA_BSTJ_MLE'][indices_dim[j],1])


    ax1.hist(data['DNA_BSTJ_CGF'][0] - data['DNA_BSTJ_MLE'][0] ,histtype = 'step',bins = 1000,color=colo[0] ,lw=1, label = "$\ell_{p}^{\mathcal{P}1} - \ell_{p}^{\mathcal{P}2}$"  ,density=1 )
    ax1.hist(data['DNA_BSTJ_CGF'][1] - data['DNA_BSTJ_MLE'][1] ,histtype = 'step',bins = 500,color=colo[0+1],lw=1, label = "$\ell_{d}^{\mathcal{P}1} - \ell_{d}^{\mathcal{P}2}$" ,density=1  )


    for j in range(6):
        ax1.scatter(data_sp['DNA_BSTJ_CGF'][indices_dim[j],0] - data_sp['DNA_BSTJ_MLE'][indices_dim[j],0],[0.39],color=color_dim[j],marker='^', s=20 )
        ax1.scatter(data_sp['DNA_BSTJ_CGF'][indices_dim[j],1] - data_sp['DNA_BSTJ_MLE'][indices_dim[j],1],[0.41],color=color_dim[j],marker='o',s=10 )
        print(dimer_files[j])
    ax0.scatter(np.mean(data['DNA_BSTJ_MLE'][0]),[0.004],color='k',marker='^', s=20 )
    ax0.scatter(np.mean(data['DNA_BSTJ_MLE'][1]),[0.004],color='k',marker='o',s=10 )
    ax1.scatter(np.mean(data['DNA_BSTJ_CGF'][0] - data['DNA_BSTJ_MLE'][0]),[0.009],color='k',marker='^', s=20 )
    ax1.scatter(np.mean(data['DNA_BSTJ_CGF'][1] - data['DNA_BSTJ_MLE'][1]),[0.009],color='k',marker='o',s=10 )

    ax1.set_xlim(0,35)
#    ax0.set_ylim(0,0.21)
    for ax in [ax0,ax1]:
        ax.legend(fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax1.set_xlabel("Base pair",fontsize=fs)


    ax0.set_yticks(np.arange(0,0.2,0.1))
    ax0.set_yticklabels(np.arange(0,0.2,0.1))
    ax1.set_yticks(np.arange(0,0.5,0.1))
    ax1.set_yticklabels(np.around(np.arange(0,0.5,0.1),1))

    fig.set_size_inches(3.25, 4)

    plt.savefig('./Plots/persistence_length_palin_article'+".pdf",dpi=600)
    plt.show()
    return data



####################################################################
###### Create and plot files for TTC correlation plot
####################################################################
def run_and_save_ttc(A,Drop_base,seq_id,label):
#    _,ttc,tt0 = A.MonteCarlo(1000000,Drop_base)
    _,ttc,tt0 = A.MonteCarlo(100000,Drop_base)
    tt = np.array(np.vstack([ttc,tt0])).T
    np.savetxt('./Data/'+label+'_'+str(seq_id+1)+'.txt',tt,fmt='%1.6f', delimiter=',')
    return tt

def print_TTC_files(NA,Drop_base,label):
#    for seq_id in range(23):
    for seq_id in range(17):
        MD = NA.choose_seq([seq_id])
        seq = MD.seq[0]
        A = cgDNA(seq,'ps2_cgf')
#        run_and_save_ttc(A,Drop_base,seq_id,label+'_ME')

        B = cgDNA(seq,'dna_mle')
#        run_and_save_ttc(B,Drop_base,seq_id,label+'_MLE')

        if comp(seq) == seq:
            print(seq_id, seq,'Palindromess--------')
            C = A
            C.ground_state =  MD.shape_sym[0]
#            C.stiff =  MD.s1b_sym[0]
#            run_and_save_ttc(B,Drop_base,seq_id,label+'_full_MD')
#            C.stiff =  MD.stiff_me_sym[0]
#            run_and_save_ttc(C,Drop_base,seq_id,label+'_trunc_MD')
            C.stiff =  MD.stiff_mre[0]
            run_and_save_ttc(C,Drop_base,seq_id,label+'_trunc_mre_MD')
        else:
            C = A
            C.ground_state =  MD.shape[0]
            C.stiff =  MD.s1b[0]
            run_and_save_ttc(C,Drop_base,seq_id,label+'_full_MD')

            C.stiff =  MD.stiff_me[0]
            run_and_save_ttc(C,Drop_base,seq_id,label+'_trunc_MD')
            print(seq_id, seq,'NOT Palindromess--------')
        print(seq_id,' --- Done' )

def insert_zero(df):
    df.loc[-1] = [0,0]
    df.index = df.index + 1
    df.sort_index(inplace=True)
    return df

def plot_TTC_files(NA,label,seq_id):
    path = '/Users/rsharma/Dropbox/cgDNAplus_py_rahul/Data/ttc_data/'
    P1 = pd.read_csv(path+'ttc_'+label+'_ME_'+str(seq_id)+'.txt',header=None)
    P2 = pd.read_csv(path+'ttc_'+label+'_MLE_'+str(seq_id)+'.txt',header=None)
    Gau = pd.read_csv(path+'ttc_'+label+'_full_MD_'+str(seq_id)+'.txt',header=None)
    T2 = pd.read_csv(path+'ttc_'+label+'_trunc_MD_'+str(seq_id)+'.txt',header=None)
    T1 = pd.read_csv(path+'sym_ttc_'+label+'_trunc_mre_MD_'+str(seq_id)+'.txt',header=None)

#    Unsym_df3 = pd.read_csv(path+'Unsym_ttc_'+label+'_full_MD_'+str(seq_id)+'.txt',header=None)
#    Unsym_df4 = pd.read_csv(path+'Unsym_ttc_'+label+'_trunc_MD_'+str(seq_id)+'.txt',header=None)

    test = pd.read_csv(path+'Alessandro_17_ME.txt',header=None)
    test = insert_zero(np.log(test))

    P1 = insert_zero(np.log(P1))
    P2 = insert_zero(np.log(P2))
    Gau = insert_zero(np.log(Gau))
    T2 = insert_zero(np.log(T2))
    T1 = insert_zero(np.log(T1))
#    Unsym_full  = insert_zero(np.log(Unsym_df3))
#    Unsym_trunc = insert_zero(np.log(Unsym_df4))

    md_f = pd.read_csv(path+'log_filter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
    md_unf = pd.read_csv(path+'log_unfilter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)

    md_f[1] = Gau[1]
    md_unf[1] = Gau[1]

    ### note that sym or unsym version of MD ttc plots are identical
    sym_md_f = pd.read_csv(path+'sym_log_filter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
    sym_md_unf = pd.read_csv(path+'sym_log_unfilter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
    sym_md_f[1] = Gau[1]
    sym_md_unf[1] = Gau[1]


    ind = np.arange(len(md_f)) + 1
    fig,ax = plt.subplots()
    label = ['$\mathcal{P}1$','$\mathcal{P}2$','$MD_{Gaussian}$','$MD_{filtered}$','$MD_{unfiltered}$']
    col = ['g','r','k','lime','royalblue']
    for enum,data in enumerate([P1,P2,Gau,sym_md_f,sym_md_unf]):
#    for enum,data in enumerate([ME,md_f]):
        mp, _ = np.polyfit(ind, data[0], 1)
        lp = str(np.around(-1/mp,1))
        md, _ = np.polyfit(ind, data[0]-data[1], 1)
        ld = str(np.around(-1/md,1))
        ax.plot(ind,data[0],label=label[enum]+' '+ ', $\ell_{p}$ = ' + lp + ', $\ell_{d}$ = ' + ld  ,color=col[enum],lw=0.5)
        ax.plot(ind,data[0]-data[1],ls='--',label='_no_legend_',color=col[enum],lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xticks(ind)
    ax.set_xticklabels(list(ind),fontsize=8)
    ax.set_title(NA.seq[seq_id-1])
    ax.set_ylabel(r"$ln langle rangle $")
    ax.tick_params(axis='both', labelsize=8,pad=3,length=3,width=0.5,direction= 'inout')
    plt.savefig('./Plots/ttc_plot_seq_'+str(seq_id)+'.pdf',dpi=600)
    None


def plot_TTC_files_all_ld(NA,label):
    nseq = 23
    ld_comp = np.zeros((nseq,5))
    for seq_id in range(1,nseq+1):
        path = '/Users/rsharma/Dropbox/cgDNAplus_py_rahul/Data/ttc_data/'
        P1 = pd.read_csv(path+'ttc_'+label+'_ME_'+str(seq_id)+'.txt',header=None)
        P2 = pd.read_csv(path+'ttc_'+label+'_MLE_'+str(seq_id)+'.txt',header=None)
        P1 = insert_zero(np.log(P1))
        P2 = insert_zero(np.log(P2))
        ### note that the sequcnes which are not palindromce -- sym is non-palindromce data
        sym_md_f = pd.read_csv(path+'sym_log_filter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
        sym_md_unf = pd.read_csv(path+'sym_log_unfilter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
        Gau = pd.read_csv(path+'ttc_'+label+'_full_MD_'+str(seq_id)+'.txt',header=None)
        Gau = insert_zero(np.log(Gau))
        sym_md_f[1] = Gau[1]
        sym_md_unf[1] = Gau[1]
        ind = np.arange(len(sym_md_f)) + 1
        for enum,data in enumerate([P1,P2,Gau,sym_md_f,sym_md_unf]):
    #    for enum,data in enumerate([ME,md_f]):
    #        mp, _ = np.polyfit(ind, data[0], 1)
    #        lp = str(np.around(-1/mp,1))
            md, _ = np.polyfit(ind, data[0]-data[1], 1)
            ld_comp[seq_id-1,enum] = np.around(-1/md,1)
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(100, 100)
    gs1.update(left=0.1, right=0.99,top=0.99,bottom=0.13)

    ax = plt.subplot(gs1[0:100,       0:100])
    ind = np.arange(nseq)
    lw = 0.75
    fs=10
    ax.plot(ind,ld_comp[:,0],label='$\ell_{d}^{\mathcal{P}1}$',color='blue',lw=lw)
    ax.plot(ind,ld_comp[:,1],label='$\ell_{d}^{\mathcal{P}2}$',color='red',lw=lw)
    ax.plot(ind,ld_comp[:,2],label='$\ell_{d}^{MD_{Gaussian}}$',color='k',lw=lw)
    ax.plot(ind,ld_comp[:,3],label='$\ell_{d}^{MD_{filtered}}$',color='green',lw=lw)
    ax.plot(ind,ld_comp[:,4],label='$\ell_{d}^{MD_{unfiltered}}$',color='green',ls='--',lw=lw)
    ax.legend(fontsize=fs,ncol=2)
    ax.set_xticks(ind)
    ax.set_xticklabels(['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','T1','T2','T3','T4','T5','T6','T7'],rotation=90)
    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_xlabel("Sequence index",fontsize=fs)
    ax.set_ylabel("Dynamic persistence length (in base-pairs)",fontsize=fs)
    plt.savefig('./Plots/compare_all_ld'+'.pdf',dpi=600)
    for i in range(5):
        for j in range(5):
            print(i,j,np.sqrt(np.mean(np.square(ld_comp[:,i] - ld_comp[:,j]))))
####################################################################
###### Above --> Create files for TTC correlation plot
####################################################################



####################################################################
###### Below --> Compare two truncation
####################################################################

def compare_two_trunc(data,seq_id,label):
    MD = data.choose_seq([seq_id])
#    wc = MD.stiff_mre[0] - MD.stiff_me_sym[0]
#    wc = MD.stiff_mre[0] -  MD.s1b_sym[0]
    res2 = cgDNA(MD.seq[0],'dna_mle')
    wc = res2.stiff.todense() - MD.stiff_mre[0]


    eig = np.linalg.eigvals(wc)
    print(eig)
    nbp = len(MD.seq[0])
    x1,x2 = stencil_42(nbp)

    x6 = np.arange(0,24*nbp-12,6)
    fig,axr = plt.subplots(1)
    sns.heatmap(wc,ax=axr,center=0,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.3,0.05
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for j in x6:
        axr.axvline(x=j,color='black', lw=lws)
        axr.axhline(y=j,color='black', lw=lws)

    ind = np.arange(0,24*nbp-18,12)
    axr.set_xticks(ind)
    axr.set_yticks(ind)
    axr.set_xticklabels(ind)
    axr.set_yticklabels(ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=6, pad=3,length=3,width=0.5,direction= 'inout')
    axr.grid(False)
    plt.savefig('./Plots/P2-Trunc2_seq_id_'+str(seq_id+1)+'.pdf',dpi=600)

def compare_two_trunc_KL(data):
    T1_T2, T2_T1, T1_T2_sym  = [], [], []
    T1_MD, MD_T1, T1_MD_sym  = [], [], []
    T2_MD, MD_T2, T2_MD_sym  = [], [], []
    T1_P1, P1_T1, T1_P1_sym  = [], [], []
    T2_P2, P2_T2, T2_P2_sym  = [], [], []
    MT1_P1, MP1_T1, MT1_P1_sym  = [], [], []
    MT2_P2, MP2_T2, MT2_P2_sym  = [], [], []


    for seq_id in range(16):
        MD = data.choose_seq([seq_id])
        P1 = cgDNA(MD.seq[0],'ps2_cgf')
        P2 = cgDNA(MD.seq[0],'dna_mle')
        ## difference between two truncation
        T1_T2.append(kl_mvn(MD.shape_sym[0], MD.stiff_me_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))
        T2_T1.append(kl_mvn(MD.shape_sym[0], MD.stiff_mre[0], MD.shape_sym[0], MD.stiff_me_sym[0]))
        T1_T2_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.stiff_me_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))

        ## T1 with MD
        T1_MD.append(kl_mvn(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))
        MD_T1.append(kl_mvn(MD.shape_sym[0], MD.stiff_mre[0], MD.shape_sym[0], MD.s1b_sym[0]))
        T1_MD_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))

        ## T2 with MD
        T2_MD.append(kl_mvn(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_me_sym[0]))
        MD_T2.append(kl_mvn(MD.shape_sym[0], MD.stiff_me_sym[0], MD.shape_sym[0], MD.s1b_sym[0]))
        T2_MD_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_me_sym[0]))

        ## T1 with P1
        T1_P1.append(kl_mvn(MD.shape_sym[0], MD.stiff_mre[0], P1.ground_state, P1.stiff.todense()))
        P1_T1.append(kl_mvn( P1.ground_state, P1.stiff.todense(), MD.shape_sym[0],  MD.stiff_mre[0]))
        T1_P1_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.stiff_mre[0], P1.ground_state, P1.stiff.todense()))

        ## T2 with P2
        T2_P2.append(kl_mvn(MD.shape_sym[0], MD.stiff_me_sym[0], P2.ground_state, P2.stiff.todense()))
        P2_T2.append(kl_mvn( P2.ground_state, P2.stiff.todense(), MD.shape_sym[0],  MD.stiff_me_sym[0]))
        T2_P2_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.stiff_me_sym[0], P2.ground_state, P2.stiff.todense()))

        ## T1 with P1
        MT1_P1.append(Mahal(MD.shape_sym[0], P1.ground_state, MD.stiff_mre[0]))
        MP1_T1.append(Mahal( P1.ground_state, MD.shape_sym[0], P1.stiff.todense()))
        MT1_P1_sym.append(Mahal_sym(MD.shape_sym[0], P1.ground_state, MD.stiff_mre[0], P1.stiff.todense()))

        ## T2 with P2
        MT2_P2.append(Mahal(MD.shape_sym[0], P2.ground_state, MD.stiff_mre[0]))
        MP2_T2.append(Mahal( P2.ground_state, MD.shape_sym[0], P2.stiff.todense()))
        MT2_P2_sym.append(Mahal_sym(MD.shape_sym[0], P2.ground_state, MD.stiff_mre[0], P2.stiff.todense()))


    for T in [T1_T2, T2_T1, T1_T2_sym, T1_MD, MD_T1, T1_MD_sym, T2_MD, MD_T2, T2_MD_sym, T1_P1, P1_T1, T1_P1_sym, T2_P2, P2_T2, T2_P2_sym,  MT1_P1, MP1_T1, MT1_P1_sym,  MT2_P2, MP2_T2, MT2_P2_sym]:
        TT = [str(np.around(i,4))+' & '  for i in T]
        TT.append(np.around(np.mean(T),4))
        TT = str(TT).replace("', '",'')
        TT = TT.replace("',","")
        TT = TT.replace("'","")
        TT = TT.replace("[","")
        TT = TT.replace("]","")
        print(TT)

    return None


def check_pos_def_prmset(ps,GC_padding,seq_kind):

    for N in range(8,13):
        if seq_kind == 'MDNA':
            seq_list = Met_all_Nmers(N)
        else:
            seq_list = all_Nmers(N)

        for seq in  tqdm.tqdm(seq_list):
            if GC_padding == True:
                res = cgDNA('GC'+seq+'GC',ps)
            else:
                try:    ##### this is required as when checking for non-GC ends, how to eliminated the one with XM or MX steps
                    res = cgDNA(seq,ps)
                except:
                    # print(seq)
                    continue

            if is_pos_def(res.stiff.todense()) == False:
                print(seq, ' ---- Not definite -----')

        print('all Nmers computed for N = ', N)


##################################---------------------------------------------
# Analysis hydroxy/methylation effect on GC islands
##################################---------------------------------------------

def stiffness_analysis_MDNA(DNA,MDNA,HDNA):

    M_D_eig = []
    H_D_eig = []
    H_M_eig = []

    for i in range(3):
        if i==0:
            M_D_eig.append(np.sort(np.linalg.eigvals(MDNA.s1b_sym[i] -  DNA.s1b_sym[0]  )))
            H_D_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b_sym[i] -  DNA.s1b_sym[0]  )))
            H_M_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b_sym[i] -  MDNA.s1b_sym[i] )))
        else:
            M_D_eig.append(np.sort(np.linalg.eigvals(MDNA.s1b[i]     -  DNA.s1b_sym[0] )))
            H_D_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b[i]     -  DNA.s1b_sym[0] )))
            H_M_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b[i]     -  MDNA.s1b[i]    )))

    ss = 24*20-18
    color = ['r','b','g']

    fs = 8
    data = [M_D_eig,H_D_eig, H_M_eig]
    fig,ax = plt.subplots(3,sharex=True)
    for k in range(3):
        for j in range(3):
            ax[k].scatter(np.arange(ss),data[k][j],s=2,color=color[j],label='# of +ve eigs = '+str(np.sum(np.array(data[k][j]) >= 0, axis=0)))
            ax[k].legend(fontsize=6.7,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.15))
            ax[k].tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
        ax[k].set_ylabel("Eigenvalues" , fontsize=fs)
        ax[k].set_xlim(-3,ss+2)

    ax[2].set_xlabel("Eigenvalue index", fontsize=fs)
    plt.tight_layout()
    plt.savefig('./Plots/stiffness_analysis_MDNA.pdf',dpi=600)
    return None



# a,b = GrooveWidths_CS(cgDNA(random_seq(22),'ps2_cgf').ground_state)
# print(a,b)


###########################################
###########################################
###############   CPG islands ############################
###########################################
def count_appearances(seq,subseq):
    sl = len(subseq)
    l = len(seq)
    count = 0
    for i in range(l):
        if subseq == seq[i:i+sl]:
            count += 1
    return count

def check_cpg_islands(seq):
    ## A CpG island is defined as a 200-bp region of DNA with a GC content higher than 50%
    ## and an observed CpG versus expected CpG ratio greater or equal to 0.6 (Gardiner-Garden and Frommer, 1987).
    ## CpG islands are characterized by CpG dinucleotide content of at least 60% of that which would be statistically expected (~4–6%)
    l = len(seq)
    G = count_appearances(seq,'G')
    C = count_appearances(seq,'C')
    GC = (G + C)/l
    obs_cpg =  count_appearances(seq,'CG')
    exp_cpg = (G*C)/l
    ratio = obs_cpg/exp_cpg
    if (GC > 0.5 and ratio > 0.6):
#        print("Yay -- cpg islands -- found --", "GC content, obs_cpg, exp_cpg, ratio = ", np.around(GC,2),np.around(obs_cpg,2),np.around(exp_cpg,2),np.around(ratio,2))
        result = True
#        print(seq)
    else:
#        print("NOT -- cpg islands -----------", "GC content, obs_cpg, exp_cpg, ratio = ", np.around(GC,2),np.around(obs_cpg,2),np.around(exp_cpg,2),np.around(ratio,2))
        result = False
    return result

def listToString(s):
    # initialize an empty string
    str1 = ""
    # return string
    return (str1.join(s))


def create_random_cpg_islands(nbr_seq,GC_content,seq_length,nbr_CpG,save_name):
    GC_content = int(np.round(GC_content*seq_length) )   ## number of C and G
    non_GC_content = seq_length - GC_content## non-GC base-pair
    non_GC_part = np.random.choice(mon[0:2],non_GC_content) ## list of non GC bases
    GC_part = np.random.choice(mon[2:4],GC_content)
    tmp_seq = list(non_GC_part) + list(GC_part)  ## list of all the bases...
    ### now from tmp_seq ... check whether a random shuffle is CPG islands
    f = open('CpG_islands/'+ save_name + ".txt","w")
    count = 0
    while count < nbr_seq:
        np.random.shuffle(tmp_seq)   ## shuffle the seq
        seq = listToString(tmp_seq)
        if check_cpg_islands(seq) == True:
            if count_appearances(seq,'CG') == nbr_CpG:
                count = count +1
                print(count)
                f.write(seq+'\n')
    f.close()
    return None


def create_random_not_cpg_islands(nbr_seq,seq_length,nbr_CpG,save_name):
    GC_content = np.random.choice([0.3,0.35,0.4,0.45,0.5])
    GC_content = int(np.round(GC_content*seq_length) )   ## number of C and G
    non_GC_content = seq_length - GC_content## non-GC base-pair
    non_GC_part = np.random.choice(mon[0:2],non_GC_content) ## list of non GC bases
    GC_part = np.random.choice(mon[2:4],GC_content)
    tmp_seq = list(non_GC_part) + list(GC_part)  ## list of all the bases...
    f = open('CpG_islands/'+ save_name + ".txt","w")
    count = 0
    while count < nbr_seq:
        np.random.shuffle(tmp_seq)   ## shuffle the seq
        seq = listToString(tmp_seq)
        if check_cpg_islands(seq) == False:
            if count_appearances(seq,'CG') == nbr_CpG:
                count = count +1
                print(count)
                f.write(seq+'\n')
        else:
            print("Wow random CpG islands found --")
    f.close()
    return None


def replacer(seq, newstring, index):
    l = len(newstring)
    return seq[:index] + newstring + seq[index + l:]

def random_methylate_seq(seq,percent,what):
    pos = [match.start() for match in re.finditer('CG', seq)]
    nbr_cpg = len(pos)
    how_many = int(nbr_cpg*percent)
    pos_to_modify = np.random.choice(pos,how_many,replace=False)
#    print(pos_to_modify)
    for p in pos_to_modify:
        seq = replacer(seq,np.random.choice(what),p)
#    print( [match.start() for match in re.finditer('MN', seq)])
    return seq


def methylate_files(file_name,percent,what,save_name):
    seq_list = pd.read_csv('CpG_islands/'+ file_name + ".txt",header=None)
    nbr_seq = seq_list.shape[0]
    for i in range(nbr_seq):
        seq_list.loc[i] = random_methylate_seq(seq_list.loc[i][0],percent,what)
    seq_list.to_csv('CpG_islands/'+ file_name[:-2] + '_' + save_name + ".txt",header=False, index=False)
    return seq_list

def read_and_rewrite_cpg_island_data():
    path = '/Users/rsharma/Dropbox/cgDNAplus_py_rahul/CpG_islands/'
    ### following data is provided by Daiva  ### each file contains 3389 sequences....
    cpg     = pd.read_csv(path+'cpg_island_200mer_seqs.txt'    ,header = None)
    not_cpg = pd.read_csv(path+'not_cpg_island_200mer_seqs.txt',header = None)
    new_cpg =[]
    new_not_cpg = []

    for i in range(3389):
        if check_cpg_islands(cpg[0].loc[i]) == True:
            new_cpg.append(cpg[0].loc[i])
        else:
            new_not_cpg.append(cpg[0].loc[i])

        if check_cpg_islands(not_cpg[0].loc[i]) == True:
            new_cpg.append(not_cpg[0].loc[i])
        else:
            new_not_cpg.append(not_cpg[0].loc[i])
    new_cpg = pd.DataFrame(new_cpg,columns=[0])
    new_not_cpg = pd.DataFrame(new_not_cpg,columns=[0])
    new_cpg.to_csv('CpG_islands/Human_CpG_islands_modified_0.txt',header=False, index=False)
    new_not_cpg.to_csv('CpG_islands/Human_NOT_CpG_islands_modified_0.txt',header=False, index=False)

    return new_cpg,new_not_cpg

##############################################################################
##############################################################################
######################## SENSITIVITY ANALYSIS ################################
##############################################################################
##############################################################################

def compute_base_prob_list_of_seq(seqs,pos1,pos2):

    prob_mat = np.zeros((pos2-pos1,4))  ### A,T,C,G
    for seq in seqs:
        for enum,base  in enumerate(seq[pos1:pos2]):
            if base == 'A':
                prob_mat[enum,0] = prob_mat[enum,0] + 1
            if base == 'T':
                prob_mat[enum,1] = prob_mat[enum,1] + 1
            if base == 'C':
                prob_mat[enum,2] = prob_mat[enum,2] + 1
            if base == 'G':
                prob_mat[enum,3] = prob_mat[enum,3] + 1
    prob_mat = prob_mat/np.sum(prob_mat[0,:])  ### division by numer of seq
    return prob_mat


def create_seq_list_sensitivity_analysis_epi():
    ttl = ttl_256()
    rand = 'GTCG' ### np.random.choice(ttl) ## this is randolmy chosen but kept fixed
    df = pd.DataFrame([],columns=['seq'])
    count = 0
    for i in ttl:
        for j in ttl:
            df.loc[count] = 'GC' + rand + i + 'CG' + j + rand + 'GC'
            count = count+1
    df.to_csv('CpG_islands/All_decamers_with_central_CpG.txt',header=False, index=False)
    return df

def epi_sensitivity_analysis_groundstate(what,ps,metric):
    seq_list = pd.read_csv('CpG_islands/All_decamers_with_central_CpG.txt',header=None)
    try:
        diff = pd.read_csv('CpG_islands/total_groundstate_change_CG_to_'+what+'_'+metric+'.txt',header=None)
    except:
        nbr_seq = seq_list.shape[0]
        diff = np.zeros(nbr_seq)
        if what in ['MN','MG']:
            w = what
        elif what in ['HK']:
            w = 'MN'
        elif what in ['HG']:
            w = 'MG'
        for i in range(nbr_seq):
            print(i)
            seq  = seq_list.loc[i][0]
            mseq = replacer(seq, w, 10)

            gs_tmp  = cgDNA(seq,ps)
            gs = gs_tmp.ground_state
            mgs_tmp = cgDNA(mseq,ps)
            mgs = mgs_tmp.ground_state
            if metric == 'abs':
                diff[i] = np.sum(np.abs(gs-mgs))
            elif metric == 'Mahal':
                gs_stiff  =  gs_tmp.stiff.todense()
                mgs_stiff = mgs_tmp.stiff.todense()
                diff[i] = Mahal_sym(gs,mgs,gs_stiff,mgs_stiff)
        diff = pd.DataFrame(diff,columns=[0])
        diff.to_csv('CpG_islands/total_groundstate_change_CG_to_'+what+'_'+metric+'.txt',header=False, index=False)
    return seq_list, diff


def plot_seq_logo_epi_sensitivity(what,ps,metric):
    lpos1,lpos2 = 6,10   #### tetramer on left
    rpos1,rpos2 = 12,16   #### tetramer on right

    seq_list, diff  = epi_sensitivity_analysis_groundstate(what,ps,metric)
    diff = diff[0].to_numpy()
    sort_diff = np.argsort(diff)
    how_many = int(4**8*0.005) # 655
    print(np.sort(diff)*510)
    ind_min = sort_diff[:how_many]
    ind_max = sort_diff[-how_many:]
    CG =  np.zeros((2,4)) + 0.25
    ldata_min = compute_base_prob_list_of_seq(seq_list[0].loc[ind_min],lpos1,lpos2) + 10**-6
    rdata_min = compute_base_prob_list_of_seq(seq_list[0].loc[ind_min],rpos1,rpos2) + 10**-6   ### this small number is added to avoid zero probability
    data_min  = np.concatenate([ldata_min,CG,rdata_min])
    ldata_max = compute_base_prob_list_of_seq(seq_list[0].loc[ind_max],lpos1,lpos2) + 10**-6
    rdata_max = compute_base_prob_list_of_seq(seq_list[0].loc[ind_max],rpos1,rpos2) + 10**-6   ### this small number is added to avoid zero probability
    data_max  = np.concatenate([ldata_max,CG,rdata_max])

    fig,ax = plt.subplots(2)
    logo_plot(data_min,'IC',ax[0],'bits')
    logo_plot(data_max,'IC',ax[1],'bits')

    lab = ['X4','X3','X2','X1','C','G','Y1','Y2','Y3','Y4']
    fs = 15
    for i in range(2):
        ax[i].set_ylim(0,2)
        ax[i].tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')
        ax[i].set_xticks(np.arange(10))
        ax[i].set_ylabel('bits',fontsize=fs)
        ax[i].set_yticks(np.arange(0,2.5,0.5))
        ax[i].set_yticklabels(np.arange(0,2.5,0.5),fontsize=fs)

    ax[0].set_xticklabels([],fontsize=fs)
    ax[1].set_xticklabels(lab,fontsize=fs)
    ax[0].set_title('Sequences with least change',fontsize=fs)
    ax[1].set_title('Sequences with most change',fontsize=fs)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    fig.savefig('./Plots/epi_sensitivity_groundstate_seq_logo_'+what+'_'+metric+'.pdf',dpi=600)
    plt.show()
    plt.close()
    return None



def compare_epi_sensitivity_on_groundstate(s1,s2,modi,modi_pos,ps,save_name):
#    lss=['-', '-','-','--','--','--']

    s1m = replacer(s1, modi, modi_pos)
    s2m = replacer(s2, modi, modi_pos)

    s1_res  = cgDNA(s1, ps).ground_state
    s2_res  = cgDNA(s2, ps).ground_state
    s1m_res = cgDNA(s1m,ps).ground_state
    s2m_res = cgDNA(s2m,ps).ground_state

    shp = [s1_res,s1m_res]
    seq = [s1]*4
    lss=['-', '--','-','--']
    color = [c1,c1,c4,c4]
    fig,ax_list = compare_shape(shp,seq,save_name,lss,color=color,type_of_var='cg+')
    sf = list(s1m)
    sf = list(map(lambda x: x.replace('M', 'C/M'), sf))
    sf = list(map(lambda x: x.replace('N', 'G/N'), sf))
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/'+save_name+modi+'_1.pdf',dpi=600)
    plt.close()

    shp = [s2_res,s2m_res]
    seq = [s2]*4
    lss=['-', '--','-','--']
    color = [c1,c1,c4,c4]
    fig,ax_list = compare_shape(shp,seq,save_name,lss,color=color,type_of_var='cg+')
    sf = list(s2m)
    sf = list(map(lambda x: x.replace('M', 'C/M'), sf))
    sf = list(map(lambda x: x.replace('N', 'G/N'), sf))
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/'+save_name+modi+'_2.pdf',dpi=600)
    plt.close()

##############################################################################
##############################################################################
######################### EPI PERSISTENCE LENGTH CODE ###################
##############################################################################
##############################################################################
def map_seq_file_list_epi_persistence():
    old_to_new_map = {}
    new_to_old_map = {}
    count = 700
    for s in range(1,8):
        count += 1
        fo = 'Sublist_'+str(s)+'_0.txt'
        tmp = pd.read_csv('./CpG_islands/'+fo,header=None)
        fn = 'Sublist_'+ str(count)+ ".txt"
        tmp.to_csv('CpG_islands/'+fn,header=False, index=False)
        old_to_new_map[fo] = fn
        new_to_old_map[fn] = fo
    for s in range(1,6):
        for p,i in zip([0.25,0.50,0.75,1],[1,2,3,4]):
            for mod in ['MN','MG','MG_MN']:
                count += 1
                fo = 'Sublist_'+str(s)+'_' +str(i) +'_'+mod+'.txt'
                tmp = pd.read_csv('./CpG_islands/'+fo,header=None)
                fn = 'Sublist_'+ str(count)+ ".txt"
                tmp.to_csv('CpG_islands/'+fn,header=False, index=False)
                old_to_new_map[fo] = fn
                new_to_old_map[fn] = fo

    for s in range(6,8):
        for p,i in zip([0.50,1],[2,4]):
            for mod in ['MN','MG','MG_MN']:
                count += 1
                fo = 'Sublist_'+str(s)+'_' +str(i) +'_'+mod+'.txt'
                tmp = pd.read_csv('./CpG_islands/'+fo,header=None)
                fn = 'Sublist_'+ str(count)+ ".txt"
                tmp.to_csv('CpG_islands/'+fn,header=False, index=False)
                old_to_new_map[fo] = fn
                new_to_old_map[fn] = fo

    for p,i in zip([0.25,0.50,0.75,1],[1,2,3,4]):
        for mod in ['MN','MG','MG_MN']:
            count += 1
            fo = 'Human_CpG_islands_modified_' +str(i) +'_'+mod+'.txt'
            tmp = pd.read_csv('./CpG_islands/' + fo,header=None)
            fn = 'Sublist_'+ str(count)+ ".txt"
            tmp.to_csv('CpG_islands/'+fn,header=False, index=False)
            old_to_new_map[fo] = fn
            new_to_old_map[fn] = fo

    for p,i in zip([0.50,1],[2,4]):
        for mod in ['MN','MG','MG_MN']:
            count += 1
            fo = 'Human_NOT_CpG_islands_modified_' +str(i) +'_'+mod+'.txt'
            tmp = pd.read_csv('./CpG_islands/'+fo,header=None)
            fn = 'Sublist_'+ str(count)+ ".txt"
            tmp.to_csv('CpG_islands/'+fn,header=False, index=False)
            old_to_new_map[fo] = fn
            new_to_old_map[fn] = fo

    count += 1
    fo = 'Human_NOT_CpG_islands_modified_0.txt'
    tmp = pd.read_csv('./CpG_islands/'+fo,header=None)
    fn = 'Sublist_'+ str(count)+ ".txt"
    tmp.to_csv('CpG_islands/'+fn,header=False, index=False)
    old_to_new_map[fo] = fn
    new_to_old_map[fn] = fo
    count += 1
    fo = 'Human_CpG_islands_modified_0.txt'
    tmp = pd.read_csv('./CpG_islands/'+fo,header=None)
    fn = 'Sublist_'+ str(count)+ ".txt"
    tmp.to_csv('CpG_islands/'+fn,header=False, index=False)
    old_to_new_map[fo] = fn
    new_to_old_map[fn] = fo

    return old_to_new_map,new_to_old_map

def tmp_load_data(mapping,s,mod,data):
    if s in range(1,6):
        p_list = np.arange(1,5)
    elif s in [6,7]:
        p_list = [2,4]

    what = []
    what.append(mapping['Sublist_'+str(s)+'_0.txt'])
    for p in p_list:
        what.append(mapping['Sublist_'+str(s)+'_'+str(p)+'_'+mod+'.txt'])


    stat = np.zeros((len(what),4))
    for enum,i in enumerate(what):
        mean = data[int(i)].mean()
        std = data[int(i)].std()
        stat[enum,0] = mean[0]
        stat[enum,1] = std[0]
        stat[enum,2] = mean[1]
        stat[enum,3] = std[1]
    return stat

def load_persistence_length_data_epi(path):
    MP, HP = {},{}
    onmap, _ = map_seq_file_list_epi_persistence()
    for i in onmap.keys():
        onmap[i] = onmap[i][8:11]
    for i in range(701,799):
        MP[i] = pd.read_csv(path+'MDNA/'+'results_'+str(i)+'.txt',header=None,delimiter=' ')
        HP[i] = pd.read_csv(path+'HDNA/'+'results_'+str(i)+'.txt',header=None,delimiter=' ')

    stat_MN, stat_MG = {}, {}
    stat_HK, stat_HG = {}, {}

    for i in range(1,8):
        stat_MN[i] = tmp_load_data(onmap,i,'MN',MP)
        stat_HK[i] = tmp_load_data(onmap,i,'MN',HP)
        stat_MG[i] = tmp_load_data(onmap,i,'MG',MP)
        stat_HG[i] = tmp_load_data(onmap,i,'MG',HP)

    fig,ax = plt.subplots(4,7,sharex=True,figsize=(13,7))
    ind = np.arange(5)
    percent = [0,25,50,75,100]
    fs2 = 13
    tit_percent = [1.3, 2.7, 5.8, 8.9,11.6,14.7,17.4]
    lab_MN = ['MN']*7+['_no_legened_']*6
    lab_MG = ['MG']*7+['_no_legened_']*6
    lab_HK = ['HK']*7+['_no_legened_']*6
    lab_HG = ['HG']*7+['_no_legened_']*6
    for enum,k in enumerate([7,6,1,2,3,4,5]):
        ind1 = ind-0.1
        ind2 = ind+0.1
        if k > 5:
            ind1 = ind1[::2]
            ind2 = ind2[::2]
        ax[0,enum].errorbar(ind1,stat_MN[k][:,0],yerr = stat_MN[k][:,1],fmt='o',label=lab_MN[enum])
        ax[1,enum].errorbar(ind1,stat_MN[k][:,2],yerr = stat_MN[k][:,3],fmt='o',label=lab_MN[enum])
        ax[0,enum].errorbar(ind2,stat_HK[k][:,0],yerr = stat_HK[k][:,1],fmt='o',label=lab_HK[enum])
        ax[1,enum].errorbar(ind2,stat_HK[k][:,2],yerr = stat_HK[k][:,3],fmt='o',label=lab_HK[enum])

        ax[2,enum].errorbar(ind1,stat_MG[k][:,0],yerr = stat_MG[k][:,1],fmt='o',label=lab_MG[enum])
        ax[3,enum].errorbar(ind1,stat_MG[k][:,2],yerr = stat_MG[k][:,3],fmt='o',label=lab_MG[enum])
        ax[2,enum].errorbar(ind2,stat_HG[k][:,0],yerr = stat_HG[k][:,1],fmt='o',label=lab_HG[enum])
        ax[3,enum].errorbar(ind2,stat_HG[k][:,2],yerr = stat_HG[k][:,3],fmt='o',label=lab_HG[enum])

        ax[0,enum].set_ylim(180,231)
        ax[1,enum].set_ylim(221,234)
        ax[2,enum].set_ylim(180,231)
        ax[3,enum].set_ylim(221,234)
        for u in range(4):
            if enum > 0:
                ax[u,enum].set_yticklabels([])
            else:
                ax[u,enum].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax[u,enum].tick_params(axis='x', labelsize=fs2, pad=3,length=4.5,width=1,direction= 'inout',rotation=45)
            ax[u,enum].tick_params(axis='y', labelsize=fs2, pad=3,length=4.5,width=1,direction= 'inout',rotation=45)
            ax[u,enum].legend(fontsize=fs2,frameon=1,ncol=2,bbox_to_anchor=(0.5, 1.1),loc=10,columnspacing=1,handlelength=0.3)
        ax[0,enum].set_xticks(ind)
        ax[0,enum].set_xticklabels(percent,fontsize=fs2,rotation='vertical')
        ax[0,0].set_ylabel('$\ell_{p}$ (in bps)',fontsize=fs2,rotation=90)
        ax[2,0].set_ylabel('$\ell_{p}$ (in bps)',fontsize=fs2,rotation=90)

        ax[1,0].set_ylabel('$\ell_{d}$ (in bps)',fontsize=fs2,rotation=90)
        ax[3,0].set_ylabel('$\ell_{d}$ (in bps)',fontsize=fs2,rotation=90)

        ax[3,enum].set_xlabel('% C$_p$G \n modification',fontsize=fs2)
        ax[0,enum].set_title(  str(tit_percent[enum])+'% C$_p$G steps',fontsize=fs2,pad=25,fontweight='bold')

    plt.tight_layout(w_pad=-0.1,h_pad=-1)
    plt.show()
    fig.savefig('./Plots/persis_length_epi.pdf',dpi=600)
    plt.close()


##############################################################################
##############################################################################
######################### EPI Grooves width ###################
##############################################################################
##############################################################################
def create_groovewidths_data_epi(what,ps):
    seq_list = pd.read_csv('CpG_islands/All_decamers_with_central_CpG.txt',header=None)
    try:
        df = pd.read_csv('./Data/grooves_data/Epi_grooves_data_'+what+'.txt')
    except:
        nbr_seq = seq_list.shape[0]
        D_width = np.zeros((nbr_seq,6))
        w1 = 'MN'
        w2 = 'MG'
        for i in tqdm.tqdm(range(nbr_seq)):
            seq  = seq_list.loc[i][0]
            mseq1 = replacer(seq, w1, 10)
            mseq2 = replacer(seq, w2, 10)
            D_width[i,0], D_width[i,1]  = GrooveWidths_CS(cgDNA(seq,ps).ground_state)
            D_width[i,2], D_width[i,3] = GrooveWidths_CS(cgDNA(mseq1,ps).ground_state)
            D_width[i,4], D_width[i,5] = GrooveWidths_CS(cgDNA(mseq2,ps).ground_state)
        col = ['minor','major','MN_minor','MN_major','MG_minor','MG_major']
        df = pd.DataFrame(np.around(D_width,3),columns=col,index = seq_list[0])
        df.to_csv('./Data/grooves_data/Epi_grooves_data_'+what+'.txt')
    return df

def create_groovewidths_data_epi_cpg(what,ps):
    sub_seq_list = ["".join(item) for item in itertools.product(['CG',what], repeat=5)]
    seq_list = []
    nbr_seq = len(sub_seq_list)
    D_width = np.zeros((nbr_seq,2))
    for i in tqdm.tqdm(range(nbr_seq)):
#        seq  = 'GC' + 'TGGT' + sub_seq_list[i] + 'ACTG' + 'GC'
        seq  = 'GC' + 'TGTG' + sub_seq_list[i] + 'CATG' + 'GC'
        seq_list.append(seq)
        D_width[i,0], D_width[i,1]  = GrooveWidths_CS(cgDNA(seq,ps).ground_state)
    col = ['minor','major']
    df = pd.DataFrame(np.around(D_width,3),columns=col,index = seq_list)
    prob_min = {0:[],1:[],2:[],3:[],4:[],5:[]}
    prob_maj = {0:[],1:[],2:[],3:[],4:[],5:[]}

    for seq in df.index:
        c = count_appearances(seq,what)
        prob_min[c].append(df.loc[seq]['minor'])
        prob_maj[c].append(df.loc[seq]['major'])
    mat = np.zeros((6,5))
    for i in range(6):
        mat[i,0] = i
        mat[i,1] = np.mean(prob_min[i])
        mat[i,2] = np.std(prob_min[i])
        mat[i,3] = np.mean(prob_maj[i])
        mat[i,4] = np.std(prob_maj[i])
    return df, mat


def create_groovewidths_data_epi_cpg2(what,ps,position,groove):
    sub_seq_list = []
    sub_seq_list_MN = []
    m = ['A','T','C','G']
    for i in m:
        for j in m:
            for k in m:
                if position == 0:
                    sub_seq_list.append('CG'+i+j+k)
                    sub_seq_list_MN.append(what+i+j+k)
                elif position == 1:
                    sub_seq_list.append(i+'CG'+j+k)
                    sub_seq_list_MN.append(i+what+j+k)
                elif position == 2:
                    sub_seq_list.append(i+j+'CG'+k)
                    sub_seq_list_MN.append(i+j+what+k)
                elif position == 3:
                    sub_seq_list.append(i+j+k+'CG')
                    sub_seq_list_MN.append(i+j+k+what)
    seq_list = []
    nbr_seq = len(sub_seq_list)
    D_width = np.zeros((nbr_seq,4))
    for i in tqdm.tqdm(range(nbr_seq)):
        if groove == 'minor':
            tmp = ''.join(np.random.choice(['A','T','C','G'],4).tolist())
            tmp1 = ''.join(np.random.choice(['A','T','C','G'],1).tolist())
        elif groove == 'major':
            tmp1 = ''.join(np.random.choice(['A','T','C','G'],4).tolist())
            tmp = ''.join(np.random.choice(['A','T','C','G'],1).tolist())
        seq     = 'GC' + 'TGTG' + tmp1 + sub_seq_list[i]    + tmp + 'CATG' + 'GC'
        seq_MN  = 'GC' + 'TGTG' + tmp1 + sub_seq_list_MN[i] + tmp + 'CATG' + 'GC'
        seq_list.append(seq_MN)
        D_width[i,0], D_width[i,1]  = GrooveWidths_CS(cgDNA(seq,ps).ground_state)
        D_width[i,2], D_width[i,3]  = GrooveWidths_CS(cgDNA(seq_MN,ps).ground_state)
    col = ['minor','major','minor_MN','major_MN']
    df = pd.DataFrame(np.around(D_width,3),columns=col,index = seq_list)
    df['minor_diff'] = - df['minor'] + df['minor_MN']
    df['major_diff'] = - df['major'] + df['major_MN']
    return df

def plot_groovewidths_data_epi_cpg2(which):
    if which == 'minor_diff':
        ylabel = "Change in Minor Groove width (in Å)"
        save_name = './Plots/modification_position_effect_minor_groove.pdf'
        groove = 'minor'
    elif which == 'major_diff':
        ylabel = "Change in Major Groove width (in Å)"
        save_name = './Plots/modification_position_effect_major_groove.pdf'
        groove = 'major'

    fig,ax = plt.subplots(figsize=(4.5,3))
    MN_mean = np.zeros(4)
    MN_std  = np.zeros(4)
    MN_ind  = np.arange(4)
    fs= 10
    xlabel = ['CGX$_4$X$_5$X$_6$','X$_2$CGX$_5$X$_6$','X$_2$X$_3$CGX$_6$','X$_2$X$_3$X$_4$CG' ]
    for i in range(4):
        df = create_groovewidths_data_epi_cpg2('MN','ps_mdna',i,groove)
        MN_mean[i] = df[which].mean()
        MN_std[i]  = df[which].std()
        print(df[df['minor_diff']<0])
    ax.errorbar(MN_ind-0.2, MN_mean, yerr=MN_std, fmt='o',color='b',label='MN')
    for i in range(4):
        df = create_groovewidths_data_epi_cpg2('MG','ps_mdna',i,groove)
        MN_mean[i] = df[which].mean()
        MN_std[i]  = df[which].std()
    ax.errorbar(MN_ind+0.1, MN_mean, yerr=MN_std, fmt='_',color='b',label='MG')

    for i in range(4):
        df = create_groovewidths_data_epi_cpg2('MN','ps_hdna',i,groove)
        MN_mean[i] = df[which].mean()
        MN_std[i]  = df[which].std()
    ax.errorbar(MN_ind-0.1, MN_mean, yerr=MN_std, fmt='o',color='r',label='HK')
    for i in range(4):
        df = create_groovewidths_data_epi_cpg2('MG','ps_hdna',i,groove)
        MN_mean[i] = df[which].mean()
        MN_std[i]  = df[which].std()
    ax.errorbar(MN_ind+0.2, MN_mean, yerr=MN_std, fmt='_',color='r',label='HG')

    ax.legend(fontsize=fs)

    ax.set_xticks(MN_ind)
    ax.set_xticklabels(xlabel,fontsize=fs)
    ax.legend(fontsize=fs)
    ax.set_xlabel("Position of C$_p$G step",fontsize=fs)
    ax.set_ylabel(ylabel,fontsize=fs-0.5)
    plt.tight_layout()
    fig.savefig(save_name,dpi=600)
    plt.close()
    return df

def plot_groovewidths_data_epi_cpg():
    MN, MN_mat = create_groovewidths_data_epi_cpg('MN','ps_mdna')
    MG, MG_mat = create_groovewidths_data_epi_cpg('MG','ps_mdna')
    HK, HK_mat = create_groovewidths_data_epi_cpg('MN','ps_hdna')
    HG, HG_mat = create_groovewidths_data_epi_cpg('MG','ps_hdna')
    fig,ax = plt.subplots(1,2,figsize=(6,3),sharey=True)
    j = 0
    i = 0
    ax[i].errorbar(MN_mat[:,0]    , MN_mat[:,1+j]-5.8, yerr=MN_mat[:,2+j],marker='o',color='r',label='MN')
    ax[i].errorbar(MG_mat[:,0]    , MG_mat[:,1+j]-5.8, yerr=MG_mat[:,2+j],marker='o',color='b',label='MG')
    i = 1
    ax[i].errorbar(HK_mat[:,0]    , HK_mat[:,1+j]-5.8, yerr=HK_mat[:,2+j],marker='o',color='r',label='HK')
    ax[i].errorbar(HG_mat[:,0]    , HG_mat[:,1+j]-5.8, yerr=HG_mat[:,2+j],marker='o',color='b',label='HG')

    ind = HK_mat[:,0]
    fs2 = 10
    percent = [0,20,40,60,80,100]
    for i in range(2):
        ax[i].set_xticks(ind)
        ax[i].set_xticklabels(percent,fontsize=fs2)
        ax[i].legend(fontsize=fs2)
        ax[i].set_xlabel("Percentage C$_p$G modification",fontsize=fs2)
        ax[i].tick_params(axis='both', labelsize=fs2, pad=3,length=3,width=0.5,direction= 'inout')

    ax[0].set_ylabel("Minor Groove width (in Å)",fontsize=fs2)
#    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
    plt.tight_layout()
    fig.savefig('./Plots/modification_percent_effect_minor_groove.pdf',dpi=600)
    plt.close()

    fig,ax = plt.subplots(1,2,figsize=(6,3),sharey=True)
    j = 2
    i = 0
    ax[i].errorbar(MN_mat[:,0]    , MN_mat[:,1+j]-5.8, yerr=MN_mat[:,2+j],marker='o',color='r',label='MN')
    ax[i].errorbar(MG_mat[:,0]    , MG_mat[:,1+j]-5.8, yerr=MG_mat[:,2+j],marker='o',color='b',label='MG')
    i = 1
    ax[i].errorbar(HK_mat[:,0]    , HK_mat[:,1+j]-5.8, yerr=HK_mat[:,2+j],marker='o',color='r',label='HK')
    ax[i].errorbar(HG_mat[:,0]    , HG_mat[:,1+j]-5.8, yerr=HG_mat[:,2+j],marker='o',color='b',label='HG')

    ind = HK_mat[:,0]
    fs2 = 10
    percent = [0,20,40,60,80,100]
    for i in range(2):
        ax[i].set_xticks(ind)
        ax[i].set_xticklabels(percent,fontsize=fs2)
        ax[i].legend(fontsize=fs2)
        ax[i].set_xlabel("Percentage C$_p$G modification",fontsize=fs2)
        ax[i].tick_params(axis='both', labelsize=fs2, pad=3,length=3,width=0.5,direction= 'inout')

    ax[0].set_ylabel("Major Groove width (in Å)",fontsize=fs2)
#    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
    plt.tight_layout()
    fig.savefig('./Plots/modification_percent_effect_major_groove.pdf',dpi=600)
    plt.close()


#################################################
#################################################
############# Seq logo demo #####################
#################################################
#################################################
def seq_logo_demo():
    seqs = ['AGTCA','AGTCT','AGCGC','ACCTG']
    prob = compute_base_prob_list_of_seq(seqs,0,5) + 10**-8
    fig,ax = plt.subplots(2,figsize=(3,2.75))
    logo_plot(prob,'IC',ax[0],'prob')
    logo_plot(prob,'IC',ax[1],'bits')

    lab = ['1','2','3','4','5']
    fs = 10
    for i in range(2):
        ax[i].tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')
        ax[i].set_xticks(np.arange(5))
        ax[i].set_yticks(np.arange(0,2.5,0.5))
        ax[i].set_yticklabels(np.arange(0,2.5,0.5),fontsize=fs)
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(0,2)
    ax[0].set_ylabel('Probability',fontsize=fs)
    ax[1].set_ylabel('bits',fontsize=fs)

    ax[0].set_xticklabels([],fontsize=fs)
    ax[1].set_xticklabels(lab,fontsize=fs)
    plt.tight_layout()
    fig.savefig('./Plots/demo_seq_logo.pdf',dpi=600)
    plt.show()
    plt.close()


##############################################################################
################################### SNPS #####################################
##############################################################################
def create_snips_data():
    tet = ttl_256()
    mat = np.zeros((4**8,6))
    count = 0
    SNIP = ['AT','AC','AG','TC','TG','CG']
    for l in tet:
        for r in tet:
            seq, res = {}, {}
            for enum,m in enumerate(['A','T','C','G']):
                seq[enum] = 'GC' + 'CT' + l + m + r + 'CG' + 'GC'
                res[enum] = cgDNA(seq[enum],'ps2_cgf')
            count2 = 0
            for k1 in range(4):
                for k2 in range(4):
                    if k1 < k2:
                            w1, w2 = res[k1].ground_state, res[k2].ground_state
                            S1, S2 = res[k1].stiff.todense(), res[k2].stiff.todense()
                            mat[count,count2] = Mahal_sym(w1,w2,S1,S2)
                            count2+=1
            print(count)
            count +=1
    df = pd.DataFrame(mat,columns=SNIP)
    df.to_csv('./Data/SNIPs/Mahal_sym_data.txt',index=False)
    return df

def signals_in_snps():
    try:
        df = pd.read_csv('./Data/SNIPs/Mahal_sym_data.txt')
    except:
        df = create_snips_data()
    df = df*390
    tet = ttl_256()
    seq_list = []
    for l in tet:
        for r in tet:
            seq_list.append(l+r)
#    SNIP = ['AT','AC','AG','TC','TG','CG']
    SNIP = ['AG','CG','AC','AT']
    SNIP_label = ['AG','GC','AC','AT']
    ## max change -- GGTA-A/T-GACC
    ## min change -- ACGT-A/G-ATCG
    df = df[SNIP]
    fs = 8
    tick_lab = [i[0] + '$\longleftrightarrow$' + i[1] for i in SNIP_label]
    fig, ax =  plt.subplots(figsize=(2.75,2.15))
    ind = np.arange(4)
    ax.errorbar(ind, df.mean(),yerr=df.std(),fmt='o')
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(tick_lab,fontsize=fs)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=None, top=0.85, wspace=None, hspace=None)
    ax.set_ylabel("  Change in groundstate \n (Mahalanobis distance)",fontsize=fs)
    fig.savefig('./Plots/SNIPS_errorbar.pdf',dpi=600)
    plt.show()
    plt.close()

    ############### seq_logo
    for enum,dim in enumerate(SNIP):
        prob = np.zeros((1,4)) + 0.25
        sort1 = np.argsort(df[dim].to_numpy())
        how_many  = int(4**8*0.0015)
        list1_min = [seq_list[i] for i in sort1[0:how_many]]
        list1_max = [seq_list[i] for i in sort1[-how_many:]]
        tmp_min = compute_base_prob_list_of_seq(list1_min,0,8) + 10**-8
        tmp_max = compute_base_prob_list_of_seq(list1_max,0,8) + 10**-8
        prob1_min = np.concatenate([tmp_min[0:4],prob,tmp_min[4:8]])
        prob1_max = np.concatenate([tmp_max[0:4],prob,tmp_max[4:8]])
        fig,ax = plt.subplots(2,figsize=(2.75,2.75),sharex=True)
        logo_plot(prob1_min,'IC',ax[0],'bits')
        logo_plot(prob1_max,'IC',ax[1],'bits')
        ax[0].set_ylim(0,2)
        ax[1].set_ylim(0,2)
        ax[0].set_ylabel('bits',fontsize=fs)
        ax[1].set_ylabel('bits',fontsize=fs)
        ax[0].set_title('Sequences with minimum change',fontsize=fs)
        ax[1].set_title('Sequences with maximum change',fontsize=fs)
        ax[1].set_xticks(np.arange(9))
        tmp = SNIP_label[enum][0] + '$\longleftrightarrow$' + SNIP_label[enum][1]
        tick_label = [1,2,3,4,tmp,6,7,8,9]
        ax[1].set_xticklabels(tick_label,fontsize=fs)
        plt.tight_layout()
        fig.savefig('./Plots/SNIPS_seq_logo_'+dim+'.pdf',dpi=600)
        plt.show()
        plt.close()

####################### stacking energy plot taken from https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.8b00643
    D = ['AA','AC','AG','AT', 'CC', 'CG', 'GC', 'TA', 'TC', 'TG']
    Stacking = np.array([ -13.37, -13.33, -13.96, -11.66, -13.81, -18.44, -16.14, -13.74, -13.22, -15.95  ])
    df = pd.DataFrame(Stacking[:,np.newaxis].T,columns = D)
    D = [ 'AT','TC','AC','AA','TA', 'CC', 'AG',  'TG', 'GC',  'CG']
    df = df[D]
    fig, ax =  plt.subplots(figsize=(2.75,2.15))
    ind = np.arange(10)
    ax.scatter(ind,df.values,s=8)
    ax.set_xticks(ind)
    D = [ 'AT','TC/GA','AC/GT','AA/TT','TA', 'CC/GG', 'AG/CT',  'TG/CA', 'GC',  'CG']
    ax.set_xticklabels(D,fontsize=fs,rotation=45)
    ax.set_ylabel('stacking energy (Kcal/mol)',fontsize=fs)
    plt.tight_layout()
    fig.savefig('./Plots/stacking_energy_dimer.pdf',dpi=600)
    plt.show()
    plt.close()

    return None

def change_in_stacking(a,b):
    D = ['AA','AC','AG','AT', 'CC', 'CG', 'GC', 'TA', 'TC', 'TG']
    Stacking = np.array([ -13.37, -13.33, -13.96, -11.66, -13.81, -18.44, -16.14, -13.74, -13.22, -15.95  ])
    s = dict(zip(D, Stacking.T))

    for dim in D:
        if dim == comp(dim):
            None
        else:
            s[comp(dim)] = s[dim]

    stack_a = 0
    stack_b = 0
    for m in ['A','T','C','G']:
            stack_a += s[a+m]
            stack_b += s[b+m]
            stack_a += s[m+a]
            stack_b += s[m+b]

    print(stack_a,stack_b)



def compare_gs_SNiPs():
#    lss=['-', '-','-','--','--','--']
    flank = 'CTGC'
    seq_maxa =  comp(flank) + 'GGTAAGACC' + flank
    seq_maxb =  comp(flank) + 'GGTATGACC' + flank
    seq_mina =  comp(flank) + 'ACGTAATCG' + flank
    seq_minb =  comp(flank) + 'ACGTGATCG' + flank


    res_maxa = cgDNA(seq_maxa,'ps2_cgf').ground_state
    res_maxb = cgDNA(seq_maxb,'ps2_cgf').ground_state
    res_mina = cgDNA(seq_mina,'ps2_cgf').ground_state
    res_minb = cgDNA(seq_minb,'ps2_cgf').ground_state


    shp = [res_maxa,res_maxb]#,res_mina,res_minb]
    seq = [seq_maxa]*2
    lss=['-', '--','--','--']
    color = [c1,c1,c4,c4]
    fig,ax_list = compare_shape(shp,seq,'test',lss,color=color,type_of_var='cg+')
    seq_l = list(seq_maxa)
    seq_l[8] = 'A/T'
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l)),minor=False)
        ax_list[a].set_xticklabels(seq_l,fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/compare_gs_SNiPs_max.pdf',dpi=600)
    plt.close()
    shp = [res_mina,res_minb]#,res_mina,res_minb]
    seq = [seq_mina]*2
    lss=['-', '--','--','--']
    color = [c1,c1,c4,c4]
    fig,ax_list = compare_shape(shp,seq,'test',lss,color=color,type_of_var='cg+')
    seq_l = list(seq_maxa)
    seq_l[8] = 'A/G'
    for enum,a in enumerate([8,9]):
        ax_list[a].set_xticks(np.arange(len(seq_l)),minor=False)
        ax_list[a].set_xticklabels(seq_l,fontsize=6,minor=False,rotation=45)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
    fig.savefig('./Plots/compare_gs_SNiPs_min.pdf',dpi=600)
    plt.close()



###############################################################################
#######################################  TEST NUMERICAL KLD ###################
###############################################################################

def test_numerical_KLD():
    mean1, mean2 = 0, 0
    std1 ,  std2 = 1, 1
    point = 100000
    rand = np.concatenate([np.random.normal(mean1,std1,point) , np.random.normal(mean2,std2,point)])
    loc = np.mean(rand)
    scale = np.std(rand)
    kernel = scipy.stats.gaussian_kde(rand)


    for u in [100,500,1000,2000,3000,5000,8000]:
        x = np.linspace(np.min(rand), np.max(rand),u)
        kde = kernel.evaluate(x)
        pdf = scipy.stats.norm.pdf(x, loc=loc, scale=scale)
        kld2 = numerical_KLD_sym(kde,pdf,x)
        print(kld2)

    fig,ax = plt.subplots()
    ax.plot(x,pdf)
    ax.plot(x,kde)

    return None




##############################################################################
###################### KLD for Gaussian error ################################
##############################################################################
#


def heat_map_Gaussian_err(data,seq,index,NA,vmax):

    cbar = np.linspace(0,vmax,25)[:,np.newaxis].T
    plt.rcParams['patch.linewidth']=1
    mat_intra = np.zeros((24,6))
    mat_phosW = np.zeros((23,6))
    mat_inter = np.zeros((23,6))
    mat_phosC = np.zeros((23,6))
    print(vmax)
    input()

    for i in range(24):
        for j in range(6):
            mat_intra[i,j] = data.loc[24*i+j]
            if  i < 23:
                mat_phosC[i,j] = data.loc[24*i+j+6]
                mat_inter[i,j] = data.loc[24*i+j+12]
                mat_phosW[i,j] = data.loc[24*i+j+18]


    print(np.mean(mat_intra),np.std(mat_intra),NA,'mat_intra')
    print(np.mean(mat_phosW),np.std(mat_phosW),NA,'mat_phosW')
    print(np.mean(mat_inter),np.std(mat_inter),NA,'mat_inter')
    print(np.mean(mat_phosC),np.std(mat_phosC),NA,'mat_phosC')

    mat_intra = mat_intra.T
    mat_phosW = mat_phosW.T
    mat_inter = mat_inter.T
    mat_phosC = mat_phosC.T

    width = 500
    fig = plt.figure(constrained_layout=False,facecolor='w', edgecolor='k')
    gs1 = gridspec.GridSpec(550, width)
    gs1.update(left=0.09, right=0.98,top=0.95,bottom=0.05)
    fs2 = 10
    vv = 125
    shift = 50
    ax0 = plt.subplot(gs1[0*vv+shift:1*vv+shift, 0:width])
    ax1 = plt.subplot(gs1[1*vv+shift:2*vv+shift, 0:width])
    ax2 = plt.subplot(gs1[2*vv+shift:3*vv+shift, 0:width])
    ax3 = plt.subplot(gs1[3*vv+shift:4*vv+shift, 0:width])
    ax4 = plt.subplot(gs1[22:42, 0:width])
    sns.cubehelix_palette(as_cmap=True)

    lw=0.2
    ax0 = sns.heatmap(mat_phosW,ax=ax0,square=True,cbar=False,cmap="Blues",vmin = 0, vmax = vmax,linewidths=lw, linecolor='black')
    ax1 = sns.heatmap(mat_inter,ax=ax1,square=True,cbar=False,cmap="Blues",vmin = 0, vmax = vmax,linewidths=lw, linecolor='black')
    ax2 = sns.heatmap(mat_phosC,ax=ax2,square=True,cbar=False,cmap="Blues",vmin = 0, vmax = vmax,linewidths=lw, linecolor='black')
    ax3 = sns.heatmap(mat_intra,ax=ax3,square=True,cbar=False,cmap="Blues",vmin = 0, vmax = vmax,linewidths=lw, linecolor='black')
    ax4 = sns.heatmap(cbar,ax=ax4,square=True,cbar=False,cmap="Blues",vmin = 0, vmax = vmax,linewidths=lw, linecolor='black')

    ax3.set_xticks(np.arange(24)+0.5)
    ax3.set_xticklabels(seq,fontsize=fs2)
    ax3.tick_params(axis='x', labelsize=fs2, pad=3,length=3,width=0.5,direction= 'inout')
    ax3.xaxis.set_tick_params(rotation=0)
    lws=1.5
    ax3.axhline(y =  0, color = 'k', linewidth = lws)
    ax3.axhline(y =  6, color = 'k', linewidth = lws)
    ax3.axvline(x =  0, color = 'k', linewidth = lws)
    ax3.axvline(x = 24, color = 'k', linewidth = lws)

    for ax in [ax0,ax1,ax2]:
        ax.axhline(y =  0, color = 'k', linewidth = lws)
        ax.axhline(y =  6, color = 'k', linewidth = lws)
        ax.axvline(x =  0, color = 'k', linewidth = lws)
        ax.axvline(x = 23, color = 'k', linewidth = lws)


    for ax in [ax0,ax1,ax2,ax3]:
        ax.set_yticks(np.arange(6)+0.5)
        ax.yaxis.set_tick_params(rotation=0)
    ax3.set_yticklabels(cgDNA_name[0:6]  ,fontsize=fs2)
    ax0.set_yticklabels(cgDNA_name[6:12] ,fontsize=fs2)
    ax1.set_yticklabels(cgDNA_name[12:18],fontsize=fs2)
    ax2.set_yticklabels(cgDNA_name[18:24],fontsize=fs2)

    ax4.set_yticks(np.arange(1)+0.5)
    ax4.set_yticklabels(['$\epsilon_{KL}^{Gauss}$'],fontsize=fs2,rotation=0)
    ax4.set_xticks(np.arange(0,25,5)+0.5)
    ax4.set_xticklabels(list(np.around(cbar[0],1))[::5],fontsize=fs2,rotation=0)

    ax4.tick_params(axis='x', which='major', labelsize=fs2, labelbottom = False, bottom=False, top = True, labeltop=True,rotation=0,pad=3,length=3,width=0.5,direction= 'inout')

    plt.show()
    fig.savefig('./Plots/Gaussian_error_'+NA+'_'+str(index)+'.pdf',dpi=600)
    plt.close()

def plot_Gaussian_error(index):
    seq = 'GCTTAGTTCAAATTTGAACTAAGC'
    seq_rna = list(seq.replace('T','U'))
    seq = list(seq)
    path = '/Users/rsharma/Dropbox/cgDNAplus_py_rahul/Data/KLD_data/'
    DNA = pd.read_csv(path+'palin.bscl.tip3p.jc.comb_'+str(index)+'.txt',header=None)
    RNA = pd.read_csv(path+'rna.ol3.tip3p.jc.comb_'+str(index)+'.txt',header=None)
    DRH = pd.read_csv(path+'hyb.dna.rna.bsc1.ol3.tip3p.jc.comb_'+str(index)+'.txt',header=None)

    vmax = np.max([DNA.max(),RNA.max(),DRH.max()])
    heat_map_Gaussian_err(DNA,seq,    index,'DNA',vmax)
    heat_map_Gaussian_err(RNA,seq_rna,index,'RNA',vmax)
    heat_map_Gaussian_err(DRH,seq,    index,'DRH',vmax)
    return None






##############################################################################################
##############################################################################################
## plots for presenations
##############################################################################################
##############################################################################################
def compare_various_errors():

    res  =np.array( [0.0027, 0.0316 , 0.0015, 0.0085 , 0.0031, 0.1032, 0.0025, 0.0311, 0.0025, 0.0309 ])
    scale =np.array( [0.0245, 0.4395, 0.0177, 0.2185, 0.0209, 0.3273 ,0.0211, 0.3378, 0.0214, 0.3449]  )
    local = np.array([0.0021, 0.0273 ,0.0013, 0.0054, 0.0020 ,0.0219, 0.0024, 0.0263, 0.0024, 0.0272])
    trunc = np.array([0.0046,0.0026,0.0049 ,0.0046, 0.0048])
    palin_err = np.array([0.0009, 0.0049, 0.0006, 0.0047, 0.0009, 0.0048, 0.0008,  0.0070] )
    indM  = np.arange(0,10,2)
    indKL = np.arange(1,10,2)
    ind = np.arange(5)
    fs2 = 14
    abc = ['DNA','RNA','DRH','MDNA', 'HDNA']

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot([0,1,3,4],palin_err[[0,2,4,6]],color='red', label = '$\epsilon_{M}^{palin, avg}$')
    ax[1].plot([0,1,3,4],palin_err[[1,3,5,7]],color='red', label = '$\epsilon_{KL}^{palin, avg}$' )
    ax[0].scatter([0,1,3,4],palin_err[[0,2,4,6]],color='red', label='_no_legend_')
    ax[1].scatter([0,1,3,4],palin_err[[1,3,5,7]],color='red', label='_no_legend_' )
    ax[0].plot(ind,scale[indM],color='blue' ,label='scale')
    ax[1].plot(ind,scale[indKL],color='blue',label='scale')
    ax[0].scatter(ind,scale[indM],color='blue' ,label='_no_legend_')
    ax[1].scatter(ind,scale[indKL],color='blue',label='_no_legend_')

    for i in range(2):
        ax[i].tick_params(axis='both', pad=5,length=3,width=0.5,direction= 'inout',rotation=0,labelsize=fs2)
        ax[i].set_xticks(ind)
        ax[i].set_xticklabels(abc,fontsize=fs2, rotation=45)
        ax[i].legend(fontsize=fs2,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.12),frameon=False)
    plt.tight_layout()

    fig.savefig('./Plots/presentation_error_comp_1.pdf',dpi=600)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot(ind,res[indM],color='red', label = '$\epsilon_{M}^{res, avg}$')
    ax[1].plot(ind,res[indKL],color='red', label = '$\epsilon_{KL}^{res, avg}$' )
    ax[0].scatter(ind,res[indM],color='red', label='_no_legend_')
    ax[1].scatter(ind,res[indKL],color='red', label='_no_legend_' )

    ax[0].plot(ind,scale[indM],color='blue' ,label='scale')
    ax[1].plot(ind,scale[indKL],color='blue',label='scale')
    ax[0].scatter(ind,scale[indM],color='blue' ,label='_no_legend_')
    ax[1].scatter(ind,scale[indKL],color='blue',label='_no_legend_')

    for i in range(2):
        ax[i].tick_params(axis='both', pad=5,length=3,width=0.5,direction= 'inout',rotation=0,labelsize=fs2)
        ax[i].set_xticks(ind)
        ax[i].set_xticklabels(abc,fontsize=fs2, rotation=45)
        ax[i].legend(fontsize=fs2,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.105),frameon=False)

    plt.tight_layout()

    fig.savefig('./Plots/presentation_error_comp_2.pdf',dpi=600)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot(ind,res[indM],color='red', label = '$\epsilon_{M}^{res, avg}$')
    ax[1].plot(ind,res[indKL],color='red', label = '$\epsilon_{KL}^{res, avg}$' )
    ax[0].scatter(ind,res[indM],color='red', label='_no_legend_')
    ax[1].scatter(ind,res[indKL],color='red', label='_no_legend_' )

    ax[0].plot(ind,local[indM],color='green', label = '$\epsilon_{M}^{local, avg}$')
    ax[1].plot(ind,local[indKL],color='green', label = '$\epsilon_{KL}^{local, avg}$' )
    ax[0].scatter(ind,local[indM],color='green', label='_no_legend_')
    ax[1].scatter(ind,local[indKL],color='green', label='_no_legend_' )

    ax[1].plot(ind,trunc,color='k', label = '$\epsilon_{KL}^{trunc, avg}$' )
    ax[1].scatter(ind,trunc,color='k', label='_no_legend_' )

    for i in range(2):
        ax[i].tick_params(axis='both', pad=5,length=3,width=0.5,direction= 'inout',rotation=0,labelsize=fs2)
        ax[i].set_xticks(ind)
        ax[i].set_xticklabels(abc,fontsize=fs2, rotation=45)
        ax[i].legend(fontsize=fs2,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()

    fig.savefig('./Plots/presentation_error_comp_3.pdf',dpi=600)
    plt.show()
    plt.close()


    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot(ind,res[indM],color='red', label = '$\epsilon_{M}^{res, avg}$')
    ax[1].plot(ind,res[indKL],color='red', label = '$\epsilon_{KL}^{res, avg}$' )
    ax[0].scatter(ind,res[indM],color='red', label='_no_legend_')
    ax[1].scatter(ind,res[indKL],color='red', label='_no_legend_' )

    ax[0].plot(ind,local[indM],color='green', label = '$\epsilon_{M}^{local, avg}$')
    ax[1].plot(ind,local[indKL],color='green', label = '$\epsilon_{KL}^{local, avg}$' )
    ax[0].scatter(ind,local[indM],color='green', label='_no_legend_')
    ax[1].scatter(ind,local[indKL],color='green', label='_no_legend_' )

    ax[1].plot(ind,trunc,color='k', label = '$\epsilon_{KL}^{trunc, avg}$' )
    ax[1].scatter(ind,trunc,color='k', label='_no_legend_' )

    ax[0].plot(ind,scale[indM],color='blue' ,label='scale')
    ax[1].plot(ind,scale[indKL],color='blue',label='scale')
    ax[0].scatter(ind,scale[indM],color='blue' ,label='_no_legend_')
    ax[1].scatter(ind,scale[indKL],color='blue',label='_no_legend_')


    for i in range(2):
        ax[i].tick_params(axis='both', pad=5,length=3,width=0.5,direction= 'inout',rotation=0,labelsize=fs2)
        ax[i].set_xticks(ind)
        ax[i].set_xticklabels(abc,fontsize=fs2, rotation=45)
        ax[i].legend(fontsize=fs2,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.2))

    plt.tight_layout()
    fig.savefig('./Plots/presentation_error_comp_4.pdf',dpi=600)
    plt.show()
    plt.close()

    return None


##################################################################
###################### figures for web article ###################
##################################################################
def compare_groundstate_two_paramsets(p, r, seq_id):
    p = p.choose_seq([seq_id])
    r = r.choose_seq([seq_id])
    s = 12
    if p.seq == r.seq:
        seq = p.seq[0]
    pc =  cgDNA(seq,'ps1')
    rc =  cgDNA(seq,'ps2_cgf')
    pw = np.array(p.shape_sym)
    rw = np.array(r.shape_sym)
    pw, rw = pw.T, rw.T
    pcw = pc.ground_state
    rcw = rc.ground_state
    for i in range(24):
        p_solid = pw[i::24]
        p_dash  = pcw[i::24]
        r_solid = rw[i::24]
        r_dash  = rcw[i::24]
        index = np.shape(p_dash)[0]
        index = np.arange(index)+1
        if i in [0,1,2,6,7,8,12,13,14,18,19,20]:
            plt.plot(index,11.46*p_solid,color='royalblue',label = 'MD1')
            plt.plot(index,11.46*r_solid,color='green',label = 'MD2')
            plt.plot(index,11.46*p_dash,color='royalblue',linestyle='dashed',label = 'PS1')
            plt.plot(index,11.46*r_dash,color='green',linestyle='dashed',label = 'PS2')
        else:
            plt.plot(index,p_solid,color='royalblue',label = 'MD1')
            plt.plot(index,r_solid,color='green',label = 'MD2')
            plt.plot(index,p_dash,color='royalblue',linestyle='dashed',label = 'PS1')
            plt.plot(index,r_dash,color='green',linestyle='dashed',label = 'PS2')

        if i in [0,1,2,12,13,14]:
            plt.ylabel(cgDNA_name_web[i] + ' (degrees)',fontsize=12)
        elif i in [6,7,8,18,19,20]:
            plt.ylabel("$"+cgDNA_name_web[i]+"$" + ' (degrees)',fontsize=12)
        elif i in [9,10,11,21,22,23]:
            plt.ylabel('$'+cgDNA_name_web[i]+'$' + ' (' + ang + ')',fontsize=12)
        else:
            plt.ylabel(cgDNA_name_web[i] + ' (' + ang + ')',fontsize=12)
        if i < 6:
            plt.xlabel("basepair",fontsize=12)
        else:
            plt.xlabel("basepair step",fontsize=12)
        plt.legend(fontsize=s, loc='upper center',fancybox=True,ncol=4, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.135))
        plt.xticks(index[::2],size=s)
        plt.yticks(size=s)
        plt.tight_layout()
        plt.savefig("./Plots/webplot_" +str(i+1)+'_seq_id_'+str(seq_id)+".pdf",dpi=600)
        plt.show()
        plt.close()
        print(seq,i+1)


    return None
