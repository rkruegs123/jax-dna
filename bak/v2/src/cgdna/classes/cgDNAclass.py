import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import time
from numpy import matlib as mb

import cgdna.modules.cgDNAUtils as tools
from cgdna.classes.dsDNAclass import dsDNA


ang = "Å"

class cgDNA:
###########################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       CLASS DEFINITION
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###########################################################################
    def __init__(self,sequence,ps='ps1'):
        if ps == 'dna_ps1':
            self.paramset = 'cgDNA+ps1.mat'
        if ps == 'dna_ps2_cur':
            self.paramset = 'cgDNA+_Curves_BSTJ_10mus_FS.mat'
        if ps == 'dna_ps2':
            self.paramset = 'Di_hmethyl_methylated-hemi_combine.mat'
        if ps == 'rna_ps2':
            self.paramset = 'Prmset_cgRNA+_OL3_CGF_10mus_int_12mus_ends.mat'
        if ps == 'drh_ps2':
            self.paramset = 'cgHYB+_CGF_OL3_BSC1_10mus_FS_GC_ends.mat'
        if ps == 'dna_ps2_mle':
            self.paramset = 'cgDNA+_MLE_ends12mus_ends_CGF_BSC1.mat'
        self.seq = sequence
        self.nbp = len(sequence)
        self.dofs = 24*self.nbp-18
        self.ground_state,self.stiff = tools.constructSeqParms(sequence,self.paramset)
        self.Struct3D = dsDNA()
        self.shapes = {}

###########################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       CLASS MEHTODS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###########################################################################

###########################################################################
#                       2D PLOTS
###########################################################################


    def frames(self):
        self.Struct3D.ConstructFromShape(self.ground_state,self.seq)


    def plot2D(self,*cgDNA,**kwargs):
        Input = ['RotDeg','LineWidth']
        RotDeg,LineWidth=0,1
        for key, value in kwargs.items():
            if value in Input:
                exec(value+'='+key)
            else:
                print('Unkown input keyword: Possible keywords are: RotDeg (default = 0 ), LineWidth (defautl = 1)')

        # Count how many DNA structures are passed in the arguments
        l = len(cgDNA)
        y1=' ('+ang+')' #this is just the axes label ... tex code for Angstrom symbol
        if RotDeg == 1:
            y2=' ($^\circ$)'
        elif RotDeg==0:
            y2=' (rad/5)'
        else:
            print('Wrong value for the Keyword RotDeg.')

        ss = {}
        ss[0] = self
        if l>1:
            for k in range(l+1):
                ss[k+1] = cgDNA[k]

        for k in range(l+1):
            if RotDeg == 1:
                intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t=tools.DecomposeCoord(ss[k].ground_state)
                intra_r,intra_t,inter_r,inter_t=(36/np.pi)*intra_r,intra_t,(36/np.pi)*inter_r,inter_t
            elif RotDeg == 0:
                intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t = tools.DecomposeCoord(ss[k].ground_state)
            else:
                print('Wrong value for the Keyword RotDeg.')

            x_intra,x_inter = range(1,ss[k].nbp+1),np.add(range(1,ss[k].nbp),0.5)
            x_pho_W, x_pho_C = range(2,ss[k].nbp+1),range(1,ss[k].nbp)

            y_intra = np.append(intra_r,intra_t, axis=0)
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.plot(x_intra,y_intra[i,:])
                plt.title('intra')
            plt.show()

            y_inter = np.append(inter_r, inter_t, axis=0)
            for i in range(6):
                plt.subplot(2,3,i+1)
                plt.plot(x_inter,y_inter[i,:])
                plt.title('inter')
            plt.show()

            y_pho_W = np.append(pho_W_r, pho_W_t, axis=0)
            for i in range(6):
                plt.subplot(2,3,i+1)
                plt.plot(x_pho_W,y_pho_W[i,:])
                plt.title('Watson')
            plt.show()

            y_pho_C = np.append(pho_C_r,pho_C_t, axis=0)
            for i in range(6):
                plt.subplot(2,3,i+1)
                plt.plot(x_pho_C,y_pho_C[i,:])
                plt.title('Crick')
            plt.show()


###########################################################################
#                       3D PLOTS
###########################################################################
    def plot3D(self):
        if self.Struct3D.empty :
            self.frames()
        self.Struct3D.plot3D()

###########################################################################
#                       MONTE CARLO
###########################################################################
    def MonteCarlo(self,NbrSamples,Drop_base=0):

        mu = np.zeros(self.dofs)
        cov = np.identity(self.dofs)

        conf = np.random.multivariate_normal(mu,cov,NbrSamples).T
        try:
            LT = sci.sparse.csc_matrix(np.linalg.cholesky(self.stiff.todense()).T)
        except:
            LT = sci.sparse.csc_matrix(np.linalg.cholesky(self.stiff).T)

        ttc = np.zeros(self.nbp-1-2*Drop_base)
        s = np.add(sci.sparse.linalg.spsolve(LT,conf), mb.repmat(self.ground_state,1,1).T)

        mu0 = self.ground_state[Drop_base*24:(self.nbp-Drop_base)*24-18]

        R0 = np.identity(3)
        s = s[Drop_base*24:(self.nbp-Drop_base)*24-18]
        for k in range(int(len(conf[0]))):
            ttc += tools.TanTan(s[:,k],R0)

        tt0 = tools.TanTan(mu0,R0)
        ttc = np.divide(ttc,NbrSamples)

        plt.plot(np.log(ttc),label='ttc',color='g')
        plt.plot(np.log(tt0),label='tt0',color='b')
        plt.plot(np.log(ttc)-np.log(tt0),label='ttc-tt0',color='r')
        plt.legend()
        plt.show()
        plt.close()
        return s,ttc,tt0

###########################################################################
#                       MAKE PDB
###########################################################################
    def makePDB(self):
        self.test = 4

###########################################################################
#                       ADDOTHERSHAPES
###########################################################################
    def addOtherShapes(self,s):

        if not self.shapes:
            self.shapes[0] = s
        else:
            n = len(self.shapes)
            self.shapes[n+1] = s

##########################################################################
#                      ENERGY
###########################################################################
    def Energy(self):
        E = {}
        n = len(self.shapes)
        for i in range(n):
            diff = np.subtract(self.shapes,self.ground_state)
            E[i] = 0.5*np.matmul(diff,np.matmul(self.stiff,diff))
        return E
