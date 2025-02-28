# ruff: noqa
# fmt: off
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import jax_dna.cgdna.utils as tools


class dsDNA:

    def __init__(self):
        self.empty = True
        self.R, self.r = {}, {}
        self.Rw, self.rw = {}, {}
        self.Rc, self.rc = {}, {}
        self.Rpw, self.rpw = {}, {}
        self.Rpc, self.rpc = {}, {}
        self.seq = {}
        self.shape = {}
        self.nbp = {}

    def ConstructFromShape(self, s, sequence):
        self.empty = False
        self.seq = sequence
        self.shape = s
        self.nbp = int(len(sequence))
        (R, r, Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc) = tools.frames(s)

        for i in range(len(sequence)):
            self.R[i], self.r[i] = R[i], r[i]
            self.Rw[i], self.rw[i] = Rw[i], rw[i]
            self.Rc[i], self.rc[i] = Rc[i], rc[i]
            self.Rpw[i], self.rpw[i] = Rpw[i], rpw[i]
            self.Rpc[i], self.rpc[i] = Rpc[i], rpc[i]

    def ConstructFromFrames(self, fraFile, pfraFile):
        return 0

    def getBaseConfig(self, bp_nbr, CompStrand):

        if CompStrand:
            dsign = -1
            R, r = self.Rc[bp_nbr], self.rc[bp_nbr]
            base = tools.comp(self.seq[bp_nbr])
        else:
            dsign = 1
            R, r = self.Rw[bp_nbr], self.rw[bp_nbr]
            base = self.seq[bp_nbr]

        if base == 'A':
            color = "r"
            dy = dsign * 6
        elif base == 'T':
            color = "b"
            dy = dsign * 4.2
        elif base == 'C':
            color = "y"
            dy = dsign * 4.2
        elif base == 'G':
            dy = dsign * 6
            color = "g"

        dz, dx, Y = 0.36, 4.20, -1.2 * dsign


        verteces_RB =[
            [-0.5*dx, -0.5*Y, -0.5*dz],
            [0.5*dx, -0.5*Y, -0.5*dz],
            [0.5*dx, dy - 0.5*Y, -0.5*dz],
            [-0.5*dx, dy - 0.5*Y, -0.5*dz],
            [-0.5*dx, -0.5*Y, 0.5*dz],
            [0.5*dx, -0.5*Y, 0.5*dz],
            [0.5*dx, dy - 0.5*Y, 0.5*dz],
            [-0.5*dx, dy - 0.5*Y, 0.5*dz]
        ]

        RB_conf = np.add(np.transpose(np.matmul(R, np.transpose(verteces_RB))),
                         np.tile(np.transpose(r), [8, 1]))

        return color, RB_conf

    def getRigidBodies(self):

        color, colorC, RB, RBC = {}, {}, {}, {}

        for i in range(self.nbp):
            color[i], RB[i] = self.getBaseConfig(i, False)
            colorC[i], RBC[i] = self.getBaseConfig(i, True)
        return RBC, colorC, RB, color

    def plot3D(self, *arg):
        if self.empty:
            raise RuntimeError(f"The object is empty.")

        fig = plt.figure()
        ax = Axes3D(fig)
        nbp = self.nbp

        RBC, colorC, RB, color = self.getRigidBodies()

        # The following loop will create rectangular patches which will represent the faces for base
        # The two loops are for watson and crick strands
        for i in range(len(colorC)):
            k, c, kc, cc = RB[i], color[i], RBC[i], colorC[i]

            x1 = [k[0][0], k[1][0], k[2][0], k[3][0]]
            y1 = [k[0][1], k[1][1], k[2][1], k[3][1]]
            z1 = [k[0][2], k[1][2], k[2][2], k[3][2]]

            x2 = [k[4][0], k[5][0], k[6][0], k[7][0]]
            y2 = [k[4][1], k[5][1], k[6][1], k[7][1]]
            z2 = [k[4][2], k[5][2], k[6][2], k[7][2]]

            f1, f2 = [list(zip(x1, y1, z1))], [list(zip(x2, y2, z2))]

            for i in f1, f2:
                alpha = 0.5
                fc = c
                pc = Poly3DCollection(i, alpha=alpha, facecolors=fc, linewidths=1)
                ax.add_collection3d(pc)

            x1 = [kc[0][0], kc[1][0], kc[2][0], kc[3][0]]
            y1 = [kc[0][1], kc[1][1], kc[2][1], kc[3][1]]
            z1 = [kc[0][2], kc[1][2], kc[2][2], kc[3][2]]

            x2 = [kc[4][0], kc[5][0], kc[6][0], kc[7][0]]
            y2 = [kc[4][1], kc[5][1], kc[6][1], kc[7][1]]
            z2 = [kc[4][2], kc[5][2], kc[6][2], kc[7][2]]

            f1, f2 = [list(zip(x1, y1, z1))], [list(zip(x2, y2, z2))]
            for i in f1, f2:
                alpha = 0.5
                fc = cc
                pc = Poly3DCollection(i, alpha=alpha, facecolors=fc, linewidths=1)
                ax.add_collection3d(pc)

        bb = np.amax(nbp)
        ax.set_xlim(-1.75*bb, 1.75*bb)  # it optimizes the 3D plot view.
        ax.set_ylim(-1.75*bb, 1.75*bb)
        ax.set_zlim(0, 3.5*bb)
        # plt.axis(axes_opt) # ON or OFF
        # plt.savefig('3D.pdf', format='png', dpi=1200)
        plt.show()
