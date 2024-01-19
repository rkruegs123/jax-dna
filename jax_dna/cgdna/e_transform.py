import numpy as np
import scipy.sparse as ssp


def Etransform():
    O_6 = np.zeros((6, 6))
    E_6 = np.diag([-1, 1, 1, -1, 1, 1 ])
    I_6 = np.eye((6))

    # E interior
    E_42 = np.concatenate((np.concatenate((O_6, O_6, O_6, O_6, O_6, O_6, I_6), axis=1),
                           np.concatenate((O_6, O_6, O_6, O_6, O_6, E_6, O_6), axis=1),
                           np.concatenate((O_6, O_6, O_6, O_6, I_6, O_6, O_6), axis=1),
                           np.concatenate((O_6, O_6, O_6, E_6, O_6, O_6, O_6), axis=1),
                           np.concatenate((O_6, O_6, I_6, O_6, O_6, O_6, O_6), axis=1),
                           np.concatenate((O_6, E_6, O_6, O_6, O_6, O_6, O_6), axis=1),
                           np.concatenate((I_6, O_6, O_6, O_6, O_6, O_6, O_6), axis=1)),
                          axis=0)

    # E 3' -> 5'
    E_36 = np.concatenate((np.concatenate((O_6, O_6, O_6, O_6, O_6, E_6), axis=1),
                           np.concatenate((O_6, O_6, O_6, O_6, I_6, O_6), axis=1),
                           np.concatenate((O_6, O_6, O_6, E_6, O_6, O_6), axis=1),
                           np.concatenate((O_6, O_6, I_6, O_6, O_6, O_6), axis=1),
                           np.concatenate((O_6, E_6, O_6, O_6, O_6, O_6), axis=1),
                           np.concatenate((I_6, O_6, O_6, O_6, O_6, O_6), axis=1)),
                          axis=0)
    E_42 = ssp.csr_matrix(E_42)
    E_36 = ssp.csr_matrix(E_36)
    return E_42, E_36


def Etrans(nbp, model="cg+"):
    E_6 = np.diag([-1, 1, 1, -1, 1, 1])
    I_6 = np.eye((6))

    if model == "cg+":
        E = np.zeros((24*nbp-18, 24*nbp-18))
        NB = 4 * nbp - 3
        for i in range(NB):
            if (i % 2) == 0:
                E[i*6:(i+1)*6, (NB-i-1)*6:(NB-i)*6] = E_6
            else:
                E[i*6:(i+1)*6, (NB-i-1)*6:(NB-i)*6] = I_6
    elif model == "cg":
        E = np.zeros((12*nbp-6, 12*nbp-6))
        NB = 2*nbp-1
        for i in range(NB):
            E[i*6:(i+1)*6, (NB-i-1)*6:(NB-i)*6] = E_6
    return E
