import numpy as np
from scipy.linalg import sqrtm


def Cay(u, *scale):

    if not scale:
        sc = 2
    else:
        sc = 2 * scale[0]

    I = np.identity(3)
    u = np.divide(u, sc)
    X = vect2mat(u)
    X2 = np.matmul(X, X)
    c = np.divide(2, 1 + np.inner(u, u))
    Q = np.add(I, np.dot(c, np.add(X, X2)))
    return Q

def Cayinv(Q, *scale):

    if not scale:
        sc = 2
    else:
        sc = 2 * scale[0]

    c = np.divide(2 * sc, np.trace(Q) + 1)
    u = np.dot(c, mat2vect(Q))

    return u

def midFrame(R1, R2):
    R = np.matmul(R1, np.real(sqrtm(R2)))
    return R

def mat2vect(X):
    a = np.zeros(3)
    a[0] = X[2, 1] - X[1, 2]
    a[1] = X[0, 2] - X[2, 0]
    a[2] = X[1, 0] - X[0, 1]
    return a

def vect2mat(u):
    X = [[0, -u[2], u[1]], [u[2], 0, -u[0]] , [-u[1], u[0], 0]]
    return X

def Cay2Quat(u, *scale):

    if not scale:
        sc = 2
    else:
        sc = 2 * scale[0]

    q = np.zeros(4)

    nu2 = np.power(np.linalg.norm(u), 2)
    sc2 = np.power(sc, 2)

    s = np.sqrt(np.divide(sc2, sc2+nu2))
    v = s * np.divide(u, sc)

    q[0], q[1:4] = s, v
    return q

def Rot2EulerAngles(R):

    q = Rot2Quat(R)
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]

    # Rotation around lab x-axis
    phi = np.arctan2(2*(qr*qi + qj*qk), 1-2*(qi*qi + qj*qj))

    # Rotation around lab y-axis
    theta = np.arcsin(2*(qr*qj - qi*qk))

    # Rotation around lab z-axis
    psi = np.arctan2(2*(qr*qk + qi*qj), 1 - 2*(qj*qj + qk*qk))

    Euler_Angles = [phi, theta, psi]
    return Euler_Angles

def Rot2Quat(R):
    u = Cayinv(R)
    q = Cay2Quat(u)
    return q

def Quat2Cay(q, *scale):

    if not scale:
        sc = 2
    else:
        sc = 2 * scale[0]

    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    u = np.dot(sc, [qi/qr, qj/qr, qk/qr])
    return u

def Quat2Rot(q):
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    qi2, qj2, qk2 = np.power(qi, 2), np.power(qj, 2), np.power(qk, 2)

    R = [
        [1 - 2 * (qj2 +qk2), 2 * (qi*qj - qk*qr), 2 * (qi*qk + qj*qr)],
        [2 * (qi*qj + qk*qr), 1 - 2 * (qi2 + qk2), 2 * (qj*qk - qi*qr)],
        [2 * (qi*qk - qj*qr), 2 * (qj*qk + qi*qr), 1 - 2 * (qi2 + qj2)],
    ]
    return R

def Quat2Rot_33(q):
    qi, qj = q[1], q[2]
    R_33 = 1 - 2 * (qi*qi + qj*qj)
    return R_33


def QuatInv(q):
    q[1:4] = -q[1:4]
    return q

def QuatVectMult(q,v):
    p = [0, v]
    vv = QuatMult(q, QuatMult(p, QuatInv(q)))
    vv = vv[1:4]
    return vv

def QuatMult(q1, q2):

    q = np.zeros(4)

    s1, s2 = q1[0], q2[0]
    v1, v2 = q1[1:4], q2[1:4]

    s = s1*s2 - np.inner(v1, v2)
    v = np.dot(s1, v2) + np.dot(s2, v1) + np.cross(v1, v2)
    q[0], q[1:4] = s, v

    return q
