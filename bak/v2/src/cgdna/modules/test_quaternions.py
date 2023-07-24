import numpy as np

import  cgdna.modules.RotationUtils as Rot


q1,q2 = np.random.rand(4),np.random.rand(4)
q1,q2 = np.divide(q1,np.linalg.norm(q1)),np.divide(q2,np.linalg.norm(q2))

R1,R2 = Rot.Quat2Rot(q1),Rot.Quat2Rot(q2)
u = Rot.Quat2Cay(q2,5)
R22 = Rot.Cay(u,5)
q22 = Rot.Cay2Quat(u,5)

R = np.dot(R1,R2)
q = Rot.QuatMult(q1,q2)
RR = Rot.Quat2Rot(q)
