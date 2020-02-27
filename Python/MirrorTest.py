import numpy as np
import matplotlib.pyplot as plt
import Methods as Mt


A = Mt.ComMatrix(0.5,(2/45)*np.pi)
B = Mt.ApplyRotation(Mt.ComMatrix(-0.2,(2)*np.pi),1/3*np.pi)
print(np.dot(A,B))