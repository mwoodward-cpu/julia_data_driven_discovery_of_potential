

from math import log, cos, sin, pi
from numpy import zeros, copy
from random import random, normalvariate
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
global D, N, h, U0, dt, X, V, A

os.chdir("/Users/dayman/mol_dynamics_node/torchdiffeq/examples/simple_sph_node/") #current working directory


def print_coordinates():  #print coordinates
  for n in range(0, N):
    print('{0:d} {1:.8f} {2:.8f}'.format(n, X[n, 0], X[n, 1]))
  print('')

def back_to_torus():  #periodic boundary conditions
  global X
  for k in range(0, batch_size):
    for n in range(0, N):
        for i in range(0, D):
            while (X[n, i, k] < 0.):
                X[n, i, k] = X[n, i, k] + 2. * pi
            while (X[n, i, k] >= 2. * pi):
                X[n, i, k] = X[n, i, k] - 2. * pi

def U(r):         #defines potential energy
  q = r / h
  if (q >= 2.):
    return 0.
  if (q >= 1.):
    return U0 * (2. - q)**3 / 4.
  return U0 * (1. - 1.5 * q * q * (1. - q / 2.));

def der_Udr(r):       # du/dr *1/r (x_i - x_j) = grad_x_i U = F
  q = r / h
  if (q >= 2.):
    return 0.
  if (q >= 1.):
    return -3. * U0 * (2. - q)**2 / (4. * r * h)   
  return U0 * (-3. + 9. * q / 4.) / h**2;


def obtain_A():
  global A
  A, tmp = zeros((N, D, batch_size)), zeros((1, D, batch_size))
  # for k in range(0,batch_size):
  for n in range(0, N):
    for m in range(0, N):    #pairwise interaction
      if (m != n):
        r2 = 0.
        for k in range(0,batch_size):
          tmp = X[n,:,k] - X[m,:,k]
          for i in range(0, D):
            if (tmp[i] < -pi):
              tmp[i] = tmp[i] + 2. * pi
            if (tmp[i] > pi):
              tmp[i] = tmp[i] - 2. * pi
            r2 = r2 + tmp[i]**2
          if (r2 < 4. * h * h):
            A[n,:,k] = A[n,:,k] - der_Udr(r2**0.5) * tmp   #dv/dt = A

#Notes 5/4/20: Used the above login for obrtain_A to feed data into neural network (replace der_Udr with grad_potential)

batch_size = 10
D, N, h, U0, dt = 2, 5, 2., 20., 0.01
X, V = zeros((N, D, batch_size)), zeros((N, D, batch_size))
T = 100
Acel, pos, vel, time = zeros((T,N,D,batch_size)), zeros((T,N,D,batch_size)), zeros((T,N,D,batch_size)), zeros((T))


#Ensemble generation:
for k in  range(0, batch_size):
    for n in range(0, N): 
        for i in range(0, D):                             
            X[n, i, k] = 2. * pi * random()
            V[n, i, k] = normalvariate(0, 1.8)

    for t in range(0, T):
        obtain_A() # velocity Verlet algorithm
        V[:,:,k] = V[:,:,k] + 0.5 * dt * A[:,:,k]
        X[:,:,k] = X[:,:,k] + dt * V[:,:,k]
        back_to_torus()
        obtain_A()
        V[:,:,k] = V[:,:,k] + 0.5 * dt * A[:,:,k]
        pos[t,:,:,k] = X[:,:,k]
        vel[t,:,:,k] = V[:,:,k]
        time[t] = t



A = Acel
np.save("small_ensemble_A.npy", A)

v = vel
np.save("small_ensemble_v.npy", v)

x = pos
np.save("small_ensemble_x.npy", x)

t = time
np.save("small_ensemble_t.npy", t)




#save as torch.tensor:

# A = torch.from_numpy(Acel).float()
# torch.save(A, 'small_ensemble_A.t')

# v = torch.from_numpy(vel).float()
# torch.save(v, 'small_ensemble_v.t')

# x = torch.from_numpy(pos).float()
# torch.save(x, 'small_ensemble_x.t')

# t = torch.from_numpy(time).float()
# torch.save(t, 'small_ensemble_t.t')

print("simulation complete")
