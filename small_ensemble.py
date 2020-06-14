from math import log, cos, sin, pi
from numpy import zeros, copy
from random import random, normalvariate
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
global D, N, h, U0, dt, X, V, A

os.chdir("/Users/dayman/julia_sph_learning/potential_learning_MD/") #current working directory

#in this code we run the verlet method forward in time with batch_size different number of initial conditions
batch_size = 7

def print_coordinates():  #print coordinates
  for n in range(0, N):
    print('{0:d} {1:.8f} {2:.8f}'.format(n, X[n, 0], X[n, 1]))
  print('')

def back_to_torus():  #periodic boundary conditions
  global X
  for n in range(0, N):
    for i in range(0, D):
        for k in range(0, batch_size):
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
  for n in range(0, N):
    for m in range(0, N):    #pairwise interaction
      if (m != n):
        r2 = 0.
        tmp = X[n] - X[m]
        for k in range(0, batch_size):
            for i in range(0, D):
                if (tmp[i,k] < -pi):
                    tmp[i,k] = tmp[i,k] + 2. * pi
                if (tmp[i,k] > pi):
                    tmp[i,k] = tmp[i,k] - 2. * pi
                r2 = r2 + tmp[i,k]**2
        if (r2 < 4. * h * h):
          A[n] = A[n] - der_Udr(r2**0.5) * tmp   #dv/dt = A


D, N, h, U0, dt = 2, 5, 2., 20., 0.01       #D: top_dim, N: #of mol 
X, V, A = zeros((N, D, batch_size)), zeros((N, D, batch_size)), zeros((N, D, batch_size))
T = 60
Acel, pos, vel, time = zeros((T,N,D,batch_size)), zeros((T,N,D,batch_size)), zeros((T,N,D,batch_size)), zeros((T))



#ensemble of INITIAL CONDs:

for k in range(0, batch_size):
    for n in range(0, N):            
        for i in range(0, D):                              
            X[n, i, k] = 2. * pi * random()
            V[n, i, k] = normalvariate(0, 1.8)


    for t in range(0, T):
        obtain_A() # velocity Verlet algorithm   obtain dv/dt
        print(A.shape)
        V = V + 0.5 * dt * A
        X = X + dt * V #+ 0.5 * dt**2 * A
        # dVdt = A
        Acel[t] = A
        pos[t] = X
        back_to_torus()
        obtain_A()
        V = V + 0.5 * dt * A
        vel[t] = V
        time[t] = t





# print(Acel.shape)
# print(Acel)




# #make a movie from the data (test 1): This is slow. Try another way. 
# plt.style.use(['dark_background'])
# def makedirs(dirname):
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)

# makedirs('mol_dyn_movie_150b2')
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(7, 7)) #, facecolor='white')
# #fig = plt.scatter(figsize=(7, 7), facecolor='white')

# ax_pos = fig.add_subplot(111, frameon=False)
# # ax_vel = fig.add_subplot(122, frameon=False)
# plt.show(block=False)





# def visualize(pos, vel, itr):

#         win_size = 2*pi

#         ax_pos.cla()
#         ax_pos.set_title('molecular dynamics particles')
#         ax_pos.set_xlabel('x')
#         ax_pos.set_ylabel('y')
#         ax_pos.plot(pos[itr, :, 0], pos[itr, :, 1], '.', color = 'w')  #plots position of all mol
#         ax_pos.set_xlim(0, win_size)
#         ax_pos.set_ylim(0, win_size)


#         # ax_pos.cla()
#         # ax_pos.set_title('molecular dynamics particles')
#         # ax_pos.set_xlabel('x')
#         # ax_pos.set_ylabel('y')
#         # ax_pos.plot(pos[itr, :, 0], pos[itr, :, 1], s= 0.25, color = 'c')  #plots position of all mol
#         # ax_pos.set_xlim(0, win_size)
#         # ax_pos.set_ylim(0, win_size)


#         # ax_pos.cla()
#         # ax_pos.set_title('molecular dynamics velocity (field?')
#         # ax_pos.set_xlabel('t')
#         # ax_pos.set_ylabel('u,v')
#         # ax_pos.plot(time[:], vel[:, 0, 0], time[:], vel[:, 0, 1])  #plots velocity of mol0 over time
#         # ax_pos.set_xlim(-win_size, win_size)
#         # ax_pos.set_ylim(-win_size, win_size)

#         fig.tight_layout()
#         plt.savefig('mol_dyn_movie_150b2/{:03d}'.format(itr))
#         plt.draw()
#         plt.pause(0.00005)


# #Making pngs for movies and display 
# for itr in range(1, T):
#   visualize(pos, vel, itr)





#save as numpy arrays:


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
# torch.save(A, 'small_5_A.t')

# v = torch.from_numpy(vel).float()
# torch.save(v, 'small_5_v.t')

# x = torch.from_numpy(pos).float()
# torch.save(x, 'small_5_x.t')

# t = torch.from_numpy(time).float()
# torch.save(t, 'small_5_time.t')