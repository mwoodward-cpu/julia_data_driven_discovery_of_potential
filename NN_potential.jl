#Author: Michael Woodward
#Updated: 06/08/20

#In this file we use a NN to approximate the potential, then use
#neural ode in order to learn F = - ∇U(|r_i - r_j|)


#Upload packages as we need them:
#not used: #using PyCall, ThArrays,
using NPZ, Flux, Zygote, DiffEqFlux



#set working directory:
cd("/Users/dayman/julia_sph_learning/potential_learning_MD")



#-------------Load data and set hyperparameters -------------------------

x = npzread("small_x.npy");   #(time, mol, coord)
v = npzread("small_v.npy");
A = npzread("small_A.npy");

#Now we need to implement neural ode where we will take the gradient of U first

Y = hcat(x,v)   #ground truth data in standard form [x y u v]






#-------------------------Misc tools ---------------------------------

batch_y = zeros(top_dim, batch_time+1 ,batch_size)
#note: for each batch: the data reads as (coord, time[s:s+batch_time])
function get_batch()
    s = sample(1:(data_size - batch_time), batch_size; replace = false, ordered = false)
    batch_u0 = u[:,s]
    batch_t = t[1:batch_time]
    for i = 0:batch_time
        batch_u[:,i+1,:] = u[:,s.+i]
    end
    return batch_u0, batch_t, batch_u
end

















#--------------NN Model for learning Potential --------------------

U = FastChain(FastDense(2,50,tanh),
              FastDense(50,1))

Θ = Flux.params(U)

#define the RHS:
function F(Θ,x,v)
    grad_u(x) = gradient(U, x)
    return hcat(v, grad_u(x))
end







#----------------Learning Algorithm ----------------------------------
