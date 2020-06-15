#in this file we approach batchhing in a different way:
#generate an ensemble of different trajectories (short time) with random initial conds.
#from this ensemble we will learn the function F (rhs) from brute force:
#Think of randomly sampling a manifold with short time trajectories run.
#This can be parallelized easily in the future?

#Steps:
#1) first we need to generate an ensemble of short time data from small.py
#2) Then we need to use the odeproblem (see universal vielhis in github)


#Author: Michael Woodward
#Updated: 06/12/20

#Try this in python...

#Upload packages as we need them:
using NPZ, Zygote
using DifferentialEquations, DiffEqSensitivity;
using Flux, DiffEqFlux;
using Test, OrdinaryDiffEq, Statistics;
using Optim, LinearAlgebra;
using Plots;




#set working directory:
cd("/Users/dayman/julia_sph_learning/potential_learning_MD")


#-------------Load data and set hyperparameters -------------------------

#Try 5 mol for 100 timespteps:
t = npzread("small_ensemble_t.npy")
x = npzread("small_ensemble_x.npy");   #(time, mol, coord, batch)
v = npzread("small_ensemble_v.npy");
A = npzread("small_ensemble_A.npy");

h = 70
dt = 0.01f0
T = size(t)[1]
tspan = (0.0f0, T)
top_dim = size(x)[3]
N = size(x)[2]
batch_size = size(x)[4]
datasize = size(t)[1]

x2 = reshape(x, (T, top_dim*N, batch_size))
v2 = reshape(v, (T, top_dim*N, batch_size))

ode_data = hcat(x2, v2)   # [time, x...y...u...v, batch]
Y0 = ode_data[1,:,:]      #u0 = [coord, batch_size] matrix:


#-------------------------Misc tools ---------------------------------

#= Try this with grad_U later...
function dudt(u,p,t)
    #only handle du being array of length 9 at the moment
    # would be nice to be able to handle batching eventually
    # for i in 1:numBatches
    # end
    Msquared = reshape(u[:],(3,3))^2;
    traceMsq = tr(Msquared);
    Msquared = reshape(Msquared,(9));
    #du[:] = reshape(-Msquared + tr(Msquared)*(1.0/3.0)*Π,(9,1));
    return -Msquared + traceMsq*dnn(u,p);
end

=#


#-----------------Neural ode Model -------
Fs = FastChain(FastDense(2*top_dim*N, h, tanh),
               FastDense(h, 2*top_dim*N))


p0 = initial_params(Fs);
solver = Tsit5()

prob_neuralode = ODEProblem(Fs,Y0,tspan,p0)
s = concrete_solve(prob_neuralode,solver,Y0,p0,saveat = t);

solver = Tsit5()
function loss_neuralode(p)
    pred = Array(concrete_solve(prob_neuralode,solver,Y0,p,saveat=t));
    loss = (1/datasize)*sum(abs2,ode_data .- pred)
    return loss;
end


# prob_neuralode = NeuralODE(Fs, tspan, Tsit5(), saveat = t)  #odeint
#
#
# function predict_neuralode(p)
#   Array(prob_neuralode(Y0, p))
#   # prob_neuralode(Y0,p)
# end
#
#
# function loss_neuralode(p)
#     pred = predict_neuralode(p)
#     loss = sum(abs2, ode_data .- pred)
#     return loss, pred
# end


#-------------visualizing learning:

theme(:juno)
test_freq = 20
iter = 0
cb = function (p,l,pred;doplot=true) #callback function to observe training
  global iter += 1
  if iter % test_freq == 0
    display(l)
    # plot current prediction against data to viz training
    # pl = plot(t,ode_data[11,:],label="data", color = "white")
    # plot!(pl,t,pred[11,:],label="prediction", color = "cyan")
    # display(plot(pl))
  end
  return false
end


#---------Training
result_neuralode1 = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = cb,
                                          maxiters = 1600)


result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                          result_neuralode1.minimizer,
                                          LBFGS(),
                                          cb = cb, maxiters = 400)




#=
#want true data to be in form [coord, time step]

Fs = FastChain(FastDense(20, 20, tanh),
               FastDense(20, 20))


prob_neuralode = NeuralODE(Fs, tspan, Tsit5(), saveat = t)  #odeint


function predict_neuralode(p)
  Array(prob_neuralode(Y0, p))
  # prob_neuralode(Y0,p)
end


function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end


#-------------visualizing learning:

theme(:juno)
test_freq = 20
iter = 0
cb = function (p,l,pred;doplot=true) #callback function to observe training
  global iter += 1
  if iter % test_freq == 0
    display(l)
    # plot current prediction against data to viz training
    pl = plot(t,ode_data[11,:],label="data", color = "white")
    plot!(pl,t,pred[11,:],label="prediction", color = "cyan")
    display(plot(pl))
  end
  return false
end




#------------Training:


result_neuralode1 = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = cb,
                                          maxiters = 1600)


result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                          result_neuralode1.minimizer,
                                          LBFGS(),
                                          cb = cb, maxiters = 400)




=#
