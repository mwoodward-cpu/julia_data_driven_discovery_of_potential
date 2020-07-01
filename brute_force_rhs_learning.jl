#in this code, we attempt to replicate what we have observed in pytorch:
#Brute force learning of RHS

#Author: Michael Woodward
#Updated: 06/09/20


#not used: #using PyCall, ThArrays,
using NPZ, Flux, Zygote, DiffEqFlux

using OrdinaryDiffEq, Optim, Plots


#set working directory:
cd("/Users/dayman/julia_sph_learning/potential_learning_MD")



#-------------Load data and set hyperparameters -------------------------

#Try 5 mol for 100 timespteps:
t = npzread("small_t.npy")
x = npzread("small_x.npy");   #(time, mol, coord)
v = npzread("small_v.npy");
A = npzread("small_A.npy");

dt = 0.01f0
T = size(t)[1]
tspan = (0.0f0, T)

x2 = reshape(x, (T, 10))'
v2 = reshape(v, (T, 10))'

ode_data = vcat(x2,v2)         #[coord, time ->] time runs along columns Ground truth
dim = size(ode_data)[1]        #for brute force F_{s}
Y0 = ode_data[:,1]

#SET HYPERPARAMATERS:
top_dim = 2
datasize = size(t)[1]
#niters = 1000  #numer of iterations through data set (epochs)
h = 70


#-------------------------Misc tools ---------------------------------





#-----------------Neural ode Model -------

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
