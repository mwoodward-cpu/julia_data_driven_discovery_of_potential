#Author: Michael Woodward
#Updated: 06/08/20

#In this file we use a NN to approximate the potential, then use
#neural ode in order to learn F = - ∇U(|r_i - r_j|)



#Upload packages as we need them:
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

Y = vcat(x2,v2)         #[coord, time ->] time runs along columns Ground truth
dim = size(ode_data)[1]        #for brute force F_{s}
N = dim/4
Y0 = ode_data[:,1]

#SET HYPERPARAMATERS:
top_dim = 2
datasize = size(t)[1]
#niters = 1000  #numer of iterations through data set (epochs)
h = 70


#-------------------------Misc tools ---------------------------------





#-----------------Neural ode Model for Potential -------

#want true data to be in form [coord, time step]

U = FastChain(FastDense(2, 30, tanh),
               FastDense(30, 1))


#compute the gradient of U with respect to input x:

grad_U = zeros(N,2)
function grad_U(p)
  for i in 1:N
    # grad_U[i,1] = gradient.NN        #x-comp
    # grad_U[i,2] = gradient.NN        #y-comp
    grad_U[i] = gradient(U,x)[1]
  end
end


#Reshape grad_U:
Fs = zeros(2*N,2)
function F(p)
  for i in 1:2*N
    F[i,1] = v2[]




prob_neuralode = NeuralODE(Fs, tspan, Tsit5(), saveat = t)


function predict_neuralode(p)
  # Array(prob_neuralode(u0, p))
  prob_neuralode(Y0,p)
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
