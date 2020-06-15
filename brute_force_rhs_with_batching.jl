

#Still needs batching for larger data sets



using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using Random, StatsBase
using NPZ, Flux, Zygote, DiffEqFlux

using OrdinaryDiffEq, Optim, Plots


#set working directory:
cd("/Users/dayman/julia_sph_learning/potential_learning_MD")



#-------------Load data and set hyperparameters -------------------------

#Try 5 mol for T = 400 timespteps:
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



#-----------------Neural ode Model -------

#want true data to be in form [coord, time step]

Fs = FastChain(FastDense(20, 100, tanh),
               FastDense(100, 20))


prob_neuralode = NeuralODE(Fs, tspan, Tsit5(), saveat = t)  #odeint


function predict_neuralode(p)
  Array(prob_neuralode(Y0, p))
  # prob_neuralode(Y0,p)
end

#loss for batching:
function loss_neuralode(p, start, k)
    pred = predict_neuralode(p)
    #loss = sum(abs2,ode_data[:,start:start+k_t] .- pred[:,start:start+k_t])
    loss = mean(abs.(pred[:,start:start+k_t] .- ode_data[:,start:start+k_t]))  #best in torchdiffeq examples
    loss,pred
end


#-------Vis

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




#------------Training with basic batching:


MAX_BATCHES = 2000
k = 200    #batch size
k_t = 35  #batch time
data2 = ((rand(1:size(ode_data)[2] -k_t), k) for i in 1:MAX_BATCHES)
# data3 = (sample((1:size(ode_data)[2] - k_t), k; replace = false, ordered = false) for i in 1:MAX_BATCHES)



opt = ADAM(7e-4)
# opt = RMSProp(0.005)
res = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, opt, data2; cb = cb)

result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                          res.minimizer,
                                          LBFGS(), data2;
                                          cb = cb)
