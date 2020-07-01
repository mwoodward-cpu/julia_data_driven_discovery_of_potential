#Still needs batching for larger data sets
#for now we will just use very small data sets (short time spans)

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
print(size(ode_data))


# dudt2 = FastChain((x, p) -> x.^3,
#                   FastDense(2, 50, tanh),
#                   FastDense(50, 2))


#----------by hand ------

# model = Chain(Dense(2, 50, tanh), Dense(50, 2))
#
# p, re = Flux.destructure(model)
# dudt!(u, p, t) = re(p)(u)
# # u0 = rand(2)
#
# prob = ODEProblem(dudt!, u0, tspan, p)
# prob_neuralode = solve(prob,Tsit5(), saveat = tsteps)




#-----------working example:

dudt2 = FastChain(FastDense(2, 70, tanh),
                  FastDense(70, 2))


prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)



#----------loss and prediction


function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    #loss = mean(abs.(pred.-ode_data))
    return loss, pred
end


theme(:juno) #set the theme for plotting

#displays only every test_freq iters:
test_freq = 50
iter = 0
cb = function (p,l,pred;doplot=true) #callback function to observe training
  global iter += 1
  if iter % test_freq == 0
    display(l)
    # plot current prediction against data to viz training
    pl = plot(tsteps,ode_data[1,:],label="data", color = "white")
    plot!(pl,tsteps,pred[1,:],label="prediction", color = "cyan")
    display(plot(pl))
  end
  return false
end



#------------Training

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = cb,
                                          maxiters = 1000)

#
# result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, p,
#                                           ADAM(0.05), cb = cb,
#                                           maxiters = 1000)

#another training bootstraped from previous:

result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                          result_neuralode.minimizer,
                                          LBFGS(),
                                          cb = cb, maxiters = 1000)



#This works the best so far... however, this still needs work:

#1) batching
#2) translate torchdiffeq into julia for batched learning





#--------Batching:

# MAX_BATCHES = 1000
# k = 15 #batch size
# data1 = ((rand(1:size(ode_data)[2] -k), k) for i in 1:MAX_BATCHES)

#
# res = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), data, cb = cb)

# result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
#                                           ADAM(0.05), data1, cb = callback)#,
