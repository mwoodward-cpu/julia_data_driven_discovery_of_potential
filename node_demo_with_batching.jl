

#Still needs batching for larger data sets



using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using Random, StatsBase

u0 = Float32[2.0; 0.0]
datasize = 340
tspan = (0.0f0, 10.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
print(size(ode_data))



dudt2 = FastChain(FastDense(2, 50, tanh),
                  FastDense(50, 2))


#batch this too?
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)


#batch this
function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
  # Array(prob_neuralode(ode_data[:,start], p))
end

# function loss_neuralode(p)
#     pred = predict_neuralode(p)
#     loss = sum(abs2, ode_data .- pred)
#     #loss = mean(abs.(pred.-ode_data))
#     return loss, pred
# end

#loss for batching:
function loss_neuralode(p, start, k)
    pred = predict_neuralode(p)
    loss = sum(abs2,ode_data[:,start:start+k_t] .- pred[:,start:start+k_t])
    #loss = mean(abs.(pred[:,start:start+k_t] .- ode_data[:,start:start+k_t]))
    #println(start)
    loss,pred
end


theme(:juno) #set the theme for plotting
test_freq = 20
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



#------------Training with basic batching:

MAX_BATCHES = 3000
k = 20    #batch size
k_t = 10  #batch time
data2 = ((rand(1:size(ode_data)[2] -k_t), k) for i in 1:MAX_BATCHES)
# data3 = (sample((1:size(ode_data)[2] - k_t), k; replace = false, ordered = false) for i in 1:MAX_BATCHES)
opt = ADAM(0.05)
# opt = RMSProp(0.005)

res = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, opt, data2; cb = cb)


result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                          res.minimizer,
                                          LBFGS(), data2;
                                          cb = cb)








# res = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, LBFGS(), data2; cb = cb)


# res = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), data, cb = cb)
# result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
#                                           ADAM(0.05), cb = cb,
#                                           maxiters = 1000)
#
#
#
# #another training bootstraped from previous:
#
#
# result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
#                                           res.minimizer,
#                                           LBFGS(), data2;
#                                           cb = cb)


#--------------- TorchDiffeq Batching
# top_dim = 2
# batch_time = 10
# batch_size = 20
# data_size = datasize
#
# batch_u = zeros(top_dim, batch_time+1 ,batch_size)
# #note: for each batch: the data reads as (coord, time[s:s+batch_time])
# function get_batch()
#     s = sample(1:(data_size - batch_time), batch_size; replace = false, ordered = false)
#     batch_u0 = ode_data[:,s]
#     batch_t = t[1:batch_time]
#     for i = 0:batch_time
#         batch_u[:,i+1,:] = ode_data[:,s.+i]
#     end
#     return batch_u0, batch_t, batch_u
# end
#
# batch_u0, batch_t, batch_u = get_batch()
#
