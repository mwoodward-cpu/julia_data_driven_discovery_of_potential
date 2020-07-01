#in this file we attempt to use the second order neural ode demo in order
#to learn the potential.

#created June 23
#Author: Michael Woodward



#action items:
#1) first load the appropriate trajectory data.
#2) 

using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqSensitivity, Zygote, RecursiveArrayTools


u0 = Float32[0.; 2.]
du0 = Float32[0.; 0.]
tspan = (0.0f0, 1.0f0)
datasize = 20
t = range(tspan[1], tspan[2], length = datasize)

F = FastChain(FastDense(2, 50, tanh), FastDense(50, 2))


p = initial_params(F)
ff(du,u,p,t) = F(u,p)
prob = SecondOrderODEProblem{false}(ff, du0, u0, tspan, p)

function predict(p)
    Array(solve(prob, Tsit5(), p=p, saveat=t))
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end



opt = ADAM(0.01)

l1 = loss_n_ode(p)

cb = function (p,l,pred)
    println(l)
    l < 0.01
end

res = DiffEqFlux.sciml_train(loss_n_ode, p, opt, cb=cb, maxiters = 1000)
