#What this does




using Flux #, Optim, DiffEqFlux

NODE = FastChain(FastDense(2, 70, tanh),
                  FastDense(70, 2), first)
# p = initial_params(NODE)
# NNODE = Chain(x -> [x],
#            Dense(1,32,tanh),
#            Dense(32,1),
#            first)

@info NNODE(1.0)

@info 1f0

g(t) = t*NNODE(t) + 1f0

using Statistics
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:3f0)



# opt = Flux.Descent(0.01)
opt = ADAM(0.05)
data2 = Iterators.repeated((), 3000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 200 == 0
    display(loss())
  end
end
display(loss())


Flux.train!(loss, Flux.params(NNODE), data2, opt; cb=cb)
# DiffEqFlux.sciml_train(loss, p, )

using Plots
t = 0:0.001:3.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True")





#one idea is to use this for solving the ode dY/dt = [v,  -sum(xi-xj)G(rij) ] = F(Y)

#replace cos above by F(Y), and replace Y by neural network
