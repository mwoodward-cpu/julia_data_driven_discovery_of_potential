using NPZ, Flux, Zygote, DiffEqFlux
using OrdinaryDiffEq, Optim, Plots, LinearAlgebra


#set working directory:
cd("/Users/dayman/julia_sph_learning/potential_learning_MD")

# T = 7
# top_dim = 2
# batch_size = 3
# N = 5
# s = T*top_dim*batch_size*N
#
#
# a = Vector(1:s)
# a1 = reshape(a, (T, N, top_dim, batch_size))
# a2 = (reshape(a, (T, top_dim*N, batch_size)))
# # a3 = reshape(a2, (T, top))
# #want initial condtion to be u0 = [coord, batch_size] matrix:
#
# a20 = a2[1,:,:]
# #-------------Load data and set hyperparameters -------------------------
#
# #Try 5 mol for 100 timespteps:
# t = npzread("small_t.npy")
# x = npzread("small_x.npy");   #(time, mol, coord)
# v = npzread("small_v.npy");
# A = npzread("small_A.npy");
#
# dt = 0.01f0
# T = size(t)[1]
# tspan = (0.0f0, T)
#
# x2 = reshape(x, (T, 10))'
# v2 = reshape(v, (T, 10))'
#
# Y = vcat(x2,v2)         #[coord, time ->] time runs along columns Ground truth
# dim = size(ode_data)[1]        #for brute force F_{s}
# N = dim/4
# Y0 = ode_data[:,1]
#
# #SET HYPERPARAMATERS:
# top_dim = 2
# datasize = size(t)[1]
# #niters = 1000  #numer of iterations through data set (epochs)
# h = 70
#
#
# #-------------------------Misc tools ---------------------------------
#
#
# f(x) = 3x^2 + 2x + 1
# df(x) = gradient(f, x)[1]
# r = df(5)
# println(r)
#
#
# #-----------------Neural ode Model for Potential -------
#
# #want true data to be in form [coord, time step]
#
# # function U(x)
# #     U = FastChain(FastDense(2, 30, tanh),
# #                   FastDense(30, 1))
# # end
#
# U = FastChain(FastDense(2, 30, tanh),
#               FastDense(30, 1))
#
#
# dU(x) = gradient(U,x)[1]
#
# r = rand(2)
#
# dU(r)

# prinln(dU(1))

# function grad_U(x)
#     grad_U = gradient(x -> U(x), x)
# end
# rad = grad_U(1)
# W = rand(2, 5); b = rand(2);
# linear(x) = W * x .+ b
# grads = gradient(() -> sum(linear(x)), Params([W, b]))
#
# prinln(grads[W])

# rad = gradient(x -> 3x^2 + 2x + 1, 5)

# println(rad)




# Y = hcat(x,v)   #ground truth data in standard form [x y u v]
#
# Y[:, 6:10, :]
#
# f(x) = 3x^2 + 2x + 1;
# df(x) = gradient(f, x)[1]; # df/dx = 6x + 2
# println(df(1))
#
#
# U = FastChain(FastDense(2,50,tanh),
#               FastDense(50,1))
#
# Θ = Flux.params(U)
#
# grad_u(x) = gradient(U, x)[1]
#
# println(grad_u(x))
