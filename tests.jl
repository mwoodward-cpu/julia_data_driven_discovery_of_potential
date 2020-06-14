#in this file we test

using NPZ, Flux, Zygote, DiffEqFlux

#set working directory:
cd("/Users/dayman/julia_sph_learning/potential_learning_MD")

#Upload the data from small.py (this is the ground truth data)

x = npzread("small_x.npy");   #(time, mol, coord)
v = npzread("small_v.npy");
A = npzread("small_A.npy");

#Now we need to implement neural ode where we will take the gradient of U first

Y = hcat(x,v)   #ground truth data in standard form [x y u v]

Y[:, 6:10, :]

f(x) = 3x^2 + 2x + 1;
df(x) = gradient(f, x)[1]; # df/dx = 6x + 2
println(df(1))


U = FastChain(FastDense(2,50,tanh),
              FastDense(50,1))

Θ = Flux.params(U)

grad_u(x) = gradient(U, x)[1]

println(grad_u(x))
