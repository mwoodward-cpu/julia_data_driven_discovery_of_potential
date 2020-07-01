using Zygote

W = ones(2, 3); x = rand(3);

# @info x
# @info W
# Z = W*x
# print(Z)
# s = sum(W*x)
# println(s)

function f(W)
    sum(W*x)
end

grad = gradient(W -> sum(W*x), W)[1]
grad2 = gradient(f, W)[1]
@info grad
@info grad2





# f'(x) = 2x
# df(x) = gradient(f,x,nest=true)[1] # df is a tuple, [1] gets the first coordinate
