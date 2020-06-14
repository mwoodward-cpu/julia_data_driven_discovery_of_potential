#In this file we compile a list of useful functions that will be used later
#some of these are just rewritten tools from torchdiffeq:
using Random, StatsBase



#batching for stochastic gradient descent (translated from torchdiffeq)
batch_u = zeros(top_dim, batch_time+1 ,batch_size)
#note: for each batch: the data reads as (coord, time[s:s+batch_time])
function get_batch()
    s = sample(1:(data_size - batch_time), batch_size; replace = false, ordered = false)
    batch_u0 = u[:,s]
    batch_t = t[1:batch_time]
    for i = 0:batch_time
        batch_u[:,i+1,:] = u[:,s.+i]
    end
    return batch_u0, batch_t, batch_u
end








#Now lets see if we can copy the learning algorithm from torchdiffeq



#=
func = ODEFunc()
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
end = time.time()

time_meter = RunningAverageMeter(0.97)
loss_meter = RunningAverageMeter(0.97)

for itr in range(1, args.niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()
    pred_y = odeint(func, batch_y0, batch_t)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()

    time_meter.update(time.time() - end)
    loss_meter.update(loss.item())

    if itr % args.test_freq == 0:
        with torch.no_grad():
            pred_y = odeint(func, true_y0, t)
            loss = torch.mean(torch.abs(pred_y - true_y))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            visualize(true_y, pred_y, func, ii)
            ii += 1

=#










#----------------------Testing the tools-----------------------:










#Where we want to emulate the python
# def get_batch():
#     s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
#     batch_y0 = true_y[s]  # (M, D)
#     batch_t = t[:batch_time]  # (T)
#     batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
#     return batch_y0, batch_t, batch_y
#bt = 3
#bsize = 4
# u = rand(2,10)
# s = sample(1:7,4 ; replace = false, ordered = false)
# r = (u[:,s])
# e = u[:,s.+1]
# #e = (u[:,s.+1:3])
# #stack = hcat(r,e)





#(dim, batch_time+1, batch_size)
# batch_y = zeros(2,4,4)
# println(batch_y)
# for i = 0:3
#     batch_y[:,i+1,:] = u[:,s.+i]
# end
#
# println(batch_y)
