import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune



'''Function that takes a tensor and group gradients together'''
def group(ts, N, C, H, W, group_size):

    #xs = ts.view(N, C//group_size, group_size, H, W)
    xs = ts.permute(0, 2, 3, 1).contiguous().view(N, H, W, C//group_size, group_size)
    
    
    return xs



'''Function that takes a grouped tensor and turn it back to original dimension'''
def ungroup(r, N, C, H, W):

    #xh = r.view(N, C, H, W)
    xh = r.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()


    return xh



'''Function that keeps only the max value in each group,
sets the rest to 0'''
def keepMax(ts, group_size):

    #   create a new tensor with the same shape as ts
    #   set each group's max value to 1 and the rest to 0
    r = F.one_hot(ts.max(4)[1], num_classes=group_size).view(ts.shape)
    print('r1: ', r)
    

    #   set all the 1s aka the max values to their original values
    r = torch.where(r < 1, r.float(), ts)
    print('\n prune: ', r)

    return r


    '''
    #?????????? another way maybe?????
    #z = torch.zeros_like(xs)
    #z = z.scatter_(2, (xs.max(4)[1]), xs.max(4)[0])
    #??? index tensor (xs.max(4)[1]) shape: 2,2,2,1 while self tensor (xs) shape 2,2,2,1,2 -> error 
    '''



'''Function that set the whole group to 0 if the group's mean < tau'''
def groupPruning(ts, dims, group_size, tau):

    #a tensor of groups' means (shape ([2, 1, 2, 2]))
    means = torch.mean(ts, dim=dims)
    print('\n means: ', means)
    
    #repeat each group's mean the same times with the # elements in the group 
    q = torch.repeat_interleave(means, repeats=group_size, dim=dims-1)

    #reshape to the same shape with original tensor
    q = q.view(ts.shape)
    print('\n tile means: ', q)

    #prune every value of a group to 0 if abs(group's mean) < tau
    boo_1 = q.abs() < tau
    ts[boo_1] = 0

    #stochastically prune every value of a group to 0/tau/-tau
    #r = torch.rand(ts.shape, device=ts.device)
    #boo_2 = q.abs() < (tau * r)                    #Error: will sometimes set items in a group to different values
    #boo_3 = q < 0

    #ts[boo_1 & boo_2] = 0
    #ts[boo_1 & (~boo_2) & boo_3] = -tau
    #ts[boo_1 & (~boo_2) & (~boo_3)] = tau

    return ts


#Function that keeps only 1 random element inside each group of a tensor and set the rest to 0
def groupStcRound(ts, group_size, bins):

    #stochastically round ts
    roud = SR(ts, bins)
    print('\n round: ', roud)

    #create a random tensor with groups of 1 (only appear once randomly) and 0 (all the rest)
    m = torch.randint(0, group_size, ts.max(4)[1].shape)           #randomly generated indices  
    #print('\n xs shape: ', ts.max(2)[1].shape, ' , m shape: ', m.shape)
    oh = F.one_hot(m, num_classes=group_size).view(ts.shape)
    print('\n one hot: ', oh)

    #change all the elements != 1 in oh to the elements with same indices in xs
    m = torch.where(oh != 1, oh.float(), roud)
    print('\n m: ', m)

    return m



def SR(x, bins):
    max_v = x.abs().max()
    sf = max_v / bins
    y = (x / sf).abs()
    frac = y - y.floor()
    rnd = torch.rand(y.shape)
    j = rnd <= frac
    y[j] = y[j].ceil()
    y[~j] = y[~j].floor()
    y = x.sign() * y

    return y * sf


def main():

    tau = 0.4

    N = 1
    C = 4
    H = 2
    W = 4

    x = torch.Tensor(N, C, H, W).normal_()
    print('\n tensor: ', x)

    #change the dimensions' order to easily work around 1 specific dim
    group_size = 4
    xs = group(x, N, C, H, W, group_size)       #shape ([2, 1, 2, 2, 2])
    print('\n group: ', xs)

    #prune everything except the max value of each group
    #r = keepMax(xs, group_size)

    

    #randomly choose one value per group to keep while setting the rest to 0
    bins = 10
    r = groupStcRound(xs, group_size, bins)

    zeros = torch.zeros_like(xs)

    rand_mat = torch.rand(xs.shape)
    print('rm: ', rand_mat.shape)
    k_th_quant = torch.topk(rand_mat, 3, 4, largest = False)[1]
    print('k quant ', k_th_quant.shape)

    res = zeros.scatter_(4, k_th_quant, xs)
    print('res: ', res.shape)

    #bool_tensor = rand_mat <= k_th_quant
    #bool_tensor = torch.eq(rand_mat, k_th_quant)
    #print('bool ', bool_tensor)
    #desired_tensor = torch.where(bool_tensor,torch.tensor(1),torch.tensor(0))
    #print(desired_tensor)


    
    #set whole group to 0 if group's mean < tau
    q = groupPruning(xs, 4, group_size, tau)
    print('\n group pruning: ', q)

    #change the dims back to original order
    xh = ungroup(r, N, H, W, C)
    print('ungroup: ', xh)
    

    #plotting histogram
    no_change = (q.abs() > tau).sum()
    print('--------')
    print(tau)
    print(no_change / q.nelement())
    print(q.shape)
    print('---------')
    plt.hist(q.view(-1).tolist(), bins=100)
    plt.show()
    


main()