import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



def group(ts, N, C, W, H, group_size):

    xs = ts.permute(0, 2, 3, 1).contiguous().view(N, W, H, C//group_size, group_size)
    
    return xs


def ungroup(r, N, C, W, H):

    xh = r.view(N, W, H, C).permute(0, 3, 1, 2).contiguous()

    return xh


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
    print('y ', y*sf)

    return y * sf



def SRG(ts, bins, group_size, num_nonzero):

    #stochastically round ts
    roud = SR(ts, bins)
    #print('\n round: ', roud)

    #all random numbers tensor with same shape as ts or roud
    rand_mat = torch.rand(roud.shape)

    idx = rand_mat.sort(4)[0][:, :, :, :, num_nonzero-1].unsqueeze(-1)
    mask = rand_mat <= idx
    res = torch.zeros_like(roud)
    res[mask] = roud[mask]

    #returns a tensor of indices of ~num_nonzero~ largest values in each group 
    #k_th_quant = torch.topk(rand_mat, num_nonzero, 4)[1]
    #print('k quant ', k_th_quant)

    #all zeros tensor with same shape as ts or roud
    #zeros = torch.zeros_like(roud)

    #copy the values at k_th_quant indices from tensor roud to tensor zeros
    #res = zeros.scatter_(4, k_th_quant, roud)
    print('res: ', res*(group_size/num_nonzero))

    '''#ONLY KEEP ONE scenario:
    ###create a random tensor with groups of 1 (only appear once randomly) and 0 (all the rest)
    m = torch.randint(0, group_size, ts.max(4)[1].shape)             #randomly generated indices for one hot tensor     
    oh = F.one_hot(m, num_classes=group_size).view(roud.shape)
    print('\n one hot: ', oh)

    #change all the elements != 1 in oh to the elements with same indices in xs
    res = torch.where(oh != 1, oh.float(), roud)
    print('\n res: ', res)
    '''

    return res * (group_size/num_nonzero)




def GSR(input, bins, num_nonzero, tile_size, tile_dim, group_size, group_dim, prune_type):

    # compute tile shapes
    shape = input.shape
    tile_shape = list(input.unsqueeze(tile_dim).shape)
    tile_shape[tile_dim] = tile_shape[tile_dim+1]//tile_size
    tile_shape[tile_dim+1] = tile_size

    # compute masks to set tiles to Dense or Sparse
    input = input.view(*tile_shape)

    rand = torch.rand(input.shape)
    tile_idx = rand.sort(tile_dim)[0][:, :, :, :, num_nonzero-1].unsqueeze(-1)
    tile_mask = rand <= tile_idx


    # compute group shapes
    input = input.view(shape)
    group_shape = list(input.unsqueeze(group_dim).shape)
    group_shape[group_dim] = group_shape[group_dim+1]//group_size
    group_shape[group_dim+1] = group_size
    
    # compute masks to set values to 0 based on num_nonzero
    input = input.view(*group_shape)

    '''
    if prune_type == PRUNE_TYPE_MAX:
        idx = input.abs().sort(group_dim+1)[0]
        idx = idx.index_select(group_dim + 1,
                               torch.tensor([num_nonzero - 1],
                                            device=input.device))
        idx = idx.repeat_interleave(group_size, group_dim+1)
        mask = rand_mat <= idx

        stochastic_scale = 1 # likely 1 is incorrect here
    elif prune_type == PRUNE_TYPE_RANDOM:
        rand_mat = torch.rand(input.shape, device=input.device)
        idx = rand_mat.sort(group_dim+1)[0]
        idx = idx.index_select(group_dim + 1,
                               torch.tensor([num_nonzero - 1],
                                            device=input.device))
        idx = idx.repeat_interleave(group_size, group_dim+1)
        mask = input.abs() > idx

        # scale based on ratio of num_nonzero
        stochastic_scale = (group_size / num_nonzero)
    else:
        raise ValueError("Invalid prune_type: {}. Options are: " \
                         "PRUNE_TYPE_RANDOM, PRUNE_TYPE_MAX")
    '''

    # set values to out.backward(torch.randn(1, 10))
    res = torch.zeros_like(input)
    res[tile_mask & mask] = input[tile_mask & mask]

    # apply stochastic scale due to pruning
    res = res * stochastic_scale

    #stochastically round res
    res = SR(res, bins)
    res = res.view(*shape)

    return res




B = 1
N = 1
C = 8
W = 4
H = 4
kW = 3
kH = 3


iters = 1000
bb = [1, 8, 16]
x = torch.Tensor(B, C, W, H).normal_()
w = torch.Tensor(N, C, kW, kH).normal_()
y = F.conv2d(x, w)

xg = group(x, B, C, W, H, 4)
wg = group(w, N, C, kW, kH, 4)

for bins in bb:
    diffs = []
    diffs_g = []
    resg = 0
    res = 0
    for i in range(iters):
        xq = SR(x, bins)
        wq = SR(w, bins)
        xgq = SRG(xg, bins, 4, 2)
        wgq = SRG(wg, bins, 4, 2)
        xgq = ungroup(xgq, B, C, W, H)
        wgq = ungroup(wgq, N, C, kW, kH)
        yq = F.conv2d(xq, wq)
        ygq = F.conv2d(xgq, wgq)
        res += yq
        resg += ygq

        diffs.append(((res / (i + 1)) - y).abs().sum())
        diffs_g.append(((resg / (i + 1)) - y).abs().sum())

    plt.plot(diffs, label='SR: {}'.format(bins))
    plt.plot(diffs_g, label='SRG: {}'.format(bins))

plt.legend(loc=0)
plt.show()