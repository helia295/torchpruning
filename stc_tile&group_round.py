import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def SR(input, bins, max_offset=0.9):
    max_v = max_offset*input.abs().max()
    sf = max_v / bins
    y = (input / sf).abs()
    frac = y - y.floor()
    rnd = torch.rand(y.shape, device=y.device)
    j = rnd <= frac
    y[j] = y[j].ceil()
    y[~j] = y[~j].floor()
    y = y.clamp(-bins, bins)
    y = input.sign() * y

    return y * sf




def GSR(input, bins, num_nonzero, tile_size, tile_dim, group_size, group_dim):

    # compute tile shapes (just like computing group shapes)
    shape = input.shape
    tile_shape = list(input.unsqueeze(tile_dim).shape)
    tile_shape[tile_dim] = tile_shape[tile_dim+1]//tile_size
    tile_shape[tile_dim+1] = tile_size

    # compute masks to set tiles to Dense or Sparse
    input = input.view(*tile_shape)

    rand = torch.rand(tile_shape[:-1])
    r = 0.5     # not sure if it works like this but am hoping r=0.5 will make half the tiles dense & half of them sparse
                # the lower r is, the less number of tiles become sparse => lower error
    tile_mask = rand <= r
    tile_mask = torch.repeat_interleave(tile_mask, repeats=tile_size, dim=tile_dim)


    # compute group shapes
    input = input.view(shape)       # make the input go back to original shape first
    group_shape = list(input.unsqueeze(group_dim).shape)
    group_shape[group_dim] = group_shape[group_dim+1]//group_size
    group_shape[group_dim+1] = group_size
    
    # compute masks to set values to 0 based on num_nonzero
    input = input.view(*group_shape)
    rand_mat = torch.rand(input.shape)
    idx = rand_mat.sort(group_dim+1)[0][:, :, :, :, num_nonzero-1].unsqueeze(-1)
    print('idx ', idx)
    mask = rand_mat <= idx

    tile_mask = tile_mask.view(mask.shape)
    #print('\n tile mask ', tile_mask)
    
    # set values to out.backward(torch.randn(1, 10))
    res = torch.zeros_like(input)
    
    res[tile_mask & mask] = input[tile_mask & mask]
    res[~tile_mask] = input[~tile_mask]
    print('result ', res)

    # apply stochastic scale due to pruning
    stochastic_scale = 1        # ??? i tried [1, r, 1/r] but 1 gave the most 'stable' low error (sometimes r resulted in really small error but not consistent)
    res = res * stochastic_scale

    #stochastically round res
    res = SR(res, bins)
    res = res.view(*shape)

    return res



B = 1
N = 1
C = 4
W = 4
H = 8
kW = 2
kH = 8


iters = 1000
bb = [1, 16, 32]
x = torch.Tensor(B, C, W, H).normal_()
w = torch.Tensor(N, C, kW, kH).normal_()
y = F.conv2d(x, w)

for bins in bb:
    diffs = []
    diffs_g = []
    resg = 0
    res = 0
    for i in range(iters):
        xq = SR(x, bins)
        wq = SR(w, bins)
        xgq = GSR(x, bins, 2, 8, 3, 4, 3)
        wgq = GSR(w, bins, 2, 8, 3, 4, 3)
        yq = F.conv2d(xq, wq)
        ygq = F.conv2d(xgq, wgq)
        res += yq
        resg += ygq

        diffs.append(((res / (i + 1)) - y).abs().sum())
        diffs_g.append(((resg / (i + 1)) - y).abs().sum())

    plt.plot(diffs, label='SR: {}'.format(bins))
    plt.plot(diffs_g, label='GSR: {}'.format(bins))

plt.legend(loc=0)
plt.show()