
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

    return y * sf



def SRG(ts, bins, group_size):

    #stochastically round ts
    roud = SR(ts, bins)
    print('\n round: ', roud)

    ###create a random tensor with groups of 1 (only appear once randomly) and 0 (all the rest)
    m = torch.randint(0, group_size, ts.max(4)[1].shape)            #randomly generated indices for one hot tensor        
    oh = F.one_hot(m, num_classes=group_size).view(roud.shape)
    print('\n one hot: ', oh)

    #change all the elements != 1 in oh to the elements with same indices in xs
    m = torch.where(oh != 1, oh.float(), roud)
    print('\n m: ', m)

    return m * group_size




B = 1
N = 1
C = 32
W = 4
H = 4
kW = 3
kH = 3


iters = 1000
bb = [1, 8, 32]

x = torch.Tensor(B, C, W, H).normal_()
w = torch.Tensor(N, C, kW, kH).normal_()
y = F.conv2d(x, w)

xg = group(x, B, C, W, H, 8)
wg = group(w, N, C, kW, kH, 8)

for bins in bb:
    diffs = []
    diffs_g = []
    resg = 0
    res = 0
    for i in range(iters):
        xq = SR(x, bins)
        wq = SR(w, bins)
        xgq = SRG(xg, bins, 8)
        wgq = SRG(wg, bins, 8)
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