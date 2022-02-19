import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune



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


mu = 0
variance = 1
sigma = math.sqrt(variance)
#x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
N = 1
C = 4
H = 2
W = 4

x = torch.Tensor(N, C, H, W).normal_()
#xr = SR(x, 10)
#plt.hist(x.view(-1).tolist(), bins=80)
#plt.hist(xr.view(-1).tolist(), bins=50)
    
plt.plot(x.view(-1).tolist(), stats.norm.pdf(x.view(-1).tolist(), mu, sigma))
#plt.plot(xr, stats.norm.pdf(xr, mu, sigma))
plt.show()