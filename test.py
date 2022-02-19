import torch
import matplotlib.pyplot as plt




def quantizeTensor(ts, sf, bin=1000):
        qT = torch.floor((ts/sf + 0.5)) * sf

        max = (bin-1)*sf

        qT = torch.clamp(qT, -max, max)

        return qT


def pruneTensor(ts, percentile):

        tau = torch.quantile(ts.abs(), percentile)

        print('tau: ', tau.item())
        print()

        ts = setValue(ts, tau)
        
        return ts


def setValue(g_vec, tau):

        tau = tau.item()
        r = torch.rand(g_vec.shape)
        ind_a = g_vec.abs() < tau
        ind_b = g_vec.abs() < tau * r
        ind_c = g_vec < 0

        g_vec[ ind_a & ind_b]  = 0
        g_vec[ ind_a & (~ ind_b) & ind_c ] = -tau 
        g_vec[ ind_a & (~ ind_b) & (~ ind_c) ] = tau

        return g_vec


def main():
        #x = torch.tensor([1. , 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        #x1 = torch.tensor([ 3., 1. , 6., 5., 8., 4., 7., 9., 2., 10.])
        #x2 = torch.tensor([1.22 , 44.2, 222.11, 0.1, 0.123, 239.2, 0.0001, 5.3, 2.111, 10.33])
        #x3 = torch.Tensor(10000).normal_()


        #print(x)
        #print()
        '''
        plt.hist(x3.tolist(), bins=100)
        plt.show()


        x_pruned = pruneTensor(x3, 0.5)


        plt.hist(x_pruned.tolist(), bins=100)
        plt.show()

        print(torch.allclose(x3, x_pruned))

        #print(x_pruned)


        '''
        #sf = 2
        #x_q = quantizeTensor(x, sf)

        #print(x)
        #print()
        #print(x_q)
        #print()
        #print(x_q / sf)


        B = 5
        N = 1
        C = 8
        W = 4
        H = 4
        kW = 3
        kH = 3

        x = torch.Tensor(B, C, W, H).normal_()
        plt.plot(x)

main()

