import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn as nn


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class WeightSVDHead(nn.Module):
    def __init__(self):
        super(WeightSVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered * weights.unsqueeze(1), src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            try:
                u, s, v = torch.svd(H[i])
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                r_det = torch.det(r)
                if r_det < 0:
                    u, s, v = torch.svd(H[i])
                    v = torch.matmul(v, self.reflect)
                    r = torch.matmul(v, u.transpose(1, 0).contiguous())
                R.append(r)

                U.append(u)
                S.append(s)
                V.append(v)
            except:
                print('svd error')

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, (weights.unsqueeze(1) * src).sum(dim=2, keepdim=True)) + (weights.unsqueeze(1) * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)


def get_groundtruth_corr(R_gt, t_gt, src, tgt, dist_threshold=2e-6, s2t=True):
    if s2t:
        src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)  # B, 3, N + B, 3, 1 -> B, 3, N
        dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)  # B,3,N,1 - B,3,1,N -> B, 3, N, N
    else:
        tgt_gt = torch.matmul(R_gt.transpose(2, 1), tgt - t_gt.unsqueeze(-1))
        dist = tgt_gt.unsqueeze(-1) - src.unsqueeze(-2)
    min_dist, min_idx = (dist ** 2).sum(1).min(-1)  # B,3,N,N -> B, N, N -> [B, npoint], [B, npoint]
    min_dist = torch.sqrt(min_dist)  # min distance from pi to pj
    # min_idx = min_idx.cpu().numpy()  # drop to cpu for numpy
    match_labels = (min_dist < dist_threshold).float()  # B, N
    # print(match_labels.sum(-1) / src.shape[-1])
    return min_dist, min_idx, match_labels


def batch_choice(data, k, p=None, replace=False):
    # data is [B, N]
    out = []
    for i in range(len(data)):
        out.append(np.random.choice(data[i], size=k, p=p[i], replace=replace))
    out = np.stack(out, 0)
    return out


def pairwise_distance(src, dst):
    # square of distance
    inner = 2 * torch.matmul(src.transpose(-1, -2).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = torch.sum(src ** 2, dim=-2, keepdim=True).transpose(-1, -2).contiguous() - inner + torch.sum(dst ** 2,
                                                                                                           dim=-2,
                                                                                                           keepdim=True)
    return distances


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None, :]) ** 2, dim=-1)









