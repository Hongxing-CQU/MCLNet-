from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import knn, index_points, WeightSVDHead
from loss import MappingMatrixLoss, OverlappingConsensusLoss, CorrespondencesConsensusLoss


def MLP(channels: list, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) for l, x in
                             zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        # x, _ = linear_attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class Aggregate(nn.Module):
    def __init__(self, dim):
        super(Aggregate, self).__init__()
        self.dim = dim
        self.proj_q = nn.Conv2d(self.dim, self.dim, 1)
        self.proj_k = nn.Conv2d(self.dim, self.dim, 1)
        self.proj_v = nn.Conv2d(self.dim, self.dim, 1)

        self.proj = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1),
            # nn.BatchNorm2d(self.dim),
            # nn.ReLU()
        )

    def forward(self, x):
        # x: b,dim,n,k
        pre = x
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)  # b,mid,n,k
        qk = torch.einsum('bcnk,bcnl->bnkl', q, k) / self.dim ** 0.5  # b,n,k,k
        prob = torch.softmax(qk, dim=-1)
        x = torch.einsum('bnkl,bcnl->bcnk', prob, v)  # b,mid,n,k
        x = self.proj(x)  # b,mid,n,k
        x = pre + x
        return x


def get_graph_feature(x, k=12):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class AGNN(nn.Module):
    def __init__(self, input_dims=3, emb_dims=256):
        super(AGNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(256, emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(emb_dims)

        self.agge1 = Aggregate(64)
        self.agge2 = Aggregate(64)
        self.agge3 = Aggregate(128)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.agge1(x)
        x1 = x.max(dim=-1)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.agge2(x)
        x2 = x.max(dim=-1)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.agge3(x)
        x3 = x.max(dim=-1)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.bn4(self.conv4(x)))

        return x


class SELayer(nn.Module):
    def __init__(self, channel_num, compress_rate):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Linear(channel_num, channel_num // compress_rate, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num // compress_rate, channel_num, bias=True),
            nn.Sigmoid()
        )

    def forward(self, feature):
        squeeze_tensor = self.gap(feature)
        squeeze_tensor = squeeze_tensor.view(squeeze_tensor.size(0), -1)
        fc_out = self.se(squeeze_tensor)
        output_tensor = feature * fc_out[:, :, None]
        return output_tensor


class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=256, num_heads=4):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads
        self.se1 = SELayer(channel_num=num_channels, compress_rate=16)

    def forward(self, feat):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5
        weight = torch.softmax(feat_attention, dim=-1)
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])
        message = self.fc_message(feat - message)
        message = self.se1(message)
        res = feat + message
        return res


class MultiScaleAttention(nn.Module):
    def __init__(self, input_dims=3, emb_dims=256):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(256, emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(emb_dims)

        self.atte1 = nn.Sequential(NonLocalBlock(64))
        self.atte2 = nn.Sequential(NonLocalBlock(64))
        self.atte3 = nn.Sequential(NonLocalBlock(128))
        self.atte4 = nn.Sequential(
            NonLocalBlock(emb_dims),
            NonLocalBlock(emb_dims),
            NonLocalBlock(emb_dims)
        )

        self.agge1 = Aggregate(64)
        self.agge2 = Aggregate(64)
        self.agge3 = Aggregate(128)

    def forward(self, x):
        x = get_graph_feature(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.agge1(x)
        x1 = x.max(dim=-1)[0]
        x1 = self.atte1(x1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.agge2(x)
        x2 = x.max(dim=-1)[0]
        x2 = self.atte2(x2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.agge3(x)
        x3 = x.max(dim=-1)[0]
        x3 = self.atte3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.atte4(x)

        return x


class CorrespondenceNet(nn.Module):
    def __init__(self):
        super(CorrespondenceNet, self).__init__()
        self.gnn = MultiScaleAttention(input_dims=10, emb_dims=256)

        self.estimate_nn = nn.Sequential(
            MLP(channels=[256, 64, 1]),
        )

    def forward(self, src, src_corr, rela_xyz, rela_dist):
        feats = torch.cat([src, src_corr, rela_xyz, rela_dist], dim=1)  #
        feats = self.gnn(feats)
        scores = self.estimate_nn(feats)

        return scores.squeeze(1)


class OverlappingConsensusLearning(nn.Module):
    def __init__(self, N_hat=0.75):
        super(OverlappingConsensusLearning, self).__init__()
        self.emb_nn = AGNN(input_dims=3, emb_dims=256)
        self.atte_gnn = AttentionalGNN(feature_dim=256, layer_names=['self', 'self', 'cross'] * 2)
        self.final_proj = nn.Conv1d(256, 256, 1, bias=True)
        self.overlap_score_estimate = nn.Sequential(
            MLP(channels=[256 * 3, 128, 1]),
        )
        self.N_hat = N_hat

    def forward(self, src_xyz, tgt_xyz):
        src_ff = self.emb_nn(src_xyz)
        tgt_ff = self.emb_nn(tgt_xyz)
        src_ff, tgt_ff = self.atte_gnn(src_ff, tgt_ff)
        src_ff = self.final_proj(src_ff)
        tgt_ff = self.final_proj(tgt_ff)
        src_global = torch.max(src_ff, dim=-1, keepdim=True).values  # b,c,1
        tgt_global = torch.max(tgt_ff, dim=-1, keepdim=True).values  # b,c,1
        src_os = self.overlap_score_estimate(torch.cat(
            [
                src_ff,
                src_global.repeat(1, 1, src_ff.shape[-1]),
                tgt_global.repeat(1, 1, src_ff.shape[-1])
            ], dim=1)).squeeze(1)  # b,n
        tgt_os = self.overlap_score_estimate(torch.cat(
            [
                tgt_ff,
                src_global.repeat(1, 1, tgt_ff.shape[-1]),
                tgt_global.repeat(1, 1, tgt_ff.shape[-1])
            ], dim=1)).squeeze(1)  # b,m

        src_idx = src_os.topk(k=int(src_xyz.shape[-1] * self.N_hat), dim=-1).indices  # b,k
        tgt_idx = tgt_os.topk(k=int(tgt_xyz.shape[-1] * self.N_hat), dim=-1).indices  # b,k

        src_overlap_xyz = torch.gather(src_xyz, dim=-1, index=src_idx[:, None].repeat(1, 3, 1))
        tgt_overlap_xyz = torch.gather(tgt_xyz, dim=-1, index=tgt_idx[:, None].repeat(1, 3, 1))
        src_overlap_ff = torch.gather(src_ff, dim=-1, index=src_idx[:, None].repeat(1, src_ff.shape[1], 1))
        tgt_overlap_ff = torch.gather(tgt_ff, dim=-1, index=tgt_idx[:, None].repeat(1, tgt_ff.shape[1], 1))

        ocl_result = {
            'src_os': src_os,
            'tgt_os': tgt_os,
            'src_overlap_xyz': src_overlap_xyz,
            'tgt_overlap_xyz': tgt_overlap_xyz,
            'src_overlap_ff': src_overlap_ff,
            'tgt_overlap_ff': tgt_overlap_ff
        }

        return ocl_result


class CorrespondencesConsensusLearning(nn.Module):
    def __init__(self, K=0.5):
        super(CorrespondencesConsensusLearning, self).__init__()
        self.corr_nn = CorrespondenceNet()
        self.K = K

    def forward(self, src_overlap_xyz, tgt_overlap_xyz, src_overlap_ff, tgt_overlap_ff):
        assert src_overlap_ff.shape[1] == tgt_overlap_ff.shape[1]
        # internal-consensus
        ff_dim = src_overlap_ff.shape[1]
        S = src_overlap_ff.permute(0, 2, 1) @ tgt_overlap_ff / ff_dim ** 0.5
        S_hat = F.softmax(S, dim=-1) * F.softmax(S, dim=-2)

        corr_idx = torch.max(S_hat, dim=-1).indices
        src_sampled = src_overlap_xyz
        src_corr_sampled = torch.gather(tgt_overlap_xyz, dim=-1, index=corr_idx[:, None].repeat(1, 3, 1))

        # external-consensus
        corr_xyz = src_sampled - src_corr_sampled
        weights = self.corr_nn(src_sampled, src_corr_sampled, corr_xyz, torch.norm(corr_xyz, dim=1, keepdim=True))
        prune_val, prune_idx = weights.topk(dim=-1, k=int(self.K * weights.shape[-1]))  # b, n//2
        src_sampled_2 = torch.gather(src_sampled, dim=-1, index=prune_idx[:, None, :].repeat(1, 3, 1))
        src_corr_sampled_2 = torch.gather(src_corr_sampled, dim=-1, index=prune_idx[:, None, :].repeat(1, 3, 1))

        if self.training:
            ccl_res = {
                'mapping_mat': S,
                'src_sampled': src_sampled,
                'src_corr_sampled': src_corr_sampled,
                'weights': weights,
                'prune_weights': prune_val,
                'prune_src_sampled': src_sampled_2,
                'prune_src_corr_sampled': src_corr_sampled_2
            }
        else:
            ccl_res = {
                'mapping_mat': None,
                'src_sampled': None,
                'src_corr_sampled': None,
                'weights': None,
                'prune_weights': prune_val,
                'prune_src_sampled': src_sampled_2,
                'prune_src_corr_sampled': src_corr_sampled_2
            }

        return ccl_res


class GHV(nn.Module):
    def __init__(self, args, group_nums=30, t=10, sigma=0.01):
        super(GHV, self).__init__()
        self.group_nums = group_nums
        self.t = t
        self.sigma = nn.Parameter(torch.Tensor([sigma]).float(), requires_grad=False)
        self.args = args

    def geometric_consensus(self, src_xyz, src_corr_xyz):
        with torch.no_grad():
            src_dist = torch.norm(src_xyz[:, :, :, None] - src_xyz[:, :, None, :], dim=1)
            tgt_dist = torch.norm(src_corr_xyz[:, :, :, None] - src_corr_xyz[:, :, None, :], dim=1)
            length_consensus = -(src_dist - tgt_dist) ** 2. / self.sigma
            length_consensus = torch.exp(length_consensus)
            length_consensus = torch.clamp(length_consensus, min=0.1)
        return length_consensus

    def one_iterate(self, src_xyz, src_corr_xyz, weights, seed_nums, group_nums=30):
        batch_size = src_xyz.shape[0]
        seed_idx = torch.multinomial(input=weights, num_samples=seed_nums, replacement=False)
        seed_weights = torch.gather(weights, index=seed_idx, dim=-1)

        seed_src = torch.gather(src_xyz, dim=-1, index=seed_idx[:, None].repeat(1, 3, 1))
        seed_src_corr = torch.gather(src_corr_xyz, dim=-1, index=seed_idx[:, None].repeat(1, 3, 1))

        length_consensus = self.geometric_consensus(seed_src, seed_src_corr)

        consensus_mat = length_consensus.view(batch_size * length_consensus.shape[-1], length_consensus.shape[-1])
        consensus_mat = consensus_mat / torch.sum(consensus_mat, dim=-1, keepdim=True)

        nn_idx = torch.multinomial(consensus_mat, num_samples=group_nums, replacement=False)
        nn_idx = nn_idx.view(batch_size, consensus_mat.shape[-1], group_nums)

        seed_src_nn = index_points(seed_src.permute(0, 2, 1), nn_idx)
        seed_src_corr_nn = index_points(seed_src_corr.permute(0, 2, 1), nn_idx)
        seed_weights_nn = index_points(seed_weights[:, :, None], nn_idx).squeeze(-1)

        seed_src_nn = seed_src_nn.permute(0, 1, 3, 2).view(batch_size * seed_nums, 3, group_nums)
        seed_src_corr_nn = seed_src_corr_nn.permute(0, 1, 3, 2).view(batch_size * seed_nums, 3,
                                                                     group_nums)
        seed_weights_nn = seed_weights_nn.view(batch_size * seed_nums, group_nums)

        src2 = seed_src_nn
        src_corr2 = seed_src_corr_nn
        w2 = seed_weights_nn

        src2_centered = src2 - src2.mean(dim=2, keepdim=True)
        src_corr2_centered = src_corr2 - src_corr2.mean(dim=2, keepdim=True)

        w2 /= w2.sum(dim=-1, keepdim=True)
        H = torch.matmul(src2_centered * w2.unsqueeze(1), src_corr2_centered.transpose(2, 1).contiguous())
        U, S, Vt = torch.svd(H.cpu())
        U, S, Vt = U.cuda(), S.cuda(), Vt.cuda()
        delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
        eye = torch.eye(3)[None, :, :].repeat(U.shape[0], 1, 1).to(U.device)
        eye[:, -1, -1] = delta_UV
        R = Vt @ eye @ U.permute(0, 2, 1)  # [b*g, 3, 3]
        t = torch.matmul(-R, src2.mean(dim=2, keepdim=True)) + src_corr2.mean(dim=2, keepdim=True)  # [b*g, 1, 3]

        R = R.view(src_xyz.shape[0], -1, 3, 3)
        t = t.view(src_xyz.shape[0], -1, 1, 3)

        pred_position = torch.einsum('bsnm,bmk->bsnk', R, src_xyz) + t.permute(0, 1, 3, 2)  # [b,g,3,s]
        L2_dis = torch.norm(pred_position - src_corr_xyz[:, None, :, :], dim=-2)  # [b,g,s]

        seedwise_fitness = torch.mean((L2_dis < self.args.seed_threshold).float(), dim=-1)  # [b, g]
        vv, batch_best_guess = seedwise_fitness.max(dim=-1)  # b
        vv = torch.mean(vv)
        final_R = R.gather(dim=1, index=batch_best_guess[:, None, None, None].expand(-1, -1, 3, 3)).squeeze(1)
        final_t = t.gather(dim=1, index=batch_best_guess[:, None, None, None].expand(-1, -1, 1, 3)).squeeze(1)
        return vv, final_R, final_t

    def forward(self, src_xyz, tgt_xyz, weights):
        weights = torch.where(torch.isnan(weights), torch.full_like(weights, 10e-8), weights)
        weights = torch.where(torch.isinf(weights), torch.full_like(weights, 1), weights)
        seed_nums = weights.shape[-1] // 2
        #
        Rs = None
        ts = None
        dist = torch.tensor([1e8]).to(src_xyz.device)
        weights = torch.softmax(weights, dim=-1)

        for it in range(self.t):
            vv, final_R, final_t = self.one_iterate(src_xyz, tgt_xyz, weights, seed_nums, group_nums=self.group_nums)
            if vv < dist:
                Rs = final_R
                ts = final_t
                dist = vv

        return Rs, ts.squeeze(1)


class MCLNet(nn.Module):
    def __init__(self, args):
        super(MCLNet, self).__init__()
        self.args = args
        self.OCL = OverlappingConsensusLearning(N_hat=args.N_hat)
        self.CCL = CorrespondencesConsensusLearning(K=args.K)
        self.GHV = GHV(args, group_nums=args.group_nums, sigma=args.sigma, t=args.t)
        self.svd_head = WeightSVDHead()

    def forward(self, src_xyz, tgt_xyz, R=None, t=None):
        if self.training is False and R is not None and t is not None:
            print('error')
            exit()
        ocl = self.OCL(src_xyz, tgt_xyz)
        ccl = self.CCL(ocl['src_overlap_xyz'], ocl['tgt_overlap_xyz'], ocl['src_overlap_ff'], ocl['tgt_overlap_ff'])
        ccl['prune_weights'] = F.softmax(ccl['prune_weights'], dim=-1)
        r_pred, t_pred = self.GHV(ccl['prune_src_sampled'], ccl['prune_src_corr_sampled'], ccl['prune_weights'])
        loss = 0
        if self.training:
            loss_m = MappingMatrixLoss(ccl['mapping_mat'], R, t, ocl['src_overlap_xyz'], ocl['tgt_overlap_xyz'],
                                       self.args.distance_threshold)
            loss_o = OverlappingConsensusLoss(ocl['src_os'], ocl['tgt_os'], R, t, src_xyz, tgt_xyz,
                                              self.args.distance_threshold)
            loss_c = CorrespondencesConsensusLoss(R, t, ccl['src_sampled'], ccl['src_corr_sampled'], ccl['weights'],
                                                  self.args.distance_threshold)
            loss = loss_m + loss_o + loss_c
        return r_pred, t_pred, loss
