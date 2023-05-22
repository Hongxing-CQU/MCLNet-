import torch
import torch.nn.functional as F
from utils import get_groundtruth_corr


def MappingMatrixLoss(S, R_gt, t_gt, src_overlap, tgt_overlap, dist_threshold=2e-6):
    P = F.softmax(S, dim=-1)
    _, Idx, Match_labels = get_groundtruth_corr(R_gt, t_gt, src_overlap, tgt_overlap, dist_threshold, s2t=True)
    tmp = torch.gather(P, index=Idx[:, :, None], dim=-1).squeeze(-1)  # b, n
    tmp = -torch.log(tmp + 1e-6) * Match_labels  # b, n
    loss = tmp.sum() / (Match_labels.sum() + 1e-6)
    return loss


def OverlappingConsensusLoss(src_os, tgt_os, R_gt, t_gt, src, tgt, dist_threshold=2e-6):
    _, min_idx, match_labels = get_groundtruth_corr(R_gt, t_gt, src, tgt, dist_threshold, True)
    _, min_idx2, match_labels2 = get_groundtruth_corr(R_gt, t_gt, src, tgt, dist_threshold, False)
    loss_os = F.binary_cross_entropy_with_logits(src_os, match_labels) \
              + F.binary_cross_entropy_with_logits(tgt_os, match_labels2)
    return loss_os


def CorrespondencesConsensusLoss(R, t, src_sampled, src_corr_sampled, weights, dist_threshold=2e-6):
    _, _, gt_label = get_groundtruth_corr(R, t, src_sampled, src_corr_sampled, dist_threshold=dist_threshold)
    loss_consensus = F.binary_cross_entropy_with_logits(weights, gt_label)
    return loss_consensus

