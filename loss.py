import torch
import torch.nn as nn
from torch.autograd.function import Function
import numpy as np


class SparseCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(SparseCenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        self.sparse_centerloss = SparseCenterLossFunction.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.centers.data.t())

    def forward(self, feat, A, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.sparse_centerloss(feat, A, label, self.centers, batch_size_tensor)
        return loss


class SparseCenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, A, label, centers, batch_size):
        ctx.save_for_backward(feature, A, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (A * (feature - centers_batch).pow(2)).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, A, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = feature - centers_batch
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        # A gradient
        grad_A = diff.pow(2) / 2.0 / batch_size

        counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), - A * diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return grad_output * A * diff / batch_size, grad_output * grad_A, None, grad_centers, None

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, A):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        inputs = inputs * A
        
        # get positive & negative dist
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an) 
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)