import torch.nn as nn
import torch


class BaseFeatureExtractor(nn.Module):
    def __init__(self, config_model, threshold):
        super(BaseFeatureExtractor, self).__init__()
        pass

    def forward(self, x):
        feat = self.backbone(x)
        return feat

    def calculate_similarity(self, feat_query, feat_template, mask, training=True):
        """
        Calculate similarity for each batch
        input:
        feat_query: BxCxHxW
        feat_template: BxCxHxW
        output: similarity Bx1
        """

        B, C, H, W = feat_query.size(0), feat_query.size(1), feat_query.size(2), feat_query.size(3)
        mask_template = mask.repeat(1, C, 1, 1)
        num_non_zero = mask.squeeze(1).sum(axis=2).sum(axis=1)
        if training:  # don't use occlusion similarity during training
            similarity = self.sim_distance(feat_query * mask_template,
                                           feat_template * mask_template).sum(axis=2).sum(axis=1) / num_non_zero
        else:  # apply occlusion aware similarity with predefined threshold
            similarity = self.sim_distance(feat_query * mask_template,
                                           feat_template * mask_template)
            similarity = self.occlusion_sim(similarity).sum(axis=2).sum(axis=1) / num_non_zero
        return similarity

    def calculate_similarity_for_search(self, feat_query, feat_templates, mask, training=True):
        """
        calculate pairwise similarity:
        input:
        feat_query: BxCxHxW
        feat_template: NxCxHxW
        output: similarity BxN
        """
        B, N, C = feat_query.size(0), feat_templates.size(0), feat_query.size(1)

        similarity = torch.zeros((B, N)).type_as(feat_query)
        for i in range(B):
            query4d = feat_query[i].unsqueeze(0).repeat(N, 1, 1, 1)
            mask_template = mask.repeat(1, C, 1, 1)
            num_feature = mask.squeeze(1).sum(axis=2).sum(axis=1)
            sim = self.sim_distance(feat_templates * mask_template,
                                    query4d * mask_template)
            if training:
                similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
            else:
                sim = self.occlusion_sim(sim)
                similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
        return similarity, sim

    def calculate_global_loss(self, positive_pair, negative_pair, neg_pair_regression=None, delta=None):
        loss = self.loss(pos_sim=positive_pair, neg_sim=negative_pair)
        if delta is not None:
            mse = nn.MSELoss()
            delta_loss = mse(neg_pair_regression, delta)
            loss[2] += delta_loss
            return loss[0], loss[1], loss[2], delta_loss
        else:
            return loss[0], loss[1], loss[2]
