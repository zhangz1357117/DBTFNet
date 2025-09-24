import torch
import torch.nn as nn
import torch.nn.functional as F


class DMKDLoss(nn.Module):
    """Dynamic Mutual Knowledge Distillation Loss (DMKD) for DBTF-Net model"""

    def __init__(self, T=2.0, smoothing_factor=3.0, lambda_m=0.5):
        super(DMKDLoss, self).__init__()
        self.T = T  # Temperature parameter for softening the distribution
        self.smoothing_factor = smoothing_factor  # Smoothing factor for dynamic weighting
        self.lambda_m = lambda_m  # Weight for the final combined loss

    def forward(self, output1, output2, target, epoch):
        """
        Compute the Dynamic Mutual Knowledge Distillation Loss.

        Parameters:
        - output1 (torch.Tensor): Predictions from the first branch (e.g., GDE-Branch).
        - output2 (torch.Tensor): Predictions from the second branch (e.g., LDA-Branch).
        - target (torch.Tensor): Ground truth labels.
        - epoch (int): Current epoch to adjust the dynamic weighting factor.

        Returns:
        - loss (torch.Tensor): The computed DMKD loss value.
        """

        # Apply softmax to get the probability distributions
        p_s = F.log_softmax(output1 / self.T, dim=1)
        p_t = F.softmax(output2 / self.T, dim=1)

        # Kullback-Leibler divergence between the two soft predictions
        kl_div = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)

        # Apply dynamic weighting based on the current epoch
        weight = 1.0 - torch.exp(-epoch / self.smoothing_factor)  # Adjust weight based on epoch

        # Binary Cross-Entropy Loss for each branch
        bce_loss1 = F.binary_cross_entropy_with_logits(output1, target)
        bce_loss2 = F.binary_cross_entropy_with_logits(output2, target)

        # Total BCE Loss is the sum of both branches
        bce_loss = bce_loss1 + bce_loss2

        # DMKD Loss is the weighted sum of BCE and KL Divergence
        dmkd_loss = bce_loss + weight * kl_div

        # Final loss function combines BCE and DMKD Loss with lambda_m weighting
        total_loss = self.lambda_m * bce_loss + (1 - self.lambda_m) * dmkd_loss

        return total_loss

