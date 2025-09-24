import torch
import torch.nn.functional as F


def dmkd_loss(output1, output2, target, epoch, temperature=2.0, smoothing_factor=3.0, lambda_m=0.5):
    """
    Dynamic Mutual Knowledge Distillation (DMKD) Loss for DBTF-Net model.

    Parameters:
    - output1 (torch.Tensor): Predictions from the first branch (e.g., GDE-Branch).
    - output2 (torch.Tensor): Predictions from the second branch (e.g., LDA-Branch).
    - target (torch.Tensor): True labels (ground truth).
    - epoch (int): Current training epoch, used to dynamically adjust distillation weight.
    - temperature (float): Temperature for softening the output distributions. Default is 2.0.
    - smoothing_factor (float): Smoothing factor for distillation, controlling the rate of change.
    - lambda_m (float): Weight for the combined loss, usually set to 0.5 for balancing.

    Returns:
    - torch.Tensor: The computed DMKD loss.
    """

    # Apply softmax to get the probability distributions
    prob1 = F.softmax(output1 / temperature, dim=-1)
    prob2 = F.softmax(output2 / temperature, dim=-1)

    # Kullback-Leibler divergence between the two soft predictions
    kl_div = F.kl_div(prob1.log(), prob2, reduction='batchmean')

    # Apply dynamic weighting based on the training epoch
    weight = 1.0 - torch.exp(-epoch / smoothing_factor)  # Dynamic weight increases as training progresses

    # Binary Cross-Entropy (BCE) Loss for each branch
    bce_loss1 = F.binary_cross_entropy_with_logits(output1, target)
    bce_loss2 = F.binary_cross_entropy_with_logits(output2, target)

    # Total BCE Loss is the sum of both branches
    bce_loss = bce_loss1 + bce_loss2

    # DMKD Loss is the weighted sum of BCE and KL Divergence
    dmkd_loss = bce_loss + weight * kl_div

    # Final total loss, combining BCE and DMKD Loss with the weight factor lambda_m
    total_loss = lambda_m * bce_loss + (1 - lambda_m) * dmkd_loss

    return total_loss


# Example usage:
output1 = torch.randn(32, 10)  # Example: Output from first branch (32 samples, 10 classes)
output2 = torch.randn(32, 10)  # Example: Output from second branch (32 samples, 10 classes)
target = torch.randint(0, 2, (32, 10)).float()  # Example: Ground truth (binary labels)
epoch = 5  # Example: Current epoch

loss = dmkd_loss(output1, output2, target, epoch)
print(f"DMKD Loss: {loss.item()}")
