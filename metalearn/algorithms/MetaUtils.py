import torch


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
# The logic for calculating dynamical per-step loss weights (step_weights) 
# during inner-loop adaptation is derived and adapted directly from the official 
# GitHub repository of the MAML++ authors:
#
# Repository: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
# Reference : A. Antoniou et al., "How to train your MAML," ICLR 2019.
# ==============================================================================


def get_per_step_loss_weights(inner_steps: int, epoch: int, epochs: int, device=None):

    loss_weights = torch.ones(inner_steps, device=device) / inner_steps
    decay_rate = 1.0 / inner_steps / epochs
    min_val = 0.03 / inner_steps

    weights_list = []
    for i in range(inner_steps - 1):
        curr_val = max(loss_weights[i].item() - epoch * decay_rate, min_val)
        weights_list.append(curr_val)
    
    final_val = min(loss_weights[-1].item() + epoch * (inner_steps - 1) * decay_rate,
                    1.0 - (inner_steps - 1) * min_val)
    weights_list.append(final_val)

    return torch.tensor(weights_list, device=device)
