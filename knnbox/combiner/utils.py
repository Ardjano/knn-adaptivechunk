r""" some utils function used for combiner """

import torch
import torch.nn.functional as F

def calculate_knn_prob(vals, distances, probability_dim, temperature, device, **kwargs):
    r"""
    How vanilla knn-mt calculates knn probs using retrieved vals and distances.
    """
    scaled_dists = - distances / temperature
    knn_weights = torch.softmax(scaled_dists, dim=-1)
    print(f"weights size: {knn_weights.size()}")
    print(f"distances size: {distances.size()}")
    print(f"vals size: {vals.size()} \n\n")
    print(f"probability dim: {probability_dim}")
    # print(vals)
    # # ONLY FOR DEBUGGING!
    # vals = vals[:, :, :, 0]

    # for i in range(vals.size()[0]):
    #     if (i % 4 == 0):
    #         print(f"Batch {i // 4}")
    #     print(f"weights {i}: {knn_weights[i]}")
    #     print(f"vals: {vals[i]}")
    #     print(f"distances: {distances[i]} \n\n")
    # assert False, "Debugged"
    B, S, K = vals.size()

    # construct prob
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    return knn_probs



def calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs):
    r"""
    How vanilla knn-mt calculate the combining probability.
    """
    neural_model_prob = F.softmax(neural_model_logit, dim=-1)
    combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)

    # some extra infomation
    extra = {}
    extra["neural_probs"] = neural_model_prob
    extra["unlog_combined_probs"] = combined_probs

    if log_probs:
        combined_probs =  torch.log(combined_probs)
    return combined_probs, extra


def calculate_knn_prob_with_merge_weight(vals, distances, merge_weights, probability_dim, temperature, device, **kwargs):
    r"""
    when the key-value pair has a merge weight.
    used by greedy-merge knn-mt
    """
    # consider merge weights here
    scaled_dists = - distances / temperature + torch.log(merge_weights.float())
    knn_weights = torch.softmax(scaled_dists, dim=-1)

    B, S, K = vals.size()

    # construct prob
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    return knn_probs

def calculate_chunk_knn_prob(padded_tokens, padded_distances, probability_dim, temperature, combiner_pad_value=-1, **kwargs):
    """
    Calculate p_kNN given padded, batched tensorized active chunks.
    """
    device=padded_tokens.device

    # padding with inf distance -> 0
    scaled_dists = -padded_distances / temperature
    knn_weights = torch.softmax(scaled_dists, dim=-1)

    # necessary trick to use scatter_add_
    scatter_indices = padded_tokens.clone()
    scatter_indices[scatter_indices == combiner_pad_value] = 0

    knn_probs = torch.zeros(
        padded_tokens.size(0),
        probability_dim, device=device)
    knn_probs.scatter_add_(
        dim=-1,
        index=scatter_indices,
        src=knn_weights)

    return knn_probs


