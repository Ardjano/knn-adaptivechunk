r"""
Combiner for chunk-based kNN MT.
Calculates p_kNN from active chunks for a single
hypothesis and combines with p_LM.
"""
import torch
import torch.nn.functional as F
from knnbox.combiner import Combiner
from knnbox.combiner.utils import calculate_chunk_knn_prob

class ChunkCombiner(Combiner):
    def __init__(self, lambda_, temperature, probability_dim, k_padded=100):
        super().__init__(lambda_, temperature, probability_dim)
        self.k_padded = k_padded
        self.combiner_pad_value = -1

    def get_knn_prob(self, padded_tokens, padded_distances, device="cuda:0", **kwargs):
        """
        Calculates p_kNN based on first token of active chunks.
        """
        # adds S = 1 dimension
        return calculate_chunk_knn_prob(padded_tokens, padded_distances, self.probability_dim, self.temperature, self.combiner_pad_value, **kwargs).unsqueeze(1)




