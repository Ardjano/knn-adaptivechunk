r"""
Combiner for chunk-based kNN MT.
Calculates p_kNN from active chunks for a single
hypothesis and combines with p_LM.
"""
import torch
import torch.nn.functional as F
from knnbox.combiner.utils import calculate_combined_prob

class ChunkCombiner:
    def  __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, active_chunks, device="cuda:0", **kwargs):
        """
        Calculates p_kNN based on first token of active chunks for
        a single hypothesis.
        """
        assert active_chunks, "No active chunks"



