r"""
Combiner for chunk-based kNN MT.
Calculates p_kNN from active chunks for a single
hypothesis and combines with p_LM.
"""
import torch
import torch.nn.functional as F
from knnbox.combiner.utils import calculate_combined_prob

class ChunkCombiner:
    def  __init__():
        pass