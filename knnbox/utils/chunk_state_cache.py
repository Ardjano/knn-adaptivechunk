import torch

class ChunkStateCache:
    """
    Dedicated cache to manage the tensorized active chunks for ChunkKNNMT during inference.
    """
    def __init__(self, max_active_chunks, pad_idx, max_chunk_size):
        self.max_active_chunks = max_active_chunks
        self.pad_idx = pad_idx
        self.max_chunk_size = max_chunk_size

        # dictionary of state tensors
        self.state = None
        self.current_bsz = 0
        self.device = None

    def is_initialized(self, bsz, device):
        return self.state is not None and self.current_bsz == bsz and self.device == device and self.state["tokens"].size(0) == bsz #review

    def init_state(self, bsz, device):
        self.current_bsz = bsz
        self.device = device

        self.state = {
            "tokens": torch.full((bsz, self.max_active_chunks, self.max_chunk_size), self.pad_idx, dtype=torch.long, device=device),
            "distances": torch.full((bsz, self.max_active_chunks), float('inf'), dtype=torch.float, device=device),
            "real_lens": torch.zeros((bsz, self.max_active_chunks), dtype=torch.long, device=device),
            "positions": torch.zeros((bsz, self.max_active_chunks), dtype=torch.long, device=device),
            "valid_mask": torch.zeros((bsz, self.max_active_chunks), dtype=torch.bool, device=device)
        }

    def get_states_tuple(self):
        if self.state is None:
            raise RuntimeError("Cache state accessed before init, shucks")
        return (self.state["tokens"], self.state["distances"], self.state["real_lens"], self.state["positions"], self.state["valid_mask"])

    def get_state_dict_for_combiner(self):
        return self.state

    def update_states_tuple(self, updated_states):
        if self.state is None:
            raise RuntimeError("Cache state not initialized before update")

        self.state["tokens"], self.state["distances"], self.state["real_lens"], self.state["positions"], self.state["valid_mask"] = updated_states

    def reorder_state(self, new_order):
        if self.state is not None:
            for key in self.state:
                # reorder rows of state tensors according to new beam order
                self.state[key] = self.state[key].index_select(0, new_order)
            self.current_bsz = new_order.size(0)

    def advance_slots(self):
        """Increments positions of all active chunks, invalidates consumed chunks."""
        if self.state is not None and self.state["valid_mask"].any():
            # cute check, only valid positions get incremented
            self.state["positions"] += self.state["valid_mask"].long()
            self.state["valid_mask"] &= (self.state["positions"] < self.state["real_lens"])

    def clear(self):
        """Called at start of new source batch."""
        self.state = None
        self.current_bsz = 0

    def prepare_combiner_inputs(self, k_padded, combiner_pad_value):
        if self.state is None:
                raise RuntimeError("Cache state must be initialized via initialize_state() before preparing combiner inputs.")

        n_hypotheses = self.current_bsz
        device = self.device

        current_mask = self.state["valid_mask"]

        padded_tokens = torch.full((n_hypotheses, k_padded), combiner_pad_value, dtype=torch.long, device=device)
        padded_distances = torch.full((n_hypotheses, k_padded), float('inf'), dtype=torch.float, device=device)

        for i in range(n_hypotheses):
            active_indices = current_mask[i].nonzero(as_tuple=True)[0]
            n_active = active_indices.numel()

            if n_active > 0:
                # review
                copylen = min(n_active, k_padded)
                copyslots = active_indices[:copylen]

                current_positions = self.state["positions"][i, copyslots]
                current_tokens = self.state["tokens"][i, copyslots].gather(1, current_positions.unsqueeze(1)).squeeze(1)

                padded_tokens[i, :copylen] = current_tokens
                padded_distances[i, :copylen] = self.state["distances"][i, copyslots]

        return padded_tokens, padded_distances











