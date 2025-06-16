import torch

class ChunkStateCache:
    """
    Dedicated cache to manage the tensorized active chunks for ChunkKNNMT during inference.
    """
    def __init__(self, max_active_chunks, pad_idx, max_chunk_size, dictionary=None):
        self.max_active_chunks = max_active_chunks
        self.pad_idx = pad_idx
        self.max_chunk_size = max_chunk_size
        self.dictionary = dictionary

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

    def prepare_combiner_inputs(self, k_padded, combiner_pad_value, temperature, sent_ids_batch=None):
        """
        For each beam hypothesis and its active chunks, retrieve target tokens at the current positions and their respective distances
        """
        if self.state is None:
                raise RuntimeError("Cache state must be initialized via initialize_state() before preparing combiner inputs.")

        n_hypotheses = self.current_bsz
        device = self.device

        current_mask = self.state["valid_mask"]

        padded_tokens = torch.full((n_hypotheses, k_padded), combiner_pad_value, dtype=torch.long, device=device)
        padded_distances = torch.full((n_hypotheses, k_padded), float('inf'), dtype=torch.float, device=device)

        # for diagnostics: calc softmax weights for all slots
        if sent_ids_batch is not None:
            scaled_dists = -self.state["distances"] / temperature
            knn_weights = torch.softmax(scaled_dists, dim=-1)

        for i in range(n_hypotheses):
            active_indices = current_mask[i].nonzero(as_tuple=True)[0]
            n_active = active_indices.numel()

            if n_active > 0:
                # diagnostics ctd
                if sent_ids_batch is not None:
                    active_positions = self.state["positions"][i, active_indices]
                    old_mask = (active_positions > 0)
                    n_old = old_mask.sum().item()
                    fraction_old = n_old / n_active

                    active_weights = knn_weights[i, active_indices]
                    prob_mass_old = active_weights[old_mask].sum().item()

                    beam_size = 4 # magic number
                    source_sent_index = i // beam_size

                    sent_id = sent_ids_batch[source_sent_index].item()

                    print(f"DIAG_SENT,{sent_id},"
                            f"fraction_old={fraction_old:.4f},"
                            f"prob_mass_old={prob_mass_old:.4f}")

                # review
                copylen = min(n_active, k_padded)
                copyslots = active_indices[:copylen]

                current_positions = self.state["positions"][i, copyslots]
                # gather along len dimension the current active positions;
                # unsqueezing required to match dimension
                current_tokens = self.state["tokens"][i, copyslots].gather(1, current_positions.unsqueeze(1)).squeeze(1)

                padded_tokens[i, :copylen] = current_tokens
                padded_distances[i, :copylen] = self.state["distances"][i, copyslots]

        return padded_tokens, padded_distances

    # # debug function written by
    def print_state_summary(
        self,
        sent_ids_batch= None, # Add sample dict
        hypothesis_idx = None,
        num_slots_to_show: int = 100,
        context_str: str = "Cache State"
    ):
        """
        Prints a summary of the cache state. If `sample` is provided, it groups
        hypotheses by their original source sentence ID.
        """
        if self.state is None:
            print(f"--- [{context_str}] Cache is not initialized. ---")
            return

        # Determine beam size to map hypotheses to source sentences
        print(f"--- [{context_str}] Summary (Device: {self.device}, Batch Size: {self.current_bsz}) ---")

        beam_size = 4 # magic number
        for i in range(self.current_bsz):
            if i % beam_size == 0:
                # --- CORRECTED SENTENCE ID LOGIC ---
                sent_id = -1 # Default
                if sent_ids_batch is not None:
                    source_sent_index = i // beam_size
                    if source_sent_index < len(sent_ids_batch):
                        sent_id = sent_ids_batch[source_sent_index].item()
                print(f"\n  --- Source Sentence ID: {sent_id} ---")
            valid_mask_hyp = self.state["valid_mask"][i]
            num_active_slots = valid_mask_hyp.sum().item()

            # Print hypothesis-specific info
            print(f"    [Hypothesis {i}] Active Slots: {num_active_slots} / {self.max_active_chunks}")

            if num_active_slots > 0 and num_slots_to_show > 0:
                print(f"      Details for first {min(num_active_slots, num_slots_to_show)} slots (Slot # | Pos/Len | Distance  | Next Token):")
                active_indices = valid_mask_hyp.nonzero(as_tuple=True)[0]

                for slot_idx in active_indices[:num_slots_to_show]:
                    pos = self.state["positions"][i, slot_idx].item()
                    real_len = self.state["real_lens"][i, slot_idx].item()
                    dist = self.state["distances"][i, slot_idx].item()

                    if pos < real_len:
                        token_id = self.state["tokens"][i, slot_idx, pos].item()
                        next_token_str = self.dictionary.symbols[token_id]
                    else:
                        next_token_str = "N/A (Consumed)"

                    print(f"        Slot {slot_idx.item():<3}:  {pos: >2}/{real_len:<2} | {dist:>9.4f} | {next_token_str}")

        print(f"--- End Summary ---")


    # def print_state_summary(
    #     self,
    #     hypothesis_idx=None,
    #     num_slots_to_show: int = 10,
    #     context_str: str = "Cache State"
    # ):
    #     """
    #     Prints a summary of the cache state.

    #     If `hypothesis_idx` is None, it prints a summary for ALL hypotheses in the batch.
    #     If `hypothesis_idx` is an integer, it prints only for that specific hypothesis.
    #     """
    #     if self.state is None:
    #         print(f"--- [{context_str}] Cache is not initialized. ---")
    #         return

    #     # Determine which hypothesis indices to loop over
    #     if hypothesis_idx is not None:
    #         # If a specific index is requested, put it in a list to use the same loop
    #         if hypothesis_idx >= self.current_bsz:
    #             print(f"--- [{context_str}] Error: Hypothesis index {hypothesis_idx} is out of bounds for batch size {self.current_bsz}. ---")
    #             return
    #         indices_to_print = [hypothesis_idx]
    #     else:
    #         # If no index is specified, loop over all hypotheses in the batch
    #         indices_to_print = range(self.current_bsz)

    #     print(f"--- [{context_str}] Summary (Device: {self.device}, Batch Size: {self.current_bsz}) ---")

    #     for i in indices_to_print:
    #         valid_mask_hyp = self.state["valid_mask"][i]
    #         num_active_slots = valid_mask_hyp.sum().item()

    #         print(f"  [Hypothesis {i}] Active Slots: {num_active_slots} / {self.max_active_chunks}")

    #         if num_active_slots > 0:
    #             print(f"    Details for first {min(num_active_slots, num_slots_to_show)} active slots (Slot # | Pos / Len | Distance  | Next Token):")
    #             active_indices = valid_mask_hyp.nonzero(as_tuple=True)[0]

    #             for slot_idx in active_indices[:num_slots_to_show]:
    #                 pos = self.state["positions"][i, slot_idx].item()
    #                 real_len = self.state["real_lens"][i, slot_idx].item()
    #                 dist = self.state["distances"][i, slot_idx].item()

    #                 # Correctly get the token symbol using the provided dictionary
    #                 if pos < real_len:
    #                     token_id = self.state["tokens"][i, slot_idx, pos].item()
    #                     next_token_str = self.dictionary.symbols[token_id]
    #                 else:
    #                     next_token_str = "N/A (Consumed)"

    #                 print(f"      Slot {slot_idx.item():<3}:  {pos: >2}/{real_len:<2} | {dist:>9.4f} | {next_token_str}")

    #     print(f"--- End Summary ---")
    # def print_state_summary(self, hypothesis_idx=0, num_slots_to_show=50, context_str="Cache State"):
    #     """Prints a summary of the cache state for a specific hypothesis."""
    #     if self.state is None:
    #         print(f"[{context_str}] Cache not initialized.")
    #         return

    #     if hypothesis_idx >= self.current_bsz:
    #         print(f"[{context_str}] Hypothesis index {hypothesis_idx} out of bounds for batch size {self.current_bsz}.")
    #         return

    #     print(f"--- [{context_str}] Summary for Hypothesis {hypothesis_idx} (Device: {self.device}, Batch Size: {self.current_bsz}) ---")
    #     valid_mask_hyp = self.state["valid_mask"][hypothesis_idx]
    #     num_active_slots = valid_mask_hyp.sum().item()
    #     print(f"  Total Active Slots: {num_active_slots} / {self.max_active_chunks}")

    #     if num_active_slots > 0:
    #         print(f"  Details for first {min(num_active_slots, num_slots_to_show)} active slots (Slot # | Pos / RealLen | Distance | Next Token ID):")
    #         active_indices = valid_mask_hyp.nonzero(as_tuple=True)[0]
    #         for i, slot_idx in enumerate(active_indices[:num_slots_to_show]):
    #             pos = self.state["positions"][hypothesis_idx, slot_idx].item()
    #             real_len = self.state["real_lens"][hypothesis_idx, slot_idx].item()
    #             dist = self.state["distances"][hypothesis_idx, slot_idx].item()
    #             next_token_id = self.state["tokens"][hypothesis_idx, slot_idx, pos].item() if pos < real_len else "N/A (Consumed)"
    #             print(f"    Slot {slot_idx.item():<3}: {pos: >3}/{real_len: <3} | {dist: >8.4f} | {self.dictionary.symbols[next_token_id]}")
    #     print(f"--- End Summary ---")
