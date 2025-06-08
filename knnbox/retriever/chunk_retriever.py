r"""
Retriever for chunk-based k-NN MT.
Operates statelessly w.r.t. active_chunks, expects
them to be passed.
"""

import torch
from knnbox.retriever.utils import retrieve_k_nearest

class ChunkRetriever:
    def __init__(self, datastore, k_new, max_active_chunks=None):
        """
        k_new here corresponds to the number of *new* chunks to retrieve; the total number of chunks can range up to max_active_chunks
        """
        self.datastore = datastore
        self.k_new = k_new
        self.max_active_chunks = max_active_chunks

        # check that we have the correct attributes
        if not hasattr(self.datastore, "faiss_index") or \
           self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys")
        # Ensure 'vals' and 'real_lens' are loaded as they are needed for retrieval content
        if "vals" not in self.datastore:
            self.datastore.load_array("vals")
        if "real_lens" not in self.datastore:
            self.datastore.load_array("real_lens")

    @torch.no_grad()
    def retrieve_and_manage_slots(self, query_tensor,
            active_slots_tokens,
            active_slots_distances,
            active_slots_real_lens,
            active_slots_positions,
            active_slots_valid_mask
            ):
        """"
        Retrieves k_new chunks for a each hypothesis (arranged into batches), added to the first available invalid slots.

        Chunks become invalid if position >= real_len (handled by decoder)

        query tensor should be of shape [B*Beam, 1, D]
        """
        device = query_tensor.device
        n_hypotheses, N_SLOTS, _ = active_slots_tokens.shape

        # maybe unnecessary?
        if self.k_new <= 0:
            return (active_slots_tokens, active_slots_distances,
                    active_slots_real_lens, active_slots_positions, active_slots_valid_mask)

        # batched kNN search
        faiss_results = retrieve_k_nearest(query_tensor, self.datastore.faiss_index["keys"], self.k_new)

        indices = faiss_results["indices"] # [B*Beam, k_new]
        distances = faiss_results["distances"]

        flattened_indices = indices.flatten()
        new_vals_padded = torch.tensor(self.datastore["vals"].data[flattened_indices], device=device).view(n_hypotheses, self.k_new, -1)
        new_real_lens = torch.tensor(self.datastore["real_lens"].data[flattened_indices], device=device).view(n_hypotheses, self.k_new)

        # vectorizes slot finding & assignment across hypotheses
        for i in range(n_hypotheses):
            # finds available invalid slots for hyp i
            invalid_slot_indices_i = (~active_slots_valid_mask[i]).nonzero(as_tuple=True)[0]

            n_invalid = invalid_slot_indices_i.size(0)
            if n_invalid == 0:
                print("No valid slots, continuing")
                continue
            n_new_chunks = min(self.k_new, n_invalid)

            # gets the first n invalid slots
            slots = invalid_slot_indices_i[:n_new_chunks]

            active_slots_tokens[i, slots] = new_vals_padded[i, :n_new_chunks]
            active_slots_distances[i, slots] = distances[i, :n_new_chunks]
            active_slots_real_lens[i, slots] = new_real_lens[i, :n_new_chunks]
            active_slots_positions[i, slots] = 0
            active_slots_valid_mask[i, slots] = True

        return (active_slots_tokens, active_slots_distances,
                active_slots_real_lens, active_slots_positions, active_slots_valid_mask)




