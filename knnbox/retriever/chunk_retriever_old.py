r"""
Retriever for chunk-based k-NN MT.
Operates statelessly w.r.t. active_chunks, expects
them to be passed.
"""

import torch
from knnbox.retriever.utils import retrieve_k_nearest

class ChunkRetriever:
    def __init__(self, datastore, k, max_active_chunks=None):
        """
        k here corresponds to the number of *new* chunks to retrieve; the total number of chunks can range up to max_active_chunks
        """
        self.datastore = datastore
        self.k = k
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
    def retrieve_and_add_chunks(self, query_tensor, batched_active_chunks):
        """"
        Retrieves k new chunks for a batch of queries and adds them
        to the corresponding active_chunks lists.

        Prunes if max_active_chunks is reached.

        query tensor should be of shape [B*Beam, 1, D]
        """
        n_hypotheses = query_tensor.shape[0]
        assert len(batched_active_chunks) == n_hypotheses, \
        "Mismatch between number of queries and batched active chunks"

        # maybe unnecessary?
        if self.k <= 0:
            return batched_active_chunks

        faiss_results = retrieve_k_nearest(query_tensor, self.datastore.faiss_index["keys"], self.k)

        indices = faiss_results["indices"] # [B*Beam, k]
        distances = faiss_results["distances"]

        new_vals_padded = torch.tensor(self.datastore["vals"].data[indices.flatten()], device=query_tensor.device).view(n_hypotheses, self.k, -1)
        new_real_lens = torch.tensor(self.datastore["real_lens"].data[indices.flatten()], device=query_tensor.device).view(n_hypotheses, self.k)

        # operates on flat batch*beam perspective; decoder
        # associates the active_chunks with the correct sequence
        for i in range(n_hypotheses):
            hyp_active_chunks = batched_active_chunks[i]

            # for every retrieved result
            for j in range(self.k):
                real_len = new_real_lens[i, j].item()
                if real_len > 0:
                    chunk_tokens = new_vals_padded[i, j, :real_len]
                    distance = distances[i, j]
                    hyp_active_chunks.append((chunk_tokens, distance, new_real_lens[i, j].clone()))

            if self.max_active_chunks is not None and len(hyp_active_chunks) > self.max_active_chunks:
                hyp_active_chunks.sort(key=lambda x: x[1].item())
                hyp_active_chunks[:] = hyp_active_chunks[:self.max_active_chunks]

        return batched_active_chunks




