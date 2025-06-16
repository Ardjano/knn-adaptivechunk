from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
import torch.nn.functional as F
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask,
    archs,
    disable_model_grad,
    enable_module_grad,
)
from knnbox.datastore import Datastore, ChunkDatastore
from knnbox.utils.chunk_state_cache import ChunkStateCache
from knnbox.retriever import Retriever, ChunkRetriever
from knnbox.combiner import Combiner, AdaptiveCombiner, ChunkCombiner

# from .adaptive_knn_mt import AdaptiveKNNMT
from .vanilla_knn_mt import VanillaKNNMT


@register_model("chunk_knn_mt")
class ChunkKNNMT(VanillaKNNMT):
    r"""
    The Chunk KNN-MT model, using chunked datastores.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add chunk knn-mt related args here
        """
        VanillaKNNMT.add_args(parser)
        # parser.add_argument("--chunk-size", type=int, metavar="N", default=4,
        #                     help="target chunk size")
        parser.add_argument("--max-chunk-size", type=int, metavar="N", default=10,
                            help="Maximum dynamic chunk size")
        parser.add_argument("--confidence-threshold", type=float, metavar="F", default=0.9,
                            help="Confidence threshold")
        parser.add_argument("--knn-k-padded", type=int, metavar="N", default=100,
                            help="Padded k for chunk combiner input, ensuring fixed tensor size for active chunks.")
        parser.add_argument("--max-active-chunks", type=int, default=100, help="Maximum number of active chunks at any point during generation")

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with ChunkKNNMTDecoder
        """
        return ChunkKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ChunkKNNMTEncoder(
            args,
            src_dict,
            embed_tokens
        )

# using this as a signal for a new batch, to clear cache
class ChunkKNNMTEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args # review

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        ret = super().forward(src_tokens, src_lengths, return_all_hiddens, token_embeddings)

        if self.args.knn_mode == "inference":
            global_vars()["chunk_knn_new_batch"] = True

        return ret


class ChunkKNNMTDecoder(TransformerDecoder):
    r"""
    The Chunk KNN-MT Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.args = args

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = ChunkDatastore(
                        path=args.knn_datastore_path,
                        dictionary_len=len(self.dictionary),
                        max_chunk_size=args.max_chunk_size,
                        confidence_threshold=args.confidence_threshold
                        # chunk_size=args.chunk_size
                    )
            self.datastore = global_vars()["datastore"]
            self.chunk_state_cache = None

        else:
            self.datastore = ChunkDatastore.load(args.knn_datastore_path, load_list=["vals", "real_lens"], load_network=True)
            if args.build_faiss_index_with_cpu:
                self.datastore.load_faiss_index("keys", move_to_gpu=False)
            else:
                self.datastore.load_faiss_index("keys")

            self.retriever = ChunkRetriever(datastore=self.datastore, k_new=args.knn_k, max_active_chunks=args.max_active_chunks)
            self.combiner = ChunkCombiner(
                lambda_=args.knn_lambda,
                temperature=args.knn_temperature,
                probability_dim=len(dictionary),
                k_padded=args.knn_k_padded
            )

            self.chunk_state_cache = ChunkStateCache(
                max_active_chunks=args.max_active_chunks,
                pad_idx=self.dictionary.pad(),
                max_chunk_size=self.datastore.max_chunk_size,
                # for debugging
                dictionary=self.dictionary
            )


    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.

        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        def _pad_chunk_and_get_length(chunk_values_tensor, max_len, pad_idx):
            real_len = chunk_values_tensor.size(0)
            padded_chunk = F.pad(chunk_values_tensor, (0, max_len-real_len), value=pad_idx)

            return padded_chunk, real_len

        if self.args.knn_mode == "build_datastore":
            def get_tgt_probs(probs, target):
                r"""
                Args:
                    probs: [B, T, dictionary]
                    target: [B, T]
                Return: [B, T]
                """
                B, T, C = probs.size(0), probs.size(1), probs.size(2)
                one_hot = torch.arange(0, C).to(target.device)[None, None].repeat(B, T, 1) == target[:, :, None]
                return (probs * one_hot.float()).sum(-1)

            # def get_entropy(probs):
            #     r"""probs: [B, T, dictionary]"""
            #     return - (probs * torch.log(probs+1e-7)).sum(-1)
            def chunk_as_string(chunk, dictionary):
                chunk_as_strings = [dictionary.string([token.item()]) for token in chunk]
                return "-".join(chunk_as_strings)

            def log_top_predictions(probs, target_ids, dictionary, k=20):
                batch_size = probs.size(0)
                seq_len = probs.size(1)

                for b in range(batch_size):
                    # print(f"Sentence {b} of {batch_size}")
                    sentence_probs = probs[b]
                    topk_probs, topk_indices = torch.topk(sentence_probs, k=k, dim=-1)

                    for t in range(seq_len):
                        actual_target_id = target_ids[b, t].item()
                        if actual_target_id == dictionary.pad():
                            continue

                #         actual_word = dictionary.string([actual_target_id])
                #         print(f"Step {t}")
                #         print(f"Actual target: {actual_word} (ID: {actual_target_id})")
                #         print(f"Top {k} predictions:")
                #         for i in range(k):
                #             pred_id = topk_indices[t, i].item()
                #             pred_prob = topk_probs[t, i].item()
                #             pred_word = dictionary.string([pred_id])
                #             print(f"      {i+1}. {pred_word:<15} (ID: {pred_id:<5}, Prob: {pred_prob:.4f})")

                #     print("-" * 40)
                # print(f"\n End of batch {b} \n")

            # # calulate probs
            output_logit = self.output_layer(x)
            output_probs = F.softmax(output_logit, dim=-1)
            # # get useful info
            target = self.datastore.get_target()
            log_top_predictions(output_probs, target, self.dictionary, k=5)
            target_prob = get_tgt_probs(output_probs, target) # [B, T]
            # entropy = get_entropy(output_probs) # [B, T]

            # # process pad
            # pad_mask = self.datastore.get_pad_mask()
            # keys = select_keys_with_pad_mask(x, pad_mask)
            # # save infomation to datastore
            # self.datastore["keys"].add(keys.half())

            pad_idx = self.dictionary.pad()
            batch_size = x.size(0)
            p_threshold = self.args.confidence_threshold
            max_chunk_size = self.args.max_chunk_size

            # identify non-padding in original target sequence
            original_pad_mask = target.ne(pad_idx)

            for b in range(batch_size):
                current_sentence_mask = original_pad_mask[b]

                s_keys = x[b, current_sentence_mask]
                s_targets = target[b, current_sentence_mask]
                s_target_probs = target_prob[b, current_sentence_mask]

                s_len = s_keys.size(0)

                if s_len == 0:
                    continue

                # initialize key, values
                current_chunk_key = s_keys[0]
                current_chunk_values = [s_targets[0]]

                for t in range(1, s_len):
                    target_t = s_targets[t]
                    prob_t = s_target_probs[t]

                    if prob_t > p_threshold and len(current_chunk_values) < max_chunk_size:
                        current_chunk_values.append(target_t)
                    else:
                        padded_chunk, real_len = _pad_chunk_and_get_length\
                            (torch.stack(current_chunk_values), max_len=max_chunk_size, pad_idx=pad_idx)

                        self.datastore["keys"].add(current_chunk_key.unsqueeze(0).half())
                        self.datastore["vals"].add(padded_chunk.unsqueeze(0))
                        # print(f"Tokens: {chunk_as_string(current_chunk_values, self.dictionary)}\nLength: {real_len}")
                        self.datastore["real_lens"].add(\
                            torch.tensor([real_len], device=x.device, dtype=torch.long))

                        current_chunk_key = s_keys[t]
                        current_chunk_values = [target_t]

                # add last chunk
                padded_chunk, real_len = _pad_chunk_and_get_length\
    (torch.stack(current_chunk_values), max_len=max_chunk_size, pad_idx=pad_idx)
                self.datastore["keys"].add(current_chunk_key.unsqueeze(0).half())
                self.datastore["vals"].add(padded_chunk.unsqueeze(0))
                self.datastore["real_lens"].add(\
                    torch.tensor([real_len], device=x.device, dtype=torch.long))

        elif self.args.knn_mode == "inference":
            if self.chunk_state_cache is None:
                raise RuntimeError("ChunkStateCache not initialized for inference mode")

            if global_vars()["chunk_knn_new_batch"]:
                self.chunk_state_cache.clear()
                global_vars()["chunk_knn_new_batch"] = False

            current_bsz = x.size(0)

            if not self.chunk_state_cache.is_initialized(current_bsz, x.device):
                self.chunk_state_cache.init_state(current_bsz, x.device)

            # advance chunk positions if more than bos
            if prev_output_tokens.size(1) > 1:
                self.chunk_state_cache.advance_slots()

            current_states_tuple = self.chunk_state_cache.get_states_tuple()
            new_states_tuple = self.retriever.retrieve_and_manage_slots(x, *current_states_tuple)
            self.chunk_state_cache.update_states_tuple(new_states_tuple)

            # pass the state through extra for get_normalized_probs
            extra["active_slots_state"] = self.chunk_state_cache.get_state_dict_for_combiner()

        if not features_only:
            x = self.output_layer(x)
        return x, extra


    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1.
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability
        """
        extra = net_output[1]
        if self.args.knn_mode == "inference" and extra and "active_slots_state" in extra:
            # debug
            current_batch_idxs = global_vars().get("current_batch_idxs", None)
            print(f"sent ids of batch: {current_batch_idxs} \n length: {len(current_batch_idxs)}")

            # 'padded' here refers to padding the number of active chunks to k_padded, to enable tensorized computation
            padded_tokens, padded_distances = self.chunk_state_cache.prepare_combiner_inputs(self.combiner.k_padded, self.combiner.combiner_pad_value, self.combiner.temperature, current_batch_idxs)

            knn_prob = self.combiner.get_knn_prob(padded_tokens, padded_distances)

            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)

            return combined_prob

        return super().get_normalized_probs(net_output, log_probs, sample)

    def reorder_incremental_state(
        self,
        incremental_state: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        super().reorder_incremental_state(incremental_state, new_order)
        if self.chunk_state_cache is not None:
            self.chunk_state_cache.reorder_state(new_order)

r""" Define some pck knn-mt's arch.
     arch name format is: chunk_knn_mt@base_model_arch
"""
@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("chunk_knn_mt", "chunk_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)
