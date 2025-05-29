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
# need to edit later
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner, AdaptiveCombiner
from .adaptive_knn_mt import AdaptiveKNNMT


@register_model("chunk_knn_mt")
class ChunkKNNMT(AdaptiveKNNMT):
    r"""
    The Chunk KNN-MT model, using chunked datastores.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add chunk knn-mt related args here
        """
        AdaptiveKNNMT.add_args(parser)
        # parser.add_argument("--chunk-size", type=int, metavar="N", default=4,
        #                     help="target chunk size")
        parser.add_argument("--max-chunk-size", type=int, metavar="N", default=10,
                            help="Maximum dynamic chunk size")
        parser.add_argument("--confidence-threshold", type=float, metavar="F", default=0.9,
                            help="Confidence threshold")

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

        else:
            self.datastore = ChunkDatastore.load(args.knn_datastore_path, load_list=["vals"], load_network=True)
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            # if args.knn_mode == "train_metak":
            if args.knn_mode == "inference":
                self.combiner = AdaptiveCombiner.load(args.knn_combiner_path)

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

        elif self.args.knn_mode == "train_metak" or self.args.knn_mode == "inference":
            ## query with x (x needn't to be half precision),
            ## save retrieved `vals` and `distances`
            self.retriever.retrieve(x, return_list=["vals", "distances"])

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
        if self.args.knn_mode == "inference" or self.args.knn_mode == "train_metak":
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


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
