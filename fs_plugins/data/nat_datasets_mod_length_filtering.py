from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TransformEosLangPairDataset,
    TruncateDataset,
    LanguagePairDataset,
    data_utils
)

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from fairseq.data.language_pair_dataset import collate

class NATLanguagePairDataset(LanguagePairDataset):
    def __init__(self, skip_src_langtok, replace_src_langtok_with_tgt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_src_langtok = skip_src_langtok
        self.replace_src_langtok_with_tgt = replace_src_langtok_with_tgt

    def filter_indices_by_size(self, indices, max_sizes, src_upsample_scale):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        print('before length filtering:', len(indices))
        indices = indices[self.src_sizes[indices] * src_upsample_scale > self.tgt_sizes[indices]]
        print('after length filtering:', len(indices))
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
    
    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if len(res) > 0:
            res["net_input"]["src_lang_id"] = res["net_input"]['src_tokens'][:, 0].clone()
            res["net_input"]["tgt_lang_id"] = res["net_input"]['prev_output_tokens'][:, 1].clone()
            if getattr(self, "replace_src_langtok_with_tgt", False) and not getattr(self, "skip_src_langtok", False):
                res["net_input"]['src_tokens'][:, 0] = res["net_input"]["tgt_lang_id"]
                
                
        # try:
            
        # except Exception as e:
        #     print("HELLO?")

        if res.get("target", None) is not None and len(res) > 0:
            res["target"][:, 0] = self.tgt_dict.bos_index
        
        if getattr(self, "skip_src_langtok", False) and len(res) > 0:
            res["net_input"]['src_tokens'][:, 0] = self.src_dict.bos_index

        return res
    

class NATAppendTokenDataset(AppendTokenDataset):
    def filter_indices_by_size(self, indices, max_sizes, src_upsample_scale):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        print('before length filtering:', len(indices))
        indices = indices[self.src_sizes[indices] * src_upsample_scale > self.tgt_sizes[indices]]
        print('after length filtering:', len(indices))
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

class NATConcatDataset(ConcatDataset):
    def filter_indices_by_size(self, indices, max_sizes, src_upsample_scale, min_sent_length, training=False):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        from fairseq.data.data_utils import collect_filtered
        import numpy as np
        def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
            def compare_leq(a, b):
                return a <= b if not isinstance(a, tuple) else max(a) <= b

            def _check_size(idx):
                if isinstance(max_positions, float) or isinstance(max_positions, int):
                    return size_fn(idx) <= max_positions
                elif isinstance(max_positions, dict):
                    idx_size = size_fn(idx)
                    assert isinstance(idx_size, dict)
                    intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
                    return all(
                        all(
                            a is None or b is None or a <= b
                            for a, b in zip(idx_size[key], max_positions[key])
                        )
                        for key in intersect_keys
                    )
                else:
                    # For MultiCorpusSampledDataset, will generalize it later
                    if not isinstance(size_fn(idx), Iterable):
                        return all(size_fn(idx) <= b for b in max_positions)
                    return all(
                        a is None or b is None or a <= b
                        for a, b in zip(size_fn(idx), max_positions)
                    )

            def _check_size_src_ratio_training(idx):
                FILTER_RATIO = 2
                MIN_LENGTH = 5
                a1, b1 = size_fn(idx)
                a2, b2 = max_positions
                return a1 <= a2 and b1 <= b2 and a1 * FILTER_RATIO > b1 and a1 < b1 * FILTER_RATIO \
                            and a1 > MIN_LENGTH and b1 > MIN_LENGTH
            
            def _check_size_src_ratio_validate(idx):
                FILTER_RATIO = 999
                MIN_LENGTH = 1
                a1, b1 = size_fn(idx)
                a2, b2 = max_positions
                return a1 <= a2 and b1 <= b2 and a1 * FILTER_RATIO > b1 and a1 < b1 * FILTER_RATIO \
                            and a1 > MIN_LENGTH and b1 > MIN_LENGTH

            ignored = []
            if training:
                itr = collect_filtered(_check_size_src_ratio_training, indices, ignored)
            else:
                itr = collect_filtered(_check_size_src_ratio_validate, indices, ignored)
            indices = np.fromiter(itr, dtype=np.int64, count=-1)
            return indices, ignored

        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        return indices, ignored



class NATPrependTokenDataset(PrependTokenDataset):
    def filter_indices_by_size(self, indices, max_sizes, src_upsample_scale):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        print('before length filtering:', len(indices))
        indices = indices[self.src_sizes[indices] * src_upsample_scale > self.tgt_sizes[indices]]
        print('after length filtering:', len(indices))
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

class NATStripTokenDataset(StripTokenDataset):
    def filter_indices_by_size(self, indices, max_sizes, src_upsample_scale):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        print('before length filtering:', len(indices))
        indices = indices[self.src_sizes[indices] * src_upsample_scale > self.tgt_sizes[indices]]
        print('after length filtering:', len(indices))
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

class NATTransformEosLangPairDataset(TransformEosLangPairDataset):
    def filter_indices_by_size(self, indices, max_sizes, src_upsample_scale):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        print('before length filtering:', len(indices))
        indices = indices[self.src_sizes[indices] * src_upsample_scale > self.tgt_sizes[indices]]
        print('after length filtering:', len(indices))
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

class NATTruncateDataset(TruncateDataset):
    def filter_indices_by_size(self, indices, max_sizes, src_upsample_scale):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        print('before length filtering:', len(indices))
        indices = indices[self.src_sizes[indices] * src_upsample_scale > self.tgt_sizes[indices]]
        print('after length filtering:', len(indices))
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

