# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
import datetime
import logging
import time
from fairseq.tasks.translation import TranslationConfig, TranslationTask, load_langpair_dataset
from fairseq.dataclass import ChoiceEnum
from fairseq import metrics
from fairseq import utils
import numpy as np

import torch
from fairseq.data import (
    FairseqDataset,
    ListDataset,
    data_utils,
    iterators,
)

from fs_plugins.data.nat_datasets_mod_length_filtering import NATLanguagePairDataset

from fs_plugins.data.multilingual_data_manager_mod import (
    MultilingualDatasetManager,
)

from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import FileContentsAction
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from fairseq import metrics, search, tokenizer, utils


EVAL_BLEU_ORDER = 4



###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])


logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch_bleu")
class TranslationMultiSimpleEpochTaskBLEU(TranslationMultiSimpleEpochTask):
    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
                            
        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        
        if getattr(args, "eval_bleu", False):
            from argparse import Namespace
            import json
            from fairseq.data import encoders
            
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--_ is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )
            try:
                gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            except Exception:
                alterred_string = getattr(args, "eval_bleu_args", "{}").replace('\'', "\"")
                gen_args = json.loads(alterred_string)
            self.sequence_generator = self.build_generator_for_bleu(
                [model], Namespace(**gen_args)
            )
            model.tokenizer_for_gleu = self.tokenizer
        
        return model

    def build_generator_for_bleu(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            for _lang in self.data_manager.langs:
                if tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(
                        _lang, tgt_langtok_spec
                    )
                    extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                    if extra_gen_cls_kwargs.get('symbols_to_strip_from_output') is None:
                        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}
                    else:
                        extra_gen_cls_kwargs["symbols_to_strip_from_output"].add(tgt_lang_tok)
        if extra_gen_cls_kwargs is not None:
            self.symbols_to_strip_from_output = extra_gen_cls_kwargs["symbols_to_strip_from_output"]
        else:
            self.symbols_to_strip_from_output = None
        return TranslationTask.build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs)


    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output



    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", torch.tensor(counts).cpu().numpy())
                metrics.log_scalar("_bleu_totals", torch.tensor(totals).cpu().numpy())
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu
        def decode(toks, escape_unk=False):
            if self.symbols_to_strip_from_output is not None:
                toks = torch.tensor([tok for tok in toks if tok.item() not in self.symbols_to_strip_from_output]).int()
            else:
                toks = torch.tensor([tok for tok in toks]).int()
            s = self.target_dictionary.string(
                toks,
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None, in_training=True)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, in_training=False
    ):
        with torch.no_grad():
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    if in_training is False:
                        tgt_lang_tok = self.data_manager.get_decoder_langtok(
                            self.args.target_lang, tgt_langtok_spec
                        )
                        src_tokens = sample["net_input"]["src_tokens"]
                        bsz = src_tokens.size(0)
                        prefix_tokens = (
                            torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                        )
                    else:
                        prefix_tokens = sample['target'][:, 0].unsqueeze(1)

                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
            else:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    if tgt_langtok_spec
                    else self.target_dictionary.eos(),
                )
    