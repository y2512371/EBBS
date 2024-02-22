from fs_plugins.others.rouge import Scorer, ROUGEConfig
import sys
import pickle
from evaluate import load
bertscore = load("bertscore")


rouge_cfg = pickle.load(open('my_scripts/eval_naus_rouge/rouge.cfg', 'rb'))

rouge_scorer = Scorer(rouge_cfg)

hyp_file = sys.argv[1]
ref_file = sys.argv[2]




# hyp_file = "tmp_eval_rouge/dat_giga10.txt.H"
# ref_file = "tmp_eval_rouge/dat_giga10.txt.T"
# src_file = "tmp_eval_rouge/gigaword/article.txt"


hyp_file_list = open(hyp_file).readlines()
ref_file_list = open(ref_file).readlines()
# src_file_list = open(src_file).readlines()

score, length = rouge_scorer.calculate_score(hyp_file_list, ref_file_list)

from evaluate import load
perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(predictions=hyp_file_list, model_id='gpt2')


len_char_list = [ len(''.join(_x.split())) for _x in hyp_file_list ]

avg_char = sum(len_char_list) / len(len_char_list)

bert_score = bertscore.compute(predictions=hyp_file_list, references=ref_file_list, lang="en")["recall"]
bert_score = sum(bert_score) / len(bert_score)

print('r-1\tr-2\tr-l\thyp_len\tref_len\tppl\t#char\tbert_score')

print('%.2f\t' % (score['ROUGE-1-F'] * 100),
      '%.2f\t' % (score['ROUGE-2-F'] * 100),
      '%.2f\t' % (score['ROUGE-L-F'] * 100),
      '%.2f\t' % length['hyp_ave_length'],
      '%.2f\t' % length['ref_ave_length'],
      '%.2f\t' % results['mean_perplexity'],
      '%.2f\t' % avg_char, 
      '%.2f\t' % bert_score)