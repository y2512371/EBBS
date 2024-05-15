import sys
from langdetect import detect, detect_langs
import sacrebleu

lang = sys.argv[1]
ref = sys.argv[2]
hyp = sys.argv[3]


ref_lines = open(ref).readlines()
hyp_lines = open(hyp).readlines()

correct_ref = []
correct_hyp = []

for ref_line, hyp_line in zip(ref_lines, hyp_lines):
    if len(hyp_line.strip()) < 2:
        continue
    hyp_lang = detect(hyp_line)
    # print(lang, langs[0])
    if lang == hyp_lang:
        correct_ref.append(ref_line)
        correct_hyp.append(hyp_line)

# print(len(ref_lines), len(hyp_lines))

if len(correct_ref) > 0:
    score = sacrebleu.corpus_bleu(correct_hyp, [correct_ref])
else:
    score = "BLEU = 0.0"
print(score, '||', round(len(correct_hyp)/len(hyp_lines), 2) )
