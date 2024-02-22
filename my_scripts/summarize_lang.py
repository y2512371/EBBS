import os
import sys

path = sys.argv[1]
out_path = sys.argv[2]

file_list = os.listdir(path)

def get_bleu(file_lines):
    for line in file_lines:
        if line.startswith("BLEU"):
            segment_list = line.split(' ')
            return float(segment_list[2])
    


bleu_dict = {}
lang_dict = {}
for file_name in file_list:
    if file_name.startswith('lang'):
        file_name_body, _ = file_name.split('.')
        _, src, trg = file_name_body.split('_')

        lines = open(os.path.join(path, file_name)).readlines()
        # print(lines)
        bleu_line = [ line.split("||")[0] for line in lines if len(line) > 0 and line.startswith("BLEU")]
        lang_line = [ line.split("||")[1] for line in lines if len(line) > 0 and line.startswith("BLEU")]

        bleu_dict[f'{src}-{trg}'] = get_bleu(bleu_line)
        lang_dict[f'{src}-{trg}'] = float(lang_line[0].strip())


bleu_dict_sorted = dict(sorted(bleu_dict.items(), key=lambda x: x))
lang_dict_sorted = dict(sorted(lang_dict.items(), key=lambda x: x))

# print(bleu_dict_sorted,lang_dict_sorted)

out_file = open(out_path, 'w')
out_file.write("AVG" + ' ')

for key in bleu_dict_sorted:
    out_file.write(key + ' ')

out_file.write('\n')

mean = sum(bleu_dict_sorted.values()) / len(bleu_dict_sorted.values())
out_file.write(str(round(mean, 2)) + ' ')

for key in bleu_dict_sorted:
    out_file.write(str(round(bleu_dict_sorted[key], 2)) + ' ')
out_file.write('\n')


mean_lang_acc = sum(lang_dict_sorted.values()) / len(lang_dict_sorted.values())
out_file.write(str(round(mean_lang_acc, 2)) + ' ')
for key in lang_dict_sorted:
    out_file.write(str(round(lang_dict_sorted[key], 2)) + ' ')
out_file.write('\n')

