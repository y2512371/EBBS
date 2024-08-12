import sys

src_file = sys.argv[1]
hyp_file = sys.argv[2]
ratio = float(sys.argv[3])
out_file = sys.argv[4]
out_file = open(out_file, 'w')

truncate_count = 0
all_count = 0
for src, hyp in zip(open(src_file), open(hyp_file)):
    all_count += 1
    src_len = len(src.split())
    max_H_len = int(src_len * ratio) + 1
    new_H = hyp.split()[:max_H_len]
    if len(hyp.split()) <= len(new_H):
        new_H = hyp.split()
    else:
        truncate_count += 1

    out_file.write(' '.join(new_H) + '\n')

print('truncate_ratio: ', truncate_count / all_count)
        