import sys
from tqdm import tqdm
import multiprocessing as mp

in_file = open(sys.argv[1], 'r')
out_file = open(sys.argv[2], 'w')


def convert_bpe(row):
    tokens  = row.split()   
    out_line = ' '.join([ 
        (token[1:] if (tokens[min(_idx+1, len(tokens)-1)].startswith('Ġ') or _idx == (len(tokens) - 1)) else  token[1:] + '@@') 
        if token.startswith('Ġ') else 
        (token if (tokens[min(_idx+1, len(tokens)-1)].startswith('Ġ') or _idx == (len(tokens) - 1)) else token + '@@')  
        for _idx, token in enumerate(tokens)])
    return out_line

pool = mp.Pool(mp.cpu_count())
print(f"Processing with {mp.cpu_count()} cores")

results = pool.map(convert_bpe, [row for row in in_file.readlines()])

pool.close()


out_file.write('\n'.join(results))
out_file.write('\n')