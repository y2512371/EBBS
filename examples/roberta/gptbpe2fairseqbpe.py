import sys
from tqdm import tqdm
in_file = open(sys.argv[1], 'r')
out_file = open(sys.argv[2], 'w')



for lines in tqdm(in_file.readlines()):

    tokens  = lines.split()   
    
    out_line = ' '.join([ 
        (token[1:] if (tokens[min(_idx+1, len(tokens)-1)].startswith('Ġ') or _idx == (len(tokens) - 1)) else  token[1:] + '@@') 
        if token.startswith('Ġ') else 
        (token if (tokens[min(_idx+1, len(tokens)-1)].startswith('Ġ') or _idx == (len(tokens) - 1)) else token + '@@')  
        for _idx, token in enumerate(tokens)])
    out_file.write(out_line + '\n')

