import sys

input_file_lines = open(sys.argv[1]).readlines()

output_file_H = open(sys.argv[1] + ".H", 'w')
output_file_T = open(sys.argv[1] + ".T", 'w')

# T-1801	<<unk>> chief calls for immediate action against regional environmental deterioration
# H-1801	1.4426950216293335	in the ####s linked to poverty and globalization
# D-1801	1.4426950216293335	in the ####s linked to poverty and globalization


H_dict = {}
T_dict = {}

for line in input_file_lines:
    try:
        if line.startswith('T'):
            tokens = line.split('\t')
            ID = int(tokens[0].split('-')[1])
            T_dict[ID] = tokens[-1].replace('<<unk>>', '<unk>')
        if line.startswith('H'):
            tokens = line.split('\t')
            
            hypo_rank_tokens = tokens[0].split('-')
            ID = int(hypo_rank_tokens[1])
            rank = 0
            if len(hypo_rank_tokens) == 3: # with mutliple items, and ranking
                rank = int(hypo_rank_tokens[2])
            
            if rank == 0:
                text = tokens[-1]
                text = text.replace('<pad>', '')
                H_dict[ID] = "<unk>" if len(text.split()) < 1 else text
    
    except Exception as e:
        print(line)

T_dict = dict(sorted(T_dict.items(), key=lambda item: int(item[0])))

for key in T_dict:
    T = T_dict[key]
    if key in H_dict:
        H = H_dict[key]
        output_file_H.write(H.strip() + '\n')
        output_file_T.write(T.strip() + '\n')
    
    
    
    
    

        
        
        