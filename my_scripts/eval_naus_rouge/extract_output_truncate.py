import sys

input_file_lines = open(sys.argv[1]).readlines()

output_file_H = open(sys.argv[1] + ".H", 'w')
output_file_T = open(sys.argv[1] + ".T", 'w')
output_file_S = open(sys.argv[1] + ".S", 'w')

ratio=float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

# T-1801	<<unk>> chief calls for immediate action against regional environmental deterioration
# H-1801	1.4426950216293335	in the ####s linked to poverty and globalization
# D-1801	1.4426950216293335	in the ####s linked to poverty and globalization

S_dict = {}
H_dict = {}
T_dict = {}

for line in input_file_lines:
    try:
        if line.startswith('S'):
            tokens = line.split('\t')
            ID = int(tokens[0].split('-')[1])
            S_dict[ID] = tokens[-1].replace('<<unk>>', '<unk>')
            
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
                # text = text.replace('<pad>', '')
                # # remote consecutative deduplicated words from text
                # text = ' '.join([x[0] for x in zip(text.split(), text.split()[1:]) if x[0] != x[1]])  
                              
                H_dict[ID] = "<unk>" if len(text.split()) < 1 else text
    
    except Exception as e:
        print(line)

T_dict = dict(sorted(T_dict.items(), key=lambda item: int(item[0])))

turncate_count = 0
all_count = 0
for key in T_dict:
    T = T_dict[key]
    if key in H_dict:
        all_count += 1
        H = H_dict[key]
        
        # Chucking
        src_len = len(S_dict[key].split())
        max_H_len = int(src_len * ratio) + 1
        
        new_H = H.split()[:max_H_len]
        if len(H.split()) <= len(new_H):
            new_H = H.split()
        else:
            turncate_count += 1
        
        new_H = ' '.join(new_H)
        
        S = S_dict[key]
        
        output_file_H.write(new_H + '\n')
        output_file_T.write(T.strip() + '\n')
        output_file_S.write(S.strip() + '\n')

print("\n----------\nTruncate_count: %d, all_count: %d, ratio: %f\n----------\n" % (turncate_count, all_count, turncate_count/all_count))
    
    
    
# def a function that removes bpe "##"
def remove_bpe(text):
    return text.replace(' ##', '').replace('##', '')

        
        
        