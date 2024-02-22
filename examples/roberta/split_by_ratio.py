import sys

MIN_LEN = 10

in_file = open(sys.argv[1], 'r')
out_file_foo = open(sys.argv[2] + '.foo', 'w')
out_file_bar = open(sys.argv[2] + '.bar', 'w')
ratio = float(sys.argv[3])

for line in in_file:
    tokens = line.strip().split()
    if len(tokens) < MIN_LEN:
        continue
    pivot = int(len(tokens) * ratio)
    out_file_foo.write(' '.join(tokens[:pivot]) + '\n')
    out_file_bar.write(' '.join(tokens[pivot:]) + '\n')


in_file.close()
out_file_foo.close()
out_file_bar.close()