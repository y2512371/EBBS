import fileinput

for line in fileinput.input():
    print(' '.join(line.split()[1:]))
