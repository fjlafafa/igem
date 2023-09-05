import argparse
import collections
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Promoter mutation')
    parser.add_argument('--input', '-i', type=str, help='input file')
    parser.add_argument('--output', '-o', type=str, help='output file')
    parser.add_argument('--mode', type=int, choices=[1, 2], help='mutation mode')
    parser.add_argument('--num', type=int, help='mutation number')
    parser.add_argument('--pos', type=tuple, help='mutation position')
    Args = parser.parse_args()
    return Args

if __name__ == '__main__':
    Args = parse_args()
    # read the first sequence in input file
    with open(Args.input, 'r') as f:
        seq = f.readlines()[0].strip().lower()
    seq = np.array(list(seq))
    if len(seq) != 50:
        raise ValueError('The length of sequence is not 50')
    counter = collections.Counter(seq)
    if counter['a'] + counter['t'] + counter['c'] + counter['g'] != 50:
        raise ValueError('The sequence contains illegal characters')
    # mutation
    nt = {'a0':'t', 'a1':'c', 'a2':'g', 'a3':'a',
          't0':'a', 't1':'c', 't2':'g', 't3':'t',
          'c0':'a', 'c1':'t', 'c2':'g', 'c3':'c',
          'g0':'a', 'g1':'t', 'g2':'c', 'g3':'g'}
    if Args.mode == 1:
        # random mutation in the whole sequence
        mutation = np.repeat(seq.reshape(1,50), Args.num, axis=0)
        i = 0
        for i in range(Args.num):
            for j in range(np.random.randint(50)):
                mutation[i][j] = nt[seq[j]+str(np.random.randint(4))]
        mutation = np.unique(mutation, axis=0)
    else:
        # mutation in specific position
        start = int(Args.pos[0])
        end = int(Args.pos[2])
        if start < 0 or end > 49:
            raise ValueError('The position is out of range')
        if start > end:
            raise ValueError('The start position is larger than the end position')
        length = end - start + 1
        num = 4 ** length
        mutation = np.repeat(seq.reshape(1,50), num, axis=0)
        count = 0
        for i in range(4 ** length):
            for j in range(length):
                mutation[i][start+j] = nt[seq[start+j]+str(int(count/(4**(length-j-1))%4))]
            count += 1
    # write the mutation sequence to output file
    with open(Args.output, 'w') as f:
        for i in range(mutation.shape[0]):
            f.write(''.join(mutation[i]) + '\n')