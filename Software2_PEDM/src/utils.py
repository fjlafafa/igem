import torch
import numpy as np
from tqdm import tqdm

AA = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'PAD', 'EOS', 'BOS']

mapping = {**{AA[i]:i for i in range(len(AA))}}

def index(aa):
    return mapping[aa]

def symbols(index):
    return AA[index]

def get_length():
    return len(AA)

# convert the aa sequence into onehot encoding
def seq_to_onehot(x):
    label = torch.tensor([mapping[aa] for aa in x])
    one_hot = torch.nn.functional.one_hot(label, num_classes=len(AA))
    return one_hot

def seqs_to_onehots(xs):
    one_hots = torch.concat(tuple([seq_to_onehot(x).unsqueeze(0) for x in xs]), 0)
    return one_hots

def seq_to_index(x):
    return torch.tensor([mapping[aa] for aa in x])

def seqs_to_indexs(xs):
    indexs = torch.concat(tuple([seq_to_index(x).unsqueeze(0) for x in xs]), 0)
    return indexs

def onehots_to_seqs(xs):
    _, label = torch.max(xs, 2)
    return [''.join([AA[j] for j in x]) for x in label]
        
# generate embedding for the aa based on chemical properties
def sequence_embedding(xs):
    hydropathy = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
    volume = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
    charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
    polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
    acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
    donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
    
    embedding = torch.tensor([
        [hydropathy[aa], volume[aa] / 100, charge[aa],
        polarity[aa], acceptor[aa], donor[aa]]
        for aa in AA
    ])
    one_hots = seqs_to_onehots(xs)
    return torch.einsum('ijk, kl-> ijl', one_hots.float(), embedding.float())

def single_random_walk(seq, max_mut):
    mut = np.random.randint(max_mut)
    seq = list(seq)
    for i in range(mut):
        seq[np.random.randint(len(seq))] = AA[np.random.randint(1, 21)]
    return seq

# generate the max_num sequences after random walk and calculate the corresponding embedding for further use
def random_walk(max_num, seq, max_mut):
    new_sequence = [single_random_walk(seq, max_mut) for i in tqdm(range(max_num))]
    # embedding = sequence_embedding(new_sequence)
    # new_sequence = torch.concat(tuple([sequence_onehot(x).unsqueeze(0) for x in new_sequence]), 0)
    return new_sequence

if __name__ == '__main__':
    sequece = ['ATISFYP', 'ATISFYP']
    print(seqs_to_indexs(sequece))

    