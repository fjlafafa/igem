import collections
import numpy as np
import pandas as pd

def create_kmer_list(input_list, kmer_num):
    """Create kmer list from input list"""
    kmer_list = []
    if len(input_list) <= kmer_num:
        kmer_list.append(input_list)
    else:
        for tmp in zip(*[input_list[i:] for i in range(kmer_num)]):
            tmp = "".join(tmp)
            kmer_list.append(tmp)
    return kmer_list

def tokenize(lines):
    """Tokenize input lines"""
    return [create_kmer_list(line, 3) for line in lines]

class Vocab:
    """Text vocabulary"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 2

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """Count tokens in the corpus."""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_data(data_dir, add_features=False):
    """Load data."""
    # Load data
    promoter = np.load(data_dir+'promoter.npy')
    strength = np.load(data_dir+'gene_expression.npy').astype(np.float32)
    # Prepare data
    labels = np.log2(strength)
    tokens = tokenize(promoter)
    vocab = Vocab(tokens)
    vocab.token_to_idx.pop('<unk>')
    vocab.idx_to_token.pop(0)
    corpus = np.array([vocab[token] for token in tokens])
    # Below is the primitive version of tokenization and vocabulary
    # vocab = dict(count_corpus(tokens))
    # idx_to_word = list(vocab.keys())
    # word_to_idx = {word:i for i, word in enumerate(idx_to_word)}
    # corpus = [[word_to_idx[word] for word in token] for token in tokens]
    structure = []
    if add_features:
        # 64-embedding
        tri_prop = pd.read_csv(data_dir+'TriStructure.csv', index_col=0)
        word_to_idx = {word.lower():i for i, word in enumerate(tri_prop.index)}
        structure = np.array([[word_to_idx[word] for word in token] for token in tokens])
        # raw numbers for linear layer
        # tri_prop = pd.read_csv(data_dir+'TriStructure.csv', index_col=0)
        # for token in tokens:
        #     structure.append([np.array(tri_prop.loc[word.upper()]) for word in token])
        # structure = np.array(structure)
    return corpus, vocab, labels, structure