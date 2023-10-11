from ast import Num
import torch
import hydra
from model import Seq2SeqModel
import pickle
from utils import sequence_embedding, seqs_to_onehots, seqs_to_indexs, index
from torch import optim
import copy
from tqdm import tqdm
import numpy as np

class DiscreteDiffusion:
    def __init__(self, num_steps, embedding_dim, num_layers, hidden_size, lr, weight_decay, device):
        self.num_steps = num_steps
        self.encoder = Seq2SeqModel(embedding_dim, num_layers, hidden_size)
        self.optimizer = optim.Adam(self.encoder.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
        self.device = device
        
    
    def step(self, epoch, seqs, eval = False, weight = None):
        bsz = len(seqs)
        if weight is None:
            weight = torch.ones(bsz)
        
        loss = self.discrete_diffusion_loss(seqs, weight)
        if not eval:
            loss.backward()
            self.optimizer.step()
        return loss
        
        
    def corrupt_seqs(self, seqs, step, method='naive_absorbing'):
        if method == 'naive_absorbing':
            indexs = seqs_to_indexs(seqs)
            mask = torch.rand(indexs.shape) < (step+1)/self.num_steps
            indexs.masked_fill(mask, 0)
            return indexs
        else:
            raise NotImplementedError()
        
    def discrete_diffusion_loss(self, seqs, weight):
        '''
        This function calculates the loss of a diffusion stepp given the weight of the samples within the batch, this is calculated by sampling the corrupted sequence at a particular timestep and estimating the original input from the corrupted sequence.
        
        return: loss after balancing
        '''
        indexs = seqs_to_indexs(seqs)
        indexs = indexs
        bsz, length = indexs.shape
        target = torch.LongTensor(bsz, length+1)
        target[:, -1].fill_(index('EOS'))
        target[:, :-1] = indexs
        loss = 0
        step = np.random.randint(self.num_steps)
        corrupted_indexs = self.corrupt_seqs(seqs, step)
        corrupted_indexs = corrupted_indexs
        source = torch.ones_like(target) * index('BOS')
        prev_outputs = copy.deepcopy(source)
        source[:, 1:] = corrupted_indexs
        prev_outputs[:, 1:] = target[:, :-1]
        source = source.to(self.device)
        prev_outputs = prev_outputs.to(self.device)
        target = target.to(self.device)
        temp_loss = self.encoder.get_loss(source, prev_outputs, target, reduce=False)
        temp_loss = temp_loss.sum(dim=1)
        # print(temp_loss)
        weight = torch.tensor(weight).to(self.device)
        loss += (weight * temp_loss).sum()/weight.size(0)
            
        return loss/self.num_steps
            
    def sample(self, num, length):
        '''
        A function for sampling the sequences given a particular length.
        '''
        seq_list = []
        for i in range(num):
            X_t = '#' * length
            for step in tqdm(range(self.num_steps)):
                X_0 = self.encoder.generate(X_t, max_len = length+1, beam_search=False)
                X_0 = X_0[1:]
                X_t_1 = ''
                for j in range(length): 
                    t = self.num_steps - step
                    if X_t[j] != '#' or np.random.random() > 1/t:
                        X_t_1 += X_t[j]
                    else:
                        X_t_1 += X_0[j]
                X_t = X_t_1
                
            seq_list.append(X_t)
            
        return seq_list
                    
    

