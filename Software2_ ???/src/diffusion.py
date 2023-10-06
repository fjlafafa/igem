import torch
import hydra
from model import Seq2SeqModel
import pickle
from utils import sequence_embedding, seqs_to_onehots, seqs_to_indexs, index
from torch import optim
import copy

class DiscreteDiffusion:
    def __init__(self, num_steps, embedding_dim, num_layers, hidden_size, lr, weight_decay, device):
        self.num_steps = num_steps
        self.model = Seq2SeqModel(embedding_dim, num_layers, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
        self.device = device
        
    
    def step(self, epoch, seqs, eval = False, weight = None):
        bsz = len(seqs)
        if weight is None:
            weight = torch.ones(bsz)
        loss = self.discrete_diffusion_loss(seqs)
        
        
    def corrupt_seqs(self, seqs, step, method='naive_absorbing'):
        if method == 'naive_absorbing':
            indexs = seqs_to_indexs(seqs)
            mask = torch.rand(indexs.shape) < (step+1)/self.num_steps
            indexs.masked_fill(mask, 0)
            return indexs
        else:
            raise NotImplementedError()
        
    def discrete_diffusion_loss(self, seqs):
        indexs = seqs_to_indexs(seqs)
        indexs = indexs
        bsz, length = indexs.shape
        target = torch.LongTensor(bsz, length+1)
        target[:, -1].fill_(index('EOS'))
        target[:, :-1] = indexs
        
        loss = 0
        for step in range(self.num_steps):
            corrupted_indexs = self.corrupt_seqs(seqs, step)
            corrupted_indexs = corrupted_indexs
            source = torch.ones_like(target) * index('BOS')
            prev_outputs = copy.deepcopy(source)
            source[:, 1:] = corrupted_indexs
            prev_outputs[:, 1:] = target[:, :-1]
            source.to(self.device)
            prev_outputs.to(self.device)
            target.to(self.device)
            
            loss += self.model.get_loss(source, prev_outputs, target, reduce=False)
            print(loss.shape)
            print(loss)
            
        return loss
            
    def sample(self, num):
        return None
    

