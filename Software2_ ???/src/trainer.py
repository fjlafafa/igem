import torch 
import hydra
import numpy as np
from utils import random_walk
import pickle
import os
from torch.utils.data import DataLoader
from data import DiffusionDataset
from diffusion import DiscreteDiffusion
from sklearn.model_selection import train_test_split
import wandb 

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_seed(cfg.seed)
        if cfg.data.generate == True:
            self.initial_sequence()
        else:
            self.load_initial_sequence()
        
        self.model = DiscreteDiffusion(**cfg.model)
        
    def set_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
            
    # Generate the initial sequence, we use this sequence 
    def initial_sequence(self):
        max_num = self.cfg.data.max_num
        seq = self.cfg.seq
        max_mut = self.cfg.data.max_mut
        new_sequences = random_walk(max_num, seq, max_mut)
        
        self.dataset = DiffusionDataset(new_sequences)
        self.train_data, self.val_data = train_test_split(self.dataset, test_size=0.25, random_state=1)
        self.train_loader = DataLoader(self.train_data, batch_size = self.cfg.data.batch_size)
        self.val_loader = DataLoader(self.val_data, batch_size = self.cfg.data.batch_size)
        
        # store data for future use
        with open(self.cfg.root_dir+'/../data/new_sequences.pkl', 'wb') as f:
            pickle.dump(new_sequences, f)
    
    def load_initial_sequence(self):
        with open(self.cfg.root_dir+'/../data/new_sequences.pkl', 'rb') as f:
            new_sequences = pickle.load(f)
        f.close()
        
        self.dataset = DiffusionDataset(new_sequences)
        self.train_data, self.val_data = train_test_split(self.dataset, test_size=0.25, random_state=1)
        self.train_loader = DataLoader(self.train_data, batch_size = self.cfg.data.batch_size)
        self.val_loader = DataLoader(self.val_data, batch_size = self.cfg.data.batch_size)
        
    def train(self): 
        step = 0
        for epoch in range(self.cfg.train.epochs):
            for batch in self.train_loader: 
                self.model.step(epoch, batch)
                step += 1
                if step % self.cfg.train.eval_per_step == 0:
                    self.eval() 

    def eval(self):
        loss = 0
        for batch in self.val_loader: 
            loss += self.model.step(0, batch, eval=True)
        loss / len(self.val_loader)
            
    def sample(self, n): 
        sequences = self.model.sample(n)
        return sequences


@hydra.main(config_path='cfg', config_name='config')
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
    