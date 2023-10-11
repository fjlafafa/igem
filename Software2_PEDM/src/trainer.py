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
from tqdm import tqdm
from torch import optim

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_seed(cfg.seed)
        if cfg.data.generate == True:
            self.initial_sequence()
        else:
            self.load_initial_sequence()
        if cfg.wandb:
            self.wandb_init()
        
        self.model = DiscreteDiffusion(**cfg.model)
        
    def wandb_init(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="igem dry lab",
            
            # track hyperparameters and run metadata
            config=self.cfg)
        
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
        self.train_data, self.val_data = train_test_split(self.dataset, test_size=0.1, random_state=1)
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
        self.train_data, self.val_data = train_test_split(self.dataset, test_size=0.1, random_state=1)
        print(len(self.val_data))
        self.train_loader = DataLoader(self.train_data, batch_size = self.cfg.data.batch_size)
        self.val_loader = DataLoader(self.val_data, batch_size = self.cfg.data.batch_size)
        
    def train(self): 
        step = 0
        for epoch in range(self.cfg.train.epochs):
            for batch in tqdm(self.train_loader):
                loss = self.model.step(epoch, batch)
                
                if self.cfg.wandb:
                    wandb.log({'train_loss':loss})
                
                if step % self.cfg.train.save_per_step == 0:
                    torch.save(self.model.encoder, self.cfg.root_dir+'/../ckpt/model_at_step_{}.pt'.format(step))
                    
                if step % self.cfg.train.eval_per_step == 0:
                    self.eval()
                    
                step += 1
                    
        torch.save(self.model.encoder, self.cfg.root_dir+'/../ckpt/model_at_step_{}.pt'.format(step))
        
            
    def eval(self):
        loss = 0
        for batch in tqdm(self.val_loader): 
            loss += self.model.step(0, batch, eval=True)
        loss = loss / len(self.val_loader)
        
        if self.cfg.wandb:
            wandb.log({'eval_loss':loss})
        
    def load(self, path=None):
        if path == None:
            path = self.cfg.root_dir+'/../ckpt/model_at_step_355.pt'
        self.model.encoder = torch.load(path)
        self.model.encoder.to(self.cfg.model.device)
            
    def sample(self): 
        sequences = self.model.sample(self.cfg.iterate.size, len(self.cfg.seq))
        file = self.cfg.root_dir+'/../new_seq/test.fasta'
        with open(file, "w", encoding='utf-8') as f:
            for i, seq in enumerate(sequences):
                f.write('>{}\n'.format(i))
                f.write(seq+'\n')
        f.close()
        return sequences

    def iterate(self): 

        temperature_list, seq_list = [], []
        
        with open(self.cfg.iterate.seq_path, "r", encoding='utf-8') as f:
            for _ in range(self.cfg.iterate.size):
                f.readline()
                seq = f.readline()
                seq_list.append(seq[:-2])
        print(seq_list)
        f.close()
        
        with open(self.cfg.iterate.temp_path, "r", encoding='utf-8') as f:
            f.readline()
            for _ in range(self.cfg.iterate.size):
                line = f.readline().split('\t')
                temperature_list.append(float(line[1][:-2]))
        print(temperature_list)
        f.close()
        for _ in range(self.cfg.iterate.numstep):
            self.model.optimizer = optim.Adam(self.model.encoder.parameters(),
                           lr=self.cfg.model.lr,
                           weight_decay=self.cfg.model.weight_decay)
            loss = self.model.step(0, seq_list, weight = np.array(temperature_list)-self.cfg.iterate.predicted_tm - 0.9*min(np.array(temperature_list)-self.cfg.iterate.predicted_tm))
            print(loss)
            
                    
        torch.save(self.model.encoder, self.cfg.root_dir+'/../ckpt/iterate.pt')

@hydra.main(config_path='cfg', config_name='config')
def main(cfg):
    trainer = Trainer(cfg)
    if trainer.cfg.task == 'train': 
        trainer.train()
    elif trainer.cfg.task == 'sample':
        trainer.load()
        trainer.sample()
    elif trainer.cfg.task == 'iterate':
        trainer.load(trainer.cfg.iterate.path)
        trainer.iterate()
        trainer.sample()
    else:
        raise NotImplementedError()
    

if __name__ == '__main__':
    main()
    