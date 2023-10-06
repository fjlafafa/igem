from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __getitem__(self, index):
        return ''.join(self.sequences[index])
    
    def __len__(self):
        return len(self.sequences) 