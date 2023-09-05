import torch
import numpy as np
from utils import load_data
from torch.utils.data import DataLoader, Dataset

class PromoterDataset(Dataset):
    def __init__(self, corpus, mode):
        self.corpus = corpus
        self.mode = mode
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        if self.mode == 1:
            return torch.tensor(self.corpus[idx],dtype=torch.int32)
        else:
            return torch.tensor(self.corpus[idx],dtype=torch.int32).unsqueeze(0)

def infer(Args):
    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    result, corpus = load_data(Args.input)
    dataset = PromoterDataset(corpus, Args.mode)
    dataloader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=False, drop_last=False)
    # Load model
    if Args.mode == 1:
        from model_1 import EPT, CLAModel, make_convblock
        convblock_1 = make_convblock(in_channels=48,
                                     out_channels=150,
                                     conv_kernel_size=5,
                                     stride=1,
                                     padding='valid',
                                     pool_kernel_size=2,
                                     dropout=0.25)
        convblock_2 = make_convblock(in_channels=150,
                                     out_channels=150,
                                     conv_kernel_size=3,
                                     stride=1,
                                     padding='valid',
                                     pool_kernel_size=2,
                                     dropout=0.25)
        base_model = CLAModel(input_size=64,
                              embedding_size=200,
                              dropout=0.25,
                              convblock_1=convblock_1,
                              convblock_2=convblock_2,
                              lstm_input_size=400,
                              lstm_hidden_size=64,
                              lstm_layers=1,
                              )
        model = EPT.load_from_checkpoint(
            Args.model, model=base_model, learning_rate=0.02).to(device)
    elif Args.mode == 2:
        from model_2 import EPT, Model
        base_model = Model(growth_rate=32,
                           block_config=(3, 3, 3, 3),
                           num_init_features=1,
                           bn_size=2,
                           compression_rate=0.5,
                           drop_rate=0.15
                           )
        model = EPT.load_from_checkpoint(
            Args.model, model=base_model, learning_rate=0.001).to(device)
    else:
        raise ValueError("Invalid mode!")
    model.eval()
    model.freeze()
    # Predict
    for i, data in enumerate(dataloader):
        data = data.to(device)
        pred = model(data)
        if i == 0:
            y_pred = pred.cpu().detach().numpy()
        else:
            y_pred = np.concatenate((y_pred, pred.cpu().detach().numpy()), axis=0)
    # Save result
    result['prediction'] = y_pred
    result.to_csv(Args.output+'output.csv', index=False)
    if Args.sort:
        result = result.sort_values(by=['prediction'], ascending=False)
        result.to_csv(Args.output+"output_sorted.csv", index=False)
