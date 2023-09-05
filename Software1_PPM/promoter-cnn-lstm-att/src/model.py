import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import PearsonCorrCoef

class AttLayer(nn.Module):
    def __init__(self, attention_dim):
        super(AttLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Parameter(torch.randn(attention_dim, attention_dim))
        self.b = nn.Parameter(torch.randn(attention_dim))
        self.u = nn.Parameter(torch.randn(attention_dim, 1))

    def forward(self, x):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = torch.tanh(torch.add(torch.matmul(x, self.W), self.b))
        ait = torch.matmul(uit, self.u)
        ait = torch.squeeze(ait, -1)
        ait = torch.exp(ait)
        ait = ait / (torch.sum(ait, axis=1, keepdims=True) + 1e-7)
        ait = torch.unsqueeze(ait, dim=-1)
        weighted_input = x * ait
        output = torch.sum(weighted_input, axis=1)
        return output
    
def make_convblock(in_channels, out_channels, conv_kernel_size, stride, padding, pool_kernel_size, dropout):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, conv_kernel_size, stride, padding),
        nn.ReLU(),
        nn.MaxPool1d(pool_kernel_size),
        nn.Dropout(dropout)
    )

class CLAModel(nn.Module):
    def __init__(self, input_size, embedding_size, dropout, convblock_1, convblock_2, lstm_input_size, lstm_hidden_size, lstm_layers):
        super(CLAModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.convblock_1 = convblock_1
        self.convblock_2 = convblock_2
        self.fc1 = nn.Linear(48, lstm_input_size)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, bidirectional=True, num_layers=lstm_layers)
        self.attention = AttLayer(lstm_hidden_size * 2)
        self.fc2 = nn.Linear(lstm_hidden_size * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.convblock_1(x)
        x = self.convblock_2(x)
        x = F.relu(self.fc1(x))
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.fc2(x)
        return x

class EPT(pl.LightningModule):
    def __init__(self, 
                 model,
                 learning_rate,
                 ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.pcc = PearsonCorrCoef()
    
    def forward(self, text):
        y_hat = self.model(text).squeeze()
        return y_hat

    def training_step(self, batch):
        x, y = batch         
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)       
        loss = F.mse_loss(y_hat, y)
        self.pcc.update(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
    
    def on_validation_epoch_end(self):
        self.log('val_pcc', self.pcc.compute(), prog_bar=True)
        self.pcc.reset()
                
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)    
        loss = F.mse_loss(y_hat, y)
        self.pcc.update(y_hat, y)
        self.log('test_loss', loss)
        
    def on_test_epoch_end(self):
        self.log('test_pcc', self.pcc.compute(), prog_bar=True)
        self.pcc.reset()
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  
        with open('predict_result.txt', 'w') as f:
            for i in y_hat:
                f.write(str(i.item()) + '\n')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        return optimizer
