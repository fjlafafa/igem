import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import PearsonCorrCoef

class TransformerBlock(pl.LightningModule):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 d_model,
                 num_heads,
                 num_layers,
                 dropout):
        super().__init__()
        self.embedding_1 = nn.Embedding(input_dim, embedding_dim)
        self.embedding_2 = nn.Linear(12, embedding_dim) # for linear
        # self.embedding_2 = nn.Embedding(input_dim, embedding_dim) # for embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads)
        self.transformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, text, structure=None):
        embedded_1 = self.embedding_1(text)
        if structure is not None:
            embedded_2 = self.embedding_2(structure)
            embedded = embedded_1 + embedded_2
        else:
            embedded = embedded_1
        output = self.transformerEncoder(embedded)
        output = self.dropout(output.flatten(1))
        return output
        
class EPT(pl.LightningModule):
    def __init__(self, 
                 transformer_block,
                 learning_rate,
                 add_feature
                 ):
        super().__init__()
        self.transformer_block = transformer_block
        self.fc = nn.Linear(48*200, 1)
        self.learning_rate = learning_rate
        self.add = add_feature        
        self.pcc = PearsonCorrCoef()
    
    def forward(self, text, structure=None):
        if self.add:
            y_hat = self.transformer_block(text, structure)
        else:
            y_hat = self.transformer_block(text)
        y_hat = self.fc(y_hat).squeeze()
        return y_hat

    def training_step(self, batch):
        if self.add:
            x, y, z = batch
            y_hat = self(x, z)
        else:
            x, y = batch
            y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.add:
            x, y, z = batch
            y_hat = self(x, z)
        else:
            x, y = batch
            y_hat = self(x)       
        loss = F.mse_loss(y_hat, y)
        self.pcc.update(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
    
    def on_validation_epoch_end(self):
        self.log('val_pcc', self.pcc.compute(), prog_bar=True)
        self.pcc.reset()
                
    def test_step(self, batch, batch_idx):
        if self.add:
            x, y, z = batch
            y_hat = self(x, z)
        else:
            x, y = batch
            y_hat = self(x)    
        loss = F.mse_loss(y_hat, y)
        self.pcc.update(y_hat, y)
        self.log('test_loss', loss)
        
    def on_test_epoch_end(self):
        self.log('test_pcc', self.pcc.compute())
        self.pcc.reset()
    
    def predict_step(self, batch):
        if self.add:
            x, y, z = batch
            y_hat = self(x, z)
        else:
            x, y = batch
            y_hat = self(x)  
        with open('pred.txt', 'w') as f:
            for i in y_hat:
                f.write(str(i.item()) + '\n')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
