import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_data
from model import EPT, Model
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings

warnings.filterwarnings("ignore")

class PromoterDataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return torch.tensor(self.text[idx],dtype=torch.int32).unsqueeze(0), \
               torch.tensor(self.label[idx],dtype=torch.float32)

class PromoterDataModule(pl.LightningDataModule):
    def __init__(self, corpus, labels, batch_size, num_workers=0):
        super().__init__()
        self.corpus = corpus
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = PromoterDataset(self.corpus[:10000], self.labels[:10000])
            self.val_data = PromoterDataset(self.corpus[10800:11884], self.labels[10800:11884])
        if stage == 'test' or stage is None:
            self.test_data = PromoterDataset(self.corpus[10000:10800], self.labels[10000:10800])
        if stage == 'predict' or stage is None:
            self.pred_data = PromoterDataset(self.corpus[10000:10800], self.labels[10000:10800])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
    
def train(Args):
    print("#"*80)
    for arg in vars(Args):
        print(arg,"--", getattr(Args, arg))
    # Set random seed
    pl.seed_everything(Args.seed, workers=True)
    # Set precision
    torch.set_float32_matmul_precision('high')
    # Load data
    corpus, vocab, labels = load_data(Args.data_dir)
    # Create data module
    dm = PromoterDataModule(corpus, labels, Args.batch_size, num_workers=Args.num_workers)
    # Checkpoint
    checkpoint_callback_pcc = ModelCheckpoint(filename='{epoch}-{val_pcc:.3f}-{val_loss:.3f}-pcc', monitor='val_pcc', save_top_k=3, mode='max')
    checkpoint_callback_epoch = ModelCheckpoint(filename='{epoch}-{val_pcc:.3f}-{val_loss:.3f}', monitor='epoch', save_top_k=5, mode='max', every_n_epochs=2)
    # Create model
    base_model = Model(growth_rate=Args.growth_rate,
                       block_config=(3, 3, 3, 3),
                       num_init_features=1,
                       bn_size=Args.bn_size,
                       compression_rate=Args.compression_rate,
                       drop_rate=Args.dropout
                       )
    model = EPT(model=base_model,
                learning_rate=Args.lr,
                )
    # Create trainer
    trainer = pl.Trainer(max_epochs=Args.epoch, 
                         devices=[Args.cuda],
                         accelerator='cuda',
                         callbacks=[checkpoint_callback_pcc, checkpoint_callback_epoch],
                         fast_dev_run=True
                        )   
    # Train model 
    if Args.train:
        trainer.fit(model, dm)
        trainer.test(model, dm)
    # Test model
    else:
        # trainer.predict(model, dm, ckpt_path=Args.ckpt)
        trainer.test(model, dm, ckpt_path=Args.ckpt)
    print("#"*80)