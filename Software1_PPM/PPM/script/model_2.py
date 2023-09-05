import lightning.pytorch as pl
import torch
import torch.nn as nn

class DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=3, padding='same', bias=False))
        self.add_module("drop", nn.Dropout(drop_rate))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features+i*growth_rate*bn_size, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)

class Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_feature, drop_rate):
        super(Transition, self).__init__()
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_feature,
                                          kernel_size=1, padding='same', bias=False))
        self.add_module("drop", nn.Dropout(drop_rate))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))
        self.add_module("norm", nn.BatchNorm2d(num_output_feature))
        
class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, num_features, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_features, num_features // ratio),
            nn.ReLU(),
            nn.Linear(num_features // ratio, num_features)
        )
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.shared_mlp(self.max_pool(x).view(x.size(0), -1))
        channel_attention = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        channel_attention = x * channel_attention
        return channel_attention
    
class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding='same')
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.conv(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention)
        spatial_attention = x * spatial_attention
        return spatial_attention    

class CBAM(nn.Module):
    """CBAM Module"""
    def __init__(self, num_features, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_features, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x1 = torch.sigmoid(self.channel_attention(x))
        x2 = torch.sigmoid(self.spatial_attention(x))
        return x1 * x2

class Model(nn.Module):
    "DenseNet-CBAM model"
    def __init__(self, growth_rate=32, block_config=(3, 3, 3, 3), num_init_features=1,
                 bn_size=1, compression_rate=0.5, drop_rate=0.2):
        super(Model, self).__init__()
        # Embedding
        self.features = nn.Sequential(nn.Embedding(64, 200))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate*bn_size
            self.features.add_module("norm%d" % (i + 1), nn.BatchNorm2d(num_features))
            if i != len(block_config) - 1:
                transition = Transition(num_features, int(num_features*compression_rate), drop_rate)
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # CBAM
        self.features.add_module("cbam", CBAM(num_features))

        # AveragePooling and Flatten   
        self.features.add_module("pool", nn.AdaptiveAvgPool2d(3))
        self.features.add_module("flatten", nn.Flatten())
        
        # MLP
        self.features.add_module("fc", nn.Linear(3240, 1))

    def forward(self, x):
        output = self.features(x)
        return output

class EPT(pl.LightningModule):
    def __init__(self, 
                 model,
                 learning_rate,
                 ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
    
    def forward(self, text):
        y_hat = self.model(text).squeeze()
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
