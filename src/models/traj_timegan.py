import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, dataset_type):
        super().__init__()
        # 根据数据集类型动态设置序列长度
        self.seq_len = 1000 if dataset_type == 'hgv' else 24
        self.latent_dim = 24
        
        if dataset_type == 'hgv':
            self.hidden_dim = 64
            self.num_layers = 6
        else:  
            self.hidden_dim = 128
            self.latent_dim = 64
            self.num_layers = 6
        
        output_dim = 6 if dataset_type == 'hgv' else 5
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, output_dim),
            nn.Tanh()
        )
        
        self.lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,  # 单方向维度
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

    # forward实现
    def forward(self, z):
        batch_size = z.size(0)
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(z.device)
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(z.device)
        
        lstm_out, _ = self.lstm(z, (h_0, c_0))
        output = self.fc(lstm_out)
        return output

class Discriminator(nn.Module):
    def __init__(self, dataset_type):
        super().__init__()
        # 特征维度根据数据集类型设定
        self.feature_dim = 6 if dataset_type == 'hgv' else 5
        self.d_hidden = 64
        
        self.feature_adjust = nn.Linear(self.feature_dim, self.feature_dim)

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.d_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.d_hidden, self.d_hidden*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.d_hidden*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(self.feature_adjust.weight.dtype)  # 添加数据类型转换
        x = self.feature_adjust(x)
        
        # 调整维度顺序为 [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        validity = self.conv_blocks(x)
        return validity

class TrajTimeGAN(nn.Module):
    def __init__(self, dataset_type):
        super().__init__()
        self.generator = Generator(dataset_type)
        self.discriminator = Discriminator(dataset_type)

    def forward(self, x):
        output, _ = self.generator(x)
        validity = self.discriminator(output)
        return output, validity

    def generate(self, noise):
        with torch.no_grad():
            output, _ = self.generator(noise)
            return output