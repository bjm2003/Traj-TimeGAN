import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from models.traj_timegan import Generator, Discriminator, TrajTimeGAN
from data_loaders.data_loader import HGVDataLoader, SineDataLoader
import yaml


class TrajTimeGANTrainer:
    def __init__(self, model, data_loader, dataset_name, dataset_type, batch_size=128, learning_rate=0.0001, epochs=100):
        self.dataset_type = dataset_type
        self.model = model
        self.data_loader = data_loader
        self.dataset_name = dataset_name

        # 硬编码训练参数
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # 初始化配置和路径

        self.prepare_paths()
        
        # 动态初始化DataLoader
        self.loader = self.data_loader.get_loader()
        self.generator = Generator(dataset_type=self.dataset_type)
        self.discriminator = Discriminator(dataset_type=self.dataset_type)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)



    def prepare_paths(self):
        self.output_root = f'outputs/{self.dataset_name}'
        os.makedirs(f'{self.output_root}/generated_data', exist_ok=True)
        os.makedirs(f'{self.output_root}/models', exist_ok=True)
        os.makedirs(f'{self.output_root}/visualization', exist_ok=True)

    def train(self):
        for epoch in range(self.epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            num_batches = 0
            for real_data in self.loader:
                # 训练判别器
                self.optimizer_D.zero_grad()
                z = torch.randn(real_data.size(0), 1000 if self.dataset_type == 'hgv' else 24, self.generator.latent_dim)
                fake_data = self.generator(z)
                
                real_loss = -torch.mean(self.discriminator(real_data))
                fake_loss = torch.mean(self.discriminator(fake_data.detach()))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_D.step()
                
                # 训练生成器
                self.optimizer_G.zero_grad()
                g_loss = -torch.mean(self.discriminator(fake_data))
                g_loss.backward()
                self.optimizer_G.step()
                
                # 累计损失
                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()
                num_batches += 1
            
            # 打印epoch统计信息
            avg_d_loss = d_loss_epoch / num_batches
            avg_g_loss = g_loss_epoch / num_batches
            print(f'Epoch [{epoch+1}/{self.epochs}] D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}')
        
        # 训练结束后保存最终结果
        self.save_artifacts(self.epochs-1)
        return self.model

    def generate(self, num_samples):
        with torch.no_grad():
            # 动态设置时间步长
            seq_len = 1000 if self.dataset_type == 'hgv' else 24
            z = torch.randn(num_samples, seq_len, self.generator.latent_dim)
            generated = self.generator(z)
            return generated.numpy()

    def save_artifacts(self, epoch):
        # 使用generate方法生成数据
        generated_data = self.generate(32)
        np.savez(f'{self.output_root}/generated_data/synthetic_data.npz', data=generated_data)
        torch.save(self.model.state_dict(), f'{self.output_root}/models/trained_model.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['hgv', 'sine'])
    args = parser.parse_args()
    
    trainer = TrajTimeGANTrainer(args.dataset)
    trainer.train()
    
    # 动态加载配置
    data_class = 'hgv' if args.dataset.startswith('hgv_') else 'sine'
    
    # 自动配置模型参数
    model_params = {
        'dataset_type': data_class
    }
    
    # 硬编码训练参数
    train_params = {
        'batch_size': 128,
        'lr': 0.0001,
        'num_epochs': 100
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = HGVDataLoader(data_path=f'data/{args.dataset}.npz') if args.dataset.startswith('hgv_') else SineDataLoader(data_path=f'data/{args.dataset}.npz')
    trainer = TrajTimeGANTrainer(
        model=TrajTimeGAN(**model_params).to(device),
        data_loader=data_loader,
        dataset_name=args.dataset
    )