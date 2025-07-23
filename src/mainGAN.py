import argparse
import os
import torch
from models.traj_timegan import TrajTimeGAN
from data_loaders.data_loader import HGVDataLoader, SineDataLoader
from train import TrajTimeGANTrainer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from metrics import MetricEvaluator, discriminative_score_metrics

def main():
    # 配置参数解析
    parser = argparse.ArgumentParser(description='TimeGAN主控制程序')
    parser.add_argument('--mode', choices=['train','generate','visualize','all'], default='all', help='运行模式')
    parser.add_argument('--dataset', required=True, help='数据集名称，如hgv_trajectories_1或sine_subsampled_train_perc_2')
    args = parser.parse_args()

    # 动态加载配置文件
    data_class = 'hgv' if args.dataset.startswith('hgv_') else 'sine' if args.dataset.startswith('sine_') else 'other'
    dataset_params = {
        'dataset_type': data_class
    }

    # 初始化组件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 统一创建输出目录
    output_base = f'outputs/{args.dataset}'
    os.makedirs(f'{output_base}/generated_data', exist_ok=True)
    os.makedirs(f'{output_base}/visualization', exist_ok=True)
    
    # 动态选择数据加载器
    DataLoaderClass = HGVDataLoader if data_class == 'hgv' else SineDataLoader
    data_loader = DataLoaderClass(
        data_path=f'data/{args.dataset}.npz'
    )
    print(f'成功加载 {args.dataset} 数据集')
    model = TrajTimeGAN(**dataset_params).to(device)
    trainer = TrajTimeGANTrainer(model, data_loader, args.dataset, dataset_type=data_class, batch_size=128, learning_rate=0.0001, epochs=100)

    # 运行流程控制
    if args.mode in ['train','all']:
        print('开始模型训练，设备：' + str(device))
        trained_model = trainer.train()
        torch.save(trained_model.state_dict(), f'outputs/{args.dataset}/models/trained_model.pth')

    if args.mode in ['generate','all']:
        print('🔧 正在生成合成数据，模式：' + ('HGV' if data_class == 'hgv' else 'SINE'))
        real_data = data_loader.get_dataset()
        num_samples = real_data.shape[0]
        fake_data = trainer.generate(num_samples)
        
        output_path = f'outputs/{args.dataset}/generated_data'
        os.makedirs(output_path, exist_ok=True)
        np.savez(f"{output_path}/synthetic_data.npz", 
                real=real_data[:1000], 
                fake=fake_data[:1000])
        print(f'生成数据已保存至 {output_path}，真实数据维度：{real_data.shape}，生成数据维度：{fake_data.shape}')

    if args.mode in ['visualize','all']:
        print('开始数据可视化，模式：' + ('HGV' if data_class == 'hgv' else 'SINE'))
        # 加载真实数据
        real_data = np.load(f'data/{args.dataset}.npz')['data']
        # 加载生成数据
        fake_data = np.load(f'outputs/{args.dataset}/generated_data/synthetic_data.npz')['fake']
        
        # 数据预处理
        real_samples = real_data[:500]
        fake_samples = fake_data[:500]
        
        # PCA可视化

        pca = PCA(n_components=2)
        
        # 处理单样本情况
        if real_samples.shape[0] == 1:
            real_reshaped = real_samples.reshape(-1, real_samples.shape[-1])
            fake_reshaped = fake_samples.reshape(-1, fake_samples.shape[-1])
            real_pca = pca.fit_transform(real_reshaped)
            fake_pca = pca.fit_transform(fake_reshaped)
        else:
            real_pca = pca.fit_transform(real_samples.reshape(real_samples.shape[0], -1))
            fake_pca = pca.fit_transform(fake_samples.reshape(fake_samples.shape[0], -1))
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.5, label='Real Data')
        plt.scatter(fake_pca[:, 0], fake_pca[:, 1], c='red', alpha=0.5, label='Generated Data')
        plt.title('PCA Visualization')
        plt.legend()
        
        # t-SNE可视化
        tsne = TSNE(n_components=2, random_state=42, verbose = 1, perplexity = 30, n_iter = 300)
        
        # 处理单样本情况
        if real_samples.shape[0] == 1:
            real_reshaped = real_samples.reshape(-1, real_samples.shape[-1])
            fake_reshaped = fake_samples.reshape(-1, fake_samples.shape[-1])
            real_tsne = tsne.fit_transform(real_reshaped)
            fake_tsne = tsne.fit_transform(fake_reshaped)
        else:
            real_tsne = tsne.fit_transform(real_samples.reshape(real_samples.shape[0], -1))
            fake_tsne = tsne.fit_transform(fake_samples.reshape(fake_samples.shape[0], -1))
        
        plt.subplot(1, 2, 2)
        plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c='blue', alpha=0.5, label='Real Data')
        plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], c='red', alpha=0.5, label='Generated Data')
        plt.title('t-SNE Visualization')
        plt.legend()
        
        # 保存可视化结果
        plt.tight_layout()
        vis_path = f'outputs/{args.dataset}/visualization'
        os.makedirs(vis_path, exist_ok=True)
        plt.savefig(f'{vis_path}/pca_tsne.png')
        plt.close()

        # 统一指标计算
        metrics = MetricEvaluator.evaluate_all(real_samples, fake_samples)
        disc_score = discriminative_score_metrics(real_samples, fake_samples)
        
        # 整合评估结果
        jsd = metrics['js_divergence']
        ssim_score = metrics['ssim_score']
        classifier_acc = metrics['classifier_accuracy']
        predictive_score = metrics['predictive_score']
        
        print('================================================\n数据质量评估结果:')
        print(f'Jensen-Shannon散度: {jsd:.4f}')
        #print(f'结构相似性 (SSIM): {ssim_score:.4f}')
        #print(f'模型依赖性 (分类器准确率): {classifier_acc:.4f}')
        print(f'预测评分: {predictive_score:.4f}')
        print(f'判别指标分数: {disc_score:.4f}')
        print(f'特征方差差异: {metrics["feature_variance_diff"]:.4f}')
        
        # 保存完整评估结果
        with open(f'{vis_path}/evaluation_results.txt', 'w') as f:
            f.write(f'Jensen-Shannon Divergence: {jsd:.4f}\n')
            f.write(f'SSIM Score: {ssim_score:.4f}\n')
            f.write(f'Classifier Accuracy: {classifier_acc:.4f}\n')
            f.write(f'Predictive Score: {predictive_score:.4f}\n')
            f.write(f'Discriminative Score: {disc_score:.4f}\n')
            f.write(f'Feature Variance Difference: {metrics["feature_variance_diff"]:.4f}\n')
        
        print(f'可视化结果和评估指标已保存至 {vis_path}')
        
        

if __name__ == "__main__":
    main()