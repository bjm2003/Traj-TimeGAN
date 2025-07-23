import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
from utils import train_test_divide, extract_time
from sklearn.preprocessing import StandardScaler

class MetricEvaluator:
    @staticmethod
    def js_divergence(real_samples, fake_samples):
        kde_real = KernelDensity(bandwidth=0.5).fit(real_samples)
        kde_fake = KernelDensity(bandwidth=0.5).fit(fake_samples)
        
        samples_real = real_samples[np.random.choice(len(real_samples), 1000, replace=True)]
        samples_fake = fake_samples[np.random.choice(len(fake_samples), 1000, replace=True)]
        
        log_p_real = kde_real.score_samples(samples_real)
        log_p_fake = kde_fake.score_samples(samples_fake)
        
        p_real = np.exp(log_p_real)
        p_fake = np.exp(log_p_fake)
        p_real /= p_real.sum()
        p_fake /= p_fake.sum()
        
        m_probs = 0.5 * (p_real + p_fake)
        return 0.5 * (entropy(p_real, m_probs) + entropy(p_fake, m_probs))

    @staticmethod
    def calculate_ssim(real, fake):
        ssim_values = []
        for r, f in zip(real, fake):
            if min(r.shape) < 7 or min(f.shape) < 7:
                continue
            ssim_values.append(ssim(r, f, data_range=f.max()-f.min(), win_size=7))
        return np.mean(ssim_values) if ssim_values else 0

    @staticmethod
    def classifier_accuracy(real, fake):
        X = np.concatenate([real, fake])
        y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
        X_train, X_test, y_train, y_test = train_test_split(
            X.reshape(X.shape[0], -1), y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    @staticmethod
    def predictive_score(real, fake):
        """计算预测评分（使用随机森林回归器）"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 数据维度验证
        if real.ndim != 3 or fake.ndim != 3:
            raise ValueError("输入数据必须是3维数组 (样本数×时间步×特征数)")
        
        # 统一时间步长度
        min_timesteps = min(real.shape[1], fake.shape[1])
        real = real[:, :min_timesteps, :]
        fake = fake[:, :min_timesteps, :]
        
        # 构建特征矩阵（保持时间连续性）
        X_train = real[:, :-1, :].reshape(-1, real.shape[2])
        y_train = real[:, 1:, :].reshape(-1, real.shape[2])
        X_test = fake[:, :-1, :].reshape(-1, fake.shape[2])
        y_test = fake[:, 1:, :].reshape(-1, fake.shape[2])
        
        # 数据有效性检查
        for data in [X_train, y_train, X_test, y_test]:
            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("输入数据包含NaN或无穷值")
        
        # 标准化处理
        scaler_X = StandardScaler().fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler().fit(y_train)
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)
        
        # 优化模型配置
        regressor = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            max_features=0.8,
            random_state=42
        )
        regressor.fit(X_train, y_train)
        
        # 预测并逆标准化
        y_pred = scaler_y.inverse_transform(regressor.predict(X_test))
        y_test = scaler_y.inverse_transform(y_test)
        
        # 过滤异常预测值
        valid_mask = (np.abs(y_pred) < 1e6).all(axis=1)
        if not valid_mask.any():
            return float('inf')
        
        return np.mean(np.square(y_test[valid_mask] - y_pred[valid_mask]))

    @staticmethod
    def feature_variance_ratio(real, fake):
        """计算特征方差比值"""
        # 处理三维数据（样本数×时间步×特征）
        real_var = np.var(real.reshape(-1, real.shape[-1]), axis=0).mean()
        fake_var = np.var(fake.reshape(-1, fake.shape[-1]), axis=0).mean()
        ratio = fake_var / (real_var + 1e-8)
        return ratio * 100  # 转换为百分比

    @classmethod
    def evaluate_all(cls, real_samples, fake_samples):
        variance_ratio = cls.feature_variance_ratio(real_samples, fake_samples)
        return {
            'js_divergence': cls.js_divergence(real_samples.mean(axis=0), fake_samples.mean(axis=0)),
            'ssim_score': cls.calculate_ssim(real_samples, fake_samples),
            'classifier_accuracy': cls.classifier_accuracy(real_samples, fake_samples),
            'predictive_score': cls.predictive_score(real_samples, fake_samples),
            'feature_variance_diff': variance_ratio
        }

def discriminative_score_metrics(ori_data, generated_data):
    """计算判别分数（使用随机森林分类器）"""
    from sklearn.ensemble import RandomForestClassifier
    
    # 数据预处理
    real_samples = ori_data.reshape(ori_data.shape[0], -1)
    fake_samples = generated_data.reshape(generated_data.shape[0], -1)
    
    # 创建标签
    X = np.vstack([real_samples, fake_samples])
    y = np.concatenate([np.zeros(len(real_samples)), np.ones(len(fake_samples))])
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 训练分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 计算判别分数
    acc = clf.score(X_test, y_test)
    # 打印训练集和测试集的标签分布
    print(f"训练集标签分布: {np.bincount(y_train.astype(int))}")
    print(f"测试集标签分布: {np.bincount(y_test.astype(int))}")
    # 打印分类器的准确率
    print(f"分类器准确率: {acc}")
    return 1 - acc