# 导入需要的包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from collections import defaultdict
from tqdm import tqdm
import copy
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义输出重定向类
class Tee:
    """同时将输出写入文件和标准输出"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# 定义一个函数来读取arff文件
def read_arff(file):            
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    return df

# 设置随机种子以确保结果可重现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)

# 设置输出目录和文件
PREFIX_NAME = 'VAE_Optimized/'
os.makedirs(PREFIX_NAME, exist_ok=True)
MODEL_PATH = os.path.join(PREFIX_NAME, 'model.pth')
output_path = os.path.join(PREFIX_NAME, 'output.txt')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, open(output_path, 'w'))
print('模型保存路径：', MODEL_PATH)
print(f'运行日志已保存至：{output_path}')

# 数据预处理函数
def preprocess_data(train_df, test_df):
    # 合并数据集
    df = pd.concat([train_df, test_df])
    print(f"合并后数据集大小: {df.shape}")
    
    # 热编码处理
    def handle_protocol(inputlist):
        protocol_list = ['tcp', 'udp', 'icmp']
        return [protocol_list.index(x) if x in protocol_list else -1 for x in inputlist]
    
    def handle_service(inputlist):
        service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                       'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames',
                       'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap',
                       'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
                       'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
                       'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i',
                       'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
        return [service_list.index(x) if x in service_list else -1 for x in inputlist]
    
    def handle_flag(inputlist):
        flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
        return [flag_list.index(x) if x in flag_list else -1 for x in inputlist]
    
    # 应用热编码
    df["'protocol_type'"] = handle_protocol(df["'protocol_type'"])
    df["'service'"] = handle_service(df["'service'"])
    df["'flag'"] = handle_flag(df["'flag'"])
    
    return df

# 数据集划分函数
def split_datasets(df, split_point=70000):
    # 为自编码器划分数据
    df1 = df.iloc[:split_point].copy()
    # 为SVM划分数据
    df2 = df.iloc[split_point:].copy()
    df2.index = pd.Series(range(len(df2)))
    
    # 重命名目标列
    new_columns = list(df1.columns)
    new_columns[-1] = 'target'
    df1.columns = new_columns
    
    # 分离正常和异常数据
    normal_df = df1[df1.target == "normal"].drop(labels='target', axis=1)
    anomaly_df = df1[df1.target != "normal"].drop(labels='target', axis=1)
    
    print(f"正常数据形状: {normal_df.shape}")
    print(f"异常数据形状: {anomaly_df.shape}")
    
    return normal_df, anomaly_df, df2

# 标准化和归一化函数
def normalize_data(normal_df, anomaly_df):
    # 标准化
    z_scaler = preprocessing.StandardScaler()
    normal_df = z_scaler.fit_transform(normal_df)
    anomaly_df = z_scaler.transform(anomaly_df)
    
    # 最小最大归一化
    m_scaler = preprocessing.MinMaxScaler()
    normal_df = m_scaler.fit_transform(normal_df)
    anomaly_df = m_scaler.transform(anomaly_df)
    
    return normal_df, anomaly_df, z_scaler, m_scaler

# 创建数据集函数
def create_dataset(df):
    # 重塑数据形状为(样本数, 1, 特征数)
    df = df.reshape(-1, 1, df.shape[1])
    sequences = df.tolist()
    dataset = [torch.tensor(s).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features

# 变分自编码器模型
class VAE(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=20, latent_dim=2, dropout_rate=0.2):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        reconstructed = self.decoder(z)
        
        return mu, logvar, reconstructed

# 损失函数
def vae_loss(reconstructed, x, mu, logvar, beta=1.0):
    # 重构损失
    recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失 = 重构损失 + beta * KL散度
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# 训练函数
def train_model(model, train_dataset, val_dataset, n_epochs, device, beta=1.0, lr=1e-4, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True)
    
    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'train_kl': [], 'val_recon': [], 'val_kl': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    no_improve = 0
    
    for epoch in range(1, n_epochs + 1):
        # 训练阶段
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kl_losses = []
        
        for seq_true in tqdm(train_dataset, desc=f"Epoch {epoch}/{n_epochs} [Train]"):
            if len(seq_true) == 1:
                continue
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            
            # 前向传播
            mu, logvar, reconstructed = model(seq_true)
            
            # 计算损失
            loss, recon_loss, kl_loss = vae_loss(reconstructed, seq_true.view(-1, 41), mu, logvar, beta)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_recon_losses.append(recon_loss.item())
            train_kl_losses.append(kl_loss.item())
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_recon_losses = []
        val_kl_losses = []
        
        with torch.no_grad():
            for seq_true in tqdm(val_dataset, desc=f"Epoch {epoch}/{n_epochs} [Val]"):
                seq_true = seq_true.to(device)
                mu, logvar, reconstructed = model(seq_true)
                
                loss, recon_loss, kl_loss = vae_loss(reconstructed, seq_true.view(-1, 41), mu, logvar, beta)
                
                val_losses.append(loss.item())
                val_recon_losses.append(recon_loss.item())
                val_kl_losses.append(kl_loss.item())
        
        # 计算平均损失
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_recon = np.mean(train_recon_losses)
        train_kl = np.mean(train_kl_losses)
        val_recon = np.mean(val_recon_losses)
        val_kl = np.mean(val_kl_losses)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)
        
        # 打印进度
        print(f'Epoch {epoch}: train loss {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}) | val loss {val_loss:.4f} (recon: {val_recon:.4f}, kl: {val_kl:.4f})')
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            
        # 早停
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs!")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    return model.eval(), history

# 预测函数
def predict(model, dataset, device):
    recon_errors = []
    latent_vectors = []
    
    with torch.no_grad():
        model.eval()
        for seq_true in tqdm(dataset, desc="Predicting"):
            seq_true = seq_true.to(device)
            mu, logvar, reconstructed = model(seq_true)
            
            # 计算重构误差
            recon_error = F.mse_loss(reconstructed, seq_true.view(-1, 41), reduction='sum').item()
            recon_errors.append(recon_error)
            
            # 保存潜在向量
            latent_vectors.append(mu.cpu().numpy())
    
    return np.array(latent_vectors), np.array(recon_errors)

# 绘制损失曲线
def plot_loss_curves(history, prefix_name):
    plt.figure(figsize=(12, 8))
    
    # 总损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('总损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 重构损失
    plt.subplot(2, 2, 2)
    plt.plot(history['train_recon'], label='训练重构损失')
    plt.plot(history['val_recon'], label='验证重构损失')
    plt.title('重构损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # KL散度
    plt.subplot(2, 2, 3)
    plt.plot(history['train_kl'], label='训练KL散度')
    plt.plot(history['val_kl'], label='验证KL散度')
    plt.title('KL散度')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(prefix_name + 'loss_curves.png')
    plt.close()

# 绘制重构误差分布
def plot_reconstruction_error_distribution(normal_errors, anomaly_errors, prefix_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(normal_errors, bins=50, kde=True, label='正常数据', color='blue', alpha=0.6)
    sns.histplot(anomaly_errors, bins=50, kde=True, label='异常数据', color='red', alpha=0.6)
    plt.xlabel('重构损失')
    plt.ylabel('频率')
    plt.title('正常数据与异常数据的重构损失分布')
    plt.legend()
    plt.savefig(prefix_name + 'reconstruction_error_distribution.png')
    plt.close()
    
    # 单独绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(normal_errors, bins=50, kde=True, ax=ax1, color='blue')
    ax1.set_title('正常数据重构损失分布')
    ax1.set_xlabel('重构损失')
    ax1.set_ylabel('频率')
    
    sns.histplot(anomaly_errors, bins=50, kde=True, ax=ax2, color='red')
    ax2.set_title('异常数据重构损失分布')
    ax2.set_xlabel('重构损失')
    ax2.set_ylabel('频率')
    
    plt.tight_layout()
    plt.savefig(prefix_name + 'reconstruction_error_separate.png')
    plt.close()

# 绘制潜在空间可视化
def plot_latent_space(normal_latent, anomaly_latent, prefix_name):
    print(f"normal_latent shape: {normal_latent.shape}")
    print(f"anomaly_latent shape: {anomaly_latent.shape}")
    if normal_latent.shape[1] < 2 or anomaly_latent.shape[1] < 2:
        print("Latent space has less than 2 dimensions, cannot plot.")
        return
    plt.figure(figsize=(12, 10))
    
    # 合并图
    plt.subplot(2, 1, 1)
    plt.scatter(normal_latent[:, 0], normal_latent[:, 1], s=2, label='正常', alpha=0.6, color='blue')
    plt.scatter(anomaly_latent[:, 0], anomaly_latent[:, 1], s=2, label='异常', alpha=0.6, color='red')
    plt.title('潜在空间分布')
    plt.xlabel('潜在维度1')
    plt.ylabel('潜在维度2')
    plt.legend()
    
    # 分开图
    plt.subplot(2, 2, 3)
    plt.scatter(normal_latent[:, 0], normal_latent[:, 1], s=2, alpha=0.6, color='blue')
    plt.title('正常数据潜在空间')
    plt.xlabel('潜在维度1')
    plt.ylabel('潜在维度2')
    
    plt.subplot(2, 2, 4)
    plt.scatter(anomaly_latent[:, 0], anomaly_latent[:, 1], s=2, alpha=0.6, color='red')
    plt.title('异常数据潜在空间')
    plt.xlabel('潜在维度1')
    plt.ylabel('潜在维度2')
    
    plt.tight_layout()
    plt.savefig(prefix_name + 'latent_space.png')
    plt.close()

# 计算最佳阈值
def find_optimal_threshold(normal_errors, anomaly_errors):
    # 合并正常和异常的重构误差
    y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    y_scores = np.concatenate([normal_errors, anomaly_errors])
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # 计算每个阈值的距离到(0,1)点
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
    
    # 找到最佳阈值索引
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, fpr, tpr, thresholds

# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, prefix_name):
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('接收者操作特征曲线 (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(prefix_name + 'roc_curve.png')
    plt.close()

# 评估模型性能
def evaluate_model(normal_errors, anomaly_errors, threshold, prefix_name):
    # 计算正常数据的准确率
    normal_correct = sum(err <= threshold for err in normal_errors)
    normal_accuracy = normal_correct / len(normal_errors)
    
    # 计算异常数据的准确率
    anomaly_correct = sum(err > threshold for err in anomaly_errors)
    anomaly_accuracy = anomaly_correct / len(anomaly_errors)
    
    print(f'正常数据检测准确率: {normal_correct}/{len(normal_errors)} = {normal_accuracy:.4f}')
    print(f'异常数据检测准确率: {anomaly_correct}/{len(anomaly_errors)} = {anomaly_accuracy:.4f}')
    
    # 绘制阈值与准确率的关系
    thresholds = np.linspace(min(min(normal_errors), min(anomaly_errors)), 
                             max(max(normal_errors), max(anomaly_errors)), 1000)
    
    normal_accuracies = []
    anomaly_accuracies = []
    
    for t in thresholds:
        normal_acc = sum(err <= t for err in normal_errors) / len(normal_errors)
        anomaly_acc = sum(err > t for err in anomaly_errors) / len(anomaly_errors)
        normal_accuracies.append(normal_acc)
        anomaly_accuracies.append(anomaly_acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, normal_accuracies, label='正常数据准确率')
    plt.plot(thresholds, anomaly_accuracies, label='异常数据准确率')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'最佳阈值 = {threshold:.4f}')
    plt.xlabel('阈值')
    plt.ylabel('准确率')
    plt.title('阈值与准确率的关系')
    plt.legend()
    plt.savefig(prefix_name + 'threshold_accuracy.png')
    plt.close()
    
    return normal_accuracy, anomaly_accuracy

# 应用模型到测试数据
def apply_model_to_test_data(model, df2, z_scaler, m_scaler, threshold, device, prefix_name):
    # 准备测试数据
    X_test = df2.iloc[:, :-1].values
    y_test = df2.iloc[:, -1].values
    
    # 标准化和归一化
    X_test = z_scaler.transform(X_test)
    X_test = m_scaler.transform(X_test)
    
    # 转换为PyTorch数据集
    X_test = X_test.reshape(-1, 1, X_test.shape[1])
    test_dataset, _, _ = create_dataset(X_test)
    
    # 预测
    latent_vectors, recon_errors = predict(model, test_dataset, device)
    
    # 根据阈值分类
    predictions = ["normal" if err <= threshold else "anomaly" for err in recon_errors]
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    # 计算性能指标
    TP = cm[0, 0]  # 真正例：正常预测为正常
    TN = cm[1, 1]  # 真负例：异常预测为异常
    FP = cm[1, 0]  # 假正例：异常预测为正常
    FN = cm[0, 1]  # 假负例：正常预测为异常
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n性能评估:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '异常'], 
                yticklabels=['正常', '异常'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig(prefix_name + 'confusion_matrix.png')
    plt.close()
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        '重构损失': recon_errors,
        '阈值': [threshold] * len(recon_errors),
        '预测': predictions,
        '真实标签': y_test,
        '是否正确': [pred == true for pred, true in zip(predictions, y_test)]
    })
    
    # 保存结果
    result_df.to_csv(prefix_name + 'prediction_results.csv', index=False)
    
    return accuracy, precision, recall, f1, result_df

# 主函数
def main():
    # 加载数据
    print("加载数据...")
    train_df = read_arff('../KDDTrain+.arff')
    test_df = read_arff('../KDDTest+.arff')
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 数据可视化
    plt.figure(figsize=(8, 6))
    plt.bar(['训练集', '测试集'], [train_df.shape[0], test_df.shape[0]])
    plt.xlabel("数据集")
    plt.ylabel('样本数量')
    plt.title("训练集和测试集的数量分布")
    plt.savefig(PREFIX_NAME + "dataset_distribution.png")
    plt.close()
    
    # 数据预处理
    print("\n数据预处理...")
    df = preprocess_data(train_df, test_df)
    
    # 数据集划分
    print("\n数据集划分...")
    normal_df, anomaly_df, df2 = split_datasets(df)
    
    # 标准化和归一化
    print("\n标准化和归一化...")
    normal_df, anomaly_df, z_scaler, m_scaler = normalize_data(normal_df, anomaly_df)
    
    # 划分训练集、验证集和测试集
    print("\n划分训练集、验证集和测试集...")
    train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=42)
    val_df, test_normal_df = train_test_split(val_df, test_size=0.33, random_state=42)
    
    print(f"训练集形状: {train_df.shape}")
    print(f"验证集形状: {val_df.shape}")
    print(f"测试集(正常)形状: {test_normal_df.shape}")
    print(f"测试集(异常)形状: {anomaly_df.shape}")
    
    # 创建数据集
    print("\n创建PyTorch数据集...")
    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, _, _ = create_dataset(val_df)
    test_normal_dataset, _, _ = create_dataset(test_normal_df)
    test_anomaly_dataset, _, _ = create_dataset(anomaly_df)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    print("创建VAE模型...")
    model = VAE(input_dim=n_features).to(device)
    print(model)

    # n开始训练...")/
    model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=100,
        device=device,
        beta=1.0,
        lr=1e-4,
        patience=10
    )

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至: {MODEL_PATH}")

    # 绘制损失曲线
    print("绘制损失曲线...")
    plot_loss_curves(history, "PREFIXn进行预测...")
    normal_latent, normal_errors = predict(model, test_normal_dataset, device)
    anomaly_latent, anomaly_errors = predict(model, test_anomaly_dataset, device)

    # 绘制重构误差分布
    print("绘制重构误差分布...")
    plot_reconstruction_error_distribution(normal_errors, anomaly_errors, PREFIX_NAME)

    # 绘制潜在空间
    print("绘制潜在空间分布...")
    plot_latent_space(normal_latent, anomaly_latent, PREFIX_NAME)

    # 寻找最佳阈值
    print("计算最佳阈值...")
    threshold, fpr, tpr, thresholds = find_optimal_threshold(normal_errors, anomaly_errors)
    print(f"最佳阈值: {threshold:.4f}")

    # 绘制ROC曲线
    print("绘制ROC曲线...")
    plot_roc_curve(fpr, tpr, PREFIX_NAME)

    # 评n评估模型性能...")
    normal_accuracy, anomaly_accuracy = evaluate_model(normal_errors, anomaly_errors, threshold, PREFIX_NAME)

    # 应用到n应用模型到测试数据...")
    accuracy, precision, recall, f1, result_df = apply_model_to_test_data(
        model, df2, z_scaler, m_scaler, threshold, device, PREFIX_NAME
    )

    # 保存最终结果
    final_results = {
        '正常数据准确率': normal_accuracy,
        '异常数据准确率': anomaly_accuracy,
        '测试集准确率': accuracy,
        '测试集精确率': precision, 
        '测试集召回率': recall,
        'F1分数': f1
    }

    # 将结果保存为CSV
    pd.DataFrame([final_results]).to_csv(PREFIX_NAME + 'final_results.csv')
    print("所有结果已保存至文件夹:", PREFIX_NAME)

if __name__ == "__main__":
    main()