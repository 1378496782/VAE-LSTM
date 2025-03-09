# 导入需要的包
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
from torch import nn,optim
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from collections import defaultdict
from tqdm import tqdm
import torch.utils.data as data
import os

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
matplotlib.rcParams['axes.unicode_minus'] = False

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

# 数据有多种格式。我们将把arff文件加载到Pandas数据帧中:
train=read_arff('../KDDTrain+.arff')
test=read_arff('../KDDTest+.arff')
# 输出数据的shape
print(train.shape)
print(test.shape)
label_list=['train','test']
y_list=[train.shape[0],test.shape[0]]
x = range(len(label_list))
plt.bar(x,y_list)  # 画出训练集和测试集的数量分布柱状图
plt.xticks([index for index in x], label_list)
plt.xlabel("数据集")
plt.ylabel('数量')
plt.show()

# 我们将把训练和测试数据合并成一个数据帧。这将给我们更多的数据来训练我们的自动编码器。我们也会重组:
df = pd.concat([train, test])
print(df.shape)
#df = df.sample(frac=1.0) # 打乱数据顺序，保证随机性
df.head(10)

# 开始热编码操作
protocol_type_list=list(df["'protocol_type'"])
service_list=list(df["'service'"])
flag_list=list(df["'flag'"])

#将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index(x,y):
    return [i for i in range(len(y)) if y[i]==x]

#定义将源文件行中3种协议类型转换成数字标识的函数
def handleProtocol(inputlist):
    protocol_list=['tcp','udp','icmp']
    new_protocol_list=[]
    for x in inputlist:
        if x in protocol_list:
            new_protocol_list.append(find_index(x,protocol_list)[0]) # 第0维返回的是热编码
    return new_protocol_list

#定义将源文件行中70种网络服务类型转换成数字标识的函数
def handleService(inputlist):
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50']
    new_service_list=[]
    for x in inputlist:
        if x in service_list:
            new_service_list.append(find_index(x,service_list)[0])
    return new_service_list


#定义将源文件行中11种网络连接状态转换成数字标识的函数
def handleFlag(inputlist):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    new_flag_list=[]
    for x in inputlist:
        if x in flag_list:
            new_flag_list.append(find_index(x,flag_list)[0])
    return new_flag_list
 
    
    
protocol_type_list=handleProtocol(protocol_type_list)
service_list=handleService(service_list)
flag_list=handleFlag(flag_list)


# 更新'protocol_type'  'service' 'flag' 三列数据
df["'protocol_type'"]=protocol_type_list
df["'service'"]=service_list
df["'flag'"]=flag_list

# 热编码后的数据
df.head(10)

df1=df.iloc[0:70000] # 前70000条用作自编码器模型的数据包含正常和不正常
df2=df.iloc[70000:]  #后70000条用做SVM的数据，包含正常和不正常

print(len(df2))
newindex=[i for i in range(0,78517)]
df2.index=pd.Series(newindex)
df2.tail(5)

#  开始进行自编码器模型实验
new_columns = list(df1.columns)
new_columns[-1] = 'target'
df1.columns = new_columns
 # 无监督学习，不需要目标列
normal_df = df1[df1.target == "normal"].drop(labels='target', axis=1)  # 正常数据
print(normal_df.shape) 
anomaly_df = df1[df1.target != "normal"].drop(labels='target', axis=1)  # 异常数据
print(anomaly_df.shape)
normal_df

# 标准化和最小最大化数据
z_scaler= preprocessing.StandardScaler()
normal_df=z_scaler.fit_transform(normal_df)
anomaly_df=z_scaler.fit_transform(anomaly_df)

m_scaler = preprocessing.MinMaxScaler()
normal_df = m_scaler.fit_transform(normal_df)
anomaly_df = m_scaler.fit_transform(anomaly_df)

print(normal_df)
print("#"*100)
print(anomaly_df)

# 正常数据集划分为训练集、验证集、测试集
train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15, # 划分比例
  random_state=1
)
val_df, test_df = train_test_split(
  val_df,
  test_size=0.33,
  random_state=1
)

print(train_df.shape)  #  正常训练集数量
print(val_df.shape)   # 正常验证集数量
print(test_df.shape)  # 正常测试集数量


anomaly_df=anomaly_df.reshape(-1,1,41) # 异常测试集数量
print(anomaly_df.shape)

# 将数据变成30923*1*41的形状
train_df=train_df.reshape(-1,1,41) 
val_df=val_df.reshape(-1,1,41)
test_df=test_df.reshape(-1,1,41)
print(train_df.shape)  #  正常训练集数量
print(val_df.shape)   # 正常验证集数量
print(test_df.shape)  # 正常测试集数量
print(type(train_df))

# 将数据转化为张量
def create_dataset(df):
  sequences = df.tolist() 
  dataset = [torch.tensor(s).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

train_dataset, seq_len, n_features = create_dataset(train_df) # 正常训练集
val_dataset, _, _ = create_dataset(val_df) # 正常验证集

test_normal_dataset, _, _ = create_dataset(test_df)  #  正常测试集
test_anomaly_dataset, _, _ = create_dataset(anomaly_df) # 异常测试集

# 自编码器模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.rnn_en1 = torch.nn.LSTM(  #41->20
            input_size=41,
            hidden_size=20,
            num_layers=1,
            batch_first=True
        )
        self.rnn_en2 = torch.nn.LSTM(  #20->10
            input_size=20,
            hidden_size=10,
            num_layers=1,
            batch_first=True
        )
        
        self.out = torch.nn.Linear(in_features=10, out_features=2) # 10->2


        self.rnn_de1 = torch.nn.LSTM(  # 2->10
            input_size=2,
            hidden_size=10,
            num_layers=1,
            batch_first=True
        )
        self.rnn_de2 = torch.nn.LSTM(  # 10->20
            input_size=10,
            hidden_size=20,
            num_layers=1,
            batch_first=True
        )
        self.out_2 = torch.nn.Linear(in_features=20, out_features=41) # 20->41
        
    def forward(self, x):
        output, (h_n, c_n) = self.rnn_en1(x) # 41->20
        output,(h_n, c_n)=self.rnn_en2(output.view(-1,1,20)) # 20->10
        encode = self.out(output) # 10->2

        output1, (h_n1, c_n1) = self.rnn_de1(encode.view(-1, 1, 2)) # 2->10
        output1,(h_n1, c_n1)=self.rnn_de2(output1.view(-1, 1, 10)) # 10->20
        decode = self.out_2(output1) # 20->41

        encode=encode.squeeze(-1)
        decode=decode.squeeze(-1)
        return encode, decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 模型加载
model = AutoEncoder().to(device)

# 训练模型
def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 优化器
  criterion = nn.MSELoss(reduction='sum').to(device) # 损失函数使用MSE
  history = dict(train=[], val=[])  # 记录训练数据的误差损失和验证集的误差损失
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0  # 初始化最大损失
  for epoch in range(1, n_epochs + 1):
    model = model.train()  # 切换到训练模式
    train_losses = []  # 训练误差
    for seq_true in train_dataset:
      optimizer.zero_grad()  # 梯度清零
      seq_true = seq_true.to(device)
      seq_compressed,seq_pred = model(seq_true.view(-1,1,41))  # 获得压缩后的数据和还原的数据
      loss = criterion(seq_pred, seq_true.view(-1,1,41)) # 求原始数据和还原后的数据的重构误差损失
      loss.backward()  # 误差反向传播
      optimizer.step()  # 更新参数
      train_losses.append(loss.item())
    val_losses = []
    model = model.eval() # 切换到eval模式  不需要梯度更新
    with torch.no_grad():  # 不需要梯度
      for seq_true in val_dataset:  # 验证集 防止过拟合
        # seq_true=seq_true.view(-1,1,41)
        seq_true = seq_true.to(device)
        _,seq_pred = model(seq_true.view(-1,1,41))
        loss = criterion(seq_pred, seq_true.view(-1,1,41))  # 验证集误差
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses) # 取误差平均
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss: # 更新损失误差
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history

def load_or_train_model(model, train_dataset, val_dataset, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # 先尝试加载已有模型
        if os.path.exists(MODEL_PATH):
            print(f"发现预训练模型 {MODEL_PATH}，正在加载...")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model = model.to(device)
            print("模型加载成功，跳过训练阶段")
            return model, None  # 返回空训练历史
        else:
            raise FileNotFoundError
    except (FileNotFoundError, RuntimeError, EOFError) as e:
        print(f"模型加载失败 ({str(e)})，开始新训练...")
        return train_model(model, train_dataset, val_dataset, n_epochs)

N_EPOCHS = 20

model = AutoEncoder().to(device)

PREFIX_NAME = 'AE3/'
os.makedirs(PREFIX_NAME, exist_ok=True)
MODEL_PATH = os.path.join(PREFIX_NAME, 'model.pth')

if not os.path.exists(MODEL_PATH):
    print("未找到预训练模型，开始训练...")
    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        n_epochs=N_EPOCHS
    )
    torch.save(model, MODEL_PATH)  # 直接保存模型对象

else:
    print("发现已有训练模型，直接加载...")
    model = torch.load(MODEL_PATH, map_location=device)  # 直接加载模型对象
    model = model.to(device)
    history = {'train': [], 'val': []}  # 没有历史数据时保持兼容


#  随着训练次数的增加 train_loss和val_loss的分布情况，曲线大致相近
x = [i for i in range(1, N_EPOCHS + 1)]
y1 = history['train']
y2 = history['val']

if len(y1) > 0 and len(y2) > 0:  # 仅当有训练历史时绘图
    x = [i for i in range(1, len(y1)+1)]  # 动态生成x轴数据
    plt.plot(x, y1, label='train_loss')
    plt.plot(x, y2, label='val_loss')
    plt.legend()
    plt.savefig(PREFIX_NAME + 'train_loss.png')
    plt.show()
else:
    print("无训练历史数据，跳过绘图步骤")
