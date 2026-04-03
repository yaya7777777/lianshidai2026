import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from loader import ChengduHouseDataset  # 导入自定义数据集类

# 下载数据集
def load_and_inspect_data():

    try:
        df = pd.read_csv('chengdu_house.csv')  # 读取CSV文件
        return df
    except FileNotFoundError:
        print("Error: Data file 'chengdu_house.csv' not found.")
        return None

# 特征工程与数据处理
def preprocess_data(df):

    # 使用自定义数据集类进行数据预处理
    dataset = ChengduHouseDataset(df)
    
    # 提取特征和标签
    X = dataset.features
    y = dataset.labels
    
    # 返回特征列信息
    feature_columns = dataset.feature_columns
    
    return X, y, feature_columns

# 数据集划分
def split_and_convert(X, y):
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分数据集，80%训练，20%测试
    
    # 对目标变量进行标准化
    y_mean = y_train.mean()  # 计算均值
    y_std = y_train.std()  # 计算标准差
    y_train_scaled = (y_train - y_mean) / y_std  # 标准化
    y_test_scaled = (y_test - y_mean) / y_std  # 使用训练集的均值和标准差标准化测试集 
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 训练集特征转换为张量
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)  # 训练集目标转换为张量，并增加维度
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # 测试集
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, y_mean, y_std

# 模型搭建
class LinearRegression(nn.Module):

    def __init__(self, input_dim):

        super(LinearRegression, self).__init__() 
        self.linear = nn.Linear(input_dim, 1)  # 1个输出
    
    def forward(self, x):

        return self.linear(x)

# 训练模型
def train_model(model, X_train, y_train, epochs=100, learning_rate=0.001, batch_size=64):
    
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
    loss_history = []  # 记录每个epoch的损失
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X_train, y_train)  # 将数据包装为数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建数据加载器，支持小批次训练
    
    print(f"Start training, total epochs: {epochs}")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:   
            optimizer.zero_grad()  # 清零梯度
            outputs = model(batch_X)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失
        
        avg_loss = total_loss / len(dataloader)  # 计算平均损失
        loss_history.append(avg_loss)  # 记录损失
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))  # 创建图形
    plt.plot(loss_history)  # 绘制损失曲线
    plt.xlabel('轮次')  # 设置x轴标签
    plt.ylabel('损失')  # 设置y轴标签
    plt.title('训练损失曲线')  # 设置标题
    plt.savefig('training_loss_curve.png')  # 保存图像
    plt.close()  # 关闭图形，释放内存
    
    return model, loss_history

# 模型评估与预测
def evaluate_model(model, X_test, y_test, y_mean, y_std):

    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度，节省内存
        predictions = model(X_test)  # 前向传播，得到预测值
        # 反标准化预测值和真实值
        predictions_original = predictions.numpy() * y_std + y_mean  # 反标准化预测值
        y_test_original = y_test.numpy() * y_std + y_mean  # 反标准化真实值
        
        mse = mean_squared_error(y_test_original, predictions_original)  # 计算均方误差
        rmse = np.sqrt(mse)  # 计算均方根误差
        mae = mean_absolute_error(y_test_original, predictions_original)  # 计算平均绝对误差
        
        print(f'RMSE: {rmse:.4f}') 
        print(f'MAE: {mae:.4f}')  
    
    return rmse, mae

def predict_new_sample(model, y_mean, y_std, feature_columns):

    # 新样本数据
    new_sample = {
        '建筑面积（平方米）': 87.22,
        '房屋户型': '2室1厅1厨1卫',
        '所在楼层': '中层',
        '户型结构': '平层',
        '建筑类型': '住宅',
        '房屋朝向': '北',
        '建筑结构': '砖混结构',
        '装修情况': '精装',
        '配备电梯': '有',
        '交易权属': '商品房',
        '房屋用途': '普通住宅'
    }
    
    # 转换为DataFrame
    new_sample_df = pd.DataFrame([new_sample])  # 将字典转换为DataFrame
    
    # 使用loader进行预处理，传递特征列信息
    new_sample_dataset = ChengduHouseDataset(new_sample_df, is_train=False, feature_columns=feature_columns)
    new_sample_features = new_sample_dataset.features
    new_sample_tensor = torch.tensor(new_sample_features, dtype=torch.float32)  # 转换为张量
    
    # 预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        prediction_scaled = model(new_sample_tensor)  # 前向传播，得到预测值
        # 反标准化预测值
        prediction_original = prediction_scaled.item() * y_std + y_mean  # 反标准化预测值
        print(f'新样本的预测单价: {prediction_original:.2f} 元/平方米')  # 打印预测结果
    
    return prediction_original


def extension_challenges(df):

    # 尝试不同的特征组合
    print("\nExtension 1: Try different feature combinations")
    
    # 使用loader处理数据
    dataset = ChengduHouseDataset(df)
    
    # 尝试不同的特征组合
    feature_combinations = [
        (None, "所有特征"),
        (10, "前10个重要特征"),
        (20, "前20个重要特征"),
        (30, "前30个重要特征")
    ]
    
    for K, description in feature_combinations:
        print(f"\n尝试特征组合: {description}")
        
        # 获取特征选择后的数据
        X = dataset.get_feature_selected_data(K)
        y = dataset.labels
        
        # 划分数据集
        X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化目标变量
        y_mean_red = y_train_red.mean()
        y_std_red = y_train_red.std()
        y_train_scaled_red = (y_train_red - y_mean_red) / y_std_red
        y_test_scaled_red = (y_test_red - y_mean_red) / y_std_red
        
        # 转换为张量
        X_train_tensor_red = torch.tensor(X_train_red, dtype=torch.float32)
        y_train_tensor_red = torch.tensor(y_train_scaled_red, dtype=torch.float32).unsqueeze(1)
        X_test_tensor_red = torch.tensor(X_test_red, dtype=torch.float32)
        y_test_tensor_red = torch.tensor(y_test_scaled_red, dtype=torch.float32).unsqueeze(1)
        
        # 训练模型
        input_dim_red = X_train_tensor_red.shape[1]
        model_red = LinearRegression(input_dim_red)
        model_red, _ = train_model(model_red, X_train_tensor_red, y_train_tensor_red, epochs=50)
        
        # 评估模型
        print(f"\n评估模型 ({description})...")
        evaluate_model(model_red, X_test_tensor_red, y_test_tensor_red, y_mean_red, y_std_red)
        
        # 获取特征重要性
        importance = dataset.get_feature_importance(model_red)
        if importance:
            print(f"\nTop 10特征重要性 ({description}):")
            for feat, imp in importance[:10]:
                print(f"{feat}: {imp:.4f}")
    
    # 2. 使用log(price)进行回归
    print("\nExtension 2: Use log(price) for regression")
    y_log = np.log1p(y)  # 对目标变量取对数，使用log1p避免log(0)的问题
    
    # 划分数据集
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # 标准化目标变量
    y_mean_log = y_train_log.mean()
    y_std_log = y_train_log.std()
    y_train_scaled_log = (y_train_log - y_mean_log) / y_std_log
    y_test_scaled_log = (y_test_log - y_mean_log) / y_std_log
    
    # 转换为张量
    X_train_tensor_log = torch.tensor(X_train_log, dtype=torch.float32)
    y_train_tensor_log = torch.tensor(y_train_scaled_log, dtype=torch.float32).unsqueeze(1)
    X_test_tensor_log = torch.tensor(X_test_log, dtype=torch.float32)
    y_test_tensor_log = torch.tensor(y_test_scaled_log, dtype=torch.float32).unsqueeze(1)
    
    # 训练模型
    input_dim_log = X_train_tensor_log.shape[1]
    model_log = LinearRegression(input_dim_log)
    model_log, _ = train_model(model_log, X_train_tensor_log, y_train_tensor_log, epochs=50)
    
    # 评估模型
    print("\nEvaluating log(price) regression model...")
    # 反标准化预测值和真实值
    model_log.eval()
    with torch.no_grad():
        predictions = model_log(X_test_tensor_log)
        # 反标准化
        predictions_original = np.expm1(predictions.numpy() * y_std_log + y_mean_log)  # 使用expm1进行反变换
        y_test_original = np.expm1(y_test_tensor_log.numpy() * y_std_log + y_mean_log)
        
        mse = mean_squared_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, predictions_original)
        
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
    
    # 可视化特征与价格的关系
    print("\nExtension 3: Visualize feature-price relationships")
    # 价格分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50)  # 绘制房价分布直方图
    plt.xlabel('Unit price (yuan/square meter)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution Histogram')
    plt.savefig('price_distribution.png')
    plt.close()
  
    
    # 面积-价格散点图   
    if '建筑面积（平方米）' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['建筑面积（平方米）'], y, alpha=0.5)  # 绘制建筑面积与房价的散点图
        plt.xlabel('Floor area (square meters)')
        plt.ylabel('Unit price (yuan/square meter)')
        plt.title('Relationship between Floor Area and Price')
        plt.savefig('area_price_scatter.png')
        plt.close()
    
def main():

    df = load_and_inspect_data()
    if df is None:
        print("数据加载失败")
        return
    print(f"数据加载成功，形状: {df.shape}")
    
    # 预处理数据
    X, y, feature_columns = preprocess_data(df)
    if X is None:
        print("数据预处理失败")
        return
    print(f"数据预处理成功，特征形状: {X.shape}")
    
    # 划分数据集
    X_train, y_train, X_test, y_test, y_mean, y_std = split_and_convert(X, y)
    print(f"数据集划分成功，训练集形状: {X_train.shape}")
    
    # 搭建模型
    input_dim = X_train.shape[1]
    model = LinearRegression(input_dim)
    print(f"模型搭建成功，输入维度: {input_dim}")
    
    # 训练模型
    model, _ = train_model(model, X_train, y_train)
    
    # 评估模型
    evaluate_model(model, X_test, y_test, y_mean, y_std)
    
    # 预测新样本
    print("\n预测新样本...")
    predict_new_sample(model, y_mean, y_std, feature_columns)
    
    # 拓展与挑战
    extension_challenges(df)  

if __name__ == "__main__":
    main()
