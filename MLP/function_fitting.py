import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 目标函数
def target_function(x, y=0, epsilon=0):

    return 2 * x**4 + y**2 + np.exp(x**2) + 6 + epsilon 

# 绘制3D曲面图
def plot_3d_surface():

    # 创建网格
    x = np.linspace(-1, 1, 100)  # 在[-1,1]范围内生成100个均匀分布的点
    y = np.linspace(-1, 1, 100)  # 在[-1,1]范围内生成100个均匀分布的点
    X, Y = np.meshgrid(x, y)    # 创建二维网格，X和Y都是100x100的矩阵
    Z = target_function(X, Y)     # 计算每个网格点的函数值
    
    # 绘制3D曲面图
    fig = plt.figure(figsize=(10, 8))  # 创建图形，设置大小为10x8英寸
    ax = fig.add_subplot(111, projection='3d')  # 创建3D子图
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')  # 绘制曲面
    ax.set_xlabel('X')  # 设置x轴标签
    ax.set_ylabel('Y')  # 设置y轴标签
    ax.set_zlabel('Z')  # 设置z轴标签
    ax.set_title('3D Surface Plot of z = 2x^4 + y^2 + e^x^2 + 6')  # 设置标题
    fig.colorbar(surf)  # 添加颜色条，显示颜色与z值的对应关系
    plt.savefig('3d_surface.png')  # 保存图像
    plt.close(fig)  # 关闭图形，释放内存
    print("3D曲面图已保存为 '3d_surface.png'")

# 构建MLP模型
class MLP(nn.Module):
    def __init__(self, hidden_size=64, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

# 生成数据集
def generate_data(noise=True, size=1000, x_range=(-1, 1), y_range=(-1, 1)):
  
    # 在指定范围内随机生成x和y值
    x = np.random.uniform(x_range[0], x_range[1], size)  # 均匀分布的x值
    y = np.random.uniform(y_range[0], y_range[1], size)  # 均匀分布的y值
    
    # 生成噪声
    if noise:
        epsilon = np.random.normal(0, 1, size)  # 高斯噪声，均值为0，标准差为1
    else:
        epsilon = np.zeros(size)  # 无噪声
    
    # 计算目标值
    z = target_function(x, y, epsilon)
    
    # 转换为PyTorch张量
    X = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)  # 将x和y合并为形状(size, 2)的张量
    y = torch.tensor(z, dtype=torch.float32).unsqueeze(1)  # 将z转换为形状(size, 1)的张量
    
    return X, y

# 训练模型
def train_model(model, train_loader, learning_rate, epochs=200):
   
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器
    loss_history = []  # 记录每个epoch的损失
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        # 遍历每个batch
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()  # 清零梯度
            outputs = model(batch_X)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # 每20个epoch打印一次损失
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return loss_history

# 绘制Loss曲线
def plot_loss_curves(loss_histories, learning_rates):
    """
    绘制不同学习率下的损失曲线
    """
    plt.figure(figsize=(10, 6))  # 创建图形
    # 绘制每条损失曲线
    for i, (history, lr) in enumerate(zip(loss_histories, learning_rates)):
        plt.plot(history, label=f'Learning Rate: {lr}')  # 绘制曲线并添加标签
    plt.xlabel('Epochs')  # 设置x轴标签
    plt.ylabel('Loss')  # 设置y轴标签
    plt.title('Loss Curves for Different Learning Rates')  # 设置标题
    plt.legend()  # 添加图例
    plt.savefig('loss_curves.png')  # 保存图像
    plt.close()  # 关闭图形，释放内存

# 测试模型
def test_model(model, test_X, test_y):
 
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度，节省内存
        predictions = model(test_X)  # 前向传播
        mse = nn.MSELoss()(predictions, test_y)  # 计算MSE
        loss = mse.item()  # loss就是MSE
        print(f'Test MSE: {loss:.4f}')  
    return loss

def main():
    
    # 绘制3D曲面图
    plot_3d_surface()
    
    # 生成训练和测试数据
    train_X, train_y = generate_data(noise=True, size=10000)  # 生成10000个带噪声的训练样本
    test_X, test_y = generate_data(noise=False, size=1000)  # 生成1000个无噪声的测试样本
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_X, train_y)  # 将数据包装为数据集
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 创建数据加载器，batch_size=64
    
    # 测试不同学习率
    learning_rates = [0.1, 0.01, 0.001, 0.0001]  # 测试4个不同的学习率
    loss_histories = []  # 记录每个学习率的损失历史
    test_mses = []  # 记录每个学习率的测试MSE
    
    for lr in learning_rates:
        print(f"\n使用学习率: {lr}")
        model = MLP(hidden_size=64, num_layers=3)  # 创建模型
        loss_history = train_model(model, train_loader, lr)  # 训练模型
        loss_histories.append(loss_history)  # 记录损失历史
        test_mse = test_model(model, test_X, test_y)  # 测试模型
        test_mses.append(test_mse)  # 记录测试MSE
    
    # 绘制Loss曲线
    plot_loss_curves(loss_histories, learning_rates)
    
    
    # 尝试不同的隐藏层大小
    hidden_sizes = [32, 64, 128]  # 测试3个不同的隐藏层大小
    for hidden_size in hidden_sizes:
        print(f"\n使用隐藏层大小: {hidden_size}")
        model = MLP(hidden_size=hidden_size, num_layers=3)  # 创建模型
        train_model(model, train_loader, 0.01, epochs=500)  # 训练500个epoch
        test_model(model, test_X, test_y)  # 测试模型
    
    # 尝试不同的层数
    num_layers_list = [2, 3, 4]  # 测试3个不同的层数
    for num_layers in num_layers_list:
        print(f"\n使用层数: {num_layers}")
        model = MLP(hidden_size=64, num_layers=num_layers)  # 创建模型
        train_model(model, train_loader, 0.01, epochs=500)  # 训练500个epoch
        test_model(model, test_X, test_y)  # 测试模型
    
    # 扩大取值范围
    print("\n扩大取值范围到 [-5, 5]...")
    # 生成更大范围的数据
    train_X_large, train_y_large = generate_data(noise=True, size=10000, x_range=(-5, 5), y_range=(-5, 5))
    test_X_large, test_y_large = generate_data(noise=False, size=1000, x_range=(-5, 5), y_range=(-5, 5))
    
    # 创建数据加载器
    train_dataset_large = TensorDataset(train_X_large, train_y_large)
    train_loader_large = DataLoader(train_dataset_large, batch_size=64, shuffle=True)
    
    # 训练模型（使用更大的模型）
    model_large = MLP(hidden_size=128, num_layers=4)  # 使用更大的模型
    print("训练模型...")
    loss_history_large = train_model(model_large, train_loader_large, 0.001, epochs=500)  # 训练500个epoch
    print("测试模型...")
    test_mse_large = test_model(model_large, test_X_large, test_y_large)  # 测试模型
    
    # 绘制扩大范围后的3D曲面图
    def plot_large_surface():

        x = np.linspace(-5, 5, 100)  # 在[-5,5]范围内生成100个点
        y = np.linspace(-5, 5, 100)  # 在[-5,5]范围内生成100个点
        X, Y = np.meshgrid(x, y)  # 创建网格
        Z = target_function(X, Y)  # 计算函数值
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Surface Plot of z = 2x^4 + y^2 + e^x^2 + 6 (Range: [-5, 5])')
        fig.colorbar(surf)
        plt.savefig('3d_surface_large.png')
        plt.close(fig)
        
    
    plot_large_surface()
    

if __name__ == "__main__":
    main()
