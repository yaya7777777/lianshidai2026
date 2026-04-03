import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet-5需要32x32x1的输入
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)  # C1: 6个5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # C3: 16个5x5卷积核
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # C5: 120个5x5卷积核
        
        # 池化层（使用最大池化，原论文使用平均池化）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(120, 84)  # F6: 84个神经元
        self.fc2 = nn.Linear(84, 10)  # 输出层: 10个神经元
        
        # 激活函数
        self.sigmoid = nn.Sigmoid(inplace=True)
    
    def forward(self, x):
        # 第一层卷积 + 池化
        x = self.pool(self.sigmoid(self.conv1(x)))
        # 第二层卷积 + 池化
        x = self.pool(self.sigmoid(self.conv2(x)))
        # 第三层卷积
        x = self.sigmoid(self.conv3(x))
        # 展平
        x = x.view(-1, 120)
        # 第一层全连接
        x = self.sigmoid(self.fc1(x))
        # 输出层
        x = self.sigmoid(self.fc2(x))   
        return x

# 初始化模型、损失函数和优化器
model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)# 学习率0.001

# 训练函数
def train(model, trainloader, criterion, optimizer, epochs=5):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for i, (inputs, labels) in enumerate(trainloader):
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
            
            # 每100批次打印一次
            if (i + 1) % 100 == 0:
                print(f'Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}')
        
        # 计算平均训练损失
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss = val_loss / len(testloader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# 测试函数
def test(model, testloader):
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 可视化损失曲线
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.show()

# 可视化预测结果
def visualize_predictions(model, testloader, num_images=10):
    model.eval()
    images, labels = next(iter(testloader))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')# 可视化灰度图像
        plt.title(f'Pred: {predicted[i].item()}, True: {labels[i].item()}')
        plt.axis('off')
    plt.savefig('predictions.png')
    plt.show()

# 主函数
if __name__ == '__main__':
    print('Training LeNet-5 on MNIST dataset...')
    # 训练模型
    train_losses, val_losses = train(model, trainloader, criterion, optimizer, epochs=15)
    
    # 测试模型
    accuracy = test(model, testloader)
    
    # 可视化结果
    plot_losses(train_losses, val_losses)
    visualize_predictions(model, testloader)
    
    # 保存模型
    torch.save(model.state_dict(), 'lenet5_model.pth')
    print('Model saved as lenet5_model.pth')