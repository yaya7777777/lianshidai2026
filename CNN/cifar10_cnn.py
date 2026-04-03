import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10的均值和标准差
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义改进的LeNet-5模型（适应彩色图像）
class LeNet5Color(nn.Module):
    def __init__(self):
        super(LeNet5Color, self).__init__()
        # 卷积层（输入通道从1改为3，适应彩色图像）
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 第一层卷积 + 池化
        x = self.pool(self.relu(self.conv1(x)))
        # 第二层卷积 + 池化
        x = self.pool(self.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 16 * 5 * 5)
        # 第一层全连接
        x = self.relu(self.fc1(x))
        # 第二层全连接
        x = self.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x



# 训练函数
def train(model, trainloader, criterion, optimizer, epochs=5):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        print(f'\nEpoch {epoch+1}/{epochs}')
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
        print(f'Train Loss: {train_loss:.4f}')
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / len(testloader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
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
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    return accuracy

# 可视化损失曲线
def plot_losses(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(f'{model_name}_loss_curves.png')
    plt.show()

# 主函数
if __name__ == '__main__':
    print('Training models on CIFAR-10 dataset...')
    
    # 训练改进的LeNet-5模型
    print('\n=== Training LeNet-5 (Color) ===')
    lenet_model = LeNet5Color()
    lenet_criterion = nn.CrossEntropyLoss()
    lenet_optimizer = optim.Adam(lenet_model.parameters(), lr=0.001)
    
    lenet_train_losses, lenet_val_losses = train(lenet_model, trainloader, lenet_criterion, lenet_optimizer, epochs=3)
    lenet_accuracy = test(lenet_model, testloader)
    plot_losses(lenet_train_losses, lenet_val_losses, 'LeNet5Color')
    torch.save(lenet_model.state_dict(), 'lenet5_color_model.pth')
    
    print(f'LeNet-5 (Color) Test Accuracy: {lenet_accuracy:.2f}%')
   