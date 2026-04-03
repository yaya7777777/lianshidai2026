import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # AlexNet需要32x32x3的输入
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10的均值和标准差（rgb通道）
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# AlexNet模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),      
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),    
            nn.Linear(4096, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# 初始化模型、损失函数和优化器
model = AlexNet()
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
    plt.savefig('alexnet_loss_curves.png')
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
        plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)) * 0.5 + 0.5)# 可视化彩色图像
        plt.title(f'Pred: {classes[predicted[i].item()]}, True: {classes[labels[i].item()]}')
        plt.axis('off')
    plt.savefig('alexnet_predictions.png')
    plt.show()

# 主函数
if __name__ == '__main__':
    print('Training AlexNet on CIFAR-10 dataset...')
    # 训练模型
    train_losses, val_losses = train(model, trainloader, criterion, optimizer, epochs=15)
    
    # 测试模型
    accuracy = test(model, testloader)
    
    # 可视化结果
    plot_losses(train_losses, val_losses)
    visualize_predictions(model, testloader)
    
    # 保存模型
    torch.save(model.state_dict(), 'alexnet_model.pth')
    print('Model saved as alexnet_model.pth')
