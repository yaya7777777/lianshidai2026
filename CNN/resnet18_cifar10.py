import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理（带数据增强）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),# 随机水平翻转
    transforms.RandomVerticalFlip(),# 随机垂直翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ResNet的核心模块（残差连接）
class BasicBlock(nn.Module):
    expansion = 1 # 残差连接的通道数倍数
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)# 归一化层1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)# 归一化层2
        
        self.shortcut = nn.Sequential()# 残差连接
        if stride != 1 or in_channels != self.expansion * out_channels:
            # 当步长不是1或输入通道数不是输出通道数的倍数时，需要添加卷积层和归一化层使维度匹配才能进行残差连接
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

# ResNet模型（主卷积路径）
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 创建ResNet18模型
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

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
    print('Training ResNet18 on CIFAR-10 dataset...')
    print('Using data augmentation to prevent overfitting')
    
    # 训练ResNet18模型
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # 添加L2正则化
    
    train_losses, val_losses = train(model, trainloader, criterion, optimizer, epochs=5)
    accuracy = test(model, testloader)
    plot_losses(train_losses, val_losses, 'ResNet18')
    torch.save(model.state_dict(), 'resnet18_model.pth')
    
    print('\nResNet18 trained successfully!')
    print(f'Test Accuracy: {accuracy:.2f}%')
