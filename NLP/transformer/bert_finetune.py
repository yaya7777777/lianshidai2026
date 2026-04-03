import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
import time

# 导入数据预处理模块
from data_preparation import create_data_loaders

# 创建数据加载器
print("Creating data loaders...")
train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 训练参数
EPOCHS = 3

# 记录训练过程
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        
        # 计算指标
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 验证函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 训练循环
print("Starting BERT fine-tuning...")
start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # 训练
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 验证
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 学习率调度
    scheduler.step(val_loss)
    
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

training_time = time.time() - start_time
print(f"\nTotal training time: {training_time:.2f} seconds")

# 测试模型
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# 保存模型
model.save_pretrained('model/bert_model')
tokenizer.save_pretrained('model/bert_tokenizer')

# 生成训练曲线
print("Generating training curves...")
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
plt.title('BERT LOSS Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accs, label='Training Accuracy')
plt.plot(range(1, EPOCHS+1), val_accs, label='Validation Accuracy')
plt.title('BERT Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('results/bert_training_curves.png')
print("Training curves saved successfully!")

# 保存训练日志
print("Saving training log...")
with open('results/bert_training_log.txt', 'w') as f:
    f.write(f"BERT Fine-Tuning Config:\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Batch Size: 32\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning Rate: 2e-5\n")
    f.write(f"\nTraining Results:\n")
    f.write(f"Total Training Time: {training_time:.2f} seconds\n")
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"\nTraining Process:\n")
    for i in range(EPOCHS):
        f.write(f"Epoch {i+1}: Training Loss={train_losses[i]:.4f}, Training Accuracy={train_accs[i]:.2f}%, Validation Loss={val_losses[i]:.4f}, Validation Accuracy={val_accs[i]:.2f}%\n")

