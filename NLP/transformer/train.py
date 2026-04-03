import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preparation import create_data_loaders
from transformer_model import TransformerEncoder

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)#注意力掩码（用于忽略填充 token）
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 计算指标
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 验证
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
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    print("=== training Transformer model ===")
    # 加载数据加载器
    try:
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)
        print(f"Data loaded successfully! Train batches count: {len(train_loader)}, Val batches count: {len(val_loader)}, Test batches count: {len(test_loader)}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # 模型参数 - 增加模型容量
    VOCAB_SIZE = 30522  # bert-base-uncased vocab size
    D_MODEL = 256  
    NUM_HEADS = 8  
    NUM_LAYERS = 3  
    D_FF = 1024  
    MAX_LEN = 256  
    DROPOUT = 0.2  
    
    # 初始化模型
    model = TransformerEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 损失函数和优化器 - 调整学习率
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 学习率调度器 - 更积极的学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
    
    # 训练参数
    EPOCHS = 20
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 训练循环
    print(f"Training Transformer model, using device: {device}")
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
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed, total time: {training_time:.2f} seconds")
    
    # 测试模型
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")    
    
    # 保存模型
    model_dir = 'NLP/transformer/model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'transformer_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"model saved to {model_path}")
    
    # 生成训练曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.title('Transformer Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), train_accs, label='Training Accuracy')
    plt.plot(range(1, EPOCHS+1), val_accs, label='Validation Accuracy')
    plt.title('Transformer Accuracy Curve') 
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    results_dir = 'NLP/transformer/results'
    os.makedirs(results_dir, exist_ok=True)
    curves_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(curves_path)
    print(f"Training curves saved to: {curves_path}")
    
    # 保存训练日志
    log_path = os.path.join(results_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"Training configuration:\n")
        f.write(f"Vocab Size: {VOCAB_SIZE}\n")
        f.write(f"D Model: {D_MODEL}\n")
        f.write(f"Num Heads: {NUM_HEADS}\n")
        f.write(f"Num Layers: {NUM_LAYERS}\n")
        f.write(f"D FF: {D_FF}\n")
        f.write(f"Max Len: {MAX_LEN}\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"Batch Size: 32\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate: 1e-4\n")
        f.write(f"Label Smoothing: 0.1\n")
        f.write(f"\nTraining results:\n")
        f.write(f"Total training time: {training_time:.2f} seconds\n")
        f.write(f"Test accuracy: {test_acc:.2f}%\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"\nTraining process:\n")
        for i in range(EPOCHS):
            f.write(f"Epoch {i+1}: Training Loss={train_losses[i]:.4f}, Training Accuracy={train_accs[i]:.2f}%, Validation Loss={val_losses[i]:.4f}, Validation Accuracy={val_accs[i]:.2f}%\n")
    print("Training log saved to NLP/Transformer/results/training_log.txt")

if __name__ == "__main__":
    main()
