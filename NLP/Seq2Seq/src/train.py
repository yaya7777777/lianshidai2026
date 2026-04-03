
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import TransformerSeq2Seq
from dataset import create_dataloaders
from tokenizer import SentencePieceTokenizer
from evaluate import LabelSmoothingCrossEntropy, calculate_bleu, calculate_chrf, calculate_comet

# 学习率调度器（Warmup + 余弦退火）
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, d_model, factor=1.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.factor = factor
        self._step = 0
    
    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _get_lr(self):

        step = max(1, self._step)
        # warmup阶段
        if step <= self.warmup_steps:
            return self.factor * (self.d_model ** -0.5) * step * (self.warmup_steps ** -1.5)
        # 余弦退火
        else:
            return self.factor * (self.d_model ** -0.5) * (step ** -0.5)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):

    model.train()
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        src_ids = batch['src_ids'].to(device)
        tgt_ids = batch['tgt_ids'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        
        # 目标输入（去掉最后一个token）和目标输出（去掉第一个token）
        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_mask, tgt_mask[:, :-1, :-1])
        
        # 计算损失（忽略pad位置）
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.contiguous().view(-1))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        total_tokens += (tgt_output != 0).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler._get_lr():.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device, tokenizer):

    model.eval()
    total_loss = 0
    all_hypotheses = []
    all_references = []
    all_sources = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)
            src_texts = batch['src_texts']
            
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            
            # 前向传播
            logits = model(src_ids, tgt_input, src_mask, tgt_mask[:, :-1, :-1])
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()
            
            # 收集用于评估的预测和参考
            preds = logits.argmax(dim=-1)
            for pred, ref, src in zip(preds, tgt_output, src_texts):
                pred_text = tokenizer.decode(pred.cpu().tolist())
                ref_text = tokenizer.decode(ref.cpu().tolist())
                all_hypotheses.append(pred_text)
                all_references.append([ref_text])
                all_sources.append(src)
    
    avg_loss = total_loss / len(data_loader)
    bleu_score = calculate_bleu(all_hypotheses, all_references)
    chrf_score = calculate_chrf(all_hypotheses, all_references)
    comet_score = calculate_comet(all_hypotheses, all_references, all_sources)
    
    return avg_loss, bleu_score, chrf_score, comet_score


def train(
    train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt,
    src_tokenizer_path, tgt_tokenizer_path, save_dir,
    batch_size, max_len, max_src_len, max_tgt_len, num_workers=0,
    d_model=128, num_heads=4, num_layers=2, d_ff=512, dropout=0.1,
    label_smoothing=0.1, clip_grad_norm=1.0, lr=0.0001, weight_decay=0.01,
    warmup_steps=100, epochs=5
):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载分词器
    print("Loading tokenizers...")
    src_tokenizer = SentencePieceTokenizer(src_tokenizer_path)
    tgt_tokenizer = SentencePieceTokenizer(tgt_tokenizer_path)
    
    # 创建数据加载器
    print("Creating dataloaders...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt,
        src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len,
        batch_size, num_workers
    )
    
    # 创建模型
    print("Creating model...")
    model = TransformerSeq2Seq(
        src_vocab_size=src_tokenizer.vocab_size(),
        tgt_vocab_size=tgt_tokenizer.vocab_size(),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数（Label Smoothing）
    if label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            ignore_index=0  # pad_id(忽略pad位置的损失)
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 优化器（AdamW）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),# 前面的参数负责梯度更新，后面的参数负责指数衰减更新
        eps=1e-9,
        weight_decay=weight_decay
    )
    
    # 学习率调度器（Warmup）
    scheduler = WarmupScheduler(optimizer, warmup_steps, d_model)
    
    # 训练记录
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_bleu': [],
        'valid_chrf': [],
        'valid_comet': [],
        'test_bleu': 0,
        'test_chrf': 0,
        'test_comet': 0
    }
    
    best_bleu = 0
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        
        print(f"Epoch {epoch}/{epochs}")
        
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        history['train_loss'].append(train_loss)
        
        # 验证
        valid_loss, valid_bleu, valid_chrf, valid_comet = evaluate(model, valid_loader, criterion, device, tgt_tokenizer)
        history['valid_loss'].append(valid_loss)
        history['valid_bleu'].append(valid_bleu)
        history['valid_chrf'].append(valid_chrf)
        history['valid_comet'].append(valid_comet)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid BLEU: {valid_bleu:.2f}")
        print(f"Valid CHRF: {valid_chrf:.2f}")
        print(f"Valid COMET: {valid_comet:.4f}")
        
        # 保存最佳模型
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_bleu': valid_bleu
            }, best_model_path)
            print(f"Saved best model (BLEU: {valid_bleu:.2f})")
    
    # 测试
    
    print("Testing best model...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_bleu, test_chrf, test_comet = evaluate(model, test_loader, criterion, device, tgt_tokenizer)
    history['test_bleu'] = test_bleu
    history['test_chrf'] = test_chrf
    history['test_comet'] = test_comet
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test BLEU: {test_bleu:.2f}")
    print(f"Test CHRF: {test_chrf:.2f}")
    print(f"Test COMET: {test_comet:.4f}")
    
    # 保存训练历史
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(history, save_dir)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    
    return history


def plot_training_curves(history, save_dir):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['valid_loss'], label='Valid Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # BLEU曲线
    axes[1].plot(history['valid_bleu'], label='Valid BLEU')
    axes[1].axhline(y=history['test_bleu'], color='r', linestyle='--', label=f'Test BLEU: {history["test_bleu"]:.2f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BLEU Score')
    axes[1].set_title('Validation BLEU Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    import traceback
    
    print("Starting training script...")
    
    try:
        # 参数
        train_src = "d:/lianshidai2026/NLP/seq2seq/data/train.zh"
        train_tgt = "d:/lianshidai2026/NLP/seq2seq/data/train.en"
        valid_src = "d:/lianshidai2026/NLP/seq2seq/data/valid.zh"
        valid_tgt = "d:/lianshidai2026/NLP/seq2seq/data/valid.en"
        test_src = "d:/lianshidai2026/NLP/seq2seq/data/test.zh"
        test_tgt = "d:/lianshidai2026/NLP/seq2seq/data/test.en"
        src_tokenizer_path = "d:/lianshidai2026/NLP/seq2seq/data/tokenizer.zh.model"
        tgt_tokenizer_path = "d:/lianshidai2026/NLP/seq2seq/data/tokenizer.en.model"
        save_dir = "d:/lianshidai2026/NLP/seq2seq/model"
        batch_size = 2
        max_len = 64
        max_src_len = 64
        max_tgt_len = 64
        num_workers = 0
        d_model = 128
        num_heads = 4
        num_layers = 2
        d_ff = 512
        dropout = 0.1
        label_smoothing = 0.1
        clip_grad_norm = 1.0
        lr = 0.0001
        weight_decay = 0.01
        warmup_steps = 100
        epochs = 5
        
        print("Parameters loaded from hardcoded values")
        print(f"Save directory: {save_dir}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        print("Save directory created")
        
        # 训练
        print("Starting training...")
        history = train(
            train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt,
            src_tokenizer_path, tgt_tokenizer_path, save_dir,
            batch_size, max_len, max_src_len, max_tgt_len, num_workers,
            d_model, num_heads, num_layers, d_ff, dropout,
            label_smoothing, clip_grad_norm, lr, weight_decay,
            warmup_steps, epochs
        )
        print("Training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
