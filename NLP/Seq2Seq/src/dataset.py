import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import numpy as np
import os


class TranslationDataset(Dataset):

    
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer, 
                 max_src_len=128, max_tgt_len=128):
    
        self.src_tokenizer = src_tokenizer  # 源语言分词器
        self.tgt_tokenizer = tgt_tokenizer  # 目标语言分词器
        self.max_src_len = max_src_len  # 最大源序列长度
        self.max_tgt_len = max_tgt_len  # 最大目标序列长度
        
        # 读取数据集
        self.data = []
        with open(src_file, 'r', encoding='utf-8') as f_src, \
             open(tgt_file, 'r', encoding='utf-8') as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_text = src_line.strip()  # 去掉首尾空格
                tgt_text = tgt_line.strip()  # 去掉首尾空格
                if src_text and tgt_text:  # 过滤空行
                    self.data.append((src_text, tgt_text))
        
        print(f"Loaded {len(self.data)} sentence pairs")
    
    def __len__(self):
    
        return len(self.data)
    
    def __getitem__(self, idx):

        src_text, tgt_text = self.data[idx]
        
        # 编码源序列
        src_ids = self.src_tokenizer.encode(src_text)
        src_ids = src_ids[:self.max_src_len]  # 截断到最大长度
        
        # 编码目标序列（添加<bos>和<eos>）
        tgt_ids = [self.tgt_tokenizer.bos_id()]  # 句子开始标记
        tgt_ids += self.tgt_tokenizer.encode(tgt_text)
        tgt_ids.append(self.tgt_tokenizer.eos_id())  # 句子结束标记
        tgt_ids = tgt_ids[:self.max_tgt_len]  # 截断到最大长度
        
        # 返回数据项
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),  # 源序列ID张量
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),  # 目标序列ID张量
            'src_text': src_text,  # 原始源文本（用于调试和评估）
            'tgt_text': tgt_text   # 原始目标文本（用于调试和评估）
        }


def collate_fn(batch, pad_id=0):
    
    # 获取批次中的最大长度
    max_src_len = max(item['src_ids'].size(0) for item in batch)
    max_tgt_len = max(item['tgt_ids'].size(0) for item in batch)
    
    # 初始化张量
    batch_size = len(batch)
    # 用pad_id填充src_ids和tgt_ids
    src_ids = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_ids = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    # src_mask初始化为False（0），表示都是pad
    src_mask = torch.zeros((batch_size, 1, max_src_len), dtype=torch.bool)
    # tgt_mask初始化为False（0）
    tgt_mask = torch.zeros((batch_size, max_tgt_len, max_tgt_len), dtype=torch.bool)
    
    # 填充数据
    for i, item in enumerate(batch):
        src_len = item['src_ids'].size(0)
        tgt_len = item['tgt_ids'].size(0)
        
        # 填充序列（将实际数据复制到填充张量中）
        src_ids[i, :src_len] = item['src_ids']
        tgt_ids[i, :tgt_len] = item['tgt_ids']
        
        # 源序列padding mask：真实位置为True，pad位置为False
        src_mask[i, 0, :src_len] = True
        
        # 目标序列causal mask：下三角矩阵（防止看到未来token）
        # torch.tril生成下三角矩阵，确保位置i只能看到位置0到i
        tgt_mask[i, :tgt_len, :tgt_len] = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
    
    return {
        'src_ids': src_ids,  # 填充后的源序列ID (batch_size, max_src_len)
        'tgt_ids': tgt_ids,  # 填充后的目标序列ID (batch_size, max_tgt_len)
        'src_mask': src_mask,  # 源序列padding mask (batch_size, 1, max_src_len)
        'tgt_mask': tgt_mask,  # 目标序列causal mask (batch_size, max_tgt_len, max_tgt_len)
        'src_texts': [item['src_text'] for item in batch],  # 原始源文本列表
        'tgt_texts': [item['tgt_text'] for item in batch]   # 原始目标文本列表
    }


def create_dataloaders(
    train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt,
    src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len,
    batch_size, num_workers=0
):

    # 创建训练数据集
    train_dataset = TranslationDataset(
        train_src, train_tgt,
        src_tokenizer, tgt_tokenizer,
        max_src_len, max_tgt_len
    )
    
    # 创建验证数据集
    valid_dataset = TranslationDataset(
        valid_src, valid_tgt,
        src_tokenizer, tgt_tokenizer,
        max_src_len, max_tgt_len
    )
    
    # 创建测试数据集
    test_dataset = TranslationDataset(
        test_src, test_tgt,
        src_tokenizer, tgt_tokenizer,
        max_src_len, max_tgt_len
    )
    
    # 创建DataLoader
    pad_id = src_tokenizer.pad_id()
    
    # 训练DataLoader：shuffle=True打乱数据
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=True  # 加速GPU数据传输
    )
    
    # 验证DataLoader：shuffle=False不打乱
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不打乱
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=True
    )
    
    # 测试DataLoader：shuffle=False不打乱
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不打乱
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader

