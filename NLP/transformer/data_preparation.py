import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import os
import traceback


# 创建数据加载器
def create_data_loaders(batch_size=32):
    # 检查本地下载的parquet文件
    data_dir = 'NLP/transformer/data'
    train_parquet = os.path.join(data_dir, 'train-00000-of-00001.parquet')
    test_parquet = os.path.join(data_dir, 'test-00000-of-00001.parquet')

    
    # 强制使用真实的parquet文件
    if True:
    
        # 从本地parquet文件加载数据集
        dataset = load_dataset('parquet', data_files={
            'train': train_parquet,
            'test': test_parquet
        })
        
        # 初始化tokenizer
    
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 设置最大序列长度
        MAX_LEN = 256
        
        # 预处理函数
        def preprocess_function(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LEN,return_tensors='pt')
        
        # 应用预处理

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        
        # 设置PyTorch格式
        tokenized_dataset = tokenized_dataset.with_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # 从训练集中划分出验证集
        train_val_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val_dataset['train']
        val_dataset = train_val_dataset['test']
        test_dataset = tokenized_dataset['test']
        
        print(f"数据集处理成功！训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0,pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0,pin_memory=False)
    
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
