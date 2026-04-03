import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

# 超参数配置
class Config:
    def __init__(self):
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_layers = 2
        self.dropout = 0.5
        self.batch_size = 64
        self.sequence_length = 35
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# 数据预处理
def load_and_preprocess_data():
    # 加载PTB数据集
    train_data, valid_data, test_data = PennTreebank(root='./data')
    
    # 转换为列表
    train_data = list(train_data)
    valid_data = list(valid_data)
    test_data = list(test_data)
    
    # 分词
    tokenizer = get_tokenizer('basic_english')
    
    # 构建词表
    counter = Counter()
    for line in train_data:
        counter.update(tokenizer(line))
    
    # 创建词表，包含<unk>和<eos>标记
    vocab = Vocab(counter, specials=['<unk>', '<eos>'])
    unk_index = vocab['<unk>']# 未知词索引
    eos_index = vocab['<eos>']# 句子结束标记索引
    
    # 数据转换为索引
    def data_to_indices(data):
        indices = []
        for line in data:
            tokens = tokenizer(line)
            # 添加<eos>标记
            tokens.append('<eos>')
            # 转换为索引
            line_indices = [vocab[token] if token in vocab else unk_index for token in tokens]
            indices.extend(line_indices)
        return indices
    
    train_indices = data_to_indices(train_data)
    valid_indices = data_to_indices(valid_data)
    test_indices = data_to_indices(test_data)
    
    return train_indices, valid_indices, test_indices, vocab

# 构建批次数据
def build_batches(indices, batch_size, sequence_length):
    # 计算总长度
    total_length = len(indices)
    # 计算每个批次的大小
    batch_size_total = batch_size * sequence_length
    # 计算可以生成的批次数量
    num_batches = total_length // batch_size_total
    # 截断数据
    indices = indices[:num_batches * batch_size_total]
    # 重塑数据为 (batch_size, num_batches * sequence_length)
    indices = torch.tensor(indices).view(batch_size, -1)
    
    batches = []
    for i in range(0, indices.shape[1] - sequence_length, sequence_length):
        # 输入序列
        input_seq = indices[:, i:i+sequence_length]
        # 目标序列（输入序列右移一位）
        target_seq = indices[:, i+1:i+sequence_length+1]
        batches.append((input_seq, target_seq))
    
    return batches

# 定义RNN语言模型
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden):
        # 嵌入层（把离散的索引投射到低维连续向量空间）
        embedded = self.dropout(self.embedding(x))
        # RNN层
        output, hidden = self.rnn(embedded, hidden)
        # 全连接层
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=config.device)

# 定义LSTM语言模型
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden):
        # 嵌入层
        embedded = self.dropout(self.embedding(x))
        # LSTM层
        output, hidden = self.lstm(embedded, hidden)
        # 全连接层
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=config.device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=config.device))

# 计算Perplexity
def evaluate(model, batches, model_type):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_seq, target_seq in batches:
            input_seq = input_seq.to(config.device)
            target_seq = target_seq.to(config.device)
            
            # 初始化隐藏状态
            if model_type == 'rnn':
                hidden = model.init_hidden(input_seq.size(0))
            else:  # lstm
                hidden = model.init_hidden(input_seq.size(0))
            
            # 前向传播
            output, _ = model(input_seq, hidden)
            
            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            total_loss += loss.item() * input_seq.numel()
            total_tokens += input_seq.numel()
    
    # 计算平均损失
    avg_loss = total_loss / total_tokens
    # 计算Perplexity
    perplexity = np.exp(avg_loss)
    return perplexity

# 训练函数
def train_model(model, train_batches, valid_batches, model_type, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_ppls = []
    valid_ppls = []
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        
        for i, (input_seq, target_seq) in enumerate(train_batches):
            input_seq = input_seq.to(config.device)
            target_seq = target_seq.to(config.device)
            
            # 初始化隐藏状态
            if model_type == 'rnn':
                hidden = model.init_hidden(input_seq.size(0))
            else:  # lstm
                hidden = model.init_hidden(input_seq.size(0))
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            output, hidden = model(input_seq, hidden)
            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            total_loss += loss.item() * input_seq.numel()
            total_tokens += input_seq.numel()
        
        # 计算训练PPL
        train_avg_loss = total_loss / total_tokens
        train_ppl = np.exp(train_avg_loss)
        train_ppls.append(train_ppl)
        
        # 计算验证PPL
        valid_ppl = compute_perplexity(model, valid_batches, model_type)
        valid_ppls.append(valid_ppl)
        
        print(f'Epoch {epoch+1}/{config.num_epochs}, Train PPL: {train_ppl:.2f}, Valid PPL: {valid_ppl:.2f}')
    
    # 保存模型
    torch.save(model.state_dict(), f'{model_name}_model.pth')
    
    # 绘制PPL曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_ppls, label='Train PPL')
    plt.plot(valid_ppls, label='Valid PPL')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title(f'{model_name} Training and Validation PPL')
    plt.legend()
    plt.savefig(f'{model_name}_ppl_curve.png')
    plt.show()
    
    return train_ppls, valid_ppls

# 主函数
def main():
    # 加载和预处理数据
    train_indices, valid_indices, test_indices, vocab = load_and_preprocess_data()
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')
    
    # 构建批次
    train_batches = build_batches(train_indices, config.batch_size, config.sequence_length)
    valid_batches = build_batches(valid_indices, config.batch_size, config.sequence_length)
    test_batches = build_batches(test_indices, config.batch_size, config.sequence_length)
    print(f'Train batches: {len(train_batches)}, Valid batches: {len(valid_batches)}, Test batches: {len(test_batches)}')
    
    # 训练RNN模型
    print('\n=== Training RNN Language Model ===')
    rnn_model = RNNLanguageModel(vocab_size, config.embedding_dim, config.hidden_dim, config.num_layers, config.dropout)
    rnn_model.to(config.device)
    rnn_train_ppls, rnn_valid_ppls = train_model(rnn_model, train_batches, valid_batches, 'rnn', 'rnn')
    
    # 训练LSTM模型
    print('\n=== Training LSTM Language Model ===')
    lstm_model = LSTMLanguageModel(vocab_size, config.embedding_dim, config.hidden_dim, config.num_layers, config.dropout)
    lstm_model.to(config.device)
    lstm_train_ppls, lstm_valid_ppls = train_model(lstm_model, train_batches, valid_batches, 'lstm', 'lstm')
    
    # 测试模型
    print('\n=== Testing Models ===')
    rnn_test_ppl = compute_perplexity(rnn_model, test_batches, 'rnn')
    lstm_test_ppl = compute_perplexity(lstm_model, test_batches, 'lstm')
    
    print(f'RNN Test PPL: {rnn_test_ppl:.2f}')
    print(f'LSTM Test PPL: {lstm_test_ppl:.2f}')
    
    

if __name__ == '__main__':
    main()