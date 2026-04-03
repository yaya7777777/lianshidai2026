import torch
import torch.nn as nn
import math
# 位置编码层（Transformer没有循环结构，无法捕捉序列顺序信息，需要位置编码）
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于生成不同频率的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数位置使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整维度并注册为缓冲区（不参与梯度计算）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 将位置编码添加到输入中
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    # 多头注意力层
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 确保模型维度可以被注意力头数量整除
        assert d_model % num_heads == 0
        # 每个注意力头的维度
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询线性变换
        self.W_k = nn.Linear(d_model, d_model)  # 键线性变换
        self.W_v = nn.Linear(d_model, d_model)  # 值线性变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出线性变换
    
    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)
        
        # 线性变换并重塑为多头结构
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)#点积
        
        # 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)# 将无效位置的分数设为-1e9（确保不被选择）
        
        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # 重塑输出并通过线性变换
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(output)
        
        return output

# 前馈神经网络层（给每个 token 提供非线性变换能力）
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)# 添加 Dropout 层，防止过拟合
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
# Transformer 编码器层（包含多头注意力机制和前馈神经网络）
class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 多头自注意力
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  
    
    def forward(self, x, mask):
    
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))# 残差连接+层归一化
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
# Transformer 编码器（包含多个编码器层）
class TransformerEncoder(nn.Module):
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos_encoding = PositionalEncoding(d_model, max_len)  # 位置编码
        # 堆叠多个编码器层（为了提取更丰富的特征）
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)  # Dropout
        self.classifier = nn.Linear(d_model, 2)  # 分类头（二分类）
    
    def forward(self, input_ids, attention_mask):
    
        # 词嵌入
        x = self.embedding(input_ids)
        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 调整注意力掩码的维度（为了匹配自注意力层的输入：(batch_size, 1, 1, seq_len, seq_len)）
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        # 使用 [CLS] token（第一个 token）的输出进行分类
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits