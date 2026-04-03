
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码张量（max_len, d_model）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 多头注意力
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0# 确保d_model能被num_heads整除
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并reshape为多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask
        if mask is not None:
            # (batch_size, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
            if mask.dim() == 3:
                #(batch_size, 1, seq_len) 或 (batch_size, seq_len, seq_len)
                if mask.size(1) == 1:
                    # (batch_size, 1, seq_len) -> (batch_size, 1, 1, seq_len)
                    mask = mask.unsqueeze(2)
                    # (batch_size, 1, 1, seq_len) -> (batch_size, num_heads, 1, seq_len)
                    mask = mask.expand(batch_size, self.num_heads, 1, mask.size(3))
                else:
                    # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                    mask = mask.unsqueeze(1)
                    # (batch_size, 1, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
                    mask = mask.expand(batch_size, self.num_heads, mask.size(2), mask.size(3))
            elif mask.dim() == 2:
                # (seq_len, seq_len) -> (1, 1, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, mask.size(0), mask.size(1))
            elif mask.dim() == 4:
                # mask已经是4维，直接使用
                pass
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        context = torch.matmul(attn, V)
        
        # 拼接多头并线性变换
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 编码器层
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# 解码器层
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力（带mask）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x, mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(x) * self.scale
        x = self.pos_encoding(x)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# 解码器（只有这个是新加的）
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(x) * self.scale
        x = self.pos_encoding(x)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x

# 把上面的编码器和解码器组合起来，形成一个Transformer Seq2Seq模型
class TransformerSeq2Seq(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    # 初始化参数
    # Xavier初始化
    # 初始化位置编码
    def _init_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):

        # 编码
        enc_output = self.encoder(src_ids, src_mask)
        
        # 解码
        dec_output = self.decoder(tgt_ids, enc_output, src_mask, tgt_mask)
        
        # 投影到词汇表
        logits = self.output_projection(dec_output)
        
        return logits
    # 编码源序列
    def encode(self, src_ids, src_mask=None):
        return self.encoder(src_ids, src_mask)
    # 单步解码（用于推理）
    def decode_step(self, tgt_ids, enc_output, src_mask=None, tgt_mask=None):
        dec_output = self.decoder(tgt_ids, enc_output, src_mask, tgt_mask)
        logits = self.output_projection(dec_output)
        return logits


if __name__ == "__main__":
    # 测试模型
    batch_size = 2
    src_len = 10
    tgt_len = 12
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_len=128,
        dropout=0.1
    )
    
    # 测试数据
    src_ids = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    logits = model(src_ids, tgt_ids)
 
