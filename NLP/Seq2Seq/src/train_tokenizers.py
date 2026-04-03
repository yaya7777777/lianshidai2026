import os
import sys
sys.path.append('src')
from tokenizer import SentencePieceTokenizer

data_dir = "d:/lianshidai2026/NLP/seq2seq/data"
# 训练英文分词器
print("Training English tokenizer...")
en_tokenizer = SentencePieceTokenizer()
en_tokenizer.train(
    input_file=os.path.join(data_dir, "train.en"),
    model_prefix=os.path.join(data_dir, "tokenizer.en"),
    vocab_size=8000,
    model_type='bpe'
)
# 训练中文分词器
print("\nTraining Chinese tokenizer...")
zh_tokenizer = SentencePieceTokenizer()
zh_tokenizer.train(
    input_file=os.path.join(data_dir, "train.zh"),
    model_prefix=os.path.join(data_dir, "tokenizer.zh"),
    vocab_size=8000,
    model_type='bpe'
)

print("\nAll tokenizers trained successfully!")