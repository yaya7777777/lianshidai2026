
import sentencepiece as spm
import os
import pickle


class SentencePieceTokenizer:
    
    def __init__(self, model_path=None):
    
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.Load(model_path)
    
    def train(self, input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    
        # 添加特殊符号
        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        
        train_args = {
            'input': input_file,
            'model_prefix': model_prefix,
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': 0.9995,  # 字符覆盖率（中文需要较高的覆盖率）
            'num_threads': 8,  # 使用8个线程加速训练
            'split_by_whitespace': True,  # 按空格分词
            'split_by_unicode_script': True,  # 按Unicode脚本分词（处理中英文混合）
            'split_by_number': True,  # 按数字分词
            'max_sentencepiece_length': 16,  # 最大子词长度
            'add_dummy_prefix': True,  # 添加虚拟前缀
            'remove_extra_whitespaces': True,  # 移除额外的空格
            'normalization_rule_name': 'nmt_nfkc_cf',  # NMT标准归一化规则
            'pad_id': 0,  # <pad>填充符ID
            'bos_id': 1,  # <bos>句子开始符ID
            'eos_id': 2,  # <eos>句子结束符ID
            'unk_id': 3,  # <unk>未知词ID
        }
        
        # 构建训练命令字符串
        cmd = f"--input={train_args['input']} " \
              f"--model_prefix={train_args['model_prefix']} " \
              f"--vocab_size={train_args['vocab_size']} " \
              f"--model_type={train_args['model_type']} " \
              f"--character_coverage={train_args['character_coverage']} " \
              f"--num_threads={train_args['num_threads']} " \
              f"--split_by_whitespace={str(train_args['split_by_whitespace']).lower()} " \
              f"--split_by_unicode_script={str(train_args['split_by_unicode_script']).lower()} " \
              f"--split_by_number={str(train_args['split_by_number']).lower()} " \
              f"--max_sentencepiece_length={train_args['max_sentencepiece_length']} " \
              f"--add_dummy_prefix={str(train_args['add_dummy_prefix']).lower()} " \
              f"--remove_extra_whitespaces={str(train_args['remove_extra_whitespaces']).lower()} " \
              f"--normalization_rule_name={train_args['normalization_rule_name']} " \
              f"--pad_id={train_args['pad_id']} " \
              f"--bos_id={train_args['bos_id']} " \
              f"--eos_id={train_args['eos_id']} " \
              f"--unk_id={train_args['unk_id']}"
        
        # 执行训练
        spm.SentencePieceTrainer.train(cmd)
        
        # 加载训练好的模型
        self.sp.Load(f"{model_prefix}.model")
        print(f"Tokenizer trained and saved to {model_prefix}.model")
    
    # 编码文本为ID序列
    def encode(self, text, add_bos=False, add_eos=False):

        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id()] + ids
        if add_eos:
            ids = ids + [self.eos_id()]
        return ids
    
    # 解码ID序列为文本
    def decode(self, ids):
        return self.sp.DecodeIds(ids)
    
    # 编码为子词片段（用于调试）
    def encode_as_pieces(self, text):

        return self.sp.EncodeAsPieces(text)
    
    # 获取词汇表大小
    def vocab_size(self):
        return self.sp.GetPieceSize()
    
    # 获取填充符ID
    def pad_id(self):
        return self.sp.pad_id()
    
    # 获取句子开始符ID
    def bos_id(self):
        return self.sp.bos_id()
    
    # 获取句子结束符ID  
    def eos_id(self):
        return self.sp.eos_id()
    
    # 获取未知词ID
    def unk_id(self):
        return self.sp.unk_id()


def prepare_tokenizer_data(src_file, tgt_file, output_file, max_lines=None):

  
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for i, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
            if max_lines and i >= max_lines:
                break
            f_out.write(src_line.strip() + '\n')
            f_out.write(tgt_line.strip() + '\n')
    
    print(f"Tokenizer training data saved to {output_file}")


if __name__ == "__main__":
    # 命令行接口：用于独立训练分词器
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SentencePiece Tokenizer')
    parser.add_argument('--src', type=str, required=True, help='Source language file')
    parser.add_argument('--tgt', type=str, required=True, help='Target language file')
    parser.add_argument('--output', type=str, default='tokenizer', help='Output model prefix')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--model-type', type=str, default='bpe', choices=['bpe', 'unigram'])
    
    args = parser.parse_args()
    
    # 准备训练数据
    train_data_file = f"{args.output}_train.txt"
    prepare_tokenizer_data(args.src, args.tgt, train_data_file)
    
    # 训练分词器
    tokenizer = SentencePieceTokenizer()
    tokenizer.train(train_data_file, args.output, args.vocab_size, args.model_type)
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
    print(f"Special tokens: PAD={tokenizer.pad_id()}, BOS={tokenizer.bos_id()}, "
          f"EOS={tokenizer.eos_id()}, UNK={tokenizer.unk_id()}")
