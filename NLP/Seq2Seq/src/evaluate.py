
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import math
import re
import subprocess
import tempfile



def calculate_bleu(hypotheses, references, max_n=4):

    # 计算n-gram精确度
    precisions = []
    
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        
        for hyp, refs in zip(hypotheses, references):
            hyp_tokens = hyp.split()
            
            # 获取参考中的最大匹配
            max_matches = 0
            max_total = 0
            
            for ref in refs:
                ref_tokens = ref.split()
                
                # 提取n-grams
                hyp_ngrams = get_ngrams(hyp_tokens, n)
                ref_ngrams = get_ngrams(ref_tokens, n)
                
                # 计算匹配数
                hyp_counts = Counter(hyp_ngrams)
                ref_counts = Counter(ref_ngrams)
                
                matches_n = sum((hyp_counts & ref_counts).values())
                total_n = len(hyp_ngrams)
                
                if matches_n > max_matches:
                    max_matches = matches_n
                    max_total = total_n
            
            matches += max_matches
            total += max_total
        
        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0)
    
    # 几何平均
    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geo_mean = 0
    
    # 长度惩罚
    hyp_len = sum(len(h.split()) for h in hypotheses)
    ref_len = sum(len(refs[0].split()) for refs in references)
    
    if hyp_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0
    
    bleu = 100 * bp * geo_mean
    return bleu

# 获取n-grams列表(用于BLEU计算)
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def length_penalty(length, alpha=1.0):
    return ((5 + length) / 6) ** alpha


def calculate_chrf(hypotheses, references, beta=2):
    
    def get_char_ngrams(text, n):
        text = '#' + text + '#'  # 边界标记
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    # 计算F分数
    def calculate_f_score(precision, recall, beta):
        if precision + recall == 0:
            return 0
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    total_score = 0
    max_n = 6  # CHRF默认使用1-6 gram
    
    for hyp, refs in zip(hypotheses, references):
        # 选择最佳参考
        best_score = 0
        for ref in refs:
            ref_score = 0
            
            # 计算1-6 gram的分数
            for n in range(1, max_n + 1):
                hyp_ngrams = get_char_ngrams(hyp, n)
                ref_ngrams = get_char_ngrams(ref, n)
                
                # 计算匹配数
                hyp_counts = Counter(hyp_ngrams)
                ref_counts = Counter(ref_ngrams)
                
                matches = sum((hyp_counts & ref_counts).values())
                precision = matches / len(hyp_ngrams) if hyp_ngrams else 0
                recall = matches / len(ref_ngrams) if ref_ngrams else 0
                
                # 计算F分数
                f_score = calculate_f_score(precision, recall, beta)
                ref_score += f_score
            
            ref_score /= max_n  # 平均
            if ref_score > best_score:
                best_score = ref_score
        
        total_score += best_score
    
    return 100 * (total_score / len(hypotheses))


def calculate_comet(hypotheses, references, sources):

    try:
        # 检查COMET是否安装
        subprocess.run(['comet-score', '--help'], capture_output=True, check=True)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_hyp:
            for hyp in hypotheses:
                f_hyp.write(hyp + '\n')
            hyp_file = f_hyp.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_ref:
            for refs in references:
                f_ref.write(refs[0] + '\n')  # 使用第一个参考
            ref_file = f_ref.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_src:
            for src in sources:
                f_src.write(src + '\n')
            src_file = f_src.name
        
        # 运行COMET
        result = subprocess.run(
            ['comet-score', 
             '--model', 'Unbabel/wmt20-comet-da',
             '--src', src_file,
             '--mt', hyp_file,
             '--ref', ref_file],
            capture_output=True,
            text=True
        )
        
        # 解析结果
        score = float(re.search(r'Average score: (\d+\.\d+)', result.stdout).group(1))
        
        # 清理临时文件
        os.unlink(hyp_file)
        os.unlink(ref_file)
        os.unlink(src_file)
        
        return score
    except Exception as e:
        print(f"COMET calculation failed: {e}")
        return 0.0


class BeamSearch:
    def __init__(self, model, tokenizer, beam_size=4, max_len=100, 
                 length_penalty_alpha=1.0, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty_alpha = length_penalty_alpha
        self.device = device
        
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()
        self.pad_id = tokenizer.pad_id()
    
    # Beam Search解码
    def search(self, src_ids, src_mask):

        batch_size = src_ids.size(0)
        assert batch_size == 1, "Beam search currently only supports batch_size=1"
        
        # 编码源序列
        with torch.no_grad():
            enc_output = self.model.encode(src_ids, src_mask)
        
        # 初始化beam
        beams = [([self.bos_id], 0.0)]  # (序列, 分数)
        completed = []
        
        for step in range(self.max_len):
            if len(beams) == 0:
                break
            
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == self.eos_id:
                    # 已完成序列
                    lp = length_penalty(len(seq), self.length_penalty_alpha)
                    completed.append((seq, score / lp))
                    continue
                
                # 准备输入
                tgt_ids = torch.tensor([seq], dtype=torch.long, device=self.device)
                tgt_mask = self._create_tgt_mask(len(seq))
                
                # 解码
                with torch.no_grad():
                    logits = self.model.decode_step(tgt_ids, enc_output, src_mask, tgt_mask)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # 获取top-k候选
                topk_log_probs, topk_ids = torch.topk(log_probs, self.beam_size * 2)
                
                for log_prob, token_id in zip(topk_log_probs[0], topk_ids[0]):
                    new_seq = seq + [token_id.item()]
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))
            
            # 选择top-k候选
            candidates.sort(key=lambda x: x[1] / length_penalty(len(x[0]), self.length_penalty_alpha), reverse=True)
            beams = candidates[:self.beam_size]
        
        # 添加未完成的序列
        for seq, score in beams:
            if seq[-1] != self.eos_id:
                lp = length_penalty(len(seq), self.length_penalty_alpha)
                completed.append((seq, score / lp))
        
        # 返回最佳序列
        if completed:
            completed.sort(key=lambda x: x[1], reverse=True)
            return completed[0][0]
        else:
            return [self.bos_id, self.eos_id]
    # 创建目标序列mask
    def _create_tgt_mask(self, length):
        mask = torch.tril(torch.ones(length, length, device=self.device)).bool()
        return mask.unsqueeze(0)  # (1, length, length)


class GreedyDecoder:

    def __init__(self, model, tokenizer, max_len=100, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()
    
    # 贪心解码
    def decode(self, src_ids, src_mask):
    
        batch_size = src_ids.size(0)
        assert batch_size == 1, "Greedy decoder currently only supports batch_size=1"
        
        # 编码源序列
        with torch.no_grad():
            enc_output = self.model.encode(src_ids, src_mask)
        
        # 初始化解码序列
        tgt_ids = [self.bos_id]
        
        for _ in range(self.max_len):
            # 准备输入
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=self.device)
            tgt_mask = self._create_tgt_mask(len(tgt_ids))
            
            # 解码
            with torch.no_grad():
                logits = self.model.decode_step(tgt_tensor, enc_output, src_mask, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1).item()
            
            tgt_ids.append(next_token)
            
            # 检查是否结束
            if next_token == self.eos_id:
                break
        
        return tgt_ids
    # 创建目标序列mask
    def _create_tgt_mask(self, length):
        mask = torch.tril(torch.ones(length, length, device=self.device)).bool()
        return mask.unsqueeze(0)

