
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import argparse
from dataset import TranslationDataset, collate_fn
from torch.utils.data import DataLoader
from evaluate import calculate_bleu, calculate_chrf, calculate_comet
from tqdm import tqdm


def evaluate_mbart(model, tokenizer, data_loader, device):
    
    model.eval()
    all_hypotheses = []
    all_references = []
    all_sources = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating M-BART"):
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_texts = batch['src_texts']
            tgt_texts = batch['tgt_texts']
            
            # 生成翻译
            generated_ids = model.generate(
                src_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # 解码
            for gen_ids, ref_text, src_text in zip(generated_ids, tgt_texts, src_texts):
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                all_hypotheses.append(pred_text)
                all_references.append([ref_text])
                all_sources.append(src_text)
    
    # 计算评估指标
    bleu_score = calculate_bleu(all_hypotheses, all_references)
    chrf_score = calculate_chrf(all_hypotheses, all_references)
    comet_score = calculate_comet(all_hypotheses, all_references, all_sources)
    
    return bleu_score, chrf_score, comet_score


def main():
    # 参数
    test_src = "d:/lianshidai2026/NLP/seq2seq/data/test.zh"
    test_tgt = "d:/lianshidai2026/NLP/seq2seq/data/test.en"
    batch_size = 2
    max_src_len = 64
    max_tgt_len = 64
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载M-BART模型和tokenizer
    print("Loading M-BART model...")
    model_name = "facebook/mbart-large-50-one-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # 设置源语言和目标语言
    tokenizer.src_lang = "zh_CN"
    
    # 创建测试数据集
    class MbartDataset(TranslationDataset):
        def __getitem__(self, idx):
            src_text, tgt_text = self.data[idx]
            
            # M-BART tokenizer编码
            src_ids = tokenizer(src_text, return_tensors="pt", padding="max_length", 
                              max_length=max_src_len).input_ids[0]
            tgt_ids = tokenizer(tgt_text, return_tensors="pt", padding="max_length", 
                              max_length=max_tgt_len).input_ids[0]
            
            return {
                'src_ids': src_ids,
                'tgt_ids': tgt_ids,
                'src_text': src_text,
                'tgt_text': tgt_text
            }
    
    test_dataset = MbartDataset(
        test_src, test_tgt,
        None, None,  # M-BART使用自己的tokenizer
        max_src_len, max_tgt_len
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_id=tokenizer.pad_token_id)
    )
    
    # 评估
    print("Evaluating M-BART model...")
    bleu_score, chrf_score, comet_score = evaluate_mbart(model, tokenizer, test_loader, device)
    
    print("\nM-BART Evaluation Results:")

    print(f"BLEU: {bleu_score:.2f}")
    print(f"CHRF: {chrf_score:.2f}")
    print(f"COMET: {comet_score:.4f}")
    
    return {
        'bleu': bleu_score,
        'chrf': chrf_score,
        'comet': comet_score
    }


if __name__ == "__main__":
    main()
