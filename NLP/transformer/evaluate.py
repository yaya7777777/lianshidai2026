import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformer_model import TransformerEncoder
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np

# 加载数据集
test_dataset = load_from_disk('NLP/Transformer/data/test_dataset')
test_loader = DataLoader(test_dataset, batch_size=32)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载自搭建的Transformer模型
VOCAB_SIZE = 30522
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 3
D_FF = 1024
MAX_LEN = 256
DROPOUT = 0.2

transformer_model = TransformerEncoder(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=DROPOUT
)
transformer_model.load_state_dict(torch.load('NLP/Transformer/model/transformer_model.pth', map_location=device))
transformer_model.to(device)
transformer_model.eval()

# 加载BERT模型
print("loading BERT model...")
bert_model = BertForSequenceClassification.from_pretrained('NLP/Transformer/model/bert_model')
bert_model.to(device)
bert_model.eval()
tokenizer = BertTokenizer.from_pretrained('NLP/Transformer/model/bert_tokenizer')

# 评估函数
def evaluate_model(model, loader, device, is_bert=False):
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    misclassified = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            if is_bert:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                logits = model(input_ids, attention_mask)
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 收集预测结果和真实标签
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # 收集误分类的样本
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append({
                        'text': batch['text'][i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'logits': logits[i].cpu().numpy()
                    })
    
    accuracy = 100. * correct / total
    return accuracy, predictions, true_labels, misclassified

# 评估自搭建的Transformer模型
print("\nevaluate Transformer model...")
transformer_acc, transformer_preds, transformer_labels, transformer_misclassified = evaluate_model(
transformer_model, test_loader, device, is_bert=False
)

# 评估BERT模型
print("evaluate BERT model...")
bert_acc, bert_preds, bert_labels, bert_misclassified = evaluate_model(
bert_model, test_loader, device, is_bert=True
)

# 计算F1分数
def calculate_f1_score(predictions, true_labels):
    from sklearn.metrics import f1_score
    return f1_score(true_labels, predictions, average='binary')

transformer_f1 = calculate_f1_score(transformer_preds, transformer_labels)
bert_f1 = calculate_f1_score(bert_preds, bert_labels)

# 分析误判样例
print("\nanalyze misclassified samples...")

# 保存误判样例分析
with open('NLP/Transformer/results/misclassification_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Misclassification Analysis\n")
    f.write("\n")
    
    f.write(f"Transformer model:\n")
    f.write(f"Accuracy: {transformer_acc:.2f}%\n")
    f.write(f"F1 score: {transformer_f1:.4f}\n")
    f.write(f"Number of misclassified samples: {len(transformer_misclassified)}\n")
    f.write("\nTop 10 misclassified samples:\n")
    for i, sample in enumerate(transformer_misclassified[:10]):
        f.write(f"Sample {i+1}:\n")
        f.write(f"  True label: {'Positive' if sample['true_label'] == 1 else 'Negative'}\n")
        f.write(f"  Predicted label: {'Positive' if sample['predicted_label'] == 1 else 'Negative'}\n")
        f.write(f"  Text: {sample['text'][:300]}...\n")
        f.write(f"  Model output: {sample['logits']}\n")
        f.write("-" * 80 + "\n")
    
    f.write("\n" + "=" * 80 + "\n")
    
    f.write(f"BERT model:\n")
    f.write(f"Accuracy: {bert_acc:.2f}%\n")
    f.write(f"F1 score: {bert_f1:.4f}\n")
    f.write(f"Number of misclassified samples: {len(bert_misclassified)}\n")
    f.write("\nTop 10 misclassified samples:\n")
    for i, sample in enumerate(bert_misclassified[:10]):
        f.write(f"Sample {i+1}:\n")
        f.write(f"  True label: {'Positive' if sample['true_label'] == 1 else 'Negative'}\n")
        f.write(f"  Predicted label: {'Positive' if sample['predicted_label'] == 1 else 'Negative'}\n")
        f.write(f"  Text: {sample['text'][:300]}...\n")
        f.write(f"  Model output: {sample['logits']}\n")
        f.write("-" * 80 + "\n")

print("Misclassification analysis saved to: NLP/Transformer/results/misclassification_analysis.txt")



