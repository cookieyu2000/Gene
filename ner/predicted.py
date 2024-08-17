import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
import matplotlib.pyplot as plt
from dataloader import valid_loader
import os
import yaml
import warnings

warnings.filterwarnings('ignore')

# 读取配置文件
with open("ner/settings.yml", 'r') as file:
    settings = yaml.safe_load(file)['ner']

# 超参数
NUM_LABELS = int(settings['num_labels'])
DEVICE = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = settings['weights_path']
MODEL_NAME = settings['model_name']
name_model = settings['name_model']
model_save_path = os.path.join(WEIGHTS_PATH, name_model)

# 加载模型
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading best model weights from {model_save_path}")
model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 最终评估并生成混淆矩阵和计算F1-score
all_labels = []
all_preds = []
mistakes = []

with torch.no_grad():
    for step, batch in enumerate(valid_loader):
        input_ids, labels = batch
        attention_mask = (input_ids != 0).int()

        input_ids = input_ids.long().to(DEVICE)
        labels = labels.long().to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        _, preds = torch.max(logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        tokens = input_ids.view(-1).cpu().numpy()

        for j in range(len(labels)):
            if labels[j] != preds[j]:
                original_text = tokenizer.decode(input_ids[j // input_ids.shape[1], :].cpu().numpy(), skip_special_tokens=True)
                readable_token = tokenizer.decode([tokens[j]]).strip()
                mistakes.append((original_text, readable_token, preds[j].cpu().numpy(), labels[j].cpu().numpy()))

# 计算并打印F1-score
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Final F1-score (weighted): {f1:.4f}")

# 打印分类报告（包含精确度、召回率和F1-score）
report = classification_report(all_labels, all_preds)
print(report)

# 生成混淆矩阵
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.title('Final Confusion Matrix')
final_cm_path = os.path.join('error', 'final_confusion_matrix.png')
plt.savefig(final_cm_path)
plt.close()

# 保存错误预测样本
mistake_file_path = os.path.join('error', 'final_mistake.txt')
with open(mistake_file_path, 'w') as f:
    for original_text, token, pred, true in mistakes:
        f.write(f"Text: {original_text}\nToken: {token}, Predicted: {pred}, Actual: {true}\n\n")

print(f"Confusion matrix saved to {final_cm_path}")
print(f"Mistakes saved to {mistake_file_path}")
