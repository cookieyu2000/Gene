'''
test.py
'''
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
from label import label_map
import yaml
import warnings

warnings.filterwarnings('ignore')

# 读取配置文件
with open("ner/settings.yml", 'r') as file:
    settings = yaml.safe_load(file)['ner_test']

# 设置随机种子
seed = settings['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# HYPER-PARAMETERS
MAX_LEN = settings['max_len']
NUM_LABELS = settings['num_labels']
name_model = settings['name_model']
model_save_path = os.path.join(settings['weights_path'], name_model)
device = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")

# 设置模型
model = AutoModelForTokenClassification.from_pretrained(settings['model_name'], num_labels=NUM_LABELS)

# 打印模型权重路径和名称
print(f"Loading model weights from: {model_save_path}")

model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(settings['model_name'])

def predict(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 截断序列
    token_ids = token_ids[:MAX_LEN]
    
    # 填充序列
    token_ids += [0] * (MAX_LEN - len(token_ids))
    
    input_ids = torch.tensor([token_ids]).to(device)
    attention_mask = (input_ids != 0).int()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    predicted_labels = [label_map[label_id] for label_id in predictions]

    return tokens, predicted_labels

def annotate_text(tokens, labels):
    annotated_text = []
    entity = []
    entity_label = None
    
    for token, label in zip(tokens, labels):
        if label != 'O':
            if label.startswith('B-') or (label.startswith('I-') and entity_label is None):
                if entity:
                    annotated_text.append(f"{''.join(entity)} ({entity_label})")
                entity = [token.replace("##", "")]
                entity_label = label[2:]
            else:
                entity.append(token.replace("##", ""))
        else:
            if entity:
                annotated_text.append(f"{''.join(entity)} ({entity_label})")
                entity = []
                entity_label = None
            annotated_text.append(token.replace("##", ""))
    
    if entity:
        annotated_text.append(f"{''.join(entity)} ({entity_label})")
    
    return ' '.join(annotated_text)

def extract_entities(tokens, labels):
    entities = []
    entity = []
    entity_label = None

    for token, label in zip(tokens, labels):
        if label != 'O':
            if label.startswith('B-') or (label.startswith('I-') and entity_label is None):
                if entity:
                    entities.append((''.join(entity), entity_label))
                entity = [token.replace("##", "")]
                entity_label = label[2:]
            else:
                entity.append(token.replace("##", ""))
        else:
            if entity:
                entities.append((''.join(entity), entity_label))
                entity = []
                entity_label = None
    
    if entity:
        entities.append((''.join(entity), entity_label))
    
    return entities

if __name__ == "__main__":
    while True:
        sample_text = input("Enter text: ")
        
        tokens, predicted_labels = predict(sample_text)
        annotated_text = annotate_text(tokens, predicted_labels)
        entities = extract_entities(tokens, predicted_labels)
        
        # print("Original Text:", sample_text)
        print("Annotated Text:", annotated_text)
        for entity, label in entities:
            print(f"Entity: {entity}, Label: {label}")
