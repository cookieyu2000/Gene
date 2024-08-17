# train.py
import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AdamW, AutoTokenizer
from tqdm import tqdm
import os
import numpy as np
import random
import yaml
from dataloader import train_loader, valid_loader
import warnings

warnings.filterwarnings('ignore')

# 读取配置文件
with open("ner/settings.yml", 'r') as file:
    settings = yaml.safe_load(file)['ner']

# 设置随机种子
seed = settings['seed']
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# 超参数
EPOCHS = int(settings['epochs'])
LEARNING_RATE = float(settings['learning_rate'])
NUM_LABELS = int(settings['num_labels'])
PATIENCE = int(settings['patience'])
LR_DECAY = float(settings['lr_decay'])
LR_TIMES = int(settings['lr_times'])

WEIGHTS_PATH = settings['weights_path']
MODEL_NAME = settings['model_name']
DEVICE = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")
name_model = settings['name_model']

if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)

# 设置模型保存路径
model_save_path = os.path.join(WEIGHTS_PATH, name_model)

# 设置模型
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 检查是否有已存在的权重文件
if os.path.exists(model_save_path):
    print(f"Loading weights from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
else:
    print("No weights found, training from scratch.")

model.to(DEVICE)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 设置损失函数
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = 0
    correct_predictions = 0
    total_predictions = 0

    print("Training started...")
    for step, batch in enumerate(tqdm(data_loader, desc="Training")):
        input_ids, labels = batch  
        attention_mask = (input_ids != 0).int()

        input_ids = input_ids.long().to(device)
        labels = labels.long().to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        loss = loss_fn(logits, labels)
        losses += loss.item()

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        loss.backward()
        optimizer.step()

    return losses / len(data_loader), correct_predictions.double() / total_predictions

def eval_model(model, data_loader, loss_fn, device, epoch):
    model.eval()
    losses = 0
    correct_predictions = 0
    total_predictions = 0

    try:
        print("Evaluating started...")
        with torch.no_grad():
            for step, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                input_ids, labels = batch
                attention_mask = (input_ids != 0).int()

                input_ids = input_ids.long().to(device)
                labels = labels.long().to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)

                loss = loss_fn(logits, labels)
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        return losses / len(data_loader), correct_predictions.double() / total_predictions

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None, None


best_accuracy = 0
no_improve_epochs = 0
lr_adjustments = 0
best_epoch = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.2%}')
    
    val_loss, val_acc = eval_model(model, valid_loader, loss_fn, DEVICE, epoch)
    if val_loss is None:
        print("Evaluation failed, stopping training.")
        break
    print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2%}')
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        no_improve_epochs = 0
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
    else:
        no_improve_epochs += 1
        print(f'No improvement for {no_improve_epochs}/{PATIENCE} epochs.')

        if no_improve_epochs >= PATIENCE:
            if lr_adjustments < LR_TIMES:
                lr_adjustments += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= LR_DECAY
                print(f'Learning rate adjusted to {optimizer.param_groups[0]["lr"]}')
                no_improve_epochs = 0
            else:
                print('Learning rate adjustments exceeded the limit, stopping training.')
                break

    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Best validation accuracy so far: {best_accuracy:.2%}")

print(f'Training complete. Best validation accuracy: {best_accuracy:.2%} at epoch {best_epoch}')



# Training samples: 130314
# Validation samples: 55850
# 186929