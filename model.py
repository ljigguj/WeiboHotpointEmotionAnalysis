import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report, accuracy_score
import torch.nn.functional as F


# ===================== 1. 加载情感词典（HowNet + NTUSD）=====================
def load_sentiment_dict():
    positive_words, negative_words = set(), set()
    for fname in ("HowNet_positive.txt", "NTUSD_positive.txt"):
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                positive_words.update(w.strip() for w in f if w.strip())
    for fname in ("HowNet_negative.txt", "NTUSD_negative.txt"):
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                negative_words.update(w.strip() for w in f if w.strip())
    return positive_words, negative_words


pos_words, neg_words = load_sentiment_dict()


# ===================== 2. 计算情感特征 =====================
def get_sentiment_features(text):
    words = list(text)
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)
    polarity = pos_count - neg_count
    intensity = polarity / (len(words) + 1)
    return [pos_count, neg_count, polarity, intensity]


# ===================== 3. BERT模型配置 =====================
MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)


# ===================== 4. 情绪三分类模型 =====================
class SentimentBERT(nn.Module):
    def __init__(self, bert_model, num_classes=3):
        super(SentimentBERT, self).__init__()
        self.bert = bert_model
        self.hidden_size = bert_model.config.hidden_size
        self.fc1 = nn.Linear(self.hidden_size + 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, sentiment_feat):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]
        sentiment_feat = sentiment_feat.to(cls_feat.device)
        concat_feat = torch.cat([cls_feat, sentiment_feat], dim=1)
        x = self.relu(self.fc1(concat_feat))
        x = self.dropout(x)
        return self.fc2(x)


# ===================== 5. Focal Loss（抑制长尾类别）=====================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


# ===================== 6. 自定义数据集 =====================
class EmotionDataset(Dataset):
    def __init__(self, texts, sentiment_feats, labels, tok, max_len=128):
        self.texts = texts
        self.sentiment_feats = sentiment_feats
        self.labels = labels
        self.tokenizer = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_feat': torch.tensor(self.sentiment_feats[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ===================== 7. 模型保存与加载 =====================
def save_model(model, tok, path="saved_model"):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model_weights.pt"))
    tok.save_pretrained(path)
    print(f"模型已保存至 {path}/")


def load_model(path="saved_model"):
    tok = BertTokenizer.from_pretrained(path)
    bert_loaded = BertModel.from_pretrained(MODEL_NAME)
    model = SentimentBERT(bert_loaded)
    model.load_state_dict(torch.load(os.path.join(path, "model_weights.pt"), map_location="cpu"))
    model.eval()
    return model, tok


# ===================== 8. 推理函数 =====================
def predict(text, model, tok, device, max_len=128):
    model.eval()
    senti_feat = torch.tensor(get_sentiment_features(text), dtype=torch.float32).unsqueeze(0).to(device)
    encoding = tok(text, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attn_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attn_mask, senti_feat)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    label_map = {0: "积极", 1: "中性", 2: "消极"}
    pred_id = int(probs.argmax())
    return label_map[pred_id], float(probs[pred_id]), probs.tolist()


# ===================== 9. 加载数据集 =====================
df = pd.read_csv("labeled_emotion_dataset.csv", encoding="utf-8-sig")
texts = df["cleaned_comment"].tolist()
labels = df["emotion"].tolist()
sentiment_feats = np.array([get_sentiment_features(t) for t in texts])

# ===================== 10. 全局配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([5.0, 1.0, 3.0], dtype=torch.float32).to(device)
BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-5
MAX_LEN = 128
N_FOLDS = 5
PATIENCE = 2  # 早停轮次

# ===================== 11. 5折交叉验证 =====================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []
best_global_f1 = 0.0
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"\n{'=' * 30} 第 {fold + 1} 折训练 {'=' * 30}")

    train_texts = [texts[i] for i in train_idx]
    val_texts   = [texts[i] for i in val_idx]
    train_senti = sentiment_feats[train_idx]
    val_senti   = sentiment_feats[val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels   = [labels[i] for i in val_idx]

    train_dataset = EmotionDataset(train_texts, train_senti, train_labels, tokenizer, MAX_LEN)
    val_dataset   = EmotionDataset(val_texts,   val_senti,   val_labels,   tokenizer, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentBERT(bert).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = FocalLoss(weight=weights).to(device)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_fold_f1 = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids     = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_feat = batch['sentiment_feat'].to(device)
            label          = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, sentiment_feat)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # --- 验证 ---
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment_feat = batch['sentiment_feat'].to(device)
                label          = batch['label'].to(device)

                logits = model(input_ids, attention_mask, sentiment_feat)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(label.cpu().numpy())

        val_f1  = f1_score(val_true, val_preds, average='weighted')
        val_acc = accuracy_score(val_true, val_preds)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1:2d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # 保存最佳模型（全局）
        if val_f1 > best_global_f1:
            best_global_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 早停
        if val_f1 > best_fold_f1:
            best_fold_f1 = val_f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  早停触发（{PATIENCE}轮无提升），提前结束当前折训练")
                break

    fold_results.append(best_fold_f1)
    print(f"\n第 {fold + 1} 折最佳 F1: {best_fold_f1:.4f}")
    print(classification_report(val_true, val_preds, target_names=["积极", "中性", "消极"], digits=4))

# ===================== 12. 保存最佳模型 =====================
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    save_model(model, tokenizer, path="saved_model")

# ===================== 13. 5折最终评估 =====================
print(f"\n{'=' * 30} 5折交叉验证最终结果 {'=' * 30}")
print(f"各折最佳 F1: {[round(f, 4) for f in fold_results]}")
print(f"平均加权 F1: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
print(f"最高 F1:    {max(fold_results):.4f}")
