import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
import torch.nn.functional as F


# ===================== 1. 加载情感词典（HowNet + NTUSD）=====================
def load_sentiment_dict():
    positive_words = set()
    with open("HowNet_positive.txt", "r", encoding="utf-8") as f:
        positive_words.update([w.strip() for w in f.readlines()])
    with open("NTUSD_positive.txt", "r", encoding="utf-8") as f:
        positive_words.update([w.strip() for w in f.readlines()])

    negative_words = set()
    with open("HowNet_negative.txt", "r", encoding="utf-8") as f:
        negative_words.update([w.strip() for w in f.readlines()])
    with open("NTUSD_negative.txt", "r", encoding="utf-8") as f:
        negative_words.update([w.strip() for w in f.readlines()])
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
    def __init__(self, bert, num_classes=3):
        super(SentimentBERT, self).__init__()
        self.bert = bert
        self.hidden_size = bert.config.hidden_size
        self.fc1 = nn.Linear(self.hidden_size + 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, sentiment_feat):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]

        # 批量情感特征处理（适配Dataloader）
        sentiment_feat = sentiment_feat.to(cls_feat.device)
        concat_feat = torch.cat([cls_feat, sentiment_feat], dim=1)

        x = self.fc1(concat_feat)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


# ===================== 5. Focal Loss（抑制长尾类别）=====================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ===================== 6. 自定义数据集（批处理训练）=====================
class EmotionDataset(Dataset):
    def __init__(self, texts, sentiment_feats, labels, tokenizer, max_len=128):
        self.texts = texts
        self.sentiment_feats = sentiment_feats
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment = self.sentiment_feats[idx]
        label = self.labels[idx]

        # BERT编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_feat': torch.tensor(sentiment, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ===================== 7. 加载数据集 =====================
df = pd.read_csv("labeled_emotion_dataset.csv", encoding="utf-8-sig")
texts = df["cleaned_comment"].tolist()
labels = df["emotion"].tolist()
sentiment_feats = np.array([get_sentiment_features(t) for t in texts])

# ===================== 8. 全局配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([5.0, 1.0, 3.0], dtype=torch.float32).to(device)  # 类别权重
BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-5
MAX_LEN = 128
N_FOLDS = 5  # 5折交叉验证

# ===================== 9. 5折交叉验证 + 训练主函数 =====================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []  # 保存每折的结果

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"\n{'=' * 30} 第 {fold + 1} 折训练 {'=' * 30}")

    # 1. 划分当前折数据
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_senti = sentiment_feats[train_idx]
    val_senti = sentiment_feats[val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    # 2. 构建数据集/加载器
    train_dataset = EmotionDataset(train_texts, train_senti, train_labels, tokenizer, MAX_LEN)
    val_dataset = EmotionDataset(val_texts, val_senti, val_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化模型/优化器/损失函数
    model = SentimentBERT(bert).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Focal Loss 替换交叉熵
    criterion = FocalLoss(weight=weights).to(device)

    # 余弦退火学习率调度器
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # ===================== 10. 单折训练循环 =====================
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        # 训练阶段
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_feat = batch['sentiment_feat'].to(device)
            label = batch['label'].to(device)

            # 前向传播
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, sentiment_feat)
            loss = criterion(logits, label)

            # 反向传播 + 梯度裁剪（防止梯度爆炸）
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪

            # 更新参数 + 学习率衰减
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment_feat = batch['sentiment_feat'].to(device)
                label = batch['label'].to(device)

                logits = model(input_ids, attention_mask, sentiment_feat)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(label.cpu().numpy())

        # 计算指标
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        print(f"Epoch {epoch + 1:2d} | 训练损失: {train_loss / len(train_loader):.4f} | 验证加权F1: {val_f1:.4f}")

    # 保存当前折最终结果
    fold_results.append(val_f1)
    print(f"\n第 {fold + 1} 折最终结果：")
    print(classification_report(val_true, val_preds, target_names=["负面", "中性", "正面"], digits=4))

# ===================== 11. 5折最终评估 =====================
print(f"\n{'=' * 30} 5折交叉验证最终结果 {'=' * 30}")
print(f"平均加权F1分数: {np.mean(fold_results):.4f}")
print(f"每折F1分数: {fold_results}")