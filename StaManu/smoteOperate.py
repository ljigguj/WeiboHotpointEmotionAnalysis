import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# ===================== 1. 加载标注数据集 =====================
df = pd.read_csv("labeled_emotion_dataset.csv", encoding="utf-8-sig")
X = df["segmented_comment"]
y = df["emotion"]

print(f"数据集大小: {len(df)} 条")
print("原始类别分布：")
print(y.value_counts().to_string())

# ===================== 2. TF-IDF 文本向量化 =====================
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# ===================== 3. 划分训练集 / 测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 4. SMOTE 过采样（仅对训练集）=====================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nSMOTE 过采样后训练集类别分布：")
print(pd.Series(y_train_res).value_counts().to_string())

# ===================== 5. 训练逻辑回归基线模型 =====================
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_res, y_train_res)
print("\n模型训练完成！")

# ===================== 6. 测试集评估 =====================
y_pred = model.predict(X_test)
acc   = accuracy_score(y_test, y_pred)
f1_w  = f1_score(y_test, y_pred, average="weighted")
f1_m  = f1_score(y_test, y_pred, average="macro")

print("\n" + "=" * 40)
print("测试集评估结果")
print("=" * 40)
print(f"  准确率 (Accuracy) : {acc:.4f}")
print(f"  加权 F1           : {f1_w:.4f}")
print(f"  宏平均 F1         : {f1_m:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=["积极", "中性", "消极"], digits=4))

print("混淆矩阵：")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["积极(真)", "中性(真)", "消极(真)"],
                     columns=["积极(预)", "中性(预)", "消极(预)"])
print(cm_df.to_string())

# ===================== 7. 保存模型与向量器 =====================
os.makedirs("baseline_model", exist_ok=True)
joblib.dump(model, "baseline_model/lr_model.pkl")
joblib.dump(tfidf, "baseline_model/tfidf_vectorizer.pkl")
print("\n基线模型已保存至 baseline_model/")


# ===================== 8. 推理函数 =====================
def predict_baseline(text: str) -> tuple:
    """使用 TF-IDF + LR 基线模型预测单条文本情绪。"""
    vec  = tfidf.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label_map = {0: "积极", 1: "中性", 2: "消极"}
    return label_map[pred], float(prob.max()), prob.tolist()
