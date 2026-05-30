import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# 1. 加载标注好的情绪数据集 3分类  正面 中性 负面 分别为0 1 2
df = pd.read_csv("labeled_emotion_dataset.csv", encoding="utf-8-sig")
X = df["segmented_comment"]  # 输入：分词后的文本
y = df["emotion"]            # 标签：情绪类别

# 2. 文本向量化（必须转数值特征，SMOTE才能处理）
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# 3. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 4. SMOTE过采样：合成少数类样本，平衡类别分布
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE过采样后训练集类别分布：")
print(pd.Series(y_train_resampled).value_counts())

# 5. 类别权重调整：模型训练时自动加权 该步解释：如果模型10000条中性评论 2000条正面评论 500条负面 模型全部判断该评论为正面也有很高的正确率
# 方式1：Sklearn内置class_weight
model = LogisticRegression(
    class_weight="balanced",  # 自动根据类别数量计算权重
    max_iter=1000
)

# 方式2：手动指定权重（适用于深度学习）
# class_weights = {0: 5.0, 1: 1.0, 2: 3.0}  # 少数类权重更高

# 6. 训练模型
model.fit(X_train_resampled, y_train_resampled)
print("模型训练完成！")