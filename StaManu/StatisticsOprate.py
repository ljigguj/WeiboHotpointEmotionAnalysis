import re
import jieba
import pandas as pd

# 加载停用词表（中文通用停用词，可自行扩展）
stop_words = set()
with open("stop_words.txt", "r", encoding="utf-8") as f:
    for word in f:
        stop_words.add(word.strip())

# 1. 加载原始数据
df = pd.read_csv("raw_hot_comments.csv", encoding="utf-8-sig")
print(f"预处理前总样本数：{len(df)}")

# 2. 去重：删除完全重复的评论
df = df.drop_duplicates(subset=["raw_comment"])
print(f"去重后样本数：{len(df)}")

# 3. 文本清洗函数：去除特殊字符、表情、链接、@用户、标点
def clean_text(text):
    if pd.isna(text):
        return ""
    # 去除@用户、链接、话题标签
    text = re.sub(r"@.*?\s|http.*?\s|#.*?#", "", text)
    # 去除表情符号、特殊字符、数字、英文
    text = re.sub(r"[\U00010000-\U0010ffff]|[^\u4e00-\u9fa5]", "", text)
    # 去除多余空格
    text = re.sub(r"\s+", "", text)
    return text

# 4. 执行清洗
df["cleaned_comment"] = df["raw_comment"].apply(clean_text)

# 5. 降噪：删除空文本、过短文本（小于1个字无情绪意义）
df = df[df["cleaned_comment"].str.len() >= 1]
print(f"清洗降噪后样本数：{len(df)}")

# 6. 中文分词 + 停用词过滤
def word_cut(text):
    # 精准分词
    words = jieba.lcut(text)
    # 过滤停用词、单字
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

df["segmented_comment"] = df["cleaned_comment"].apply(word_cut)

# 7. 保存预处理后数据
df = df[df["segmented_comment"].str.len() > 0]  # 过滤分词后空数据
df.to_csv("preprocessed_comments.csv", index=False, encoding="utf-8-sig")
print(f"预处理完成，最终有效样本：{len(df)}")
print("预处理数据已保存：preprocessed_comments.csv")