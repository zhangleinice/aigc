# 随机森林（RandomForest）算法，做了一次训练和测试

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 读取数据
training_data = pd.read_parquet("data/test_all_with_embeddings.parquet")

# 从数据中随机抽样50000行作为训练集（500行test）
df = training_data.sample(n=500, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.category, test_size=0.2, random_state=42
)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train, y_train)

# 在测试集上进行预测
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

# 输出分类报告
report = classification_report(y_test, preds)
print(report)
