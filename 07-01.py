
# 用 K-Means 算法来进行聚类
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import openai, os

# 读取包含嵌入向量的DataFrame
embedding_df = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 将嵌入向量转换为矩阵
matrix = np.vstack(embedding_df.embedding.values)

# 设置聚类的簇数
num_of_clusters = 20

# 使用K-means算法进行聚类
kmeans = KMeans(n_clusters=num_of_clusters, init="k-means++", n_init=10, random_state=42)
kmeans.fit(matrix)

# 获取每个样本所属的簇标签
labels = kmeans.labels_

# 将簇标签添加到DataFrame中
embedding_df["cluster"] = labels





# 统计每个cluster的数量
new_df = embedding_df.groupby('cluster')['cluster'].count().reset_index(name='count')

# 统计这个cluster里最多的分类的数量
title_count = embedding_df.groupby(['cluster', 'title']).size().reset_index(name='title_count')
first_titles = title_count.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
first_titles = first_titles.reset_index(drop=True)
new_df = pd.merge(new_df, first_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
new_df = new_df.rename(columns={'title': 'rank1', 'title_count': 'rank1_count'})

# 统计这个cluster里第二多的分类的数量
second_titles = title_count[~title_count['title'].isin(first_titles['title'])]
second_titles = second_titles.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
second_titles = second_titles.reset_index(drop=True)
new_df = pd.merge(new_df, second_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
new_df = new_df.rename(columns={'title': 'rank2', 'title_count': 'rank2_count'})
new_df['first_percentage'] = (new_df['rank1_count'] / new_df['count']).map(lambda x: '{:.2%}'.format(x))
# 将缺失值替换为 0
new_df.fillna(0, inplace=True)
# 输出结果
from IPython.display import display
# display(new_df)



items_per_cluster = 10
COMPLETIONS_MODEL = "text-davinci-003"

openai.api_key = os.environ.get("OPENAI_API_KEY")

for i in range(num_of_clusters):
    cluster_name = new_df[new_df.cluster == i].iloc[0].rank1
    print(f"Cluster {i}, Rank 1: {cluster_name}, Theme:", end=" ")

    content = "\n".join(
        embedding_df[embedding_df.cluster == i].text.sample(items_per_cluster, random_state=42).values
    )
    response = openai.Completion.create(
        model=COMPLETIONS_MODEL,
        prompt=f'''我们想要给下面的内容，分组成有意义的类别，以便我们可以对其进行总结。请根据下面这些内容的共同点，总结一个50个字以内的新闻组的名称。比如 “PC硬件”\n\n内容:\n"""\n{content}\n"""新闻组名称：''',
        temperature=0,
        max_tokens=100,
        top_p=1,
    )
    print(response["choices"][0]["text"].replace("\n", ""))
