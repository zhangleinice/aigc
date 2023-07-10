# data/20_newsgroup.csv  ==> parquet

from openai.embeddings_utils import get_embeddings
import openai, os, tiktoken, backoff
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def twenty_newsgroup_to_csv():
    # 从sklearn.datasets中获取20个新闻组数据集的训练子集
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    # 创建DataFrame来存储数据
    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    # 创建目标变量DataFrame
    targets = pd.DataFrame(newsgroups_train.target_names, columns=['title'])

    # 将目标变量与文本数据进行合并
    out = pd.merge(df, targets, left_on='target', right_index=True)
    
    # 将数据保存为CSV文件
    out.to_csv('data/20_newsgroup.csv', index=False)
    
# twenty_newsgroup_to_csv()

# 设置OpenAI API密钥
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 设置嵌入模型和编码
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # 这是text-embedding-ada-002的编码方式
batch_size = 200
max_tokens = 8000  # text-embedding-ada-002的最大标记数为8191

# 从CSV文件读取数据
df = pd.read_csv('data/20_newsgroup.csv')

print("过滤空值前的行数:", len(df))

# 过滤掉空值
df = df[df['text'].isnull() == False]

encoding = tiktoken.get_encoding(embedding_encoding)

# 计算每个文本的标记数
df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))

print("过滤标记数前的行数:", len(df))

# 根据最大标记数过滤数据
df = df[df.n_tokens <= max_tokens]

print("使用的数据行数:", len(df))


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    # 使用指数退避策略处理OpenAI API的速率限制错误
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings

# 获取所有文本数据
prompts = df.text.tolist()

# 将文本数据分组为批次
prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

embeddings = []
for batch in prompt_batches:
    # 获取嵌入向量，处理速率限制错误
    batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
    embeddings += batch_embeddings

# 将嵌入向量添加到DataFrame中
df["embedding"] = embeddings

# 将数据保存为parquet格式
df.to_parquet("data/20_newsgroup_with_embedding.parquet", index=False)
