
# 大批量数据转embedding处理
# 并采用 parquet 这个序列化的格式存储

import pandas as pd
import tiktoken
import openai
import os
import backoff
from openai.embeddings_utils import get_embedding, get_embeddings

openai.api_key = os.environ.get("OPENAI_API_KEY")

# embedding 模型参数
embedding_model = "text-embedding-ada-002"

# embedding 编码方式
embedding_encoding = "cl100k_base"

# text-embedding-ada-002 的最大标记数量为 8191
max_tokens = 8000

# 从 CSV 文件读取数据并进行预处理
df = pd.read_csv('data/test.txt', sep='_!_', names=['id', 'code', 'category', 'title', 'keywords'], engine='python')
df = df.fillna("")
df["combined"] = "标题: " + df.title.str.strip() + "; 关键字: " + df.keywords.str.strip()

print("过滤前的文本行数：", len(df))

# 获取编码器
encoding = tiktoken.get_encoding(embedding_encoding)

# 过滤超过最大标记数量的文本行
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens]

print("过滤后的文本行数：", len(df))

# 批量处理参数
batch_size = 200

# 使用 backoff 库处理速率限制错误
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings

# 对随机抽样的数据进行处理
df_all = df
prompts = df_all.combined.tolist()
prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

embeddings = []
for batch in prompt_batches:
    batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
    embeddings += batch_embeddings

df_all["embedding"] = embeddings

# 以 parquet 格式保存处理后的数据
df_all.to_parquet("data/test_all_with_embeddings.parquet", index=True)

print('finished')
