# 

# llama-index版本更新api变化比较大
# pip3 install --force-reinstall -v "llama-index==0.5.27"

import openai, os
import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTFaissIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser

openai.api_key = os.environ.get("OPENAI_API_KEY")

# 定义文本分割器和节点解析器，用于将文本切割成块，并从文档中提取节点
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
parser = SimpleNodeParser(text_splitter=text_splitter)

# 加载数据，将文档内容读取为节点
documents = SimpleDirectoryReader('./data/faq/').load_data()
nodes = parser.get_nodes_from_documents(documents)

# 定义文本嵌入模型，这里使用了 HuggingFace 的 sentence-transformers/paraphrase-multilingual-mpnet-base-v2 模型，用于将文本转换为语义向量嵌入
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
))

# 创建服务上下文，将文本嵌入模型设置为默认的服务上下文
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# 定义 Faiss 索引，使用 faiss.IndexFlatIP 创建了一个内积（inner product）型的 Faiss 索引对象
dimension = 768
faiss_index = faiss.IndexFlatIP(dimension)

# 创建 GPTFaiss 索引，将节点、Faiss 索引和服务上下文传递给 GPTFaissIndex 类，用于构建索引
index = GPTFaissIndex(nodes=nodes, faiss_index=faiss_index, service_context=service_context)



from llama_index import QueryMode

response = index.query(
    "请问你们海南能发货吗？", 
    mode=QueryMode.EMBEDDING,
    verbose=True, 
)
print(response)