# 使用python3安装
# pip3 install spacy
# python3 -m spacy download zh_core_web_sm

import openai, os
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import SpacyTextSplitter
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

openai.api_key = os.environ.get("OPENAI_API_KEY")


# 导入所需的模块和类

# 定义LLM预测器
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))

# 创建文本分割器
text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=2048)

# 创建节点解析器
parser = SimpleNodeParser(text_splitter=text_splitter)

# 从文档中获取节点
documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
nodes = parser.get_nodes_from_documents(documents)


# 创建服务上下文
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# 创建GPT列表索引
list_index = GPTListIndex(nodes=nodes, service_context=service_context)


query_engine = list_index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("下面鲁迅先生以第一人称‘我’写的内容，请你用中文总结一下:")
print(response)
