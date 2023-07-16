
# 利用开源模型进行 FAQ 问答

# Langchain（语言链）是一个用于处理文本生成任务的工具库。它基于 OpenAI GPT 系列模型和 Faiss 索引库，提供了一套高效的文本检索和生成解决方案。

import openai, os
import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTFaissIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser

from langchain.llms.base import LLM
from llama_index import LLMPredictor
from typing import Optional, List, Mapping, Any

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

# 自定义的 LLM 类，用于处理文本生成任务
class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, history = model.chat(tokenizer, prompt, history=[])
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": "chatglm-6b-int4"}

    @property
    def _llm_type(self) -> str:
        return "custom"


from langchain.text_splitter import SpacyTextSplitter

# 创建 LLMPredictor，使用自定义的 LLM
llm_predictor = LLMPredictor(llm=CustomLLM())

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

# 创建服务上下文，将文本嵌入模型设置为默认的服务上下文，同时也传入自定义的 LLMPredictor
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)

# 定义 Faiss 索引，使用 faiss.IndexFlatIP 创建了一个内积（inner product）型的 Faiss 索引对象
dimension = 768
faiss_index = faiss.IndexFlatIP(dimension)

# 创建 GPTFaiss 索引，将节点、Faiss 索引和服务上下文传递给 GPTFaissIndex 类，用于构建索引
index = GPTFaissIndex(nodes=nodes, faiss_index=faiss_index, service_context=service_context)

from llama_index import QuestionAnswerPrompt
from llama_index import QueryMode

# 定义问题回答模板
QA_PROMPT_TMPL = (
    "{context_str}"
    "\n\n"
    "根据以上信息，请回答下面的问题：\n"
    "Q: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

# 执行查询操作，传入查询字符串、查询模式和问题回答模板
response = index.query(
    "请问你们海南能发货吗？", 
    mode=QueryMode.EMBEDDING,
    text_qa_template=QA_PROMPT,
    verbose=True, 
)
print(response)
