import re
import json
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent, tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#  VectorDBQA 更新为 RetrievalQA
from langchain.chains import RetrievalQA

llm = OpenAI(temperature=0)

# 问答llmchain
loader = TextLoader('./data/faq/ecommerce_faq.txt')

documents = loader.load()

text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
print('embeddings', embeddings)

docsearch = FAISS.from_documents(texts, embeddings)

# 替换原来的 VectorDBQA 为 RetrievalQA
faq_chain = VectorDBQA.from_chain_type(
    llm=llm, vectorstore=docsearch, verbose=True, 
    # 简单搜索
    search_type="similarity"
)

@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)


res = faq('如何更改帐户信息')
print('res', res)



# # 定义问题翻译成英文的模板
# en_to_zh_prompt = PromptTemplate(
#     template="请把下面这句话翻译成英文： \n\n {question}?", input_variables=["question"]
# )


# # 使用 LLMChain 执行问题翻译成英文
# question_translate_chain = LLMChain(llm=llm, prompt=en_to_zh_prompt, output_key="english_question")
# english = question_translate_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
# # print(english)


# prompt = PromptTemplate(
#     template="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc. \n\n {question}?", input_variables=["question"]
# )

# faq1_chain = VectorDBQA.from_chain_type(
#     llm=llm, vectorstore=docsearch, verbose=True, chain_type_kwargs={'prompt': prompt})

# res1 = faq1_chain.run(question='如何更改帐户信息')
# print('res1', res1)


# template = """Use the following pieces of context to answer the question at the end. If the answer can't be determined using only the information in the provided context simply output "NO ANSWER", just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)




