
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

docsearch = FAISS.from_documents(texts, embeddings)

# 替换原来的 VectorDBQA 为 RetrievalQA
faq_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=docsearch, verbose=True)


#  商品推荐 llMchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader

product_loader = CSVLoader('./data/faq/ecommerce_products.csv')
product_documents = product_loader.load()
product_text_splitter = CharacterTextSplitter(chunk_size=1024, separator="\n")
product_texts = product_text_splitter.split_documents(product_documents)
product_search = FAISS.from_documents(product_texts, OpenAIEmbeddings())
# 替换原来的 VectorDBQA 为 RetrievalQA
product_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=product_search, verbose=True)


# 订单查询
import json

ORDER_1 = "20230101ABC"
ORDER_2 = "20230101EFG"

ORDER_1_DETAIL = {
    "order_number": ORDER_1,
    "status": "已发货",
    "shipping_date" : "2023-01-03",
    "estimated_delivered_date": "2023-01-05",
} 

ORDER_2_DETAIL = {
    "order_number": ORDER_2,
    "status": "未发货",
    "shipping_date" : None,
    "estimated_delivered_date": None,
}


import re

answer_order_info = PromptTemplate(
    template="请把下面的订单信息回复给用户： \n\n {order}?", input_variables=["order"]
)
answer_order_llm = LLMChain(llm=ChatOpenAI(temperature=0),  prompt=answer_order_info)

# return_direct=True：不要再经过 Thought 那一步思考，直接把我们的回答给到用户就好了
# 设了这个参数之后，你就会发现 AI 不会在没有得到一个订单号的时候继续去反复思考，尝试使用工具，而是会直接去询问用户的订单号
@tool("Search Order", return_direct=True)
def search_order(input:str)->str:

    # 加提示语：找不到订单的时候，防止重复调用 OpenAI 的思考策略，来敷衍用户

    """useful for when you need to answer questions about customers orders"""
    pattern = r"\d+[A-Z]+"
    match = re.search(pattern, input)

    order_number = input
    if match:
        order_number = match.group(0)
    else:
        return "请问您的订单号是多少？"
    if order_number == ORDER_1:        
        return answer_order_llm.run(json.dumps(ORDER_1_DETAIL))
    elif order_number == ORDER_2:
        return answer_order_llm.run(json.dumps(ORDER_2_DETAIL))
    else:
        return f"对不起，根据{input}没有找到您的订单"

@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)

@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)


tools = [
    search_order,
    recommend_product,
    faq
]

chatllm=ChatOpenAI(temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# conversational-react-description： 支持多轮对话
conversation_agent = initialize_agent(
    tools, 
    chatllm, 
    agent="conversational-react-description", 
    memory=memory, 
    verbose=True
)

# zero-shot-react-description：零样本
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# question = "请问你们的货，能送到三亚吗？大概需要几天？"
# result = conversation_agent.run(question)
# print(result)


# question = "我想买一件衣服，想要在春天去公园穿，但是不知道哪个款式好看，你能帮我推荐一下吗？"
# answer = conversation_agent.run(question)
# print(answer)


# question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
# answer = conversation_agent.run(question)
# print(answer)


# question1 = "我有一张订单，一直没有收到，能麻烦帮我查一下吗？"
# answer1 = conversation_agent.run(question1)
# print(answer1)


# question2 = "我的订单号是20230101ABC"
# answer2 = conversation_agent.run(question2)
# print(answer2)


question3 = "你们的退货政策是怎么样的？"
answer3 = conversation_agent.run(question3)
print(answer3)







