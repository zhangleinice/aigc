from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, tool
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
import json
import re


llm = OpenAI(temperature=0)

loader = TextLoader("./data/faq/ecommerce_faq.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

faq_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)


@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)


# res = faq('如何更改帐户信息')
# print('res', res)



#  商品推荐 llMchain
loader = CSVLoader(file_path='./data/faq/ecommerce_products.csv')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

product_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)


@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)

# res = recommend_product('我想买一件衣服，想要在春天去公园穿，但是不知道哪个款式好看，你能帮我推荐一下吗？')
# print('res', res)


# 订单查询
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

answer_order_info = PromptTemplate(
    template="请把下面的订单信息回复给用户： \n\n {order}?", input_variables=["order"]
)
answer_order_llm = LLMChain(llm=OpenAI(temperature=0),  prompt=answer_order_info)

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

tools = [
    search_order,
    recommend_product,
    faq
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools, 
    # 用OpenAI，不要用ChatOpenAI，会报错 Could not parse LLM output: `{text}`
    llm, 
    agent="conversational-react-description", 
    memory=memory, 
    verbose=True
)

question3 = "你们的退货政策是怎么样的"
answer3 = agent.run(question3)
print(answer3)


