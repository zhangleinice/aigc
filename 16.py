
import openai, os

openai.api_key = os.environ.get("OPENAI_API_KEY")

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template
)

# 记住三轮对话
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)

memory.load_memory_variables({})

llm_chain = LLMChain(
    llm=OpenAI(), 
    prompt=prompt, 
    # 记住上下文
    memory=memory,
    verbose=True
)
llm_chain.predict(human_input="你是谁？")


llm_chain.predict(human_input="鱼香肉丝怎么做？")
llm_chain.predict(human_input="那宫保鸡丁呢？")
llm_chain.predict(human_input="我问你的第一句话是什么？")
