
# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.memory import ConversationSummaryBufferMemory

# SUMMARIZER_TEMPLATE = """请将以下内容逐步概括所提供的对话内容，并将新的概括添加到之前的概括中，形成新的概括。

# EXAMPLE
# Current summary:
# Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量。

# New lines of conversation:
# Human：为什么你认为人工智能是一种积极的力量？
# AI：因为人工智能将帮助人类发挥他们的潜能。

# New summary:
# Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量，因为它将帮助人类发挥他们的潜能。
# END OF EXAMPLE

# Current summary:
# {summary}

# New lines of conversation:
# {new_lines}

# New summary:"""

# SUMMARY_PROMPT = PromptTemplate(
#     input_variables=["summary", "new_lines"], template=SUMMARIZER_TEMPLATE
# )

# memory = ConversationSummaryBufferMemory(llm=OpenAI(), prompt=SUMMARY_PROMPT, max_token_limit=40)

# memory.save_context(
#     {"input": "你好"}, 
#     {"ouput": "你好，我是客服李四，有什么我可以帮助您的么"}
#     )

# memory.save_context(
#     {"input": "我叫张三，在你们这里下了一张订单，订单号是 2023ABCD，我的邮箱地址是 customer@abc.com，但是这个订单十几天了还没有收到货"}, 
#     {"ouput": "好的，您稍等，我先为您查询一下您的订单"}
#     )

# llm_chain = LLMChain(
#     llm=OpenAI(), 
#     prompt=SUMMARY_PROMPT, 
#     # 记住上下文
#     memory=memory,
#     verbose=True
# )

# history = memory.load_memory_variables({})
# print(history)


from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

entityMemory = ConversationEntityMemory(llm=OpenAI())

conversation = ConversationChain(
    llm=OpenAI(), 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=entityMemory
)

answer=conversation.predict(input="我叫张老三，在你们这里下了一张订单，订单号是 2023ABCD，我的邮箱地址是 customer@abc.com，但是这个订单十几天了还没有收到货")
# print(answer)


answer=conversation.predict(input="我刚才的订单号是多少？")
print(answer)

