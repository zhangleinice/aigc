
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

import openai, os
openai.api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(temperature=0)

# ai总结上下文
memory = ConversationSummaryMemory(llm=OpenAI())

prompt_template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内

{history}
Human: {input}
AI:"""


prompt = PromptTemplate(
    input_variables=["history", "input"], template=prompt_template
)
conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=memory,
    prompt=prompt,
    verbose=True
)

answer = conversation_with_summary.predict(input="你好")
print(answer)
