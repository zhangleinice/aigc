import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

openai.api_key = os.environ.get("OPENAI_API_KEY")

# 初始化 OpenAI 模型
llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

# 定义问题翻译成英文的模板
en_to_zh_prompt = PromptTemplate(
    template="请把下面这句话翻译成英文： \n\n {question}?", input_variables=["question"]
)

# 定义问题的模板
question_prompt = PromptTemplate(
    template="{english_question}", input_variables=["english_question"]
)

# 定义答案翻译成中文的模板
zh_to_cn_prompt = PromptTemplate(
    input_variables=["english_answer"],
    template="请把下面这一段翻译成中文： \n\n{english_answer}?",
)

# 使用 LLMChain 执行问题翻译成英文
question_translate_chain = LLMChain(llm=llm, prompt=en_to_zh_prompt, output_key="english_question")
# english = question_translate_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
# print(english)

# 使用 LLMChain 执行问题的回答获取
qa_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="english_answer")
# english_answer = qa_chain.run(english_question=english)
# print(english_answer)

# 使用 LLMChain 执行答案翻译成中文
answer_translate_chain = LLMChain(llm=llm, prompt=zh_to_cn_prompt)
# answer = answer_translate_chain.run(english_answer=english_answer)
# print(answer)

# 使用链式调用

from langchain.chains import SimpleSequentialChain

# 创建简单顺序链，包含问题翻译成英文、问题回答获取和答案翻译成中文的链条
chinese_qa_chain = SimpleSequentialChain(
    chains=[question_translate_chain, qa_chain, answer_translate_chain], input_key="question",
    verbose=True)

# 运行链式调用来完成整个问题和回答的过程
answer = chinese_qa_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
print(answer)
