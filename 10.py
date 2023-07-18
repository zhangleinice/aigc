
# import openai, os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader,GPTSimpleVectorIndex

# openai.api_key = os.environ.get("OPENAI_API_KEY")

# documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
# index = GPTSimpleVectorIndex.from_documents(documents)

# index.save_to_disk('./data/index_mr_fujino.json')



index = GPTVectorStoreIndex.load_from_disk('./data/index_mr_fujino.json')
# response = index.query("鲁迅先生在日本学习医学的老师是谁？")
# print(response)


response = index.query("鲁迅先生去哪里学的医学？")
print(response)



# from llama_index import QuestionAnswerPrompt

# 定义查询字符串
# query_str = "鲁迅先生去哪里学的医学？"

# # 定义默认的文本问题回答模板
# DEFAULT_TEXT_QA_PROMPT_TMPL = (
#     "Context information is below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the question: {query_str}\n"
# )

# # 创建QuestionAnswerPrompt实例，传入问题回答模板
# QA_PROMPT = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)

# # 执行查询操作，传入查询字符串和文本问题回答模板
# response = index.query(query_str, text_qa_template=QA_PROMPT)

# # 打印查询结果
# print(response)
