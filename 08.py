import tiktoken
import openai
import os

# 设置OpenAI的API密钥
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 指定使用的模型
COMPLETION_MODEL = "text-davinci-003"

# 获取编码器
encoding = tiktoken.get_encoding('p50k_base')

# 对关键词"灾害"进行编码
token_ids = encoding.encode("灾害")

print(token_ids)

# 长文本输入
long_text = """在这个快节奏的现代社会中，我们每个人都面临着各种各样的挑战和困难。在这些挑战和困难中，有些是由外部因素引起的，例如经济萧条、全球变暖和自然灾害等。还有一些是由内部因素引起的，例如情感问题、健康问题和自我怀疑等。面对这些挑战和困难，我们需要采取积极的态度和行动来克服它们。这意味着我们必须具备坚韧不拔的意志和创造性思维，以及寻求外部支持的能力。只有这样，我们才能真正地实现自己的潜力并取得成功。"""

# # 偏置映射字典
# bias_map = {}
# for token_id in token_ids:
#     bias_map[token_id] = -100

# # 将文本改写为短小精悍的版本
# def make_text_short(text):
#     messages = []
#     messages.append({"role": "system", "content": "你是一个用来将文本改写得短的AI助手，用户输入一段文本，你给出一段意思相同，但是短小精悍的结果"})
#     messages.append({"role": "user", "content": text})
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=0.5,
#         max_tokens=2048,
#         n=3,
#         presence_penalty=0,
#         frequency_penalty=2,
#         logit_bias=bias_map,
#     )
#     return response

# # 生成短小精悍的版本
# short_version = make_text_short(long_text)

# index = 1
# for choice in short_version["choices"]:
#     print(f"version {index}: " + choice["message"]["content"])
#     index += 1



def translate(text):
    messages = []
    messages.append( {"role": "system", "content": "你是一个翻译，把用户的话翻译成英文"})
    messages.append( {"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.5, max_tokens=2048,        n=1
    )
    return response["choices"][0]["message"]["content"]

chinese = long_text
english = translate(chinese)

num_of_tokens_in_chinese = len(encoding.encode(chinese))
num_of_tokens_in_english = len(encoding.encode(english))
print(english)
print(f"chinese: {num_of_tokens_in_chinese} tokens")
print(f"english: {num_of_tokens_in_english} tokens")