# print('hello world')

# list = [1,2,3,4,5]
# print(list[1:2])


import spacy

# 加载英语语言模型
nlp = spacy.load('en')

# 处理文本
doc = nlp("这是一段示例文本。")
for token in doc:
    print(token.text)
