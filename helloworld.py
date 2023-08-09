# # print('hello world')

# list = [1,2,3,4,5]
# print(list[1:2])


# # import spacy

# # # 加载英语语言模型
# # nlp = spacy.load('en')

# # # 处理文本
# # doc = nlp("这是一段示例文本。")
# # for token in doc:
# #     print(token.text)

# from pathlib import Path

# # 使用 Path 对象创建文件
# filename = "example.txt"

# # 使用 Path.touch() 方法创建文件，如果文件不存在则会创建新文件
# path = Path(filename)
# path.touch()

# # 使用 Path.write_text() 方法写入内容
# content = "这是一个示例文件。\nHello, world!\nPython 文件创建示例。\n"
# path.write_text(content)

import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单的图像
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 显示图像
plt.plot(x, y)
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('简单的Matplotlib示例')


# 保存图像到文件
plt.savefig("./data/simple_plot.png")

plt.show()
