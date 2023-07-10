
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

history = """User : 你是谁？
Assistant : 我是一个人工智能语言模型，可以回答各种问题，包括做菜的问题。

User : 请问鱼香肉丝怎么做？
Assistant : 鱼香肉丝的做法：

1. 猪里脊肉切丝，加入盐、淀粉、料酒搅拌均匀，腌制10分钟
2. 胡萝卜、青椒、木耳切丝备用
3. 热锅凉油，放入蒜末、姜末煸炒出香味，加入肉丝翻炒至变色
4. 加入胡萝卜、青椒、木耳翻炒，加入酱油、糖、醋、盐、味精、水淀粉搅拌均匀，倒入锅中翻炒均匀即可。

提示：鱼香肉丝的关键是调料的配比，酱油、糖、醋、盐、味精的比例要适合自己的口味。

User : 那蚝油牛肉呢？
Assistant : 蚝油牛肉的做法：

1. 将牛肉切成薄片，加入盐、淀粉、料酒腌制10分钟
2. 葱姜切末，蒜切片，青红椒切丝备用
3. 热锅凉油，放入葱姜末、蒜片煸炒出香味，加入牛肉翻炒至变色
4. 加入蚝油、酱油、糖、水淀粉调味，加入青红椒翻炒均匀即可。

提示：蚝油牛肉的关键是牛肉要切薄，翻炒时火候要快，保证牛肉口感鲜嫩。调味时，蚝油和酱油的比例也要适合自己的口味。
"""


class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]
        # 通过 API 计算 Token 数量
        # num_of_tokens = response['usage']['total_tokens']
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            del self.messages[1:3]  # 删除第一轮对话的部分内容
        return message

def summarize(text, max_tokens=200):
    response = openai.Completion.create(
        model=COMPLETION_MODEL,
        prompt=text + "\n\n请总结一下上面User和Assistant聊了些什么：\n",
        max_tokens=max_tokens,
    )
    return response["choices"][0]["text"]

summarized = summarize(history)
print(summarized)


prompt = summarized + "\n\n请你根据已经聊了的内容，继续对话："
conversation = Conversation(prompt, 5)

question = "那宫保鸡丁呢？"
answer = conversation.ask(question)
print("User : %s" % question)
print("Assistant : %s\n" % answer)