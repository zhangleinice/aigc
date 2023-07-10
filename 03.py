# 智能客服

import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2021AEDG，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'

def get_response(prompt, temperature = 1.0):
    completions = openai.Completion.create (
        # 引擎选择
        engine=COMPLETION_MODEL,
        prompt=prompt,
        # 输入和返回最多多少个token
        max_tokens=1024,
        # 生成几条数据供选择
        n=1,
        # 希望模型输出的内容在遇到什么内容的时候就停下来
        stop=None,
        # 输入范围是 0-2 之间的浮点数，代表输出结果的随机性或者说多样性
        # 也就是还是让每次生成的内容都有些不一样。你也可以把这个参数设置为 0，这样，每次输出的结果的随机性就会比较小。
        # 这个参数该怎么设置，取决于实际使用的场景。如果对应的场景比较严肃，不希望出现差错，那么设得低一点比较合适，比如银行客服的场景。如果场景没那么严肃，有趣更加重要，比如讲笑话的机器人，那么就可以设置得高一些。
        temperature=temperature,
    )
    message = completions.choices[0].text
    print(completions)
    return message


# print(get_response(prompt))


question =  """
Q : 鱼香肉丝怎么做？
A : 
"""
print(get_response(question))
    