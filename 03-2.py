
# Few-Shots Learning（少样本学习）,举一反三

import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

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
    return message


prompts = """判断一下用户的评论情感上是正面的还是负面的
评论：买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质
情感：正面

评论：随意降价，不予价保，服务态度差
情感：负面
"""

good_case = prompts + """
评论：外形外观：苹果审美一直很好，金色非常漂亮
拍照效果：14pro升级的4800万像素真的是没的说，太好了，
运行速度：苹果的反应速度好，用上三五年也不会卡顿的，之前的7P用到现在也不卡
其他特色：14pro的磨砂金真的太好看了，不太高调，也不至于没有特点，非常耐看，很好的
情感：
"""


bad_case = prompts + """
评论：信号不好电池也不耐电不推荐购买
情感
"""

print(get_response(bad_case))

print(get_response(good_case))