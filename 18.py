
import os,openai,backoff
import pandas as pd
import subprocess

openai.api_key = os.getenv("OPENAI_API_KEY")

# 数据生成
# openai.api_key = os.getenv("OPENAI_API_KEY")
# dynasties= ['唐', '宋', '元', '明', '清', '汉', '魏', '晋', '南北朝']
# super_powers = ['隐形', '飞行', '读心术', '瞬间移动', '不死之身', '喷火']
# story_types = ['轻松', '努力', '艰难']

# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
# def gpt35(prompt, max_tokens=2048, temperature=0.5, top_p=1, frequency_penalty=0, presence_penalty=0):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         frequency_penalty=frequency_penalty,
#         presence_penalty=presence_penalty)
#     return response["choices"][0]["text"]

# def prepare_stories(dynasties, super_powers, story_types, output_file="data/ultraman_stories.csv"):
#     df = pd.DataFrame()
#     repeat = 3
#     for dynasty in dynasties:
#         for super_power in super_powers:
#             for story_type in story_types:
#                    for i in range(repeat):
#                         prompt = f"""请你用中文写一段300字的故事，情节跌宕起伏，讲述一位{dynasty}朝时期的英雄人物，穿越到现代，拥有了{super_power}这样的超能力，通过{story_type}的战斗，帮助奥特曼一起打败了怪兽的故事。"""
#                         story = gpt35(prompt)
#                         row = {"dynasty": dynasty, "super_power": super_power, "story_type": story_type, "story": story}
#                         row = pd.DataFrame([row])
#                         df = pd.concat([df, row], axis=0, ignore_index=True)

#     df.to_csv("data/ultraman_stories.csv")

# prepare_stories(dynasties, super_powers, story_types)




# 数据处理
# df = pd.read_csv("data/ultraman_stories.csv")

# df['sub_prompt'] = df['dynasty'] + "," + df['super_power'] + "," + df['story_type']

# prepared_data = df.loc[:,['sub_prompt','story']]

# # 用 Prompt 和 Completion 作为列名存储成了一个 CSV
# prepared_data.rename(columns={'sub_prompt':'prompt', 'story':'completion'}, inplace=True)

# prepared_data.to_csv('data/prepared_data.csv',index=False)

# # 把上面的 CSV 文件，转化成了一个 JSONL 格式的文件
# subprocess.run('openai tools fine_tunes.prepare_data --file data/prepared_data.csv --quiet'.split())


# # 通过 subprocess 来提交微调的指令
# subprocess.run('openai api fine_tunes.create --training_file data/prepared_data_prepared.jsonl --model curie --suffix "ultraman"'.split())




# # 定义Fine-tuning命令
# # 我们选用了 Curie 作为基础模型，模型后缀我给它取了一个 ultraman 的名字
# fine_tune_command = "openai api fine_tunes.create --training_file data/prepared_data_prepared.jsonl --model curie --suffix 'ultraman'"

# # 通过 subprocess 来提交微调的指令
# # 由于timeout参数设置为None，命令行将一直运行，直到Fine-tuning任务完成
# process = subprocess.run(fine_tune_command.split(), stdout=subprocess.PIPE, text=True, timeout=None)

# print(process.stdout)



# fine_tune_id = "ft-M3CIjWSOFQlQTKZdEFv08v1X"

# # 查询Fine-tuning任务状态
# response = openai.FineTune.retrieve(id=fine_tune_id)
# print(response)  


# 找出所有我们微调的模型
subprocess.run('openai api fine_tunes.list'.split())


# 使用微调模型
# def write_a_story(prompt):
#     response = openai.Completion.create(
#         model="curie",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=2000,
#         top_p=1,
#         stop=["."])
#     return response["choices"][0]["text"]

# story = write_a_story("宋,发射激光,艰难 ->\n")
# print(story)