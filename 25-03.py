
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

pipeline = DiffusionPipeline.from_pretrained(
    "Linaqruf/anything-v3.0").to("cuda")

prompt = "masterpiece, best quality, masterpiece, (1girl), <lora:LLCharV2-8:1>, (LLChar), (takami chika:1.1), (orange hair, red eyes), standing, (school uniform, pleated skirt, grey skirt, short sleeves, white shirt, red neckerchief), happy"

negative_prompt = "(EasyNegative:1.4)"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    # steps=32,
    # cfg_scale=7,
    # speed=2411835165,
    # sampler_index='Euler',
    height=512,
    width=512,
).images[0]

# print(image)

# 显示生成的图片
plt.imshow(image)
plt.axis('off')  # 不显示坐标轴

# 保存图片
plt.savefig("./data/sd/25-03.png")  # 保存图片
# # plt.show()
