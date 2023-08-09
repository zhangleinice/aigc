from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
from diffusers.utils import load_image

# 通过简笔画来画出好看的图片
controlnet = ControlNetModel.from_pretrained(
    # Scribble模型：简笔画
    "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()



def draw_image_grids(images, rows, cols, file_name):
    # 创建 rows x cols 的网格以显示图像
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for row in range(rows):
        for col in range(cols):
            axes[row, col].imshow(images[col + row * cols])
    for ax in axes.flatten():
        ax.axis('off')
    # 显示图像网格
    plt.savefig(f"./data/sd/{file_name}.png")
    plt.show()


image_file = "./data/scribble_dog.png"
scribble_image = load_image(image_file)

generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(4)]
prompt = "dog"
prompt = [prompt + t for t in [" in a room", " near the lake", " on the street", " in the forrest"]]
output = pipe(
    prompt,
    scribble_image,
    negative_prompt=["lowres, bad anatomy, worst quality, low quality"] * 4,
    generator=generator,
    num_inference_steps=50,
)

draw_image_grids(output.images, 2, 2, "25-02-dog")