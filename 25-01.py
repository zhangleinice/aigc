# 通过“动态捕捉”来画人物图片


# 1.通过 OpenposeDetector 先捕捉一下图片里面的人物姿势
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import matplotlib.pyplot as plt

# 从预训练模型加载 OpenposeDetector
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# 加载第一张图片并使用 Openpose 检测器进行处理
image_file1 = "./data/rodin.jpg"
original_image1 = load_image(image_file1)
openpose_image1 = openpose(original_image1)

# 加载第二张图片并使用 Openpose 检测器进行处理
image_file2 = "./data/discobolos.jpg"
original_image2 = load_image(image_file2)
openpose_image2 = openpose(original_image2)

# 将原始图片和 Openpose 处理后的图片放入列表
images = [original_image1, openpose_image1, original_image2, openpose_image2]

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

# 调用函数在网格中显示图片
# draw_image_grids(images, 2, 2, "posture")



# 2.加载模型
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# 3.基于这些姿势来画画
poses = [openpose_image1, openpose_image2, openpose_image1, openpose_image2]

generator = [torch.Generator(device="cpu").manual_seed(42) for i in range(4)]
prompt1 = "batman character, best quality, extremely detailed"
prompt2 = "ironman character, best quality, extremely detailed"

output = pipe(
    [prompt1, prompt1, prompt2, prompt2],
    poses,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    generator=generator,
    num_inference_steps=20,
)

draw_image_grids(output.images, 2, 2, "superheroes")



